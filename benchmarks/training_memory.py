"""Training time, peak memory and accuracy by solver vs N (Fig. S3).

Trains on nested subsets of the Beer lab GM12878 sequence sets (300bp DHS vs
null set 1, -t 2 l=10 k=6 d=3, C=1: the GkmExplain paper's -t 2 settings) with:

  libsvm   precomputed N x N Gram in host RAM (tiles computed on --device) +
           sklearn's LIBSVM solver. What solver="auto" uses while the Gram
           fits in 75% of RAM.
  smo      column-cached SMO, O(cache_size x N) memory. auto's fallback.
  nystrom  low-rank Nystrom approximation (approximate).
  lsgkm    LS-GKM gkmtrain at -T --threads, scored with gkmpredict.

Every run is a separate process, so peak_rss_mb is that run's own peak.
Held-out accuracy uses --n-test positives and negatives excluded from every
training set. r_vs_exact is the Pearson correlation of held-out decision
values with the exact gkmsvm-lite model at the same N (libsvm, else smo).
libsvm runs whose Gram would not fit in RAM are recorded as "skipped".

--data ctcf (Fig. S3B) trains on the H1-hESC CTCF sets from ctcf_data.py
instead, LS-GKM's own scaling data (Lee 2016, Fig. 1: 5k-90k), with test
sequences from the held-out chr1/chr2. --lsgkm-caches runs gkmtrain at each
kernel cache size (-m, MB), as that study did with 100 MB and 8 GB.

Usage:
  python benchmarks/training_memory.py --device cuda
  python benchmarks/training_memory.py --device cuda --sizes 1000,5000,max
  python benchmarks/training_memory.py --device cpu --sizes 500,1000 --no-lsgkm
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from _common import (
    LSGKM_THREADS,
    RESULTS,
    SEQ_DIR,
    emit_result,
    gpu_pool_mb,
    gpu_sync,
    lsgkm_binary,
    lsgkm_default_threads,
    lsgkm_has_option,
    lsgkm_sources,
    merge_parts,
    read_seqs,
    rewrite_table,
    run_measured,
    set_cpu_threads,
    warm_binaries,
    warm_jit_cache,
    worker_result,
    write_seqs,
    write_tsv,
)

SOLVERS = ("libsvm", "smo", "nystrom", "lsgkm")
DEFAULT_SIZES = "1000,2000,5000,10000,20000,30000,max"


# ── gkmsvm-lite worker (child process) ─────────────────────────────


def _worker(spec: dict) -> None:
    from gkmsvm import one_hot_encode, train_gkmsvm
    from gkmsvm.backend import to_cpu

    work = Path(spec["work"])
    pos = read_seqs(work / f"pos_{spec['N']}.fa")
    neg = read_seqs(work / f"neg_{spec['N']}.fa")
    test = read_seqs(work / "test.fa")
    params = dict(
        kernel_type="estimated",
        l=spec["l"],
        k=spec["k"],
        d=spec["d"],
        C=spec["C"],
        solver=spec["solver"],
        cache_size=spec["cache_size"],
        device=spec["device"],
        verbose=False,
    )
    # Untimed: load this solver's kernels from the JIT caches (warm_jit_cache
    # compiled them) and start CUDA, so train_s holds training alone
    train_gkmsvm(pos[:50], neg[:50], **params)
    gpu_sync()

    t0 = time.perf_counter()
    model = train_gkmsvm(pos, neg, **params)
    gpu_sync()
    train_s = time.perf_counter() - t0
    pool = gpu_pool_mb()

    X = np.stack([one_hot_encode(s) for s in test])
    if spec["device"] == "cuda":
        model.cuda()
    scores = to_cpu(model(X, batch_size=256)).ravel()
    np.save(work / f"scores_{spec['solver']}_{spec['N']}.npy", scores)
    emit_result(train_s=train_s, n_sv=int(model.num_support_vectors), gpu_pool_mb=pool)


def _merge(args) -> None:
    """Concatenate two parts and fill r_vs_exact for rows whose part had no
    exact model (libsvm, else smo) at their N, from the scores in --arrays."""
    import pandas as pd

    out = merge_parts(dict(zip(("cpu", "gpu"), args.merge_parts)), args.output)
    if not args.arrays:
        return
    arrays = Path(args.arrays)
    df = pd.read_csv(out, sep="\t", comment="#")
    for i, r in df.iterrows():
        if r["status"] != "ok" or pd.notna(r.get("r_vs_exact")):
            continue
        ours = arrays / f"scores_{r['run']}_{r['N']}.npy"
        exact = next(
            (
                p
                for p in (arrays / f"scores_{s}_{r['N']}.npy" for s in ("libsvm", "smo"))
                if p.exists()
            ),
            None,
        )
        if ours.exists() and exact is not None:
            df.loc[i, "r_vs_exact"] = float(
                np.corrcoef(np.load(ours), np.load(exact))[0, 1]
            )
    rewrite_table(out, df.astype({c: "Int64" for c in ("n_sv", "threads") if c in df}))


# ── Main ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device for gkmsvm-lite kernel computation",
    )
    parser.add_argument(
        "--sizes",
        default=DEFAULT_SIZES,
        help="Comma-separated training set sizes (total N, balanced); "
        "'max' = all sequences not held out",
    )
    parser.add_argument(
        "--solvers",
        default=",".join(SOLVERS),
        help=f"Comma-separated subset of {', '.join(SOLVERS)}",
    )
    parser.add_argument(
        "--data",
        default="gm12878",
        choices=["gm12878", "ctcf"],
        help="gm12878: Beer lab DHS sets (S3); ctcf: ctcf_data.py (S3B)",
    )
    parser.add_argument(
        "--lsgkm-caches",
        default="100",
        help="Comma-separated gkmtrain kernel cache sizes in MB (-m; 100 is "
        "LS-GKM's default)",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=500,
        help="Held-out positives and negatives each (default: 500)",
    )
    parser.add_argument("-l", type=int, default=10)
    parser.add_argument("-k", type=int, default=6)
    parser.add_argument("-d", type=int, default=3)
    parser.add_argument("-C", type=float, default=1.0)
    parser.add_argument(
        "--cache-size", type=int, default=2048, help="SMO kernel-column cache (columns)"
    )
    parser.add_argument("--conda-env", default="lsgkm")
    parser.add_argument(
        "--lsgkm-bin",
        default=None,
        help="Directory with LS-GKM binaries the conda env lacks",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=lsgkm_default_threads(),
        choices=LSGKM_THREADS,
        help="LS-GKM -T and gkmsvm-lite CPU threads",
    )
    parser.add_argument("--no-lsgkm", action="store_true")
    parser.add_argument(
        "--no-warm-jit",
        action="store_true",
        help="Skip compiling kernels before the timed runs (for smoke tests; "
        "the first run of each kernel then includes compiling it)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=6 * 3600,
        help="Per-run timeout in seconds (default: 6 h)",
    )
    parser.add_argument(
        "--arrays",
        default=None,
        help="Directory to save held-out scores in, so --merge-parts can "
        "compare one part's runs with the other's exact model",
    )
    parser.add_argument(
        "--merge-parts",
        nargs=2,
        metavar=("PART_TSV", "PART_TSV"),
        help="Write --output from two parts (e.g. LS-GKM on CPU, gkmsvm-lite "
        "on GPU), filling r_vs_exact from --arrays",
    )
    parser.add_argument("--output", default=str(RESULTS / "training_memory.tsv"))
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        _worker(json.loads(args.worker))
        return
    if args.merge_parts:
        _merge(args)
        return
    # Workers inherit this through the environment
    lite_threads = set_cpu_threads(args.threads)
    if not args.no_warm_jit:
        warm_jit_cache(args.device)

    solvers = [
        s for s in args.solvers.split(",") if not (args.no_lsgkm and s == "lsgkm")
    ]
    bad = [s for s in solvers if s not in SOLVERS]
    if bad:
        parser.error(f"unknown solver(s): {', '.join(bad)}")

    bins = {}
    if "lsgkm" in solvers:
        for name in ("gkmtrain", "gkmpredict"):
            bins[name] = lsgkm_binary(name, args.conda_env, args.lsgkm_bin)
        if not all(bins.values()):
            sys.exit(
                f"gkmtrain/gkmpredict not found in "
                f"{args.lsgkm_bin or 'conda env ' + args.conda_env}; use --no-lsgkm "
                "to skip LS-GKM"
            )
        predict_threads = (
            args.threads if lsgkm_has_option(bins["gkmpredict"], "-T") else None
        )
        # Untimed: page in the binaries before their first timed run
        warm_binaries(*bins.values())

    from gkmsvm.train import _gram_fits_on_cpu
    from sklearn.metrics import roc_auc_score

    import dsqtl_auprc as bench

    rng = np.random.default_rng(0)

    def shuffled(path):
        seqs = read_seqs(path)
        return [seqs[i] for i in rng.permutation(len(seqs))]

    if args.data == "ctcf":
        from ctcf_data import OUT as CTCF

        if not (CTCF / "train_pos.fa").exists():
            sys.exit("CTCF sets missing: run python benchmarks/ctcf_data.py")
        pos, neg = shuffled(CTCF / "train_pos.fa"), shuffled(CTCF / "train_neg.fa")
        test_pos = shuffled(CTCF / "test_pos.fa")[: args.n_test]
        test_neg = shuffled(CTCF / "test_neg.fa")[: args.n_test]
    else:
        bench.download_sequence_sets()
        pos = shuffled(SEQ_DIR / "gm12878_shared.fa")
        neg = shuffled(SEQ_DIR / "nullseqs_gm12878_shared.1.1.fa")
        test_pos, pos = pos[: args.n_test], pos[args.n_test :]
        test_neg, neg = neg[: args.n_test], neg[args.n_test :]
    caches = [float(c) for c in args.lsgkm_caches.split(",")]
    # One run per (solver, LS-GKM cache)
    runs = [(s, c) for s in solvers for c in (caches if s == "lsgkm" else [None])]
    n_max = 2 * min(len(pos), len(neg))
    sizes = sorted(
        {n_max if s == "max" else min(int(s), n_max) for s in args.sizes.split(",")}
    )
    y_test = np.r_[np.ones(args.n_test), np.zeros(args.n_test)]
    print(f"Sizes: {sizes}  (max {n_max:,}), held out {2 * args.n_test:,}")

    work = Path(tempfile.mkdtemp(prefix="training_memory_"))
    write_seqs(work / "test.fa", test_pos + test_neg, "t")

    rows = []
    for N in sizes:
        half = N // 2
        write_seqs(work / f"pos_{N}.fa", pos[:half], "p")
        write_seqs(work / f"neg_{N}.fa", neg[:half], "n")
        gram_gb = N * N * 8 / 1024**3
        print(f"\n── N = {N:,} (Gram {gram_gb:.1f} GB) ──")
        scores = {}
        for solver, cache in runs:
            name = solver if cache is None or len(caches) == 1 else f"lsgkm_{cache:g}"
            row = {
                "N": N,
                "solver": solver,
                "gram_gb": gram_gb,
                "device": "cpu" if solver == "lsgkm" else args.device,
                "train_s": None,
                "gpu_pool_mb": None,
                "n_sv": None,
                "threads": lite_threads if args.device == "cpu" else None,
                "lsgkm_cache_mb": cache,
            }
            # Both build a dense N x N (approximate, for Nystrom) Gram in host RAM
            if solver in ("libsvm", "nystrom") and not _gram_fits_on_cpu(N):
                rows.append(
                    {**row, "status": "skipped", "wall_s": None, "peak_rss_mb": None}
                )
                print(f"  {solver}: skipped, Gram exceeds 75% of RAM")
                continue
            if solver == "lsgkm":
                prefix = work / f"lsgkm_{N}_{cache:g}"
                m = run_measured(
                    [
                        bins["gkmtrain"],
                        "-t",
                        "2",
                        "-l",
                        str(args.l),
                        "-k",
                        str(args.k),
                        "-d",
                        str(args.d),
                        "-c",
                        str(args.C),
                        "-T",
                        str(args.threads),
                        "-m",
                        f"{cache:g}",
                        str(work / f"pos_{N}.fa"),
                        str(work / f"neg_{N}.fa"),
                        str(prefix),
                    ],
                    timeout=args.timeout,
                )
                row.update(train_s=m.wall_s, threads=args.threads)
                if m.status == "ok":
                    model_file = Path(f"{prefix}.model.txt")
                    with open(model_file) as f:
                        nr_sv = next(line for line in f if line.startswith("nr_sv"))
                    row["n_sv"] = sum(int(v) for v in nr_sv.split()[1:])
                    cmd = [bins["gkmpredict"]]
                    if predict_threads:
                        cmd += ["-T", str(predict_threads)]
                    out = work / f"lsgkm_{N}_pred.txt"
                    p = run_measured(
                        cmd + [str(work / "test.fa"), str(model_file), str(out)]
                    )
                    if p.status == "ok":
                        with open(out) as f:
                            scores[name] = np.array(
                                [float(x.split("\t")[1]) for x in f]
                            )
            else:
                spec = {
                    "solver": solver,
                    "N": N,
                    "work": str(work),
                    "device": args.device,
                    "l": args.l,
                    "k": args.k,
                    "d": args.d,
                    "C": args.C,
                    "cache_size": args.cache_size,
                }
                m = run_measured(
                    [sys.executable, __file__, "--worker", json.dumps(spec)],
                    timeout=args.timeout,
                )
                res = worker_result(m.stdout) or {}
                row.update(
                    train_s=res.get("train_s"),
                    n_sv=res.get("n_sv"),
                    gpu_pool_mb=res.get("gpu_pool_mb"),
                )
                if m.status == "ok":
                    scores[solver] = np.load(work / f"scores_{solver}_{N}.npy")
            if m.status != "ok":
                print(f"  {name} {m.status}: {m.stderr[-300:]}")
            row.update(
                status=m.status, wall_s=m.wall_s, peak_rss_mb=m.peak_rss_mb, run=name
            )
            rows.append(row)

        if args.arrays:
            Path(args.arrays).mkdir(parents=True, exist_ok=True)
            for name, s in scores.items():
                np.save(Path(args.arrays) / f"scores_{name}_{N}.npy", s)
        exact = scores.get("libsvm", scores.get("smo"))
        for row in rows:
            if row["N"] != N or row.get("run") not in scores:
                continue
            s = scores[row["run"]]
            row["test_auroc"] = float(roc_auc_score(y_test, s))
            row["r_vs_exact"] = (
                float(np.corrcoef(s, exact)[0, 1]) if exact is not None else None
            )
        for row in (r for r in rows if r["N"] == N):
            ts = f"{row['train_s']:.1f}s" if row.get("train_s") else "—"
            mem = f"{row['peak_rss_mb']:.0f} MB" if row.get("peak_rss_mb") else "—"
            auc = f"AUROC {row['test_auroc']:.4f}" if row.get("test_auroc") else ""
            r = f"r {row['r_vs_exact']:.5f}" if row.get("r_vs_exact") else ""
            print(
                f"  {row.get('run', row['solver']):<11s} {ts:>9s}  peak {mem:>9s}  "
                f"SVs {row['n_sv'] or '—'}  {auc}  {r}  [{row['status']}]"
            )

    import pandas as pd

    df = pd.DataFrame(rows)
    df = df.astype({c: "Int64" for c in ("n_sv", "threads") if c in df})
    write_tsv(
        args.output,
        df,
        "Training time and memory by solver",
        args.device,
        kernel=f"-t 2 -l {args.l} -k {args.k} -d {args.d} -c {args.C}",
        smo_cache_columns=args.cache_size,
        held_out=2 * args.n_test,
        data=args.data,
        lsgkm_caches_mb=args.lsgkm_caches if "lsgkm" in solvers else None,
        lite_cpu_threads=lite_threads,
        lsgkm=None
        if "lsgkm" not in solvers
        else lsgkm_sources(bins, args.conda_env),
    )
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
