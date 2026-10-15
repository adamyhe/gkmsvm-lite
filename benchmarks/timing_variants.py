"""Time to score the dsQTL variant set, gkmsvm-lite vs LS-GKM (Fig. 1C).

Uses one of the GkmExplain paper's lsgkm models (default -t 3 gkmrbf, negset 1)
and the 28,309 dsQTL variants, timing the three scoring methods of Fig. 1D:

  ISM         LS-GKM: gkmpredict on the 51bp ref and alt FASTAs (two runs)
              gkmsvm-lite: score_variants(method="kernel")
  GkmExplain  LS-GKM: gkmexplain -m 5 on the 51bp ref FASTA. It reports the
              impact of all three substitutions at the center base; the one
              matching the alt allele is the variant score.
              gkmsvm-lite: score_variants(method="gkmexplain"), alt allele only
  deltaSVM    Building the 10-mer weight table.
              LS-GKM: gkmpredict on the 524,800 non-redundant 10-mers (as the
              paper's pipeline does with nrkmers.py)
              gkmsvm-lite: to_deltasvm(query_norm=True), all 4^10 10-mers

Every run is a separate process. wall_s includes model loading and, for
gkmsvm-lite, JIT compilation; compute_s excludes both (for LS-GKM it equals
wall_s). peak_rss_mb is that process's peak RSS, gpu_pool_mb the CuPy memory
pool size at the end of the run. max_abs_diff compares each gkmsvm-lite result
with LS-GKM's output from the same run set.

LS-GKM's gkmexplain is not in every ls-gkm build. If the conda env lacks it,
build lsgkm from source and pass --lsgkm-bin <dir with gkmpredict, gkmexplain>.

Usage:
  python benchmarks/timing_variants.py --device cuda
  python benchmarks/timing_variants.py --device cuda --lsgkm-bin ~/lsgkm/bin
  python benchmarks/timing_variants.py --device cpu --n-variants 500 --methods ISM
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
    emit_result,
    gpu_pool_mb,
    gpu_sync,
    lsgkm_binary,
    lsgkm_default_threads,
    lsgkm_has_option,
    lsgkm_version,
    run_measured,
    set_cpu_threads,
    worker_result,
    write_seqs,
    write_tsv,
)

METHODS = ("ISM", "GkmExplain", "deltaSVM")
PAPER_TAGS = {"t2": "t2_l10_k6_d3_t16", "t3": "t3_l10_k6_d3_c10_g2_t16"}
L = 10
WARMUP = 16


# ── Inputs ─────────────────────────────────────────────────────────


def _decode(X: np.ndarray) -> list[str]:
    """[N, 4, L] one-hot → strings, all-zero columns as N."""
    idx = X.argmax(axis=1)
    idx[X.sum(axis=1) == 0] = 4
    lut = np.array(list("ACGTN"))
    return ["".join(row) for row in lut[idx]]


def _nonredundant_lmers(l: int = L) -> np.ndarray:
    """Indices (first base most significant) of l-mers with index <= their RC's."""
    idx = np.arange(4**l, dtype=np.int64)
    rc = np.zeros_like(idx)
    rest = idx.copy()
    for _ in range(l):
        rc = rc * 4 + (3 - rest % 4)
        rest //= 4
    return idx[idx <= rc]


def _lmer_strings(idx: np.ndarray, l: int = L) -> list[str]:
    powers = 4 ** np.arange(l - 1, -1, -1, dtype=np.int64)
    digits = (idx[:, None] // powers[None, :]) % 4
    lut = np.array(list("ACGT"))
    return ["".join(row) for row in lut[digits]]


def _prepare(args, work: Path) -> dict:
    import dsqtl_auprc as bench

    bench.download_data()
    _, X51, _, _ = bench.load_test_variants()
    ref, alt = X51
    if args.n_variants:
        ref, alt = ref[: args.n_variants], alt[: args.n_variants]
    tag = PAPER_TAGS[args.tag]
    bench._load_paper_model(tag, args.negset)  # downloads the model file
    model_path = bench.MODEL_DIR / f"gkmsvm_{tag}_negset{args.negset}.model.txt"

    np.savez(work / "variants.npz", ref=ref, alt=alt)
    write_seqs(work / "ref.fa", _decode(ref), "v")
    write_seqs(work / "alt.fa", _decode(alt), "v")
    nr = _nonredundant_lmers()
    np.save(work / "lmers.npy", nr)
    if "deltaSVM" in args.methods:
        write_seqs(work / "lmers.fa", _lmer_strings(nr), "k")
    center = ref.shape[2] // 2
    alt_base = alt[:, :, center].argmax(axis=1)
    return {"model": str(model_path), "n_variants": len(ref), "alt_base": alt_base}


# ── gkmsvm-lite worker (runs in a child process) ───────────────────


def _worker(spec: dict) -> None:
    from gkmsvm.backend import to_cpu
    from gkmsvm.importers.lsgkm import load_lsgkm_model

    t0 = time.perf_counter()
    model = load_lsgkm_model(spec["model"], dtype=np.float64)
    if spec["device"] == "cuda":
        model.cuda()
    data = np.load(Path(spec["work"]) / "variants.npz")
    ref, alt = data["ref"], data["alt"]
    method, bs = spec["method"], spec["batch_size"]

    # Warm up JIT / kernel compilation and packed-SV caches on a few variants
    if method == "deltaSVM":
        _ = model(ref[:WARMUP], batch_size=bs)
    else:
        _ = model.score_variants(ref[:WARMUP], alt[:WARMUP], batch_size=bs,
                                 method="kernel" if method == "ISM" else "gkmexplain")
    gpu_sync()
    load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    if method == "deltaSVM":
        out = np.asarray(model.to_deltasvm(device="cpu", query_norm=True).weights,
                         dtype=np.float64)
        out = out + float(model.bias)  # gkmpredict scale
    else:
        out = to_cpu(model.score_variants(
            ref, alt, batch_size=bs, method="kernel" if method == "ISM" else "gkmexplain",
        )).ravel()
    gpu_sync()
    compute_s = time.perf_counter() - t0

    np.save(Path(spec["work"]) / f"lite_{method}_{spec['device']}.npy", out)
    emit_result(load_s=load_s, compute_s=compute_s, gpu_pool_mb=gpu_pool_mb())


# ── LS-GKM runs ────────────────────────────────────────────────────


def _read_scores(path: Path, col: int = 1) -> np.ndarray:
    with open(path) as f:
        return np.array([float(line.rstrip("\n").split("\t")[col]) for line in f])


def _lsgkm_runs(method: str, bins: dict, threads: dict, work: Path, model: str) -> list[tuple]:
    """Commands for one method as (cmd, output path) pairs."""

    def predict(fa, out):
        cmd = [bins["gkmpredict"]]
        if threads["gkmpredict"]:
            cmd += ["-T", str(threads["gkmpredict"])]
        return cmd + [str(work / fa), model, str(work / out)], work / out

    if method == "ISM":
        return [predict("ref.fa", "lsgkm_ref.txt"), predict("alt.fa", "lsgkm_alt.txt")]
    if method == "deltaSVM":
        return [predict("lmers.fa", "lsgkm_lmers.txt")]
    cmd = [bins["gkmexplain"], "-m", "5"]
    if threads["gkmexplain"]:
        cmd += ["-T", str(threads["gkmexplain"])]
    out = work / "lsgkm_explain.txt"
    return [(cmd + [str(work / "ref.fa"), model, str(out)], out)]


def _lsgkm_scores(method: str, work: Path, alt_base: np.ndarray) -> np.ndarray:
    if method == "ISM":
        return _read_scores(work / "lsgkm_alt.txt") - _read_scores(work / "lsgkm_ref.txt")
    if method == "deltaSVM":
        return _read_scores(work / "lsgkm_lmers.txt")
    with open(work / "lsgkm_explain.txt") as f:
        impacts = np.array([[float(v) for v in line.rstrip("\n").split("\t")[2].split(",")]
                            for line in f])
    return impacts[np.arange(len(impacts)), alt_base]


# ── Main ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"],
                        help="cuda times gkmsvm-lite on CPU and GPU; cpu on CPU only")
    parser.add_argument("--methods", default=",".join(METHODS),
                        help=f"Comma-separated subset of {', '.join(METHODS)}")
    parser.add_argument("--tag", default="t3", choices=list(PAPER_TAGS),
                        help="Paper model kernel (default: t3)")
    parser.add_argument("--negset", type=int, default=1)
    parser.add_argument("--n-variants", type=int, default=None,
                        help="Use only the first N variants (default: all 28,309)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--conda-env", default="lsgkm")
    parser.add_argument("--lsgkm-bin", default=None,
                        help="Directory with LS-GKM binaries (overrides --conda-env)")
    parser.add_argument("--threads", type=int, default=lsgkm_default_threads(),
                        choices=LSGKM_THREADS,
                        help="LS-GKM -T where supported, and gkmsvm-lite CPU threads")
    parser.add_argument("--no-lsgkm", action="store_true", help="Skip LS-GKM runs")
    parser.add_argument("--timeout", type=float, default=6 * 3600,
                        help="Per-run timeout in seconds (default: 6 h)")
    parser.add_argument("--keep-work", action="store_true",
                        help="Keep the temporary directory with inputs and outputs")
    parser.add_argument("--output", default=str(RESULTS / "timing_variants.tsv"))
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        _worker(json.loads(args.worker))
        return
    # Workers inherit this through the environment
    lite_threads = set_cpu_threads(args.threads)

    args.methods = args.methods.split(",")
    bad = [m for m in args.methods if m not in METHODS]
    if bad:
        parser.error(f"unknown method(s): {', '.join(bad)}")

    bins, threads = {}, {}
    if not args.no_lsgkm:
        for name in ("gkmpredict", "gkmexplain"):
            bins[name] = lsgkm_binary(name, args.conda_env, args.lsgkm_bin)
            threads[name] = (args.threads if bins[name] and lsgkm_has_option(bins[name], "-T")
                             else None)
        need = {"gkmpredict"} | ({"gkmexplain"} if "GkmExplain" in args.methods else set())
        missing = [n for n in need if not bins[n]]
        if missing:
            where = args.lsgkm_bin or f"conda env '{args.conda_env}'"
            sys.exit(f"{', '.join(missing)} not found in {where}. Build lsgkm from source "
                     "and pass --lsgkm-bin, or use --no-lsgkm.")
        print(f"LS-GKM: {bins}  threads: {threads}")

    work = Path(tempfile.mkdtemp(prefix="timing_variants_"))
    print(f"Work dir: {work}")
    info = _prepare(args, work)
    print(f"{info['n_variants']:,} variants, model {Path(info['model']).name}")

    devices = ["cpu", "cuda"] if args.device == "cuda" else ["cpu"]
    n_items = {"ISM": info["n_variants"], "GkmExplain": info["n_variants"],
               "deltaSVM": 4**L}
    rows = []
    for method in args.methods:
        print(f"\n── {method} ──")
        ref_scores = None
        if not args.no_lsgkm:
            runs = _lsgkm_runs(method, bins, threads, work, info["model"])
            wall, peak, status = 0.0, 0.0, "ok"
            for cmd, _ in runs:
                m = run_measured(cmd, timeout=args.timeout)
                wall += m.wall_s
                peak = max(peak, m.peak_rss_mb or 0.0)
                if m.status != "ok":
                    status = m.status
                    print(f"  LS-GKM {m.status}: {m.stderr[-300:]}")
                    break
            if status == "ok":
                ref_scores = _lsgkm_scores(method, work, info["alt_base"])
            binary = "gkmexplain" if method == "GkmExplain" else "gkmpredict"
            n = len(_nonredundant_lmers()) if method == "deltaSVM" else info["n_variants"]
            rows.append({
                "method": method, "implementation": "LS-GKM", "device": "cpu",
                "threads": threads[binary] or 1, "n_items": n, "status": status,
                "wall_s": wall, "compute_s": wall, "peak_rss_mb": peak,
                "gpu_pool_mb": None, "max_abs_diff": None,
            })
            print(f"  LS-GKM ({threads[binary] or 1} threads): {wall:.1f}s  "
                  f"peak {peak:.0f} MB  [{status}]")

        for device in devices:
            spec = {"method": method, "device": device, "model": info["model"],
                    "work": str(work), "batch_size": args.batch_size}
            m = run_measured([sys.executable, __file__, "--worker", json.dumps(spec)],
                             timeout=args.timeout)
            res = worker_result(m.stdout) or {}
            diff = None
            if m.status == "ok" and ref_scores is not None:
                ours = np.load(work / f"lite_{method}_{device}.npy")
                if method == "deltaSVM":
                    ours = ours[np.load(work / "lmers.npy")]
                diff = float(np.abs(ours - ref_scores).max())
            if m.status != "ok":
                print(f"  gkmsvm-lite {device} {m.status}: {m.stderr[-300:]}")
            rows.append({
                "method": method, "implementation": "gkmsvm-lite", "device": device,
                "threads": lite_threads if device == "cpu" else None,
                "n_items": n_items[method], "status": m.status,
                "wall_s": m.wall_s, "compute_s": res.get("compute_s"),
                "peak_rss_mb": m.peak_rss_mb, "gpu_pool_mb": res.get("gpu_pool_mb"),
                "max_abs_diff": diff,
            })
            r = rows[-1]
            print(f"  gkmsvm-lite {device}: {r['wall_s']:.1f}s wall, "
                  f"{r['compute_s'] or float('nan'):.1f}s compute, "
                  f"peak {r['peak_rss_mb']:.0f} MB"
                  + (f", GPU pool {r['gpu_pool_mb']:.0f} MB" if r["gpu_pool_mb"] else "")
                  + (f", max|Δ| vs LS-GKM {diff:.1e}" if diff is not None else "")
                  + f"  [{m.status}]")

    import pandas as pd

    df = pd.DataFrame(rows).astype({"threads": "Int64"})
    write_tsv(args.output, df, "dsQTL variant scoring time", args.device,
              model=Path(info["model"]).name, n_variants=info["n_variants"],
              batch_size=args.batch_size, lite_cpu_threads=lite_threads,
              lsgkm=None if args.no_lsgkm else (
                  lsgkm_version(args.conda_env) if not args.lsgkm_bin
                  else f"built from source ({args.lsgkm_bin})"))
    if args.keep_work:
        print(f"Kept {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
