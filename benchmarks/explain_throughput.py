"""GkmExplain and ISM throughput on whole sequences, with precision checks (Fig. S4B-F).

Attributes the first --sizes (ISM: --ism-sizes) GM12878 DHS sequences (300 bp, gm12878_shared.fa,
22,384 of them) with two models:

  t2  ENCODE ENCFF579AOX (-t 2 l=11 k=7 d=3, 72,145 SVs): through the explain
      table once a call is large enough (use_explain_table="auto"), else the
      l-mer path
  t3  The GkmExplain paper's dsQTL model (-t 3 l=10 k=6 d=3 gamma=2, negset 1):
      the l-mer index path, SV weights depending on the query

and three operations, each [B, 4, L]:

  hypothetical  gkmexplain(mode="hypothetical")     LS-GKM gkmexplain -m 1
  perturbation  gkmexplain(mode="perturbation")     LS-GKM gkmexplain -m 4
  ISM           ism()                               LS-GKM gkmpredict on each
                                                    sequence's 3L substitutions,
                                                    minus gkmpredict on it

Each gkmsvm-lite run is a separate process. Before timing, it runs the same
operation on 8 sequences with a second copy of the model (compiling kernels,
not filling the timed model's caches), and for the explain table a forced
build on an l=6 model, so neither the table's kernels nor anything else is
compiled in a timed call. first_s is the first call on the freshly loaded
model, including building its caches (packed SVs, l-mer index or V, explain
table); steady_s is a second call (--repeats 1 skips it, steady_s = first_s).
LS-GKM's times include loading its model file, as in normal use; they run on
the first --c-max-seqs sequences (--c-ism-max-seqs for ISM: 900 gkmpredict
queries per sequence) and are reported per sequence.

Precision checks, against a float64 CPU reference computed once per model in
an untimed process (default paths, the same SVs and coefficients):

  ref_max_abs / ref_max_rel      the first --n-check sequences' output against
                                 the reference, absolute and relative to the
                                 reference's largest |value|. CPU rows measure
                                 path agreement (explain table vs l-mer path,
                                 ~1e-14); MLX rows float32.
  delta_max_abs / delta_max_rel  summation-to-delta: an identity the output
                                 must satisfy, checked on the float64 scores.
                                 hypothetical: the importance (hypothetical at
                                 each sequence's own base) sums to
                                 f(x) - b - exp(-gamma) sum(coef) (the last
                                 term for -t 3 only), over every timed
                                 sequence. ISM: each entry equals
                                 f(mutant) - f(x), on 12 substitutions in each
                                 of the first --n-check sequences, scored
                                 directly. Relative to the largest |target|.
                                 perturbation approximates score differences by
                                 design (lsgkm's mode 4), so it has no identity.

C rows compare with the same reference (ref_*). MLX computes in float32, so
its ref_* and delta_* columns are the float32 error: what a user of the Apple
GPU should expect for TF-MoDISco inputs or score differences.

To run the CPU and GPU halves on different machines (benchmarks/sherlock/),
run --device cpu on one and --device cuda --gpu-only on the other, then
--merge-parts CPU_TSV GPU_TSV. Each half computes its own reference.

Usage:
  python benchmarks/explain_throughput.py --device cuda --lsgkm-bin ~/lsgkm/bin
  python benchmarks/explain_throughput.py --device mlx --threads 4 --cpu-threads 4,10
  python benchmarks/explain_throughput.py --device cpu --sizes 100 --no-lsgkm \\
      --kernels t2 --ops hypothetical
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from _common import (
    LSGKM_THREADS,
    MLSGKM,
    REPO,
    RESULTS,
    SEQ_DIR,
    c_threads_opts,
    emit_result,
    gpu_pool_mb,
    gpu_reset_peak,
    gpu_sync,
    lsgkm_binary,
    lsgkm_default_threads,
    lsgkm_has_option,
    lsgkm_sources,
    merge_parts,
    mlsgkm_binary,
    mlsgkm_version,
    read_seqs,
    run_measured,
    set_cpu_threads,
    to_device,
    warm_binaries,
    warm_jit_cache,
    worker_result,
    write_seqs,
    write_tsv,
)

OPS = ("hypothetical", "perturbation", "ISM")
LSGKM_MODE = {"hypothetical": "1", "perturbation": "4"}
DEFAULT_SIZES = "100,1000,10000"
# ISM's cost per sequence doesn't depend on how many there are (no table to
# amortize), and -t 3 ISM takes ~1 s per sequence on an M4: 10k would be ~3 h
# per call
DEFAULT_ISM_SIZES = "100,1000"
WARMUP_SEQS = 8
# Substitutions per sequence in the ISM delta check
ISM_SAMPLES = 12
BASES = "ACGT"


def _model_path(kernel: str) -> Path:
    """The model file for a kernel tag, downloaded if needed."""
    if kernel == "t2":
        return REPO / "tests" / "fixtures" / "encode_ENCFF579AOX.model.txt.gz"
    import dsqtl_auprc as bench
    from timing_variants import PAPER_TAGS

    tag = PAPER_TAGS["t3"]
    bench._load_paper_model(tag, 1)  # downloads the model file
    return bench.MODEL_DIR / f"gkmsvm_{tag}_negset1.model.txt"


def _load(path: str):
    from gkmsvm.importers.lsgkm import load_lsgkm_model

    return load_lsgkm_model(str(path))


def _rbf_const(model) -> float:
    """exp(-gamma) sum(coef) for RBF models, 0 otherwise: the part of
    f(x) - b that importance attributions leave out (as lsgkm does)."""
    if model.kernel is model.kernel.base:
        return 0.0
    from gkmsvm.backend import to_cpu

    coef = np.asarray(to_cpu(model.coefficients), dtype=np.float64)
    return float(np.exp(-model.kernel.gamma) * coef.sum())


def _run_op(model, X, op: str, batch_size: int | None):
    from gkmsvm import gkmexplain, ism
    from gkmsvm.backend import to_cpu

    if op == "ISM":
        out = ism(model, X, batch_size=batch_size)
    else:
        out = gkmexplain(model, X, mode=op, batch_size=batch_size or 50)
    out = np.asarray(to_cpu(out), dtype=np.float64)
    gpu_sync()
    return out


# ── Reference (untimed child process) ──────────────────────────────


def _reference(spec: dict) -> None:
    """Float64 CPU outputs, scores and ISM spot checks for one model."""
    work = Path(spec["work"])
    kernel = spec["kernel"]
    model = _load(spec["model"])
    codes = np.load(work / "codes.npy")
    X = codes[: spec["n_ref"]]
    for op in spec["ops"]:
        np.save(work / f"ref_{kernel}_{op}.npy", _run_op(model, X, op, None))
    from gkmsvm.backend import to_cpu

    scores = np.asarray(to_cpu(model(codes[: spec["n_scores"]], batch_size=1000)))
    np.save(work / f"scores_{kernel}.npy", scores.ravel().astype(np.float64))

    # ISM spot checks: ISM_SAMPLES substitutions in each of the first n_check
    # sequences, scored directly
    rng = np.random.default_rng(0)
    n, L = min(spec["n_check"], len(codes)), codes.shape[1]
    seq = np.repeat(np.arange(n), ISM_SAMPLES)
    pos = rng.integers(0, L, len(seq))
    base = (codes[seq, pos] + rng.integers(1, 4, len(seq))) % 4  # never the ref base
    mut = codes[seq].copy()
    mut[np.arange(len(seq)), pos] = base
    f_mut = np.asarray(to_cpu(model(mut, batch_size=1000))).ravel()
    np.savez(
        work / f"ism_check_{kernel}.npz",
        seq=seq,
        pos=pos,
        base=base,
        delta=f_mut - scores.ravel()[seq],
    )
    (work / f"meta_{kernel}.json").write_text(
        json.dumps({"bias": float(model.bias), "rbf_const": _rbf_const(model)})
    )


# ── gkmsvm-lite worker (child process) ─────────────────────────────


def _warm_table(model, op: str, device: str, X) -> None:
    """Compile the explain table's kernels: a forced build on an l=6 copy."""
    from gkmsvm import GkmSVM
    from gkmsvm.backend import to_cpu

    tiny = GkmSVM(
        to_cpu(model.support_sequences)[:200],
        to_cpu(model.coefficients)[:200],
        model.bias,
        model.kernel_type,
        {**model.kernel_params, "L": 6, "k": 4, "d": 2},
    )
    to_device(tiny, device)
    tiny.use_explain_table = True
    _run_op(tiny, X[:2], op, None)


def _checks(spec: dict, out: np.ndarray, codes: np.ndarray) -> dict:
    """ref_* and delta_* columns for one output (see the module docstring)."""
    work = Path(spec["work"])
    kernel, op = spec["kernel"], spec["op"]
    res = {}
    ref_path = work / f"ref_{kernel}_{op}.npy"
    if ref_path.exists():
        ref = np.load(ref_path)
        n = min(len(out), len(ref), spec["n_check"])
        err = np.abs(out[:n] - ref[:n]).max()
        res |= {
            "ref_n": n,
            "ref_max_abs": float(err),
            "ref_max_rel": float(err / np.abs(ref[:n]).max()),
        }
    meta = json.loads((work / f"meta_{kernel}.json").read_text())
    if op == "hypothetical":
        scores = np.load(work / f"scores_{kernel}.npy")[: len(out)]
        target = scores - meta["bias"] - meta["rbf_const"]
        # Importance = hypothetical at each sequence's own base
        got = np.take_along_axis(out, codes[: len(out), None, :], axis=1).sum(axis=(1, 2))
        err = np.abs(got - target)
        res |= {"delta_check": "sum(importance) = f(x) - b - c", "delta_n": len(out)}
    elif op == "ISM":
        chk = np.load(work / f"ism_check_{kernel}.npz")
        keep = chk["seq"] < len(out)
        target = chk["delta"][keep]
        got = out[chk["seq"][keep], chk["base"][keep], chk["pos"][keep]]
        err = np.abs(got - target)
        res |= {"delta_check": "ISM = f(mutant) - f(x)", "delta_n": int(keep.sum())}
    else:
        return res
    res |= {
        "delta_max_abs": float(err.max()),
        "delta_max_rel": float(err.max() / np.abs(target).max()),
    }
    return res


def _worker(spec: dict) -> None:
    import numba

    numba.set_num_threads(min(spec["threads"], numba.config.NUMBA_NUM_THREADS))
    device, op = spec["device"], spec["op"]
    codes = np.load(Path(spec["work"]) / "codes.npy")
    # Users pass one-hot NumPy arrays: time moving them to the device too
    from gkmsvm.codec import codes_to_one_hot

    X = codes_to_one_hot(codes[: spec["n_seqs"]]).astype(np.float32)

    warm = to_device(_load(spec["model"]), device)
    _run_op(warm, X[:WARMUP_SEQS], op, spec["batch_size"])
    rbf = warm.kernel is not warm.kernel.base
    if op != "ISM" and not rbf:
        _warm_table(warm, op, device, X)
    del warm
    gc.collect()
    gpu_reset_peak()

    model = to_device(_load(spec["model"]), device)
    times = []
    for _ in range(spec["repeats"]):
        t0 = time.perf_counter()
        out = _run_op(model, X, op, spec["batch_size"])
        times.append(time.perf_counter() - t0)
    table = bool(getattr(model, "_explain_tables", None))
    emit_result(
        first_s=times[0],
        steady_s=times[-1],
        gpu_mem_mb=gpu_pool_mb(),
        explain_table=table,
        **_checks(spec, out, codes),
    )


# ── LS-GKM runs ────────────────────────────────────────────────────


def _read_explain(path: Path) -> np.ndarray:
    """gkmexplain output: name, score, then A,C,G,T per position split by ';'."""
    with open(path) as f:
        rows = [line.rstrip("\n").split("\t") for line in f]
    return np.stack(
        [
            np.array([[float(v) for v in p.split(",")] for p in r[2].split(";")]).T
            for r in rows
        ]
    )


def _read_scores(path: Path) -> np.ndarray:
    with open(path) as f:
        return np.array([float(line.rstrip("\n").split("\t")[1]) for line in f])


def _ism_fasta(seqs: list[str], path: Path) -> None:
    """Every substitution of every sequence: position-major, bases A, C, G, T
    minus the reference base."""
    with open(path, "w") as f:
        for i, s in enumerate(seqs):
            for p, ref in enumerate(s):
                for b in BASES:
                    if b != ref:
                        f.write(f">m{i}_{p}_{b}\n{s[:p]}{b}{s[p + 1:]}\n")


def _ism_from_scores(seqs: list[str], ref: np.ndarray, mut: np.ndarray) -> np.ndarray:
    L = len(seqs[0])
    out = np.zeros((len(seqs), 4, L))
    k = 0
    for i, s in enumerate(seqs):
        for p, r in enumerate(s):
            for j, b in enumerate(BASES):
                if b != r:
                    out[i, j, p] = mut[k] - ref[i]
                    k += 1
    return out


def _c_run(op, impl, bins, n_thr, work: Path, model_txt: Path, seqs, timeout):
    """Time one C implementation on seqs; (status, wall_s, peak_rss_mb, output)."""
    opts = c_threads_opts(impl, n_thr)
    fa = work / "c_query.fa"
    write_seqs(fa, seqs, "q")
    if op == "ISM":
        mut_fa = work / "c_mutants.fa"
        _ism_fasta(seqs, mut_fa)
        cmds = [
            [bins["gkmpredict"], *opts, str(f), str(model_txt), str(work / f"{n}.txt")]
            for f, n in ((fa, "c_ref"), (mut_fa, "c_mut"))
        ]
    else:
        out = work / "c_explain.txt"
        cmds = [
            [bins["gkmexplain"], "-m", LSGKM_MODE[op], *opts, str(fa), str(model_txt),
             str(out)]
        ]
    wall, peak = 0.0, 0.0
    for cmd in cmds:
        m = run_measured(cmd, timeout=timeout)
        wall += m.wall_s
        peak = max(peak, m.peak_rss_mb or 0.0)
        if m.status != "ok":
            print(f"  {impl} {m.status}: {(m.stderr or m.stdout)[-300:]}")
            return m.status, wall, peak, None
    try:
        if op == "ISM":
            res = _ism_from_scores(
                seqs, _read_scores(work / "c_ref.txt"), _read_scores(work / "c_mut.txt")
            )
        else:
            res = _read_explain(work / "c_explain.txt")
    except (OSError, ValueError, IndexError) as e:
        print(f"  {impl}: unreadable output ({e})")
        return "failed", wall, peak, None
    return "ok", wall, peak, res


# ── Main ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda", "mlx"],
        help="cuda / mlx: run gkmsvm-lite on CPU and that GPU (default: cuda)",
    )
    parser.add_argument(
        "--sizes", default=DEFAULT_SIZES, help="Sequence counts for GkmExplain"
    )
    parser.add_argument(
        "--ism-sizes", default=DEFAULT_ISM_SIZES, help="Sequence counts for ISM"
    )
    parser.add_argument("--kernels", default="t2,t3")
    parser.add_argument("--ops", default=",".join(OPS))
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="batch_size for gkmexplain/ism (default: the library's)",
    )
    parser.add_argument("--repeats", type=int, default=2, choices=[1, 2])
    parser.add_argument(
        "--n-check", type=int, default=64, help="Sequences in the precision checks"
    )
    parser.add_argument(
        "--c-max-seqs",
        type=int,
        default=100,
        help="Sequences LS-GKM/mLS-GKM gkmexplain run on",
    )
    parser.add_argument(
        "--c-ism-max-seqs",
        type=int,
        default=10,
        help="Sequences LS-GKM/mLS-GKM ISM runs on (900 gkmpredict queries each)",
    )
    parser.add_argument("--conda-env", default="lsgkm")
    parser.add_argument(
        "--lsgkm-bin",
        default=None,
        help="Directory with LS-GKM binaries the conda env lacks (gkmexplain: "
        "kundajelab/lsgkm)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=lsgkm_default_threads(),
        choices=LSGKM_THREADS,
        help="LS-GKM -T / mLS-GKM -t, and gkmsvm-lite CPU threads",
    )
    parser.add_argument(
        "--cpu-threads",
        default=None,
        help="Comma-separated gkmsvm-lite CPU thread counts (any; default "
        "--threads), e.g. 4,10 on a 10-core Apple chip",
    )
    parser.add_argument("--no-lsgkm", action="store_true")
    parser.add_argument("--mlsgkm-bin", default=None, help="mLS-GKM bin directory")
    parser.add_argument("--no-warm-jit", action="store_true")
    parser.add_argument("--timeout", type=float, default=6 * 3600)
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="Time only gkmsvm-lite on --device's GPU, for a GPU part",
    )
    parser.add_argument(
        "--merge-parts",
        nargs=2,
        metavar=("CPU_TSV", "GPU_TSV"),
        help="Write --output from a --device cpu file and a --gpu-only file",
    )
    parser.add_argument("--output", default=str(RESULTS / "explain_throughput.tsv"))
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--reference", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        _worker(json.loads(args.worker))
        return
    # Progress shows up in a log as it happens (stdout to a file is block-buffered)
    sys.stdout.reconfigure(line_buffering=True)
    if args.reference:
        _reference(json.loads(args.reference))
        return
    if args.merge_parts:
        merge_parts(dict(zip(("cpu", "gpu"), args.merge_parts)), args.output)
        return
    if args.gpu_only and args.device == "cpu":
        parser.error("--gpu-only needs --device cuda or mlx")
    ops = args.ops.split(",")
    if bad := [o for o in ops if o not in OPS]:
        parser.error(f"unknown op(s): {', '.join(bad)}")
    kernels = args.kernels.split(",")
    if bad := [k for k in kernels if k not in ("t2", "t3")]:
        parser.error(f"unknown kernel(s): {', '.join(bad)}")

    cpu_threads = (
        [int(t) for t in args.cpu_threads.split(",")] if args.cpu_threads else [args.threads]
    )
    # Workers inherit NUMBA_NUM_THREADS from this, and Numba can't run more
    # threads than its pool: size it for the largest count, and let each worker
    # set its own (numba.set_num_threads)
    pool = set_cpu_threads(max([args.threads, *cpu_threads]))
    lite_threads = min(args.threads, pool)
    cpu_threads = [min(t, pool) for t in cpu_threads]
    if not args.no_warm_jit:
        warm_jit_cache(args.device)
    lite_runs = ([] if args.gpu_only else [("cpu", t) for t in cpu_threads]) + (
        [(args.device, lite_threads)] if args.device != "cpu" else []
    )

    # C implementations: name -> (binaries, threads per binary)
    c_impls = {}
    if not args.no_lsgkm and not args.gpu_only:
        bins = {
            n: lsgkm_binary(n, args.conda_env, args.lsgkm_bin)
            for n in ("gkmpredict", "gkmexplain")
        }
        need = {"gkmpredict" if op == "ISM" else "gkmexplain" for op in ops}
        if missing := [n for n in need if not bins[n]]:
            sys.exit(
                f"{', '.join(missing)} not found; build kundajelab/lsgkm and pass "
                "--lsgkm-bin, or use --no-lsgkm"
            )
        c_impls["LS-GKM"] = (
            bins,
            {
                n: args.threads if b and lsgkm_has_option(b, "-T") else 1
                for n, b in bins.items()
            },
        )
    if args.mlsgkm_bin and not args.gpu_only:
        bins = {n: mlsgkm_binary(n, args.mlsgkm_bin) for n in ("gkmpredict", "gkmexplain")}
        c_impls[MLSGKM] = (bins, {n: args.threads for n in bins})
    warm_binaries(*(b for bins, _ in c_impls.values() for b in bins.values()))

    seqs = read_seqs(SEQ_DIR / "gm12878_shared.fa")

    def counts(arg):
        return sorted({min(int(s), len(seqs)) for s in arg.split(",")})

    op_sizes = {op: counts(args.ism_sizes if op == "ISM" else args.sizes) for op in ops}
    sizes = sorted({n for v in op_sizes.values() for n in v})
    seqs = seqs[: max(sizes)]
    from gkmsvm.codec import encode_codes

    work = Path(tempfile.mkdtemp(prefix="explain_throughput_"))
    np.save(work / "codes.npy", np.stack([encode_codes(s) for s in seqs]))
    print(
        f"Work dir: {work}\nSizes {op_sizes}, gkmsvm-lite runs {lite_runs}, "
        f"C: {list(c_impls)}"
    )

    rows = []
    try:
        for kernel in kernels:
            src = _model_path(kernel)
            # Every run reads the model from the work dir (node-local under
            # Slurm). gkmexplain needs it uncompressed, and the kundajelab fork
            # rejects a "norc 0" line (RC on is its default; see
            # dev/lsgkm_wgkm_fixtures.py)
            model_txt = work / f"{kernel}.model.txt"
            with gzip.open(src, "rt") if src.suffix == ".gz" else open(src) as f:
                model_txt.write_text("".join(ln for ln in f if ln.strip() != "norc 0"))
            rbf = "rbf" in next(
                line.split()[1]
                for line in model_txt.read_text().splitlines()
                if line.startswith("kernel_type")
            )
            c_caps = [
                args.c_ism_max_seqs if op == "ISM" else args.c_max_seqs for op in ops
            ]
            n_ref = min(len(seqs), max([args.n_check, *c_caps] if c_impls else [args.n_check]))
            print(f"\n── {kernel}: {model_txt.name}; float64 reference on {n_ref} seqs ──")
            ref_spec = {
                "work": str(work),
                "kernel": kernel,
                "model": str(model_txt),
                "ops": ops,
                "n_ref": n_ref,
                "n_scores": max(sizes),
                "n_check": args.n_check,
            }
            m = run_measured(
                [sys.executable, __file__, "--reference", json.dumps(ref_spec)]
            )
            if m.status != "ok":
                sys.exit(f"reference for {kernel} {m.status}: {m.stderr[-1000:]}")
            print(f"  reference: {m.wall_s:.0f}s")

            for op in ops:
                print(f"\n── {kernel} {op} ──")
                base = {"kernel": kernel, "model": src.name, "op": op}
                ref = np.load(work / f"ref_{kernel}_{op}.npy")
                for impl, (bins, thr) in c_impls.items():
                    binary = "gkmpredict" if op == "ISM" else "gkmexplain"
                    # mLS-GKM's gkmexplain returns zeros for non-RBF models
                    # (2bb06be; see timing_variants.py)
                    if impl == MLSGKM and binary == "gkmexplain" and not rbf:
                        print(f"  {impl} gkmexplain skipped: wrong for non-RBF models")
                        continue
                    if not bins.get(binary):
                        print(f"  {impl}: no {binary}")
                        continue
                    n = min(
                        len(seqs), args.c_ism_max_seqs if op == "ISM" else args.c_max_seqs
                    )
                    print(f"  {impl} ({thr[binary]} threads, {n} seqs) running...", flush=True)
                    status, wall, peak, out = _c_run(
                        op, impl, bins, thr[binary], work, model_txt, seqs[:n], args.timeout
                    )
                    row = {
                        **base,
                        "n_seqs": n,
                        "implementation": impl,
                        "device": "cpu",
                        "threads": thr[binary],
                        "status": status,
                        "first_s": wall,
                        "steady_s": wall,
                        "peak_rss_mb": peak,
                    }
                    if out is not None:
                        err = np.abs(out - ref[:n]).max()
                        row |= {
                            "ref_n": n,
                            "ref_max_abs": float(err),
                            "ref_max_rel": float(err / np.abs(ref[:n]).max()),
                        }
                    rows.append(row)
                    print(
                        f"  {impl}: {wall:.1f}s ({wall / n:.3g} s/seq)"
                        + (f", max|Δ| vs reference {row['ref_max_abs']:.1e}"
                           if "ref_max_abs" in row else "")
                        + f"  [{status}]"
                    )

                for n_seqs in op_sizes[op]:
                    for device, n_thr in lite_runs:
                        spec = {
                            "work": str(work),
                            "kernel": kernel,
                            "model": str(model_txt),
                            "op": op,
                            "device": device,
                            "threads": n_thr,
                            "n_seqs": n_seqs,
                            "batch_size": args.batch_size,
                            "repeats": args.repeats,
                            "n_check": args.n_check,
                        }
                        m = run_measured(
                            [sys.executable, __file__, "--worker", json.dumps(spec)],
                            timeout=args.timeout,
                        )
                        res = worker_result(m.stdout) or {}
                        if m.status != "ok":
                            print(f"  {device} {n_seqs} {m.status}: {m.stderr[-300:]}")
                        rows.append(
                            {
                                **base,
                                "n_seqs": n_seqs,
                                "implementation": "gkmsvm-lite",
                                "device": device,
                                "threads": n_thr if device == "cpu" else None,
                                "status": m.status,
                                "peak_rss_mb": m.peak_rss_mb,
                                **res,
                            }
                        )
                        if res:
                            print(
                                f"  {device}{f' {n_thr}T' if device == 'cpu' else ''}, "
                                f"{n_seqs:,} seqs: first {res['first_s']:.2f}s, steady "
                                f"{res['steady_s']:.2f}s "
                                f"({n_seqs / res['steady_s']:,.1f} seq/s"
                                + (", explain table" if res.get("explain_table") else "")
                                + f"); ref rel {res.get('ref_max_rel', float('nan')):.1e}"
                                + (f", delta rel {res['delta_max_rel']:.1e}"
                                   if "delta_max_rel" in res else "")
                            )
    finally:
        if args.keep_work:
            print(f"Kept {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)

    import pandas as pd

    df = pd.DataFrame(rows)
    df["seqs_per_sec"] = df["n_seqs"] / df["steady_s"]
    write_tsv(
        args.output,
        df.astype({"threads": "Int64"}),
        "GkmExplain and ISM throughput on 300 bp sequences",
        args.device,
        sequences="gm12878_shared.fa",
        n_check=args.n_check,
        batch_size=args.batch_size,
        lite_cpu_threads=",".join(str(t) for t in cpu_threads),
        lsgkm=lsgkm_sources(c_impls["LS-GKM"][0], args.conda_env)
        if "LS-GKM" in c_impls
        else None,
        mlsgkm=None if args.gpu_only else mlsgkm_version(args.mlsgkm_bin),
    )


if __name__ == "__main__":
    main()
