"""Prediction throughput vs number of support vectors (Fig. S2).

Scores --n-seqs GM12878 DHS sequences (300 bp) with models of increasing size,
built from nested random subsets of the ENCODE ENCFF579AOX model's support
vectors (72,145 SVs, l=11 k=7 d=3):

  -t 2  gkm_esttrunc, as trained. gkmsvm-lite scores through the exact
        l-mer weight table, whose cost doesn't depend on the number of SVs.
  -t 3  gkmrbf (gamma 1) on the same SVs and coefficients. gkmsvm-lite
        scores through the cached l-mer index (per-SV kernel values for
        the RBF), whose cost grows with the matching (window, SV) pairs.

For both, path "masked join" turns the table and the index off, the kernel
path before them (an SV index is built on every call), and LS-GKM's
gkmpredict scores the same models at -T --threads. The -t 3 coefficients
were trained for -t 2; only speed is measured.

Each gkmsvm-lite run is a separate process. first_s is the first call on a
freshly loaded model, including building its caches (packed SVs, table or
index); steady_s is a second call. JIT compilation is excluded by first
scoring with a small model of the same kernel. LS-GKM's time includes
loading its model file, as in normal use.

Usage:
  python benchmarks/throughput_sweep.py --device cuda
  python benchmarks/throughput_sweep.py --device cpu --sizes 1000,4000 --no-lsgkm
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
    REPO,
    RESULTS,
    SEQ_DIR,
    emit_result,
    gpu_sync,
    lsgkm_binary,
    lsgkm_default_threads,
    lsgkm_has_option,
    read_seqs,
    run_measured,
    set_cpu_threads,
    worker_result,
    write_seqs,
    write_tsv,
)

MODEL_PATH = REPO / "tests" / "fixtures" / "encode_ENCFF579AOX.model.txt.gz"
DEFAULT_SIZES = "1000,4000,16000,max"
KERNELS = {"t2": ("gkm_esttrunc", {}), "t3": ("gkmrbf", {"gamma": 1.0})}
# Path a default model() call takes, per kernel
DEFAULT_PATH = {"t2": "l-mer table", "t3": "l-mer index"}
WARMUP_SVS = 600


def _subset_model(full, idx, kernel: str):
    from gkmsvm import GkmSVM

    kernel_type, extra = KERNELS[kernel]
    # Positive-coefficient SVs first, as LIBSVM-format files list them by class
    idx = idx[np.argsort(full.coefficients[idx] <= 0, kind="stable")]
    return GkmSVM(full.support_sequences[idx], full.coefficients[idx], full.bias,
                  kernel_type, {**full.kernel_params, **extra})


# ── gkmsvm-lite worker (child process) ─────────────────────────────


def _worker(spec: dict) -> None:
    from gkmsvm import load_model, one_hot_encode
    from gkmsvm.backend import to_cpu

    X = np.stack([one_hot_encode(s) for s in read_seqs(spec["queries"])])

    def prepare(path):
        model = load_model(path)
        if spec["path"] == "masked join":
            model.use_lmer_table = False
            model.use_lmer_index = False
        if spec["device"] == "cuda":
            model.cuda()
        return model

    warm = prepare(spec["warmup_model"])  # compiles kernels on a small model
    _ = to_cpu(warm(X[:32]))
    gpu_sync()

    model = prepare(spec["model"])
    times = []
    for _ in range(2):
        t0 = time.perf_counter()
        _ = to_cpu(model(X, batch_size=spec["batch_size"]))
        gpu_sync()
        times.append(time.perf_counter() - t0)
    emit_result(first_s=times[0], steady_s=times[1])


# ── Main ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"],
                        help="cuda: run gkmsvm-lite on CPU and CUDA (default: cuda)")
    parser.add_argument("--sizes", default=DEFAULT_SIZES,
                        help="SV counts, comma separated; 'max' = all 72,145")
    parser.add_argument("--kernels", default="t2,t3")
    parser.add_argument("--n-seqs", type=int, default=1000, help="Query sequences")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--conda-env", default="lsgkm")
    parser.add_argument("--lsgkm-bin", default=None,
                        help="Directory with gkmpredict (default: from --conda-env)")
    parser.add_argument("--threads", type=int, default=lsgkm_default_threads(),
                        choices=LSGKM_THREADS,
                        help="LS-GKM -T and gkmsvm-lite CPU threads")
    parser.add_argument("--no-lsgkm", action="store_true")
    parser.add_argument("--timeout", type=float, default=6 * 3600)
    parser.add_argument("--output", default=str(RESULTS / "throughput.tsv"))
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        _worker(json.loads(args.worker))
        return

    import pandas as pd
    from gkmsvm import load_lsgkm_model

    lite_threads = set_cpu_threads(args.threads)
    gkmpredict = None if args.no_lsgkm else lsgkm_binary("gkmpredict", args.conda_env,
                                                         args.lsgkm_bin)
    if not args.no_lsgkm and gkmpredict is None:
        sys.exit("gkmpredict not found; pass --lsgkm-bin or --no-lsgkm")
    # Some builds' gkmpredict has no -T (single-threaded)
    lsgkm_threads = (args.threads if gkmpredict and lsgkm_has_option(gkmpredict, "-T")
                     else 1)

    full = load_lsgkm_model(MODEL_PATH, dtype=np.float64)
    S = full.num_support_vectors
    sizes = sorted({S if s == "max" else min(int(s), S) for s in args.sizes.split(",")})
    order = np.random.default_rng(0).permutation(S)
    kernels = args.kernels.split(",")
    devices = ["cpu", "cuda"] if args.device == "cuda" else ["cpu"]

    work = Path(tempfile.mkdtemp(prefix="throughput_"))
    queries = work / "queries.fa"
    write_seqs(queries, read_seqs(SEQ_DIR / "gm12878_shared.fa")[:args.n_seqs], "q")
    print(f"Work dir: {work}\n{args.n_seqs} queries, SV counts {sizes}, "
          f"{lite_threads} CPU threads")

    rows = []
    try:
        for kernel in kernels:
            warm_npz = work / f"{kernel}_warmup.npz"
            _subset_model(full, order[:WARMUP_SVS], kernel).save(str(warm_npz))
            for n_sv in sizes:
                model = _subset_model(full, order[:n_sv], kernel)
                npz = work / f"{kernel}_{n_sv}.npz"
                model.save(str(npz))
                print(f"\n── {kernel}, {n_sv:,} SVs ──")
                base = {"kernel": kernel, "n_sv": n_sv, "n_seqs": args.n_seqs}

                if gkmpredict is not None:
                    txt = work / f"{kernel}_{n_sv}.model.txt"
                    model.save(str(txt), fmt="lsgkm")
                    opts = ["-T", str(lsgkm_threads)] if lsgkm_threads > 1 else []
                    m = run_measured([gkmpredict, *opts, str(queries), str(txt),
                                      str(work / "pred.txt")], timeout=args.timeout)
                    if m.status != "ok":
                        print(f"  LS-GKM {m.status}: {(m.stderr or m.stdout)[-500:]}")
                    rows.append({**base, "implementation": "LS-GKM", "path": "gkmpredict",
                                 "device": "cpu", "threads": lsgkm_threads,
                                 "status": m.status, "first_s": m.wall_s,
                                 "steady_s": m.wall_s, "peak_rss_mb": m.peak_rss_mb})
                    print(f"  LS-GKM: {m.wall_s:.1f}s [{m.status}]")

                for path in (DEFAULT_PATH[kernel], "masked join"):
                    for device in devices:
                        spec = {"model": str(npz), "warmup_model": str(warm_npz),
                                "queries": str(queries), "path": path, "device": device,
                                "batch_size": args.batch_size}
                        m = run_measured([sys.executable, __file__, "--worker",
                                          json.dumps(spec)], timeout=args.timeout)
                        res = worker_result(m.stdout) or {}
                        if m.status != "ok":
                            print(f"  {path} {device} {m.status}: {m.stderr[-300:]}")
                        rows.append({**base, "implementation": "gkmsvm-lite", "path": path,
                                     "device": device,
                                     "threads": lite_threads if device == "cpu" else None,
                                     "status": m.status, "first_s": res.get("first_s"),
                                     "steady_s": res.get("steady_s"),
                                     "peak_rss_mb": m.peak_rss_mb})
                        if res:
                            print(f"  {path}, {device}: first {res['first_s']:.2f}s, "
                                  f"steady {res['steady_s']:.2f}s "
                                  f"({args.n_seqs / res['steady_s']:,.0f} seq/s)")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    df = pd.DataFrame(rows)
    df["seqs_per_sec"] = df["n_seqs"] / df["steady_s"]
    write_tsv(args.output, df, "Prediction throughput vs number of SVs", args.device,
              model=MODEL_PATH.name, lsgkm_threads=lsgkm_threads if gkmpredict else None,
              lite_cpu_threads=lite_threads)


if __name__ == "__main__":
    main()
