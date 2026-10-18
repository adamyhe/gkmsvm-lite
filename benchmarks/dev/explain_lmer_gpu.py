"""Is GkmExplain's l-mer path worth running on CUDA?

Times gkmexplain (hypothetical mode) per sequence on random 300 bp
sequences, l=11 k=7 d=3, for each kernel and path:

  cpu-lmer    the numba l-mer path (_explain_lmer_cpu / _explain_index_cpu)
  cpu-fused   the numba per-SV kernel (_explain_cpu)
  cuda-lmer   the CUDA l-mer path (_explain_lmer_gpu)
  cuda-fused  the CUDA per-SV kernel (fused_explain)

The fused paths are forced with model.use_lmer_index = False; for -t 0
(143,980 neighbors at dmax 5) the l-mer paths are forced by lowering
explain._EXPLAIN_SV_WINDOWS_PER_NEIGHBOR. Each row also reports the largest
difference from cpu-lmer relative to its peak.

Usage (GPU server):
  python benchmarks/dev/explain_lmer_gpu.py --sv 2000 10000 --n 500
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from gkmsvm import GkmSVM, explain, gkmexplain

KERNELS = {"-t 0": "gkm_cnt", "-t 2": "gkm_esttrunc", "-t 3": "gkmrbf"}
PARAMS = {"L": 11, "k": 7, "d": 3, "gamma": 2.0}


def onehot(rng, n, length):
    return np.eye(4, dtype=np.float32)[rng.integers(0, 4, (n, length))].transpose(0, 2, 1).copy()


def run(model, x, device, lmer, batch_size):
    import cupy as cp

    model.use_lmer_index = lmer
    model.clear_caches()
    if device == "cuda":
        model.cuda()
        x = cp.asarray(x)
    else:
        model.cpu()
    gkmexplain(model, x[:2], mode="hypothetical")  # JIT, caches
    if device == "cuda":
        cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    out = gkmexplain(model, x, mode="hypothetical", batch_size=batch_size)
    if device == "cuda":
        cp.cuda.Device().synchronize()
    dt = time.perf_counter() - t0
    return dt, cp.asnumpy(out) if device == "cuda" else out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sv", type=int, nargs="+", default=[2000, 10000])
    ap.add_argument("--n", type=int, default=500, help="query sequences")
    ap.add_argument("--len", type=int, default=300)
    ap.add_argument("--batch", type=int, default=500, help="CUDA batch_size")
    ap.add_argument("--kernels", nargs="+", default=list(KERNELS))
    ap.add_argument("--skip-fused", action="store_true",
                    help="skip the per-SV paths (slow for -t 0 at large S)")
    args = ap.parse_args()

    explain._EXPLAIN_SV_WINDOWS_PER_NEIGHBOR = 0
    rng = np.random.default_rng(0)
    x = onehot(rng, args.n, args.len)
    print(f"{'kernel':6} {'S':>6} {'path':11} {'ms/seq':>9} {'max rel diff':>13}")
    for name in args.kernels:
        for S in args.sv:
            model = GkmSVM(onehot(rng, S, args.len), rng.normal(size=S), 0.1,
                           KERNELS[name], PARAMS)
            paths = [("cpu-lmer", "cpu", True, 50), ("cuda-lmer", "cuda", True, args.batch)]
            if not args.skip_fused:
                paths += [("cpu-fused", "cpu", False, 50), ("cuda-fused", "cuda", False, args.batch)]
            ref = None
            for label, device, lmer, bs in paths:
                dt, out = run(model, x, device, lmer, bs)
                if ref is None:
                    ref = out
                diff = np.abs(out - ref).max() / np.abs(ref).max()
                print(f"{name:6} {S:>6} {label:11} {dt / args.n * 1e3:>9.3f} {diff:>13.2e}",
                      flush=True)


if __name__ == "__main__":
    main()
