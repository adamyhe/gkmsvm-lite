"""Check and time the CUDA masked join against the CUDA brute-force kernel.

Computes the -t 2 self-Gram of N GM12878 sequences on the GPU both ways
(kernel.use_masked toggled) and reports agreement and time. Run on a machine
with CuPy and an NVIDIA GPU.

Usage:
  python benchmarks/dev/masked_gpu_check.py
  python benchmarks/dev/masked_gpu_check.py --sizes 1000,5000 -l 10 -k 6
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import SEQ_DIR, gpu_sync


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sizes", default="1000,2000,5000")
    parser.add_argument("-l", type=int, default=11)
    parser.add_argument("-k", type=int, default=7)
    parser.add_argument("-d", type=int, default=3)
    parser.add_argument("--lanes", default=None,
                        help="Comma-separated join lane-group sizes to compare (1-32)")
    parser.add_argument("--profile", action="store_true",
                        help="Time index build and join per mask size (adds syncs)")
    args = parser.parse_args()

    import cupy as cp

    from gkmsvm import one_hot_encode, read_fasta
    from gkmsvm.gram import compute_gram
    from gkmsvm.kernels import masked
    from gkmsvm.kernels.esttrunc import EstTruncGkmKernel

    sizes = [int(s) for s in args.sizes.split(",")]
    lane_options = ([int(x) for x in args.lanes.split(",")] if args.lanes
                    else [masked.GPU_JOIN_LANES])
    half = max(sizes) // 2
    pos = [s for _, s in read_fasta(str(SEQ_DIR / "gm12878_shared.fa"))][:half]
    neg = [s for _, s in read_fasta(str(SEQ_DIR / "nullseqs_gm12878_shared.1.1.fa"))][:half]
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}, "
          f"-t 2 l={args.l} k={args.k} d={args.d}")

    def gram(X, masked):
        kern = EstTruncGkmKernel(args.l, args.k, d=args.d)
        kern.use_masked = masked
        gpu_sync()
        t0 = time.perf_counter()
        G = compute_gram(kern, X)
        gpu_sync()
        return G, time.perf_counter() - t0

    # Compile and warm up both paths
    Xw = cp.asarray(np.stack([one_hot_encode(s) for s in pos[:40]]))
    gram(Xw, True)
    gram(Xw, False)

    for N in sizes:
        seqs = pos[: N // 2] + neg[: N // 2]
        X = cp.asarray(np.stack([one_hot_encode(s) for s in seqs]))
        G_b, t_b = gram(X, False)
        print(f"N={N:>6}: brute {t_b:8.2f}s")
        for lanes in lane_options:
            masked.GPU_JOIN_LANES = lanes
            if args.profile:
                masked.GPU_PROFILE = {}
            G_m, t_m = gram(X, True)
            prof, masked.GPU_PROFILE = masked.GPU_PROFILE, None
            rel = np.abs(G_m - G_b).max() / np.abs(G_b).max()
            print(f"    masked, {lanes:>2} lanes: {t_m:8.2f}s  speedup {t_b / t_m:5.1f}x  "
                  f"max |diff|/max|K| {rel:.1e}")
            if prof:
                print("      " + ", ".join(f"{k} {v:.2f}s" for k, v in sorted(prof.items())))


if __name__ == "__main__":
    main()
