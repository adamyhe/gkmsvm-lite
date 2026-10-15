"""Where does the CUDA masked join spend its time? Ablate it without ncu.

Times variants of the join kernel on one realistic column panel (-t 2 l=11
k=7 d=3, size-8 masks, which dominate):

  full       the production kernel (shared-memory atomic counts)
  no-atomic  same scans, but each thread sums counts in a register
             (removes atomic contention; result is wrong on purpose)
  offsets    keys and bucket-offset reads only, no bucket scan
  keys       masked-key computation only

full - no-atomic ~ cost of the shared atomics; no-atomic - offsets ~ cost of
scanning buckets (colseq reads); offsets - keys ~ cost of the scattered
offset loads.

Usage (GPU server):
  python benchmarks/dev/masked_gpu_ablation.py --rows 5000 --cols 1000
  nsys profile --stats=true -o masked_gram \\
      python benchmarks/timing_train.py --device cuda --sizes 5000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import SEQ_DIR

_VARIANTS = r"""
__device__ __forceinline__ unsigned int masked_key(
    unsigned int w, const int* __restrict__ pos, int j)
{
    unsigned int key = 0;
    for (int q = 0; q < j; q++)
        key |= ((w >> (2 * pos[q])) & 3u) << (2 * q);
    return key;
}

// mode 0 full, 1 no-atomic, 2 offsets only, 3 keys only
extern "C" __global__ void join_variant(
    const unsigned int* __restrict__ bx, int Wx,
    const int* __restrict__ pos, int M, int j, long long K,
    const int* __restrict__ offsets, const int* __restrict__ colseq,
    long long n_win, int S, int lanes, int mode, int* __restrict__ sink)
{
    extern __shared__ int counts[];
    int b = blockIdx.x;
    for (int s = threadIdx.x; s < S; s += blockDim.x) counts[s] = 0;
    __syncthreads();
    int items = M * Wx;
    int group = threadIdx.x / lanes, lane = threadIdx.x % lanes;
    int n_groups = blockDim.x / lanes;
    int acc = 0;
    for (int it = group; it < items; it += n_groups) {
        int m = it / Wx;
        int i = it - m * Wx;
        unsigned int key = masked_key(bx[(long long)b * Wx + i], pos + m * j, j);
        if (mode == 3) { acc += key; continue; }
        const int* off = offsets + m * (K + 1);
        int lo = off[key], hi = off[key + 1];
        if (mode == 2) { acc += hi - lo; continue; }
        const int* cs = colseq + m * n_win;
        for (int t = lo + lane; t < hi; t += lanes) {
            if (mode == 0) atomicAdd(&counts[cs[t]], 1);
            else acc += cs[t];
        }
    }
    __syncthreads();
    if (mode == 0) {
        for (int s = threadIdx.x; s < S; s += blockDim.x) acc += counts[s];
    }
    atomicAdd(&sink[b], acc);  // keep the compiler from dropping the work
}
"""


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rows", type=int, default=5000, help="Query sequences (panel rows)")
    parser.add_argument("--cols", type=int, default=1000, help="Column sequences (one tile)")
    parser.add_argument("--masks", type=int, default=32, help="Size-8 masks per launch")
    parser.add_argument("--lanes", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    import cupy as cp

    from gkmsvm import one_hot_encode, read_fasta
    from gkmsvm.codec import reverse_complement
    from gkmsvm.kernels.esttrunc import EstTruncGkmKernel
    from gkmsvm.kernels.masked import _build_column_index, mask_plan, mask_runs

    n = max(args.rows, args.cols)
    seqs = [s for _, s in read_fasta(str(SEQ_DIR / "gm12878_shared.fa"))][:n]
    X = np.stack([one_hot_encode(s) for s in seqs])
    kern = EstTruncGkmKernel(11, 7, d=3)

    def pack(a):
        return kern.pack(a)

    bx = pack(X[: args.rows])
    cols = X[: args.cols]
    by = np.concatenate([pack(cols), pack(reverse_complement(cols))])
    seq_of_row = np.tile(np.arange(args.cols, dtype=np.int32), 2)
    plan = mask_plan(tuple(kern._mismatch_table), kern.l, kern._min_matches)
    j, _, positions = plan[0]  # size-8 masks
    chunk = positions[: args.masks]
    offsets, colseq = _build_column_index(by, seq_of_row, mask_runs(chunk), j)

    mod = cp.RawModule(code=_VARIANTS)
    fn = mod.get_function("join_variant")
    d_bx = cp.asarray(bx)
    d_pos = cp.asarray(chunk.astype(np.int32))
    d_off = cp.asarray(offsets)
    d_cs = cp.asarray(colseq)
    sink = cp.zeros(args.rows, dtype=cp.int32)
    B, Wx = bx.shape
    K = 1 << (2 * j)
    n_win = by.size
    gpu = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"{gpu}: {B} query x {args.cols} column sequences, {len(chunk)} size-{j} masks, "
          f"{args.lanes} lanes")

    times = {}
    for mode, name in enumerate(["full", "no-atomic", "offsets", "keys"]):
        def launch(mode=mode):
            fn((B,), (256,), (d_bx, np.int32(Wx), d_pos, np.int32(len(chunk)), np.int32(j),
                              np.int64(K), d_off, d_cs, np.int64(n_win), np.int32(args.cols),
                              np.int32(args.lanes), np.int32(mode), sink),
               shared_mem=args.cols * 4)
        launch()
        cp.cuda.Device().synchronize()
        best = np.inf
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            launch()
            cp.cuda.Device().synchronize()
            best = min(best, time.perf_counter() - t0)
        times[name] = best
        print(f"  {name:10s} {best * 1e3:8.2f} ms")
    print(f"  atomics ~{(times['full'] - times['no-atomic']) * 1e3:.1f} ms, "
          f"bucket scans ~{(times['no-atomic'] - times['offsets']) * 1e3:.1f} ms, "
          f"offset loads ~{(times['offsets'] - times['keys']) * 1e3:.1f} ms, "
          f"keys {times['keys'] * 1e3:.1f} ms")


if __name__ == "__main__":
    main()
