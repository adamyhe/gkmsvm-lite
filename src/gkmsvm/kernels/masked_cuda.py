"""The masked join on CUDA (CuPy RawKernels): the column index and the join
of kernels/masked.py on the GPU."""

from __future__ import annotations

from functools import cache

import numpy as np

from gkmsvm.kernels.direct import row_stride
from gkmsvm.kernels.masked import _column_weights, mask_runs

# ---------------------------------------------------------------------------
# CUDA (CuPy)
# ---------------------------------------------------------------------------

_MASKED_CUDA = r"""
// Masks are given as contiguous runs: runs[(m * R + r) * 3 + {0,1,2}] =
// (shift, bitmask, destination); unused runs have bitmask 0 (see mask_runs).
__device__ __forceinline__ unsigned int masked_key(
    unsigned int w, const unsigned int* __restrict__ runs, int R)
{
    unsigned int key = 0;
    for (int r = 0; r < R; r++)
        key |= ((w >> runs[3 * r]) & runs[3 * r + 1]) << runs[3 * r + 2];
    return key;
}

#define PAD_WINDOW 0xFFFFFFFFu  // past a shorter sequence's windows: skipped

// counts[m, key + 1] += 1 for every column window (m = mask in chunk)
extern "C" __global__ void masked_hist(
    const unsigned int* __restrict__ by, long long n_win,
    const unsigned int* __restrict__ runs, int R, int M, long long K,
    int* __restrict__ counts)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)M * n_win) return;
    int m = (int)(idx / n_win);
    long long w = idx % n_win;
    if (by[w] == PAD_WINDOW) return;
    unsigned int key = masked_key(by[w], runs + m * R * 3, R);
    atomicAdd(&counts[m * (K + 1) + key + 1], 1);
}

// colseq[m, slot] = column sequence of each window (uint16: column tiles are
// at most 8192 sequences), slots from fill (bucket starts)
extern "C" __global__ void masked_scatter(
    const unsigned int* __restrict__ by, long long n_win, int Wy,
    const int* __restrict__ seq_of_row,
    const unsigned int* __restrict__ runs, int R, int M, long long K,
    int* __restrict__ fill, unsigned short* __restrict__ colseq)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)M * n_win) return;
    int m = (int)(idx / n_win);
    long long w = idx % n_win;
    if (by[w] == PAD_WINDOW) return;
    unsigned int key = masked_key(by[w], runs + m * R * 3, R);
    int slot = atomicAdd(&fill[m * K + key], 1);
    colseq[m * n_win + slot] = (unsigned short)seq_of_row[w / Wy];
}

// One block per query sequence b. Counts matches per column sequence in
// shared memory (exact int32), then result[b, :] += coef * counts. Only this
// block writes row b, so the final update needs no atomics.
// Lane groups of `lanes` threads share each bucket scan (contiguous reads);
// a group takes one (mask, window) item. The chunk's runs sit in shared
// memory after the counters.
extern "C" __global__ void masked_join(
    const unsigned int* __restrict__ bx, int Wx,
    const unsigned int* __restrict__ runs, int R, int M, long long K,
    const int* __restrict__ offsets, const unsigned short* __restrict__ colseq,
    long long n_win, int S, double coef, double* __restrict__ result, int lanes)
{
    extern __shared__ int smem[];
    int* counts = smem;
    unsigned int* s_runs = (unsigned int*)(smem + S);
    int b = blockIdx.x;
    for (int s = threadIdx.x; s < S; s += blockDim.x) counts[s] = 0;
    for (int t = threadIdx.x; t < M * R * 3; t += blockDim.x) s_runs[t] = runs[t];
    __syncthreads();
    int group = threadIdx.x / lanes, lane = threadIdx.x % lanes;
    int n_groups = blockDim.x / lanes;
    for (int it = group; it < M * Wx; it += n_groups) {
        int m = it / Wx, i = it % Wx;
        unsigned int u = bx[(long long)b * Wx + i];
        if (u == PAD_WINDOW) continue;
        unsigned int key = masked_key(u, s_runs + m * R * 3, R);
        const int* off = offsets + m * (K + 1);
        const unsigned short* cs = colseq + m * n_win;
        int hi = off[key + 1];
        for (int t = off[key] + lane; t < hi; t += lanes) atomicAdd(&counts[cs[t]], 1);
    }
    __syncthreads();
    for (int s = threadIdx.x; s < S; s += blockDim.x) {
        int c = counts[s];
        if (c) result[(long long)b * S + s] += coef * c;
    }
}

// Weighted windows (-t 4/5): masked_scatter that also records each slot's
// window weight (wby: [n_win] uint8)
extern "C" __global__ void masked_scatter_w(
    const unsigned int* __restrict__ by, long long n_win, int Wy,
    const int* __restrict__ seq_of_row, const unsigned char* __restrict__ wby,
    const unsigned int* __restrict__ runs, int R, int M, long long K,
    int* __restrict__ fill, unsigned short* __restrict__ colseq,
    unsigned char* __restrict__ colwt)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)M * n_win) return;
    int m = (int)(idx / n_win);
    long long w = idx % n_win;
    if (by[w] == PAD_WINDOW) return;
    unsigned int key = masked_key(by[w], runs + m * R * 3, R);
    int slot = atomicAdd(&fill[m * K + key], 1);
    colseq[m * n_win + slot] = (unsigned short)seq_of_row[w / Wy];
    colwt[m * n_win + slot] = wby[w];
}

// masked_join with window weights: counts[s] += wx[b, i] * colwt[t], exact in
// uint64 (weights are integers <= 255); wx_stride 0 when every row shares one
// [Wx] row of weights
extern "C" __global__ void masked_join_w(
    const unsigned int* __restrict__ bx, int Wx, const unsigned char* __restrict__ wx,
    int wx_stride,
    const unsigned int* __restrict__ runs, int R, int M, long long K,
    const int* __restrict__ offsets, const unsigned short* __restrict__ colseq,
    const unsigned char* __restrict__ colwt,
    long long n_win, int S, double coef, double* __restrict__ result, int lanes)
{
    extern __shared__ unsigned long long smem64[];
    unsigned long long* counts = smem64;
    unsigned int* s_runs = (unsigned int*)(smem64 + S);
    int b = blockIdx.x;
    for (int s = threadIdx.x; s < S; s += blockDim.x) counts[s] = 0;
    for (int t = threadIdx.x; t < M * R * 3; t += blockDim.x) s_runs[t] = runs[t];
    __syncthreads();
    int group = threadIdx.x / lanes, lane = threadIdx.x % lanes;
    int n_groups = blockDim.x / lanes;
    for (int it = group; it < M * Wx; it += n_groups) {
        int m = it / Wx, i = it % Wx;
        unsigned int u = bx[(long long)b * Wx + i];
        if (u == PAD_WINDOW) continue;
        unsigned int key = masked_key(u, s_runs + m * R * 3, R);
        const int* off = offsets + m * (K + 1);
        const unsigned short* cs = colseq + m * n_win;
        const unsigned char* cw = colwt + m * n_win;
        unsigned long long a = wx[(long long)b * wx_stride + i];
        int hi = off[key + 1];
        for (int t = off[key] + lane; t < hi; t += lanes)
            atomicAdd(&counts[cs[t]], a * cw[t]);
    }
    __syncthreads();
    for (int s = threadIdx.x; s < S; s += blockDim.x) {
        unsigned long long c = counts[s];
        if (c) result[(long long)b * S + s] += coef * (double)c;
    }
}
"""

# Column sequences per join launch: the shared-memory counter row is S int32
# (uint64 with window weights)
_GPU_COL_TILE = 8192
_GPU_COL_TILE_W = 4096
# Masks per index build: bounds index memory (offsets + one id per window per mask)
_GPU_MASK_CHUNK = 32
# Threads that share one bucket scan in the join (power of two <= 32)
_GPU_JOIN_LANES = 16  # best on a Titan Xp (sweep 1-32: 16 > 8 > 4 > 32 > 1)
# One (mask, window) item per lane group beat one query window looping over
# the masks (Titan Xp N=5000 Gram: 1.60 s vs 1.85 s; also slower on an M4)


@cache
def _cuda_kernels():
    import cupy as cp

    module = cp.RawModule(code=_MASKED_CUDA)
    names = (
        "masked_hist",
        "masked_scatter",
        "masked_join",
        "masked_scatter_w",
        "masked_join_w",
    )
    return tuple(module.get_function(n) for n in names)


def masked_pairwise_gpu(
    bx_packed, by_packed, plan, by_rc_packed=None, wx=None, wy=None, wy_rc=None
):
    """CUDA version of masked_pairwise(); inputs and result are CuPy arrays.

    bx_packed: [B, Wx] uint32; by_packed, by_rc_packed: [S, Wy] uint32, PAD
    past shorter sequences' windows (skipped).
    wx, wy, wy_rc: optional row_weights (NumPy uint8, [1 or rows, W]), as in
    masked_pairwise.
    Returns [B, S] float64.
    """
    import cupy as cp

    hist, scatter, join, scatter_w, join_w = _cuda_kernels()
    weighted = wx is not None
    bx = cp.ascontiguousarray(bx_packed, dtype=cp.uint32)
    B, Wx = bx.shape
    S_total, Wy = by_packed.shape
    result = cp.zeros((B, S_total), dtype=cp.float64)
    block = 256
    tile = _GPU_COL_TILE_W if weighted else _GPU_COL_TILE
    if weighted:
        wx_np = np.ascontiguousarray(wx, dtype=np.uint8)
        wx_args = (cp.asarray(wx_np), np.int32(row_stride(wx_np, Wx)))
        blocks_w = [np.asarray(wy, dtype=np.uint8)] + (
            [np.asarray(wy_rc, dtype=np.uint8)] if by_rc_packed is not None else []
        )

    for c0 in range(0, S_total, tile):
        c1 = min(c0 + tile, S_total)
        S = c1 - c0
        rows = [by_packed[c0:c1]] + (
            [by_rc_packed[c0:c1]] if by_rc_packed is not None else []
        )
        by = cp.ascontiguousarray(cp.concatenate(rows), dtype=cp.uint32)
        seq_of_row = cp.tile(cp.arange(S, dtype=cp.int32), len(rows))
        n_win = by.shape[0] * Wy
        col_args = (by, np.int64(n_win), np.int32(Wy), seq_of_row)
        if weighted:  # the rows' weights, forward block then RC block
            tile_w = [w if w.shape[0] == 1 else w[c0:c1] for w in blocks_w]
            wby = cp.asarray(_column_weights(tile_w, S, interleave=False).ravel())
        res_tile = cp.zeros((B, S), dtype=cp.float64)
        for j, coef, positions in plan:
            K = 1 << (2 * j)
            # the joins' trailing arguments
            tail = (np.int64(n_win), np.int32(S), np.float64(coef), res_tile)
            tail += (np.int32(_GPU_JOIN_LANES),)
            for m0 in range(0, len(positions), _GPU_MASK_CHUNK):
                runs_np = mask_runs(positions[m0 : m0 + _GPU_MASK_CHUNK])
                M, R = runs_np.shape[0], runs_np.shape[1]
                runs = cp.asarray(runs_np.ravel())
                mask_args = (runs, np.int32(R), np.int32(M), np.int64(K))
                counts = cp.zeros((M, K + 1), dtype=cp.int32)
                grid = (M * n_win + block - 1) // block
                hist((grid,), (block,), (by, np.int64(n_win), *mask_args, counts))
                offsets = cp.cumsum(counts, axis=1, dtype=cp.int32)
                # A copy: with one mask the slice is already contiguous, and the
                # scatter's atomics would otherwise overwrite offsets
                fill = offsets[:, :K].copy()
                colseq = cp.empty((M, n_win), dtype=cp.uint16)
                if weighted:
                    colwt = cp.empty((M, n_win), dtype=cp.uint8)
                    scatter_w(
                        (grid,),
                        (block,),
                        (*col_args, wby, *mask_args, fill, colseq, colwt),
                    )
                    join_w(
                        (B,),
                        (block,),
                        (bx, np.int32(Wx), *wx_args, *mask_args, offsets, colseq)
                        + (colwt, *tail),
                        shared_mem=S * 8 + M * R * 3 * 4,
                    )
                    continue
                scatter((grid,), (block,), (*col_args, *mask_args, fill, colseq))
                join(
                    (B,),
                    (block,),
                    (bx, np.int32(Wx), *mask_args, offsets, colseq, *tail),
                    shared_mem=(S + M * R * 3) * 4,
                )
        result[:, c0:c1] = res_tile
    return result
