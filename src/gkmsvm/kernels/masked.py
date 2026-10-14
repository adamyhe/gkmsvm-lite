"""Masked-join pairwise kernel for tables with a mismatch cutoff (CPU).

The packed kernels compare every window pair: K_raw(x, y) = sum_{i,j}
h(m(x_i, y_j)), with h(m) = 0 for m > dmax. Writing

    h(m) = sum_t a_t * C(l - m, l - dmax + t),   t = 0..dmax,

which holds for every m (see docs/design/mask-join-kernel.md), and noting that
C(l - m, j) counts the size-j position subsets on which two l-mers with m
mismatches agree, gives

    K_raw(x, y) = sum_t a_t * #{(i, j, S) : |S| = l - dmax + t, x_i|S == y_j|S}.

Each term is an exact-match join on masked keys, so pairs beyond dmax are
never visited. Results equal the brute-force kernel up to floating-point
summation order.
"""

from __future__ import annotations

import time
from functools import cache
from itertools import combinations
from math import comb

import numpy as np
from numba import njit, prange

# Largest number of masks worth joining; above this the brute-force loop wins
_MAX_MASKS = 1000
# Masks per chunk: bounds the column-side index memory per call
_MASK_CHUNK = 16


@cache
def mask_plan(table_key: tuple[float, ...], l: int, min_matches: int):
    """Mask levels for a weight table, or None if the masked join doesn't apply.

    Returns a list of (j, a_t, positions [M, j] int64) for each level with a
    nonzero coefficient, where positions lists every size-j subset of 0..l-1.
    """
    table = np.asarray(table_key)
    dmax = l - min_matches
    if dmax < 0 or min_matches < 1:
        return None
    js = list(range(l - dmax, l + 1))
    if sum(comb(l, j) for j in js) > _MAX_MASKS:
        return None
    A = np.array([[comb(l - m, j) for j in js] for m in range(dmax + 1)], dtype=np.float64)
    a = np.linalg.solve(A, table[: dmax + 1])
    tol = 1e-12 * np.abs(a).max()
    plan = []
    for j, coef in zip(js, a):
        if abs(coef) <= tol:
            continue
        positions = np.array(list(combinations(range(l), j)), dtype=np.int64)
        plan.append((j, float(coef), positions))
    return plan


@njit(inline="always")
def _masked_key(w, pos, j):
    key = np.uint32(0)
    for i in range(j):
        key |= ((w >> np.uint32(2 * pos[i])) & np.uint32(3)) << np.uint32(2 * i)
    return key


@njit(parallel=True, cache=True)
def _build_column_index(by, seq_of_row, positions, j):
    """Counting sort of column windows by masked key, one index per mask.

    by: [R, W] packed windows; row r belongs to column sequence seq_of_row[r].
    Returns offsets [M, 4^j + 1] and colseq [M, R*W] (column sequence ids in
    key order).
    """
    R, W = by.shape
    M = positions.shape[0]
    K = 1 << (2 * j)
    offsets = np.zeros((M, K + 1), dtype=np.int32)
    colseq = np.empty((M, R * W), dtype=np.int32)
    for m in prange(M):
        pos = positions[m]
        for r in range(R):
            for w in range(W):
                offsets[m, _masked_key(by[r, w], pos, j) + 1] += 1
        for key in range(K):
            offsets[m, key + 1] += offsets[m, key]
        fill = offsets[m, :K].copy()
        for r in range(R):
            s = seq_of_row[r]
            for w in range(W):
                key = _masked_key(by[r, w], pos, j)
                colseq[m, fill[key]] = s
                fill[key] += 1
    return offsets, colseq


@njit(parallel=True, cache=True, fastmath=True)
def _join_rows(bx, positions, j, offsets, colseq, coef, result):
    """result[b, s] += coef for every (query window, column window, mask) match.

    Each query sequence's row is owned by one thread, so there are no races.
    """
    B, Wx = bx.shape
    M = positions.shape[0]
    for b in prange(B):
        row = result[b]
        for m in range(M):
            pos = positions[m]
            for i in range(Wx):
                key = _masked_key(bx[b, i], pos, j)
                for t in range(offsets[m, key], offsets[m, key + 1]):
                    row[colseq[m, t]] += coef


def masked_pairwise(bx_packed, by_packed, plan, by_rc_packed=None):
    """Raw kernel [B, S] from packed windows via masked joins.

    bx_packed: [B, Wx] uint32 query windows.
    by_packed: [S, Wy] uint32 column windows.
    by_rc_packed: [S, Wy] uint32 windows of the column sequences' reverse
        complements; included in the same sum (the RC block of K_raw).
    plan: from mask_plan().
    """
    bx = np.ascontiguousarray(bx_packed)
    S = by_packed.shape[0]
    if by_rc_packed is not None:
        by = np.ascontiguousarray(np.concatenate([by_packed, by_rc_packed]))
        seq_of_row = np.concatenate([np.arange(S), np.arange(S)]).astype(np.int32)
    else:
        by = np.ascontiguousarray(by_packed)
        seq_of_row = np.arange(S, dtype=np.int32)
    result = np.zeros((bx.shape[0], S), dtype=np.float64)
    for j, coef, positions in plan:
        for start in range(0, len(positions), _MASK_CHUNK):
            chunk = positions[start : start + _MASK_CHUNK]
            offsets, colseq = _build_column_index(by, seq_of_row, chunk, j)
            _join_rows(bx, chunk, j, offsets, colseq, coef, result)
    return result


# ---------------------------------------------------------------------------
# CUDA (CuPy)
# ---------------------------------------------------------------------------

_MASKED_CUDA = r"""
__device__ __forceinline__ unsigned int masked_key(
    unsigned int w, const int* __restrict__ pos, int j)
{
    unsigned int key = 0;
    for (int q = 0; q < j; q++)
        key |= ((w >> (2 * pos[q])) & 3u) << (2 * q);
    return key;
}

// counts[m, key + 1] += 1 for every column window (m = mask in chunk)
extern "C" __global__ void masked_hist(
    const unsigned int* __restrict__ by, long long n_win,
    const int* __restrict__ pos, int M, int j, long long K,
    int* __restrict__ counts)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)M * n_win) return;
    int m = (int)(idx / n_win);
    long long w = idx % n_win;
    unsigned int key = masked_key(by[w], pos + m * j, j);
    atomicAdd(&counts[m * (K + 1) + key + 1], 1);
}

// colseq[m, slot] = column sequence of each window, slots from fill (offsets)
extern "C" __global__ void masked_scatter(
    const unsigned int* __restrict__ by, long long n_win, int Wy,
    const int* __restrict__ seq_of_row,
    const int* __restrict__ pos, int M, int j, long long K,
    int* __restrict__ fill, int* __restrict__ colseq)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)M * n_win) return;
    int m = (int)(idx / n_win);
    long long w = idx % n_win;
    unsigned int key = masked_key(by[w], pos + m * j, j);
    int slot = atomicAdd(&fill[m * K + key], 1);
    colseq[m * n_win + slot] = seq_of_row[w / Wy];
}

// One block per query sequence b. Counts matches per column sequence in
// shared memory (exact int32), then result[b, :] += coef * counts. Only this
// block writes row b, so the final update needs no atomics.
extern "C" __global__ void masked_join(
    const unsigned int* __restrict__ bx, int Wx,
    const int* __restrict__ pos, int M, int j, long long K,
    const int* __restrict__ offsets, const int* __restrict__ colseq,
    long long n_win, int S, double coef, double* __restrict__ result)
{
    extern __shared__ int counts[];
    int b = blockIdx.x;
    for (int s = threadIdx.x; s < S; s += blockDim.x) counts[s] = 0;
    __syncthreads();
    int items = M * Wx;
    for (int it = threadIdx.x; it < items; it += blockDim.x) {
        int m = it / Wx;
        int i = it - m * Wx;
        unsigned int key = masked_key(bx[(long long)b * Wx + i], pos + m * j, j);
        const int* off = offsets + m * (K + 1);
        const int* cs = colseq + m * n_win;
        int hi = off[key + 1];
        for (int t = off[key]; t < hi; t++) atomicAdd(&counts[cs[t]], 1);
    }
    __syncthreads();
    for (int s = threadIdx.x; s < S; s += blockDim.x) {
        int c = counts[s];
        if (c) result[(long long)b * S + s] += coef * c;
    }
}
"""

_cuda_module = None
# Column sequences per join launch: the shared-memory counter row is S int32
_GPU_COL_TILE = 8192
# Masks per index build: bounds index memory (offsets + one id per window per mask)
_GPU_MASK_CHUNK = 32
# Set to a dict to accumulate per-stage seconds (synchronizes between stages)
GPU_PROFILE: dict | None = None


def _stage(name, t0):
    """Add the time since t0 to GPU_PROFILE[name]; returns the new start time."""
    import cupy as cp

    if GPU_PROFILE is None:
        return t0
    cp.cuda.get_current_stream().synchronize()
    now = time.perf_counter()
    GPU_PROFILE[name] = GPU_PROFILE.get(name, 0.0) + now - t0
    return now


def _cuda_kernels():
    global _cuda_module
    import cupy as cp

    if _cuda_module is None:
        _cuda_module = cp.RawModule(code=_MASKED_CUDA)
    return (_cuda_module.get_function("masked_hist"),
            _cuda_module.get_function("masked_scatter"),
            _cuda_module.get_function("masked_join"))


def masked_pairwise_gpu(bx_packed, by_packed, plan, by_rc_packed=None):
    """CUDA version of masked_pairwise(); inputs and result are CuPy arrays.

    bx_packed: [B, Wx] uint32; by_packed, by_rc_packed: [S, Wy] uint32.
    Returns [B, S] float64.
    """
    import cupy as cp

    hist, scatter, join = _cuda_kernels()
    bx = cp.ascontiguousarray(bx_packed, dtype=cp.uint32)
    B, Wx = bx.shape
    S_total, Wy = by_packed.shape
    result = cp.zeros((B, S_total), dtype=cp.float64)
    block = 256

    for c0 in range(0, S_total, _GPU_COL_TILE):
        c1 = min(c0 + _GPU_COL_TILE, S_total)
        S = c1 - c0
        rows = [by_packed[c0:c1]] + ([by_rc_packed[c0:c1]] if by_rc_packed is not None else [])
        by = cp.ascontiguousarray(cp.concatenate(rows), dtype=cp.uint32)
        seq_of_row = cp.tile(cp.arange(S, dtype=cp.int32), len(rows))
        n_win = by.shape[0] * Wy
        res_tile = cp.zeros((B, S), dtype=cp.float64)
        for j, coef, positions in plan:
            K = 1 << (2 * j)
            for m0 in range(0, len(positions), _GPU_MASK_CHUNK):
                t0 = time.perf_counter()
                pos = cp.asarray(positions[m0 : m0 + _GPU_MASK_CHUNK], dtype=cp.int32)
                M = pos.shape[0]
                counts = cp.zeros((M, K + 1), dtype=cp.int32)
                grid = (M * n_win + block - 1) // block
                hist((grid,), (block,),
                     (by, np.int64(n_win), pos, np.int32(M), np.int32(j), np.int64(K), counts))
                offsets = cp.cumsum(counts, axis=1, dtype=cp.int32)
                # A copy: with one mask the slice is already contiguous, and the
                # scatter's atomics would otherwise overwrite offsets
                fill = offsets[:, :K].copy()
                colseq = cp.empty((M, n_win), dtype=cp.int32)
                scatter((grid,), (block,),
                        (by, np.int64(n_win), np.int32(Wy), seq_of_row, pos, np.int32(M),
                         np.int32(j), np.int64(K), fill, colseq))
                t0 = _stage(f"index j={j}", t0)
                join((B,), (block,),
                     (bx, np.int32(Wx), pos, np.int32(M), np.int32(j), np.int64(K),
                      offsets, colseq, np.int64(n_win), np.int32(S), np.float64(coef),
                      res_tile),
                     shared_mem=S * 4)
                _stage(f"join j={j}", t0)
        result[:, c0:c1] = res_tile
    return result
