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


def mask_runs(positions: np.ndarray) -> np.ndarray:
    """Masks as contiguous runs: [M, R, 3] uint32 (shift, bitmask, destination).

    A mask's sorted positions split into maximal runs of consecutive
    positions; each run moves as one shift-and-mask, so a key costs at most
    l - j + 1 operations (4 for a size-8 mask of an 11-mer) instead of j
    gathers. Unused runs have bitmask 0.
    """
    M = positions.shape[0]
    runs = []
    for m in range(M):
        pos = sorted(int(p) for p in positions[m])
        out, q, start = [], 0, 0
        for i in range(1, len(pos) + 1):
            if i == len(pos) or pos[i] != pos[i - 1] + 1:
                n = i - start
                out.append((2 * pos[start], (1 << (2 * n)) - 1, 2 * q))
                q += n
                start = i
        runs.append(out)
    R = max(len(r) for r in runs)
    arr = np.zeros((M, R, 3), dtype=np.uint32)
    for m, r in enumerate(runs):
        arr[m, : len(r)] = r
    return arr


@njit(inline="always")
def _masked_key(w, runs):
    """Key of packed window w under one mask's runs ([R, 3])."""
    key = np.uint32(0)
    for r in range(runs.shape[0]):
        key |= ((w >> runs[r, 0]) & runs[r, 1]) << runs[r, 2]
    return key


@njit(parallel=True, cache=True)
def _build_column_index(by, seq_of_row, runs, j):
    """Counting sort of column windows by masked key, one index per mask.

    by: [R, W] packed windows; row r belongs to column sequence seq_of_row[r].
    Returns offsets [M, 4^j + 1] and colseq [M, R*W] (column sequence ids in
    key order).
    """
    R, W = by.shape
    M = runs.shape[0]
    K = 1 << (2 * j)
    offsets = np.zeros((M, K + 1), dtype=np.int32)
    colseq = np.empty((M, R * W), dtype=np.int32)
    for m in prange(M):
        mr = runs[m]
        for r in range(R):
            for w in range(W):
                offsets[m, _masked_key(by[r, w], mr) + 1] += 1
        for key in range(K):
            offsets[m, key + 1] += offsets[m, key]
        fill = offsets[m, :K].copy()
        for r in range(R):
            s = seq_of_row[r]
            for w in range(W):
                key = _masked_key(by[r, w], mr)
                colseq[m, fill[key]] = s
                fill[key] += 1
    return offsets, colseq


@njit(parallel=True, cache=True, fastmath=True)
def _join_rows(bx, runs, offsets, colseq, coef, result):
    """result[b, s] += coef for every (query window, column window, mask) match.

    Each query sequence's row is owned by one thread, so there are no races.
    """
    B, Wx = bx.shape
    M = runs.shape[0]
    for b in prange(B):
        row = result[b]
        for m in range(M):
            mr = runs[m]
            for i in range(Wx):
                key = _masked_key(bx[b, i], mr)
                for t in range(offsets[m, key], offsets[m, key + 1]):
                    row[colseq[m, t]] += coef


@njit(parallel=True, cache=True, fastmath=True)
def _join_rows_upper(bx, runs, offsets, colseq, coef, result, row_min):
    """_join_rows, skipping columns below row_min[b] for row b.

    Needs each bucket's column ids in ascending order, which the counting
    sort gives when the indexed rows are one block of column sequences in
    order. Scans each bucket backwards and stops below row_min[b].
    """
    B, Wx = bx.shape
    M = runs.shape[0]
    for idx in prange(B):
        # Alternate rows from both ends: rows late in a diagonal block skip
        # most columns, and prange hands each thread a contiguous range
        b = idx // 2 if idx % 2 == 0 else B - 1 - idx // 2
        row = result[b]
        cmin = row_min[b]
        for m in range(M):
            mr = runs[m]
            for i in range(Wx):
                key = _masked_key(bx[b, i], mr)
                # Ids ascend within a bucket: scan from the end and stop at
                # the first column below cmin (cheaper than a binary search)
                lo = offsets[m, key]
                t = offsets[m, key + 1] - 1
                while t >= lo:
                    c = colseq[m, t]
                    if c < cmin:
                        break
                    row[c] += coef
                    t -= 1


def _masked_pairwise_upper(bx, by_blocks, plan, diag_offset):
    """Rows 0..B-1 against columns diag_offset.. of the same sequences.

    Only entries with column >= row are computed (the rest stay 0).
    by_blocks: column window arrays (forward, and RC if any). Their rows are
    interleaved per sequence, so one index serves both and each bucket's
    column ids come out ascending.
    """
    B = bx.shape[0]
    S, W = by_blocks[0].shape
    by = np.ascontiguousarray(np.stack(by_blocks, axis=1).reshape(-1, W))
    seq_of_row = np.repeat(np.arange(S, dtype=np.int32), len(by_blocks))
    row_min = np.maximum(0, np.arange(B) - diag_offset).astype(np.int64)
    result = np.zeros((B, S), dtype=np.float64)
    for j, coef, positions in plan:
        for start in range(0, len(positions), _MASK_CHUNK):
            runs = mask_runs(positions[start : start + _MASK_CHUNK])
            offsets, colseq = _build_column_index(by, seq_of_row, runs, j)
            _join_rows_upper(bx, runs, offsets, colseq, coef, result, row_min)
    return result


def masked_pairwise(bx_packed, by_packed, plan, by_rc_packed=None, diag_offset=None):
    """Raw kernel [B, S] from packed windows via masked joins.

    bx_packed: [B, Wx] uint32 query windows.
    by_packed: [S, Wy] uint32 column windows.
    by_rc_packed: [S, Wy] uint32 windows of the column sequences' reverse
        complements; included in the same sum (the RC block of K_raw).
    plan: from mask_plan().
    diag_offset: the rows are sequences 0..B-1 and the columns sequences
        diag_offset..diag_offset+S-1 of one set (a self-Gram column panel);
        entries below the diagonal are skipped and left 0.
    """
    bx = np.ascontiguousarray(bx_packed)
    if diag_offset is not None:
        blocks = [by_packed] + ([by_rc_packed] if by_rc_packed is not None else [])
        return _masked_pairwise_upper(bx, blocks, plan, diag_offset)
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
            runs = mask_runs(chunk)
            offsets, colseq = _build_column_index(by, seq_of_row, runs, j)
            _join_rows(bx, runs, offsets, colseq, coef, result)
    return result


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
    unsigned int key = masked_key(by[w], runs + m * R * 3, R);
    int slot = atomicAdd(&fill[m * K + key], 1);
    colseq[m * n_win + slot] = (unsigned short)seq_of_row[w / Wy];
}

// One block per query sequence b. Counts matches per column sequence in
// shared memory (exact int32), then result[b, :] += coef * counts. Only this
// block writes row b, so the final update needs no atomics.
// Lane groups of `lanes` threads share each bucket scan (contiguous reads).
// window_outer = 1: a group takes one query window, loads it once and loops
// over the masks; 0: a group takes one (mask, window) item. The chunk's runs
// sit in shared memory after the counters.
extern "C" __global__ void masked_join(
    const unsigned int* __restrict__ bx, int Wx,
    const unsigned int* __restrict__ runs, int R, int M, long long K,
    const int* __restrict__ offsets, const unsigned short* __restrict__ colseq,
    long long n_win, int S, double coef, double* __restrict__ result, int lanes,
    int window_outer)
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
    int items = window_outer ? Wx : M * Wx;
    for (int it = group; it < items; it += n_groups) {
        int i = window_outer ? it : it % Wx;
        int m0 = window_outer ? 0 : it / Wx;
        int m1 = window_outer ? M : m0 + 1;
        unsigned int w = bx[(long long)b * Wx + i];
        for (int m = m0; m < m1; m++) {
            unsigned int key = masked_key(w, s_runs + m * R * 3, R);
            const int* off = offsets + m * (K + 1);
            const unsigned short* cs = colseq + m * n_win;
            int hi = off[key + 1];
            for (int t = off[key] + lane; t < hi; t += lanes) atomicAdd(&counts[cs[t]], 1);
        }
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
# Threads that share one bucket scan in the join (power of two <= 32)
GPU_JOIN_LANES = 16  # best on a Titan Xp (sweep 1-32: 16 > 8 > 4 > 32 > 1)
# 0: one (mask, window) item per lane group; 1: each group loads a query
# window once and loops over the masks (slower on a Titan Xp and an M4)
GPU_JOIN_WINDOW_OUTER = 0  # Titan Xp N=5000 Gram: item 1.60 s, window 1.85 s


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
                runs_np = mask_runs(positions[m0 : m0 + _GPU_MASK_CHUNK])
                M, R = runs_np.shape[0], runs_np.shape[1]
                runs = cp.asarray(runs_np.ravel())
                counts = cp.zeros((M, K + 1), dtype=cp.int32)
                grid = (M * n_win + block - 1) // block
                hist((grid,), (block,),
                     (by, np.int64(n_win), runs, np.int32(R), np.int32(M), np.int64(K), counts))
                offsets = cp.cumsum(counts, axis=1, dtype=cp.int32)
                # A copy: with one mask the slice is already contiguous, and the
                # scatter's atomics would otherwise overwrite offsets
                fill = offsets[:, :K].copy()
                colseq = cp.empty((M, n_win), dtype=cp.uint16)
                scatter((grid,), (block,),
                        (by, np.int64(n_win), np.int32(Wy), seq_of_row, runs, np.int32(R),
                         np.int32(M), np.int64(K), fill, colseq))
                t0 = _stage(f"index j={j}", t0)
                join((B,), (block,),
                     (bx, np.int32(Wx), runs, np.int32(R), np.int32(M), np.int64(K),
                      offsets, colseq, np.int64(n_win), np.int32(S), np.float64(coef),
                      res_tile, np.int32(GPU_JOIN_LANES), np.int32(GPU_JOIN_WINDOW_OUTER)),
                     shared_mem=(S + M * R * 3) * 4)
                _stage(f"join j={j}", t0)
        result[:, c0:c1] = res_tile
    return result


# ---------------------------------------------------------------------------
# Center-weighted kernel (-t 4) and self-kernels
# ---------------------------------------------------------------------------


def weighted_mask_plan(l: int, k: int, pos_weights: np.ndarray):
    """Plan for the center-weighted kernel: g(u, v) = e_k(weights on agreeing positions).

    e_k sums prod(w_T) over the size-k subsets T of the agreeing positions,
    so every size-k mask is a level with coefficient prod(w_T). Masks with
    equal coefficients share a level (products are taken over sorted weights,
    so equal multisets give identical values), which keeps each GPU join's
    counts in exact integers.
    """
    w = np.asarray(pos_weights, dtype=np.float64)
    groups: dict[float, list[tuple[int, ...]]] = {}
    for T in combinations(range(l), k):
        coef = float(np.prod(np.sort(w[list(T)])))
        groups.setdefault(coef, []).append(T)
    return [(k, coef, np.array(Ts, dtype=np.int64)) for coef, Ts in groups.items()]


@njit(parallel=True, cache=True)
def _self_join_counts(bx, brc, runs, coef, out):
    """out[b] += coef * #(window pairs of b, and of b vs rc(b), agreeing on a mask).

    brc has zero columns when there is no RC term.
    """
    B, W = bx.shape
    Wr = brc.shape[1]
    M = runs.shape[0]
    for b in prange(B):
        keys = np.empty(W, dtype=np.uint32)
        rkeys = np.empty(Wr, dtype=np.uint32)
        acc = 0
        for m in range(M):
            mr = runs[m]
            for i in range(W):
                keys[i] = _masked_key(bx[b, i], mr)
            for i in range(Wr):
                rkeys[i] = _masked_key(brc[b, i], mr)
            keys.sort()
            rkeys.sort()
            # sum over keys of n_fwd * (n_fwd + n_rc), by merging sorted runs
            i = 0
            r = 0
            while i < W:
                key = keys[i]
                n = 0
                while i < W and keys[i] == key:
                    n += 1
                    i += 1
                while r < Wr and rkeys[r] < key:
                    r += 1
                nr = 0
                while r < Wr and rkeys[r] == key:
                    nr += 1
                    r += 1
                acc += n * (n + nr)
        out[b] += coef * acc


def masked_self_kernel(bx_packed, plan, bx_rc_packed=None):
    """K_raw(x, x) for each row (CPU): forward block plus the RC cross block."""
    bx = np.ascontiguousarray(bx_packed)
    brc = (np.ascontiguousarray(bx_rc_packed) if bx_rc_packed is not None
           else np.zeros((bx.shape[0], 0), dtype=np.uint32))
    out = np.zeros(bx.shape[0], dtype=np.float64)
    for j, coef, positions in plan:
        _self_join_counts(bx, brc, mask_runs(positions), coef, out)
    return out


# ---------------------------------------------------------------------------
# Metal (MLX)
# ---------------------------------------------------------------------------

# Same structure as masked_join (CUDA). Metal has no float64, so each launch
# writes exact int32 counts per (query, column) and the caller combines
# levels in float64 on the CPU (unified memory makes that copy cheap).
_METAL_MASKED_JOIN = """
    uint b = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint tg = threads_per_threadgroup.x;
    int Wx = params[0], M = params[1], R = params[2], S = params[3], lanes = params[4];
    uint K1 = (uint)params[5];
    uint n_win = (uint)params[6];
    threadgroup atomic_int counts[TILE];
    for (int s = tid; s < S; s += tg)
        atomic_store_explicit(&counts[s], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Each lane group takes one (mask, window) item and scans its bucket
    // (runs: shift, bitmask, destination per run). Measured on an M4 this
    // beats loading each window once and looping masks inside (0.60 s vs
    // 0.65 s at 2000x2000).
    int group = tid / lanes, lane = tid % lanes, n_groups = tg / lanes;
    for (int it = group; it < M * Wx; it += n_groups) {
        int m = it / Wx;
        int i = it - m * Wx;
        uint w = bx[b * Wx + i];
        {
            uint key = 0;
            for (int r = 0; r < R; r++) {
                uint o = (uint)(m * R + r) * 3;
                key |= ((w >> runs[o]) & runs[o + 1]) << runs[o + 2];
            }
            uint base = (uint)m * K1;
            int hi = offsets[base + key + 1];
            uint cbase = (uint)m * n_win;
            for (int t = offsets[base + key] + lane; t < hi; t += lanes)
                atomic_fetch_add_explicit(&counts[colseq[cbase + t]], 1, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = tid; s < S; s += tg)
        out[b * S + s] = atomic_load_explicit(&counts[s], memory_order_relaxed);
"""

_METAL_HIST = """
    uint idx = thread_position_in_grid.x;
    uint M = (uint)params[0], R = (uint)params[1], K1 = (uint)params[2], n_win = (uint)params[3];
    if (idx >= M * n_win) return;
    uint m = idx / n_win, w = idx - m * n_win;
    uint v = by[w], key = 0;
    for (uint r = 0; r < R; r++) {
        uint o = (m * R + r) * 3;
        key |= ((v >> runs[o]) & runs[o + 1]) << runs[o + 2];
    }
    atomic_fetch_add_explicit(&counts[m * K1 + key + 1], 1, memory_order_relaxed);
"""

# slot = bucket start + atomic counter; colseq is atomic only because MLX makes
# every output of an atomic kernel atomic
_METAL_SCATTER = """
    uint idx = thread_position_in_grid.x;
    uint M = (uint)params[0], R = (uint)params[1], K1 = (uint)params[2], n_win = (uint)params[3];
    uint Wy = (uint)params[4];
    if (idx >= M * n_win) return;
    uint m = idx / n_win, w = idx - m * n_win;
    uint v = by[w], key = 0;
    for (uint r = 0; r < R; r++) {
        uint o = (m * R + r) * 3;
        key |= ((v >> runs[o]) & runs[o + 1]) << runs[o + 2];
    }
    int slot = offsets[m * K1 + key]
             + atomic_fetch_add_explicit(&fill[m * (K1 - 1) + key], 1, memory_order_relaxed);
    atomic_store_explicit(&colseq[m * n_win + slot], seq_of_row[w / Wy], memory_order_relaxed);
"""

_metal_hist = None
_metal_scatter = None
# Build the column index on the GPU (True) or with Numba on the CPU (False)
METAL_INDEX = True


def _metal_index(by_mx, seq_of_row_mx, chunk, j, Wy):
    """Column index (offsets [M*(K+1)], colseq [M*n_win]) built in Metal."""
    import mlx.core as mx

    global _metal_hist, _metal_scatter
    if _metal_hist is None:
        _metal_hist = mx.fast.metal_kernel(
            name="gkm_masked_hist", input_names=["by", "runs", "params"],
            output_names=["counts"], source=_METAL_HIST, atomic_outputs=True)
        _metal_scatter = mx.fast.metal_kernel(
            name="gkm_masked_scatter",
            input_names=["by", "seq_of_row", "runs", "offsets", "params"],
            output_names=["fill", "colseq"], source=_METAL_SCATTER, atomic_outputs=True)
    runs_np = mask_runs(chunk)
    M, R = runs_np.shape[0], runs_np.shape[1]
    K = 1 << (2 * j)
    n_win = by_mx.shape[0]
    pos = mx.array(runs_np.ravel())
    params = mx.array([M, R, K + 1, n_win, Wy], dtype=mx.int32)
    threads = 256
    grid = ((M * n_win + threads - 1) // threads) * threads
    counts = _metal_hist(
        inputs=[by_mx, pos, params], template=[], grid=(grid, 1, 1),
        threadgroup=(threads, 1, 1), output_shapes=[(M * (K + 1),)],
        output_dtypes=[mx.int32], init_value=0)[0]
    offsets = mx.cumsum(counts.reshape(M, K + 1), axis=1).reshape(-1)
    _, colseq = _metal_scatter(
        inputs=[by_mx, seq_of_row_mx, pos, offsets, params], template=[],
        grid=(grid, 1, 1), threadgroup=(threads, 1, 1),
        output_shapes=[(M * K,), (M * n_win,)], output_dtypes=[mx.int32, mx.int32],
        init_value=0)
    return offsets, colseq


_metal_join = None
# Columns per launch: the threadgroup counter row (32 KB limit) is TILE int32
_METAL_COL_TILE = 4096
_METAL_LANES = 16
_METAL_THREADS = 256


def _get_metal_join():
    global _metal_join
    if _metal_join is None:
        import mlx.core as mx

        _metal_join = mx.fast.metal_kernel(
            name="gkm_masked_join",
            input_names=["bx", "runs", "offsets", "colseq", "params"],
            output_names=["out"],
            source=_METAL_MASKED_JOIN,
        )
    return _metal_join


def masked_pairwise_mlx(bx_packed, by_packed, plan, by_rc_packed=None) -> np.ndarray:
    """Metal version of masked_pairwise(); the index is built in Metal too
    unless METAL_INDEX is False (then with Numba on the CPU).

    Inputs may be NumPy or MLX uint32 arrays. Returns [B, S] float64 NumPy,
    combined exactly from per-level integer counts.
    """
    import mlx.core as mx

    join = _get_metal_join()
    bx = np.ascontiguousarray(np.asarray(bx_packed), dtype=np.uint32)
    by_all = np.asarray(by_packed, dtype=np.uint32)
    by_rc_all = None if by_rc_packed is None else np.asarray(by_rc_packed, dtype=np.uint32)
    B, Wx = bx.shape
    S_total, Wy = by_all.shape
    bx_mx = mx.array(bx.ravel())
    result = np.zeros((B, S_total), dtype=np.float64)

    for c0 in range(0, S_total, _METAL_COL_TILE):
        c1 = min(c0 + _METAL_COL_TILE, S_total)
        S = c1 - c0
        rows = [by_all[c0:c1]] + ([by_rc_all[c0:c1]] if by_rc_all is not None else [])
        by = np.ascontiguousarray(np.concatenate(rows))
        seq_of_row = np.tile(np.arange(S, dtype=np.int32), len(rows))
        n_win = by.shape[0] * Wy
        if METAL_INDEX:
            by_mx, seq_mx = mx.array(by.ravel()), mx.array(seq_of_row)
        for j, coef, positions in plan:
            K1 = (1 << (2 * j)) + 1
            level = None
            for m0 in range(0, len(positions), _MASK_CHUNK):
                chunk = positions[m0 : m0 + _MASK_CHUNK]
                if METAL_INDEX:
                    offsets, colseq = _metal_index(by_mx, seq_mx, chunk, j, Wy)
                else:
                    offsets, colseq = _build_column_index(by, seq_of_row, mask_runs(chunk), j)
                    offsets, colseq = mx.array(offsets.ravel()), mx.array(colseq.ravel())
                runs_np = mask_runs(chunk)
                params = mx.array([Wx, runs_np.shape[0], runs_np.shape[1], S, _METAL_LANES,
                                   K1, n_win], dtype=mx.int32)
                counts = join(
                    inputs=[bx_mx, mx.array(runs_np.ravel()), offsets, colseq, params],
                    template=[("TILE", _METAL_COL_TILE)],
                    grid=(B * _METAL_THREADS, 1, 1),
                    threadgroup=(_METAL_THREADS, 1, 1),
                    output_shapes=[(B * S,)],
                    output_dtypes=[mx.int32],
                )[0]
                level = counts if level is None else level + counts
            result[:, c0:c1] += coef * np.array(level, dtype=np.float64).reshape(B, S)
    return result
