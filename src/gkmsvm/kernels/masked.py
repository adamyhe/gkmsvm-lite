"""Masked-join pairwise kernel for tables with a mismatch cutoff (CPU, CUDA, Metal).

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

from functools import cache
from itertools import combinations
from math import comb

import numpy as np
from numba import njit, prange

from gkmsvm.kernels.direct import _spans, row_stride

# Largest number of masks worth joining; above this the brute-force loop wins
_MAX_MASKS = 1000
# Masks per chunk: bounds the column-side index memory per call
_MASK_CHUNK = 16
# CPU index memory per chunk (offsets + one int32 id per column window per
# mask); fewer masks per chunk when the columns are large
_INDEX_BYTES = 512 * 2**20


def _cpu_mask_chunk(n_win: int, j: int) -> int:
    per_mask = 4 * (n_win + (1 << (2 * j)) + 1)
    return max(1, min(_MASK_CHUNK, _INDEX_BYTES // per_mask))


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
    A = np.array(
        [[comb(l - m, j) for j in js] for m in range(dmax + 1)], dtype=np.float64
    )
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
def _build_column_index(by, seq_of_row, runs, j, wby):
    """Counting sort of column windows by masked key, one index per mask.

    by: [R, W] packed windows; row r belongs to column sequence seq_of_row[r].
    PAD_WINDOW (padding past a shorter sequence's windows) is left out.
    wby: [R, W] uint8 window weights, or [0, 0] for unweighted windows.
    Returns offsets [M, 4^j + 1], colseq [M, R*W] (column sequence ids in
    key order; the tail past offsets[m, -1] unused) and colwt (their
    weights; [M, 0] when unweighted).
    """
    R, W = by.shape
    M = runs.shape[0]
    K = 1 << (2 * j)
    weighted = wby.shape[0] > 0
    offsets = np.zeros((M, K + 1), dtype=np.int32)
    colseq = np.empty((M, R * W), dtype=np.int32)
    colwt = np.empty((M, R * W if weighted else 0), dtype=np.uint8)
    sp = _spans(by)  # each row's non-PAD windows
    for m in prange(M):
        mr = runs[m]
        for r in range(R):
            for w in range(sp[r, 0], sp[r, 1]):
                offsets[m, _masked_key(by[r, w], mr) + 1] += 1
        for key in range(K):
            offsets[m, key + 1] += offsets[m, key]
        fill = offsets[m, :K].copy()
        for r in range(R):
            s = seq_of_row[r]
            for w in range(sp[r, 0], sp[r, 1]):
                key = _masked_key(by[r, w], mr)
                colseq[m, fill[key]] = s
                if weighted:
                    colwt[m, fill[key]] = wby[r, w]
                fill[key] += 1
    return offsets, colseq, colwt


@njit(parallel=True, cache=True, fastmath=True)
def _join_rows(bx, runs, offsets, colseq, coef, result, wx, colwt):
    """result[b, s] += coef for every (query window, column window, mask) match,
    times wx[b, i] colwt[t] for weighted windows (colwt with 0 columns:
    unweighted; wx [1, Wx] when every row has the same weights). PAD query
    windows are skipped.

    Each query sequence's row is owned by one thread, so there are no races.
    """
    B = bx.shape[0]
    M = runs.shape[0]
    weighted = colwt.shape[1] > 0
    sx = _spans(bx)  # each row's non-PAD windows
    for b in prange(B):
        row = result[b]
        wb = wx[min(np.int64(b), wx.shape[0] - 1)]
        for m in range(M):
            mr = runs[m]
            for i in range(sx[b, 0], sx[b, 1]):
                key = _masked_key(bx[b, i], mr)
                if weighted:
                    c = coef * wb[i]
                    for t in range(offsets[m, key], offsets[m, key + 1]):
                        row[colseq[m, t]] += c * colwt[m, t]
                else:
                    for t in range(offsets[m, key], offsets[m, key + 1]):
                        row[colseq[m, t]] += coef


@njit(parallel=True, cache=True, fastmath=True)
def _join_rows_upper(bx, runs, offsets, colseq, coef, result, row_min, wx, colwt):
    """_join_rows, skipping columns below row_min[b] for row b.

    Needs each bucket's column ids in ascending order, which the counting
    sort gives when the indexed rows are one block of column sequences in
    order. Scans each bucket backwards and stops below row_min[b].
    """
    B = bx.shape[0]
    M = runs.shape[0]
    weighted = colwt.shape[1] > 0
    sx = _spans(bx)  # each row's non-PAD windows
    for idx in prange(B):
        # Alternate rows from both ends: rows late in a diagonal block skip
        # most columns, and prange hands each thread a contiguous range
        b = idx // 2 if idx % 2 == 0 else B - 1 - idx // 2
        row = result[b]
        cmin = row_min[b]
        wb = wx[min(np.int64(b), wx.shape[0] - 1)]
        for m in range(M):
            mr = runs[m]
            for i in range(sx[b, 0], sx[b, 1]):
                key = _masked_key(bx[b, i], mr)
                # Ids ascend within a bucket: scan from the end and stop at
                # the first column below cmin (cheaper than a binary search)
                lo = offsets[m, key]
                t = offsets[m, key + 1] - 1
                ci = coef * wb[i] if weighted else coef
                while t >= lo:
                    c = colseq[m, t]
                    if c < cmin:
                        break
                    row[c] += ci * colwt[m, t] if weighted else ci
                    t -= 1


def _column_weights(blocks, S, interleave):
    """[R, W] uint8 weights of the column rows built from blocks (row_weights
    of S column sequences each: [S, W], or [1, W] shared, or None): stacked
    block after block, or with interleave alternating per sequence; [0, 0]
    without weights."""
    if blocks[0] is None:
        return np.zeros((0, 0), dtype=np.uint8)
    full = [np.broadcast_to(b, (S, b.shape[1])) for b in blocks]
    w = (
        np.stack(full, axis=1).reshape(-1, full[0].shape[1])
        if interleave
        else np.concatenate(full)
    )
    return np.ascontiguousarray(w, dtype=np.uint8)


def _row_weights(wx):
    """Query row weights as float64 [1 or B, Wx], [1, 0] when unweighted."""
    return np.zeros((1, 0)) if wx is None else np.asarray(wx, dtype=np.float64)


def _column_indexes(by, seq_of_row, wby, plan):
    """(coef, runs, offsets, colseq, colwt) for each chunk of each level's
    masks: the column index of _build_column_index."""
    for j, coef, positions in plan:
        step = _cpu_mask_chunk(by.size, j)
        for start in range(0, len(positions), step):
            runs = mask_runs(positions[start : start + step])
            yield coef, runs, *_build_column_index(by, seq_of_row, runs, j, wby)


def _masked_pairwise_upper(bx, by_blocks, plan, diag_offset, wx, wy_blocks):
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
    wby = _column_weights(wy_blocks, S, interleave=True)
    row_min = np.maximum(0, np.arange(B) - diag_offset).astype(np.int64)
    result = np.zeros((B, S), dtype=np.float64)
    wx = _row_weights(wx)
    for coef, runs, offsets, colseq, colwt in _column_indexes(
        by, seq_of_row, wby, plan
    ):
        _join_rows_upper(bx, runs, offsets, colseq, coef, result, row_min, wx, colwt)
    return result


def masked_pairwise(
    bx_packed,
    by_packed,
    plan,
    by_rc_packed=None,
    diag_offset=None,
    wx=None,
    wy=None,
    wy_rc=None,
):
    """Raw kernel [B, S] from packed windows via masked joins.

    bx_packed: [B, Wx] uint32 query windows.
    by_packed: [S, Wy] uint32 column windows.
    by_rc_packed: [S, Wy] uint32 windows of the column sequences' reverse
        complements; included in the same sum (the RC block of K_raw).
    plan: from mask_plan().
    diag_offset: the rows are sequences 0..B-1 and the columns sequences
        diag_offset..diag_offset+S-1 of one set (a self-Gram column panel);
        entries below the diagonal are skipped and left 0.
    wx, wy, wy_rc: row_weights of the rows, columns and RC columns ([1, W]
        when shared by every row, [R, W] for padded rows), or None.
    Rows and columns may hold PAD_WINDOW past a shorter sequence's windows;
    it is skipped.
    """
    bx = np.ascontiguousarray(bx_packed)
    if diag_offset is not None:
        blocks = [by_packed] + ([by_rc_packed] if by_rc_packed is not None else [])
        w_blocks = [wy] + ([wy_rc] if by_rc_packed is not None else [])
        return _masked_pairwise_upper(bx, blocks, plan, diag_offset, wx, w_blocks)
    S = by_packed.shape[0]
    if by_rc_packed is not None:
        by = np.ascontiguousarray(np.concatenate([by_packed, by_rc_packed]))
        seq_of_row = np.concatenate([np.arange(S), np.arange(S)]).astype(np.int32)
        wby = _column_weights([wy, wy_rc], S, interleave=False)
    else:
        by = np.ascontiguousarray(by_packed)
        seq_of_row = np.arange(S, dtype=np.int32)
        wby = _column_weights([wy], S, interleave=False)
    result = np.zeros((bx.shape[0], S), dtype=np.float64)
    wx = _row_weights(wx)
    for coef, runs, offsets, colseq, colwt in _column_indexes(
        by, seq_of_row, wby, plan
    ):
        _join_rows(bx, runs, offsets, colseq, coef, result, wx, colwt)
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


# ---------------------------------------------------------------------------
# Metal (MLX)
# ---------------------------------------------------------------------------

# Key of packed window w under one mask's R runs (shift, bitmask,
# destination; see mask_runs). A template: MLX passes small inputs (runs) in
# the constant address space, larger ones in device
_METAL_MASKED_KEY_HEADER = """
template <typename P>
inline uint masked_key(uint w, P runs, int R) {
    uint key = 0;
    for (int r = 0; r < R; r++)
        key |= ((w >> runs[3 * r]) & runs[3 * r + 1]) << runs[3 * r + 2];
    return key;
}
"""

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
    // Each lane group takes one (mask, window) item and scans its bucket.
    // Measured on an M4 this
    // beats loading each window once and looping masks inside (0.60 s vs
    // 0.65 s at 2000x2000).
    int group = tid / lanes, lane = tid % lanes, n_groups = tg / lanes;
    for (int it = group; it < M * Wx; it += n_groups) {
        int m = it / Wx;
        int i = it - m * Wx;
        uint w = bx[b * Wx + i];
        if (w != 0xFFFFFFFFu) {  // PAD past a shorter sequence's windows
            uint key = masked_key(w, runs + m * R * 3, R);
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

# With window weights (-t 4/5): counts[s] += wx[b, i] * the entry's weight
# (packed in colseq's top bits by _METAL_SCATTER_W), exact in 64 bits: a low
# uint32 word and a carry word, which the adding thread bumps when its add
# wraps the low word (atomic_fetch_add returns the word before its own add)
_METAL_MASKED_JOIN_W = """
    uint b = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint tg = threads_per_threadgroup.x;
    int Wx = params[0], M = params[1], R = params[2], S = params[3], lanes = params[4];
    uint K1 = (uint)params[5];
    uint n_win = (uint)params[6];
    uint wx_stride = (uint)params[7];
    threadgroup atomic_uint lo[TILE];
    threadgroup atomic_uint hi[TILE];
    for (int s = tid; s < S; s += tg) {
        atomic_store_explicit(&lo[s], 0u, memory_order_relaxed);
        atomic_store_explicit(&hi[s], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int group = tid / lanes, lane = tid % lanes, n_groups = tg / lanes;
    for (int it = group; it < M * Wx; it += n_groups) {
        int m = it / Wx;
        int i = it - m * Wx;
        uint w = bx[b * Wx + i];
        if (w != 0xFFFFFFFFu) {  // PAD past a shorter sequence's windows
            uint key = masked_key(w, runs + m * R * 3, R);
            uint a = (uint)wx[b * wx_stride + i];
            uint base = (uint)m * K1;
            int hi_t = offsets[base + key + 1];
            uint cbase = (uint)m * n_win;
            for (int t = offsets[base + key] + lane; t < hi_t; t += lanes) {
                uint e = (uint)colseq[cbase + t];
                uint s = e & 0xFFFFu, v = a * (e >> 16);
                uint old = atomic_fetch_add_explicit(&lo[s], v, memory_order_relaxed);
                if (old + v < old)
                    atomic_fetch_add_explicit(&hi[s], 1u, memory_order_relaxed);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = tid; s < S; s += tg) {
        out_lo[b * S + s] = atomic_load_explicit(&lo[s], memory_order_relaxed);
        out_hi[b * S + s] = atomic_load_explicit(&hi[s], memory_order_relaxed);
    }
"""

_METAL_HIST = """
    uint idx = thread_position_in_grid.x;
    uint M = (uint)params[0], R = (uint)params[1], K1 = (uint)params[2], n_win = (uint)params[3];
    if (idx >= M * n_win) return;
    uint m = idx / n_win, w = idx - m * n_win;
    uint v = by[w];
    if (v == 0xFFFFFFFFu) return;  // PAD
    uint key = masked_key(v, runs + m * R * 3, (int)R);
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
    uint v = by[w];
    if (v == 0xFFFFFFFFu) return;  // PAD
    uint key = masked_key(v, runs + m * R * 3, (int)R);
    int slot = offsets[m * K1 + key]
             + atomic_fetch_add_explicit(&fill[m * (K1 - 1) + key], 1, memory_order_relaxed);
    // per_window: seq_of_row holds one entry per window (weighted), else per row
    uint e = (uint)params[5] ? (uint)w : w / Wy;
    atomic_store_explicit(&colseq[m * n_win + slot], seq_of_row[e], memory_order_relaxed);
"""


@cache
def _metal_index_kernels():
    import mlx.core as mx

    hist = mx.fast.metal_kernel(
        name="gkm_masked_hist",
        input_names=["by", "runs", "params"],
        output_names=["counts"],
        source=_METAL_HIST,
        header=_METAL_MASKED_KEY_HEADER,
        atomic_outputs=True,
    )
    scatter = mx.fast.metal_kernel(
        name="gkm_masked_scatter",
        input_names=["by", "seq_of_row", "runs", "offsets", "params"],
        output_names=["fill", "colseq"],
        source=_METAL_SCATTER,
        header=_METAL_MASKED_KEY_HEADER,
        atomic_outputs=True,
    )
    return hist, scatter


def _metal_index(by_mx, seq_of_row_mx, chunk, j, Wy):
    """Column index (offsets [M*(K+1)], colseq [M*n_win]) built in Metal, and
    the chunk's mask_runs. With window weights, seq_of_row_mx holds each
    column row's sequence id in its low 16 bits and, per window, the weight
    in the top bits ([n_win]; see masked_pairwise_mlx), and colseq gets
    those entries."""
    import mlx.core as mx

    hist, scatter = _metal_index_kernels()
    runs_np = mask_runs(chunk)
    M, R = runs_np.shape[0], runs_np.shape[1]
    K = 1 << (2 * j)
    n_win = by_mx.shape[0]
    pos = mx.array(runs_np.ravel())
    per_window = int(seq_of_row_mx.shape[0] == n_win)  # (with Wy = 1 both agree)
    params = mx.array([M, R, K + 1, n_win, Wy, per_window], dtype=mx.int32)
    threads = 256
    grid = ((M * n_win + threads - 1) // threads) * threads
    counts = hist(
        inputs=[by_mx, pos, params],
        template=[],
        grid=(grid, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(M * (K + 1),)],
        output_dtypes=[mx.int32],
        init_value=0,
    )[0]
    offsets = mx.cumsum(counts.reshape(M, K + 1), axis=1).reshape(-1)
    _, colseq = scatter(
        inputs=[by_mx, seq_of_row_mx, pos, offsets, params],
        template=[],
        grid=(grid, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(M * K,), (M * n_win,)],
        output_dtypes=[mx.int32, mx.int32],
        init_value=0,
    )
    return offsets, colseq, runs_np


# Columns per launch: the threadgroup counter row (32 KB limit) is TILE int32
# (two uint32 words per column with window weights)
_METAL_COL_TILE = 4096
_METAL_COL_TILE_W = 2048
_METAL_LANES = 16
_METAL_THREADS = 256


@cache
def _metal_join(weighted: bool):
    import mlx.core as mx

    if weighted:
        return mx.fast.metal_kernel(
            name="gkm_masked_join_w",
            input_names=["bx", "wx", "runs", "offsets", "colseq", "params"],
            output_names=["out_lo", "out_hi"],
            source=_METAL_MASKED_JOIN_W,
            header=_METAL_MASKED_KEY_HEADER,
        )
    return mx.fast.metal_kernel(
        name="gkm_masked_join",
        input_names=["bx", "runs", "offsets", "colseq", "params"],
        output_names=["out"],
        source=_METAL_MASKED_JOIN,
        header=_METAL_MASKED_KEY_HEADER,
    )


def masked_pairwise_mlx(
    bx_packed, by_packed, plan, by_rc_packed=None, wx=None, wy=None, wy_rc=None
) -> np.ndarray:
    """Metal version of masked_pairwise(); the column index is built in Metal too.

    Inputs may be NumPy or MLX uint32 arrays; wx, wy, wy_rc: row_weights as
    in masked_pairwise. Returns [B, S] float64 NumPy, combined exactly from
    per-level integer counts (64-bit with window weights).
    """
    import mlx.core as mx

    weighted = wx is not None
    join = _metal_join(weighted)
    tile = _METAL_COL_TILE_W if weighted else _METAL_COL_TILE
    bx = np.ascontiguousarray(np.asarray(bx_packed), dtype=np.uint32)
    by_all = np.asarray(by_packed, dtype=np.uint32)
    by_rc_all = (
        None if by_rc_packed is None else np.asarray(by_rc_packed, dtype=np.uint32)
    )
    B, Wx = bx.shape
    S_total, Wy = by_all.shape
    bx_mx = mx.array(bx.ravel())
    result = np.zeros((B, S_total), dtype=np.float64)
    if weighted:
        wx_np = np.ascontiguousarray(wx, dtype=np.uint32)
        wx_mx = mx.array(wx_np.ravel())
        wx_stride = row_stride(wx_np, Wx)
        blocks_w = [wy] + ([wy_rc] if by_rc_all is not None else [])

    for c0 in range(0, S_total, tile):
        c1 = min(c0 + tile, S_total)
        S = c1 - c0
        rows = [by_all[c0:c1]] + ([by_rc_all[c0:c1]] if by_rc_all is not None else [])
        by = np.ascontiguousarray(np.concatenate(rows))
        seq_of_row = np.tile(np.arange(S, dtype=np.int32), len(rows))
        n_win = by.shape[0] * Wy
        if weighted:  # one entry per window: sequence id | weight << 16
            tile_w = [w if w.shape[0] == 1 else w[c0:c1] for w in blocks_w]
            wby = _column_weights(tile_w, S, interleave=False).astype(np.int32)
            seq_of_row = (np.repeat(seq_of_row, Wy) | (wby.ravel() << 16)).astype(
                np.int32
            )
        by_mx, seq_mx = mx.array(by.ravel()), mx.array(seq_of_row)
        for j, coef, positions in plan:
            K1 = (1 << (2 * j)) + 1
            level = None
            for m0 in range(0, len(positions), _MASK_CHUNK):
                chunk = positions[m0 : m0 + _MASK_CHUNK]
                offsets, colseq, runs_np = _metal_index(by_mx, seq_mx, chunk, j, Wy)
                params = mx.array(
                    [
                        Wx,
                        runs_np.shape[0],
                        runs_np.shape[1],
                        S,
                        _METAL_LANES,
                        K1,
                        n_win,
                    ],
                    dtype=mx.int32,
                )
                runs_mx = mx.array(runs_np.ravel())
                if weighted:
                    params = mx.concatenate(
                        [params, mx.array([wx_stride], dtype=mx.int32)]
                    )
                    lo, hi = join(
                        inputs=[bx_mx, wx_mx, runs_mx, offsets, colseq, params],
                        template=[("TILE", tile)],
                        grid=(B * _METAL_THREADS, 1, 1),
                        threadgroup=(_METAL_THREADS, 1, 1),
                        output_shapes=[(B * S,), (B * S,)],
                        output_dtypes=[mx.uint32, mx.uint32],
                    )
                    counts = (
                        np.array(hi).astype(np.uint64) << np.uint64(32)
                    ) | np.array(lo).astype(np.uint64)
                else:
                    counts = join(
                        inputs=[bx_mx, runs_mx, offsets, colseq, params],
                        template=[("TILE", tile)],
                        grid=(B * _METAL_THREADS, 1, 1),
                        threadgroup=(_METAL_THREADS, 1, 1),
                        output_shapes=[(B * S,)],
                        output_dtypes=[mx.int32],
                    )[0]
                level = counts if level is None else level + counts
            result[:, c0:c1] += coef * np.array(level, dtype=np.float64).reshape(B, S)
    return result
