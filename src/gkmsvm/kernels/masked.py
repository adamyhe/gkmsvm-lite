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

from gkmsvm.kernels.primitives import _spans

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
