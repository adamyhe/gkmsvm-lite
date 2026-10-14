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
