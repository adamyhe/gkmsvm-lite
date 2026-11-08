"""Numba kernels: the per-SV pair loop and the l-mer path (SV l-mer weights V,
or RBF weights over the l-mer index)."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from gkmsvm.kernels.packed import PAD_WINDOW
from gkmsvm.kernels.primitives import _packed_mismatches


@njit(parallel=True, cache=True, fastmath=True)
def _explain_cpu(
    packed_x,
    packed_y,
    table_a,
    table_b,
    eff_coef,
    l,
    min_matches,
    seqlen,
    perturbation,
    wx,
    wy,
):
    """[B, 4, seqlen] attributions of packed_x [B, W] against packed_y [S, Wy];
    wx [B or 1, W], wy [S or 1, Wy]: window weights ([1, 0] when unweighted).
    PAD windows (past shorter sequences' windows) are skipped."""
    B, W = packed_x.shape
    S_c, Wy = packed_y.shape
    weighted = wx.shape[1] > 0
    result = np.zeros((B, 4, seqlen), dtype=np.float64)
    for b in prange(B):
        wxb = wx[min(np.int64(b), wx.shape[0] - 1)]
        for s in range(S_c):
            ec = eff_coef[b, s]
            wys = wy[min(np.int64(s), wy.shape[0] - 1)]
            for i in range(W):
                bx_val = packed_x[b, i]
                if bx_val == PAD_WINDOW:
                    continue
                for j in range(Wy):
                    by_val = packed_y[s, j]
                    if by_val == PAD_WINDOW:
                        continue
                    mm = _packed_mismatches(bx_val, by_val)
                    if l - mm < min_matches:
                        continue
                    ew = ec * wxb[i] * wys[j] if weighted else ec
                    av = ew * table_a[mm]
                    bv = ew * table_b[mm]
                    for k in range(l):
                        bx_k = int((bx_val >> np.uint32(2 * k)) & np.uint32(3))
                        by_k = int((by_val >> np.uint32(2 * k)) & np.uint32(3))
                        pos = k + i
                        if not perturbation:
                            result[b, by_k, pos] += av if bx_k == by_k else bv
                        elif bx_k == by_k:
                            for h in range(4):
                                if h != by_k:
                                    result[b, h, pos] += av
                        else:
                            result[b, by_k, pos] += bv
    return result


@njit(inline="always")
def _distribute(acc, i, mask, v, wv, a, b, l, perturbation):
    """Add one (query window i, l-mer v = u ^ mask) pair to acc [4, seqlen], as
    _explain_cpu does for an SV window equal to v with effective coefficient wv;
    a, b are table_a / table_b at the pair's mismatch count."""
    av = wv * a
    bv = wv * b
    for k in range(l):
        v_k = int((v >> np.uint32(2 * k)) & np.uint32(3))
        same = ((mask >> np.uint32(2 * k)) & np.uint32(3)) == 0
        pos = k + i
        if not perturbation:
            acc[v_k, pos] += av if same else bv
        elif same:
            for h in range(4):
                if h != v_k:
                    acc[h, pos] += av
        else:
            acc[v_k, pos] += bv


@njit(parallel=True, cache=True, fastmath=True)
def _explain_lmer_cpu(
    packed_x,
    inv_dx,
    V,
    masks,
    levels,
    table_a,
    table_b,
    l,
    seqlen,
    perturbation,
    n_split,
    wx,
):
    """[B, 4, seqlen] attributions from the SVs' l-mer weights V (see _explain_lmer).

    Each query window u visits every l-mer v = u ^ mask within the mismatch
    cutoff and adds V[v] times the alpha/kappa (or perturbation) weight of
    their mismatch count to each of its positions, as _explain_cpu does for
    every SV window equal to v. Rows are split into n_split window chunks.
    wx: the query windows' weights ([B or 1, W], [1, 0] when unweighted; V
    holds the SVs'). PAD query windows have no neighbors (u ^ mask >= 4^l).
    """
    B, W = packed_x.shape
    n_lmers = V.shape[0]
    weighted = wx.shape[1] > 0
    part = np.zeros((B * n_split, 4, seqlen))
    step = (W + n_split - 1) // n_split
    for task in prange(B * n_split):
        b, c = task // n_split, task % n_split
        acc = part[task]
        wxb = wx[min(np.int64(b), wx.shape[0] - 1)]
        for i in range(c * step, min(W, (c + 1) * step)):
            u = packed_x[b, i]
            a = wxb[i] if weighted else 1.0
            for t in range(masks.shape[0]):
                v = u ^ masks[t]
                if np.int64(v) >= n_lmers or V[v] == 0.0:
                    continue
                _distribute(
                    acc,
                    i,
                    masks[t],
                    v,
                    a * V[v],
                    table_a[levels[t]],
                    table_b[levels[t]],
                    l,
                    perturbation,
                )
    out = part.reshape(B, n_split, 4, seqlen).sum(axis=1)
    for b in range(B):
        out[b] *= inv_dx[b]
    return out


@njit(parallel=True, cache=True, fastmath=True)
def _explain_index_cpu(
    packed_x,
    inv_dx,
    offsets,
    ids,
    w,
    masks,
    levels,
    table_a,
    table_b,
    l,
    seqlen,
    perturbation,
    n_split,
    wx,
    vals,
):
    """As _explain_lmer_cpu with per-query SV weights w [B, S]: V[v] for row b is
    the sum of w[b, s] over the index's entries for v (RBF models), times each
    entry's window weight in vals (empty when unweighted)."""
    B, W = packed_x.shape
    n_lmers = offsets.shape[0] - 1
    weighted = vals.shape[0] > 0
    part = np.zeros((B * n_split, 4, seqlen))
    step = (W + n_split - 1) // n_split
    for task in prange(B * n_split):
        b, c = task // n_split, task % n_split
        acc = part[task]
        wb = w[b]
        wxb = wx[min(np.int64(b), wx.shape[0] - 1)]
        for i in range(c * step, min(W, (c + 1) * step)):
            u = packed_x[b, i]
            a = wxb[i] if weighted else 1.0
            for t in range(masks.shape[0]):
                v = u ^ masks[t]
                if np.int64(v) >= n_lmers:
                    continue
                wv = 0.0
                if weighted:
                    for e in range(offsets[v], offsets[v + 1]):
                        wv += wb[ids[e]] * vals[e]
                else:
                    for e in range(offsets[v], offsets[v + 1]):
                        wv += wb[ids[e]]
                if wv == 0.0:
                    continue
                _distribute(
                    acc,
                    i,
                    masks[t],
                    v,
                    a * wv,
                    table_a[levels[t]],
                    table_b[levels[t]],
                    l,
                    perturbation,
                )
    out = part.reshape(B, n_split, 4, seqlen).sum(axis=1)
    for b in range(B):
        out[b] *= inv_dx[b]
    return out
