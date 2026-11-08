"""Numba kernels for the l-mer index (kernels/lmer_index.py): CSR build, query,
and the fused query + reduce, mutation impact and ISM passes."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _counting_sort(keys, ids, offsets):
    """ids reordered by key (stable), given the key offsets."""
    fill = offsets[:-1].copy()
    out = np.empty_like(ids)
    for i in range(keys.shape[0]):
        k = keys[i]
        out[fill[k]] = ids[i]
        fill[k] += 1
    return out


@njit(inline="always")
def _add_window(acc, w, scale, offsets, ids, vals, masks, weights, n_lmers, lo=0):
    """acc[s] += scale * K_raw(window w, s) for columns s >= lo; vals: the
    entries' window weights (empty: all 1). Each l-mer's entries are sorted by
    column, so its scan runs down from the last and stops below lo."""
    weighted = vals.shape[0] > 0
    if np.int64(w) >= n_lmers:  # padding window
        return
    for t in range(masks.shape[0]):
        v = np.int64(w ^ masks[t])
        wt = scale * weights[t]
        for e in range(offsets[v + 1] - 1, offsets[v] - 1, -1):
            s = ids[e]
            if s < lo:
                break
            acc[s] += wt * vals[e] if weighted else wt


@njit(parallel=True, cache=True)
def _query_cpu(
    bx, offsets, ids, vals, masks, weights, n_lmers, S, n_split, wx, diag_offset
):
    """Each row's windows in n_split chunks (so few rows still use every
    thread), each accumulating into its own row of partial sums. wx: the
    query windows' weights, [B or 1, W] ([1, 0]: all 1). diag_offset >= 0:
    row b only gets columns >= diag_offset + b (the rest stay 0)."""
    B, W = bx.shape
    part = np.zeros((B * n_split, S), dtype=np.float64)
    step = (W + n_split - 1) // n_split
    for task in prange(B * n_split):
        # Rows interleaved first and last (0, B-1, 1, B-2, ...): the workqueue
        # layer gives each thread one contiguous range of tasks, and rows
        # above a diagonal shrink with b
        j, c = task // n_split, task % n_split
        b = j // 2 if j % 2 == 0 else B - 1 - j // 2
        acc = part[b * n_split + c]
        wb = wx[min(np.int64(b), wx.shape[0] - 1)]
        lo = diag_offset + b if diag_offset >= 0 else 0
        for i in range(c * step, min(W, (c + 1) * step)):
            scale = wb[i] if wb.shape[0] > 0 else 1.0
            _add_window(
                acc, bx[b, i], scale, offsets, ids, vals, masks, weights, n_lmers, lo
            )
    if n_split == 1:
        return part
    return part.reshape(B, n_split, S).sum(axis=1)


@njit(parallel=True, cache=True, fastmath={"nsz", "arcp", "contract", "reassoc"})
def _query_reduce_cpu(
    bx,
    offsets,
    ids,
    vals,
    masks,
    weights,
    n_lmers,
    wx,
    inv_dx,
    inv_ds,
    coef,
    gamma,
    n_tasks,
):
    """[B]: sum_s coef_s f(K_raw[b, s] * inv_dx[b] * inv_ds[s]), without the [B, S] block.

    f(k) = exp(gamma (k - 1)) when gamma is not NaN (RBF), else k. Each task
    accumulates one row's hits in a length-S buffer that stays in cache and is
    zeroed as it is reduced.
    """
    B, W = bx.shape
    S = inv_ds.shape[0]
    rbf = not np.isnan(gamma)
    floor = np.exp(-gamma) if rbf else 1.0
    out = np.zeros(B)
    for task in prange(n_tasks):
        acc = np.zeros(S)
        for b in range(task, B, n_tasks):
            wb = wx[min(np.int64(b), wx.shape[0] - 1)]
            for i in range(W):
                a = wb[i] if wb.shape[0] > 0 else 1.0
                _add_window(
                    acc, bx[b, i], a, offsets, ids, vals, masks, weights, n_lmers
                )
            scale = inv_dx[b]
            total = 0.0
            if rbf:
                g = gamma * scale
                for j in range(S):
                    total += coef[j] * np.exp(g * acc[j] * inv_ds[j])
                    acc[j] = 0.0
                total *= floor
            else:
                for j in range(S):
                    total += coef[j] * acc[j] * inv_ds[j]
                    acc[j] = 0.0
                total *= scale
            out[b] = total
    return out


@njit(parallel=True, cache=True, fastmath={"nsz", "arcp", "contract", "reassoc"})
def _query_impact_cpu(
    bx,
    offsets,
    ids,
    vals,
    masks,
    weights,
    n_lmers,
    wx,
    inv_dr,
    inv_ds,
    coef,
    gamma,
    n_tasks,
):
    """[n] mutation impact (explain.mutation_impact) of rows bx = [ref; alt].

    Each task accumulates K_raw(ref) and K_raw(alt) - K_raw(ref) in two length-S
    buffers; the difference visits only window slots where the alleles differ
    (all of them past an indel), so a SNP costs about one query. With gamma
    (not NaN) each SV's dK scales by (exp(gamma (k - 1)) - exp(-gamma)) / k of
    its normalized k (0 when k <= 0), else by 1.
    """
    W = bx.shape[1]
    n = bx.shape[0] // 2
    S = inv_ds.shape[0]
    rbf = not np.isnan(gamma)
    floor = np.exp(-gamma) if rbf else 0.0
    out = np.zeros(n)
    for task in prange(n_tasks):
        acc = np.zeros(S)
        delta = np.zeros(S)
        for b in range(task, n, n_tasks):
            wr = wx[min(np.int64(b), wx.shape[0] - 1)]
            wa = wx[min(np.int64(n + b), wx.shape[0] - 1)]
            for i in range(W):
                r = bx[b, i]
                a = bx[n + b, i]
                ar = wr[i] if wr.shape[0] > 0 else 1.0
                aa = wa[i] if wa.shape[0] > 0 else 1.0
                _add_window(acc, r, ar, offsets, ids, vals, masks, weights, n_lmers)
                if r != a or ar != aa:
                    _add_window(
                        delta, r, -ar, offsets, ids, vals, masks, weights, n_lmers
                    )
                    _add_window(
                        delta, a, aa, offsets, ids, vals, masks, weights, n_lmers
                    )
            scale = inv_dr[b]
            total = 0.0
            for j in range(S):
                dk = delta[j]
                if dk != 0.0:
                    z = scale * inv_ds[j]
                    dk *= z
                    if rbf:
                        k = acc[j] * z
                        dk = (
                            dk * (np.exp(gamma * (k - 1.0)) - floor) / k
                            if k > 0
                            else 0.0
                        )
                    total += coef[j] * dk
                    delta[j] = 0.0
                acc[j] = 0.0
            out[b] = total
    return out


@njit(inline="always")
def _add_hits(acc, row, sign, rowwt, offsets, ids, vals, masks, weights, n_lmers):
    """acc[s] += sign * K_raw(row, s) for the packed windows in row; rowwt: their
    window weights (empty: all 1)."""
    for i in range(row.shape[0]):
        scale = sign * rowwt[i] if rowwt.shape[0] > 0 else sign
        _add_window(acc, row[i], scale, offsets, ids, vals, masks, weights, n_lmers)


@njit(parallel=True, cache=True, fastmath={"nsz", "arcp", "contract", "reassoc"})
def _query_mutants_cpu(
    old,
    new,
    ref,
    ref_score,
    offsets,
    ids,
    vals,
    masks,
    weights,
    n_lmers,
    slot_wt,
    inv_dx,
    inv_dmut,
    inv_ds,
    coef,
    gamma,
    n_tasks,
):
    """[4, L, B] ISM deltas: f(mutant) - ref_score[b], with
    f = sum_s coef_s f(K_raw(mutant, s) inv_dmut inv_ds[s]) as in _query_reduce_cpu.

    K_raw(mutant, s) = ref[b, s] + K_raw(new[v, p, b], s) - K_raw(old[p, b], s): old
    [L, B, l] and new [4, L, B, l] hold the windows covering position p. A base
    with the reference's windows and norm (the reference base) gets 0; an N
    replaced by A keeps the forward windows (N packs as A) but not the norm.
    slot_wt [L, 1 or B, l]: the window weights of those windows (0 for padding;
    [L, 1, 0] when unweighted).
    """
    L, B = old.shape[0], old.shape[1]
    S = inv_ds.shape[0]
    rbf = not np.isnan(gamma)
    floor = np.exp(-gamma) if rbf else 1.0
    out = np.zeros((4, L, B))
    for task in prange(n_tasks):
        d_old = np.zeros(S)
        tmp = np.empty(S)
        for item in range(task, L * B, n_tasks):
            p, b = item // B, item % B
            rowwt = slot_wt[p, min(np.int64(b), slot_wt.shape[1] - 1)]
            _add_hits(
                d_old,
                old[p, b],
                -1.0,
                rowwt,
                offsets,
                ids,
                vals,
                masks,
                weights,
                n_lmers,
            )
            for v in range(4):
                # Window 0 always covers p
                if new[v, p, b, 0] == old[p, b, 0] and inv_dmut[v, p, b] == inv_dx[b]:
                    continue
                tmp[:] = d_old
                _add_hits(
                    tmp,
                    new[v, p, b],
                    1.0,
                    rowwt,
                    offsets,
                    ids,
                    vals,
                    masks,
                    weights,
                    n_lmers,
                )
                scale = inv_dmut[v, p, b]
                total = 0.0
                if rbf:
                    g = gamma * scale
                    for j in range(S):
                        total += coef[j] * np.exp(g * (ref[b, j] + tmp[j]) * inv_ds[j])
                    total *= floor
                else:
                    for j in range(S):
                        total += coef[j] * (ref[b, j] + tmp[j]) * inv_ds[j]
                    total *= scale
                out[v, p, b] = total - ref_score[b]
            d_old[:] = 0.0
    return out
