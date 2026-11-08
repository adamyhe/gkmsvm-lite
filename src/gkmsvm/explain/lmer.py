"""The l-mer path: each query window visits its neighbors' summed SV weights V
(or, for RBF, the per-query SV weights summed over the l-mer index), on
CPU, CUDA or Metal."""

from __future__ import annotations

import numpy as np

from gkmsvm.backend import get_array_module
from gkmsvm.explain.common import (
    _batches,
    _explain_V,
    _inv_dx,
    _rbf_weights,
    _sv_weights,
)
from gkmsvm.explain.cpu import _explain_index_cpu, _explain_lmer_cpu
from gkmsvm.explain.cuda import _explain_lmer_gpu
from gkmsvm.explain.metal import _explain_lmer_metal

# Past lmer_index.MAX_NEIGHBORS, the l-mer path is taken when the SVs have at
# least this many windows per neighbor: -t 0 l=11 k=7 without d (143,980 neighbors, 300 bp
# SVs) breaks even near 1,000 SVs on CPU (Apple M4, Xeon) and 2,000 on a Titan
# Xp, and is 3.6-4.7x faster at 10,000 (benchmarks/dev/explain_lmer_gpu.py)
_EXPLAIN_SV_WINDOWS_PER_NEIGHBOR = 6
# Metal (an M4): the per-SV kernel (_explain_sv_metal) keeps up to about 2,900
# such SVs (32 x 300 bp: 0.57 s vs 0.81 s at 2,000 SVs, 2.3 s vs 0.83 s at 8,000)
_EXPLAIN_SV_WINDOWS_PER_NEIGHBOR_METAL = 12


def _explain_lmer(
    model,
    x,
    lengths,
    table_a,
    table_b,
    dmax,
    perturbation,
    gamma,
    batch_size,
    verbose,
    metal=False,
):
    """GkmExplain through the SVs' l-mer weights, or None when it doesn't apply.

    The attribution is linear in the SVs: with normalization each SV window
    v contributes coef_s / sqrt(K(sv, sv) K(x, x)) times a weight that
    depends only on the query window u and v. Summing over SVs first gives
    V[v] = sum_s coef_s / sqrt(K(sv, sv)) * (occurrences of v in sv_s and its
    RC), and a query window then visits only the l-mers within dmax
    mismatches. A call costs nothing per SV; V (4^l float64) is cached.

    RBF models (gamma given) scale each SV by _rbf_scale of its kernel value
    with the query, so V depends on the query: each neighbor sums the
    per-query SV weights over the l-mer index's entries instead (about one
    more index query per call). CuPy input runs in CUDA (_explain_lmer_cuda),
    NumPy on CPU, or in Metal with metal (MLX input arrives as NumPy).
    """
    import numba

    from gkmsvm.kernels import lmer_index

    base = model.kernel.base
    l = base.l
    if not model.use_lmer_index or l > lmer_index.MAX_L or dmax < 0:
        return None
    n_neighbors = lmer_index.neighbor_count(l, dmax)
    if n_neighbors > lmer_index.MAX_NEIGHBORS:
        # A query window visits n_neighbors l-mers here and every SV window
        # in the per-SV kernel
        packed = model._sv_packed()
        sv_windows = sum(p.size for p in packed if p is not None)
        per_neighbor = (
            _EXPLAIN_SV_WINDOWS_PER_NEIGHBOR_METAL
            if metal
            else _EXPLAIN_SV_WINDOWS_PER_NEIGHBOR
        )
        if n_neighbors * per_neighbor > sv_windows:
            return None
    coef, inv_ds = _sv_weights(model)
    index = V = None
    if gamma is not None:
        index = model._sv_index()  # its arrays live on CPU whatever the model's device
        if index is None:
            return None
    else:
        V = _explain_V(model)
    # Cached so the Metal path's device copies (keyed by the masks) are reused
    key = ("explain_masks", perturbation, dmax)
    if key not in model._sv_cache:
        nonzero = np.maximum(np.abs(table_a), np.abs(table_b))
        model._sv_cache[key] = lmer_index.neighbor_masks(l, nonzero, dmax)
    masks, levels = model._sv_cache[key]

    if get_array_module(x) is not np:  # (MLX reaches here as NumPy, metal=True)
        return _explain_lmer_cuda(
            model,
            x,
            lengths,
            V,
            index,
            coef,
            inv_ds,
            masks,
            levels,
            table_a,
            table_b,
            gamma,
            batch_size,
            verbose,
            perturbation,
        )
    seqlen = x.shape[-1]
    parts = []
    # Metal: each batch's kernel runs while the CPU queries the next batch's SV
    # weights (RBF) and packs it; its partials are folded after that
    pending = None
    for _, packed_x, packed_x_rc, wx in _batches(base, x, lengths, batch_size, verbose):
        inv_dx = _inv_dx(base, packed_x, packed_x_rc)
        B_b, W = packed_x.shape
        n_split = max(1, min(W, -(-numba.get_num_threads() // B_b)))
        # (on Metal too, the CPU query: exact in float64, and it runs while the
        # GPU explains the previous batch; the Metal query alone is ~1.4x the
        # CPU's on the dsQTL -t 3 model, M5, but would queue behind that kernel)
        w = None
        if index is not None:
            w = _rbf_weights(index, packed_x, inv_dx, coef, inv_ds, gamma)
        if metal:
            job = _explain_lmer_metal(
                packed_x,
                inv_dx,
                V,
                index,
                w,
                masks,
                levels,
                table_a,
                table_b,
                l,
                seqlen,
                perturbation,
                wx,
                model._sv_cache,
                defer=True,
            )
            if pending is not None:
                parts.append(pending())
            pending = job
        elif index is None:
            parts.append(
                _explain_lmer_cpu(
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
                )
            )
        else:
            parts.append(
                _explain_index_cpu(
                    packed_x,
                    inv_dx,
                    index.offsets,
                    index.ids,
                    np.ascontiguousarray(w),
                    masks,
                    levels,
                    table_a,
                    table_b,
                    l,
                    seqlen,
                    perturbation,
                    n_split,
                    wx,
                    index.vals,
                )
            )
    if pending is not None:
        parts.append(pending())
    return np.concatenate(parts)


def _explain_lmer_cuda(
    model,
    x,
    lengths,
    V,
    index,
    coef,
    inv_ds,
    masks,
    levels,
    table_a,
    table_b,
    gamma,
    batch_size,
    verbose,
    perturbation,
):
    """_explain_lmer's loop on CUDA (_explain_lmer_gpu); V, the masks and the
    l-mer index's arrays are uploaded once and kept in the model's caches."""
    import cupy as cp

    base = model.kernel.base
    l = base.l
    c = model._sv_cache
    if index is None:
        if "explain_V_cuda" not in c:
            c["explain_V_cuda"] = cp.asarray(V)
        V = c["explain_V_cuda"]
        offsets = ids = vals = None
    else:
        offsets, ids, _, _, vals = index._arrays("cuda")
        coef_d = cp.asarray(coef)
        inv_ds_d = cp.asarray(inv_ds)
    masks, levels = cp.asarray(masks), cp.asarray(levels)
    table_a = cp.asarray(table_a, dtype=cp.float64)
    table_b = cp.asarray(table_b, dtype=cp.float64)

    B, seqlen = x.shape[0], x.shape[-1]
    out = cp.empty((B, 4, seqlen), dtype=cp.float64)
    for b0, packed_x, packed_x_rc, wx in _batches(
        base, x, lengths, batch_size, verbose
    ):
        inv_dx = _inv_dx(base, packed_x, packed_x_rc)
        w = None
        if index is not None:
            w = _rbf_weights(index, packed_x, inv_dx, coef_d, inv_ds_d, gamma)
        out[b0 : b0 + batch_size] = _explain_lmer_gpu(
            packed_x,
            inv_dx,
            V,
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
            wx,
            vals,
        )
    return out
