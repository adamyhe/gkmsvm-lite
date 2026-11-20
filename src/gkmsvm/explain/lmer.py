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
from gkmsvm.explain.cpu import _dense_V, _explain_index_cpu, _explain_lmer_cpu
from gkmsvm.explain.cuda import _explain_lmer_gpu
from gkmsvm.explain.metal import _explain_lmer_metal
from gkmsvm.kernels.packed import PAD_WINDOW

# Past lmer_index.MAX_NEIGHBORS, the l-mer path is taken when the SVs have at
# least this many windows per neighbor: -t 0 l=11 k=7 without d (143,980 neighbors, 300 bp
# SVs) breaks even near 1,000 SVs on CPU (Apple M4, Xeon) and 2,000 on a Titan
# Xp, and is 3.6-4.7x faster at 10,000 (benchmarks/dev/explain_lmer_gpu.py)
_EXPLAIN_SV_WINDOWS_PER_NEIGHBOR = 6
# Metal (an M4): the per-SV kernel (_explain_sv_metal) keeps up to about 2,900
# such SVs (32 x 300 bp: 0.57 s vs 0.81 s at 2,000 SVs, 2.3 s vs 0.83 s at 8,000)
_EXPLAIN_SV_WINDOWS_PER_NEIGHBOR_METAL = 12

# RBF: building each row's V (_dense_V, one pass over the l-mer index) and
# taking the V path, against summing an l-mer's index entries at every visit.
# Per device (a, b, c, d): the per-row V saves visits * (a + b e) and costs
# rows * 4^l * (c + d e) seconds, e the index entries per l-mer and visits
# the batch's (query window, neighbor) pairs. Least-squares fit (CPU; it
# picks the faster path in all 32 cases) and the threshold separating every
# case (Metal, where the CPU builds V while the GPU runs the previous batch:
# per-row V wins when visits * e > 17 * rows * 4^l) over 500-32,715 SVs of
# the dsQTL -t 3 model, 19-300 bp, both modes; Apple M4, 10 threads
# (benchmarks/dev/explain_dense_v.py). CUDA keeps the index path.
_DENSE_V_COST = {
    "cpu": (-0.22e-9, 0.65e-9, 0.66e-9, 0.11e-9),
    "metal": (0.0, 1.0, 17.0, 0.0),
}
# Bytes of per-row V built at once (float64; float32 for Metal)
_DENSE_V_BYTES = 512 * 2**20
# None: by cost; True / False force it (benchmarks, tests)
_DENSE_V = None


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
    more index query per call), or, when the visits would read more entries
    than one pass over the index costs (long queries, dense indexes:
    _use_dense_V), each row's V is built in that pass (_dense_V) and the V
    path runs with it (CPU and Metal). CuPy input runs in CUDA (_explain_lmer_cuda),
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
    # Metal: each batch's kernels run while the CPU queries the next batch's SV
    # weights (RBF) and packs it; their partials are folded after that
    pending = []
    for _, packed_x, packed_x_rc, wx in _batches(base, x, lengths, batch_size, verbose):
        inv_dx = _inv_dx(base, packed_x, packed_x_rc)
        B_b, W = packed_x.shape
        n_split = max(1, min(W, -(-numba.get_num_threads() // B_b)))
        jobs = []
        if index is None:
            jobs.append((slice(None), V, None))
        else:
            # (on Metal too, the CPU query: exact in float64, and it runs while
            # the GPU explains the previous batch; the Metal query alone is
            # ~1.4x the CPU's on the dsQTL -t 3 model, M5, but would queue
            # behind that kernel)
            w = _rbf_weights(index, packed_x, inv_dx, coef, inv_ds, gamma)
            visits = int(np.count_nonzero(packed_x != PAD_WINDOW)) * len(masks)
            if _use_dense_V(index, visits, B_b, l, metal):
                # Rows in blocks of at most _DENSE_V_BYTES of V (float32 for
                # Metal, built so: no conversion pass before the upload)
                dtype = np.float32 if metal else np.float64
                rows = max(1, _DENSE_V_BYTES // (np.dtype(dtype).itemsize * 4**l))
                wT = np.ascontiguousarray(w.T)
                for r0 in range(0, B_b, rows):
                    r = slice(r0, min(B_b, r0 + rows))
                    Vb = np.empty((r.stop - r0, 4**l), dtype=dtype)
                    _dense_V(index.offsets, index.ids, index.vals, wT[:, r], Vb)
                    jobs.append((r, Vb, None))
            else:
                jobs.append((slice(None), None, w))
        for r, Vr, w in jobs:
            wr = wx if wx.shape[0] == 1 else wx[r]
            if metal:
                job = _explain_lmer_metal(
                    packed_x[r],
                    inv_dx[r],
                    Vr,
                    index,
                    w,
                    masks,
                    levels,
                    table_a,
                    table_b,
                    l,
                    seqlen,
                    perturbation,
                    wr,
                    model._sv_cache,
                    defer=True,
                )
                parts.extend(f() for f in pending)
                pending = [job]
            elif Vr is not None:
                parts.append(
                    _explain_lmer_cpu(
                        packed_x[r],
                        inv_dx[r],
                        Vr if Vr.ndim == 2 else Vr[None],
                        masks,
                        levels,
                        table_a,
                        table_b,
                        l,
                        seqlen,
                        perturbation,
                        n_split,
                        wr,
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
    parts.extend(f() for f in pending)
    return np.concatenate(parts)


def _use_dense_V(index, visits, rows, l, metal):
    """Whether an RBF batch builds each row's V (_dense_V) and takes the V path
    rather than summing the index entries at every neighbor visit
    (_DENSE_V_COST)."""
    if _DENSE_V is not None:
        return _DENSE_V
    e = len(index.ids) / (len(index.offsets) - 1)
    a, b, c, d = _DENSE_V_COST["metal" if metal else "cpu"]
    return visits * (a + b * e) > rows * 4**l * (c + d * e)
