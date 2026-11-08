"""Helpers shared by the explain paths: input batches, query norms, the RBF
scale, the SVs' l-mer weights V, and launch slicing for the GPU kernels."""

from __future__ import annotations

import numpy as np

from gkmsvm.backend import get_array_module
from gkmsvm.svm import _progress, _rows

# Threads to aim for per launch; mask slices (grid.y) make up for few windows
_EXPLAIN_LMER_THREADS = 1 << 18


def _launch_slices(n, per_slice, row_threads, B, max_slices=None):
    """(n_slices, chunk): n masks (or SV windows) split into grid.y slices of
    chunk, as many as it takes B rows of row_threads threads to reach
    _EXPLAIN_LMER_THREADS, but no fewer than per_slice items each (and at
    most max_slices)."""
    caps = [-(-n // per_slice), -(-_EXPLAIN_LMER_THREADS // (B * row_threads))]
    if max_slices is not None:
        caps.append(max_slices)
    chunk = -(-n // max(1, min(caps)))
    return -(-n // chunk), chunk


def _rbf_scale(k_norm, gamma, xp=np):
    """LS-GKM's RBF attribution factor per SV: (exp(gamma (K - 1)) - exp(-gamma)) / K,
    0 where K <= 0, for normalized base kernel values K."""
    pos = k_norm > 0
    safe = xp.where(pos, k_norm, 1.0)
    return xp.where(pos, (xp.exp(gamma * (safe - 1)) - np.exp(-gamma)) / safe, 0.0)


def _query_wts(base, packed):
    """Window weights of packed query rows as float64 [R or 1, W] on CPU
    (row_weights), [1, 0] when every window counts once."""
    w = base.row_weights(packed)
    return np.zeros((1, 0)) if w is None else w.astype(np.float64)


def _batches(base, x, lengths, batch_size, verbose):
    """(b0, packed_x, packed_x_rc, wx) for each batch of x's rows: their packed
    windows (packed_x contiguous) and _query_wts, with a tqdm bar when verbose."""
    xp = get_array_module(x)
    B = x.shape[0]
    starts = _progress(range(0, B, batch_size), verbose, "GkmExplain")
    for b0 in starts:
        packed_x, packed_x_rc = base.pack_both(
            x[b0 : b0 + batch_size], _rows(lengths, b0, batch_size)
        )
        packed_x = xp.ascontiguousarray(packed_x)
        yield b0, packed_x, packed_x_rc, _query_wts(base, packed_x)


def _inv_dx(base, packed_x, packed_x_rc):
    """1 / sqrt(K(x, x)) [B] on packed_x's device; ones without normalization."""
    xp = get_array_module(packed_x)
    if not base.normalize:
        return xp.ones(len(packed_x))
    return 1 / xp.sqrt(
        xp.clip(base.raw_diag_packed(packed_x, packed_x_rc), 1e-20, None)
    )


def _rbf_weights(index, packed_x, inv_dx, coef, inv_ds, gamma):
    """Per-query SV weights [B, S] of an RBF model's l-mer path: coef_s /
    sqrt(K(sv, sv)) times _rbf_scale of the normalized kernel value (index
    query on packed_x's device)."""
    k_norm = index.pairwise(packed_x) * inv_dx[:, None] * inv_ds[None, :]
    return coef * inv_ds * _rbf_scale(k_norm, gamma, get_array_module(packed_x))


def _sv_weights(model):
    """(coef, 1 / sqrt(K(sv, sv))) on CPU in float64; ones without normalization."""
    from gkmsvm.backend import to_cpu

    coef = to_cpu(model.coefficients).astype(np.float64)
    inv_ds = (
        1 / np.sqrt(np.clip(to_cpu(model._sv_diag()).astype(np.float64), 1e-20, None))
        if model.kernel.base.normalize
        else np.ones(len(coef))
    )
    return coef, inv_ds


def _explain_V(model):
    """V[v] = sum_s coef_s / sqrt(K(sv, sv)) * (occurrences of l-mer v in sv_s and
    its RC, each counted with its window weight), 4^l float64 on CPU (cached)."""
    from gkmsvm.backend import to_cpu
    from gkmsvm.lmer_table import sv_lmer_weights

    c = model._sv_cache
    if "explain_V" not in c:
        coef, inv_ds = _sv_weights(model)
        fwd, rc = (None if p is None else to_cpu(p) for p in model._sv_packed())
        c["explain_V"] = sv_lmer_weights(model.kernel.base, fwd, rc, coef * inv_ds)
    return c["explain_V"]
