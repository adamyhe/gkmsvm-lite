"""In silico mutagenesis for gapped k-mer SVMs.

Window-delta optimization: when a single base changes, only ~l of the
W = L - l + 1 windows are affected. Instead of recomputing the full kernel,
we compute the delta from the affected windows only. With window weights
(-t 4/-t 5) each affected window keeps the weight of its position in the
sequence (_slot_weights).
"""

from __future__ import annotations

import numpy as np

from gkmsvm.backend import float_dtype, get_array_module, is_mlx, to_cpu
from gkmsvm.kernels.packed import PAD_WINDOW, device_of
from gkmsvm.kernels.rbf import RbfKernel
from gkmsvm.svm import (
    GkmSVM,
    _per_position,
    _progress,
    _rows,
    _short_rows,
    as_batch,
)


def ism(
    model: GkmSVM,
    x,
    *,
    lengths=None,
    batch_size: int | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Compute score change for every single-base substitution.

    Args:
        model: A GkmSVM model.
        x: [B, 4, L] one-hot encoded reference sequences, or a list of
            [4, L_i] arrays or DNA strings of different lengths.
        lengths: [B] lengths of x's rows when they differ (x padded to the
            longest).
        batch_size: Process inputs in batches of this size.  ``None``
            means process all at once (when ``verbose=False``) or
            one at a time (when ``verbose=True``).
        verbose: Show tqdm progress bar over input batches.

    Returns:
        [B, 4, L] score deltas; 0 past each sequence's length, NaN for
        sequences shorter than l (with a warning).
    """
    x, lengths = as_batch(x, lengths)
    x = model._match_device(x)
    short = _short_rows(lengths, model.kernel.base.l, "ISM deltas")

    def rows(s, e):
        return _ism_packed(model, x[s:e], _rows(lengths, s, e - s))

    bs = batch_size if batch_size is not None else (1 if verbose else None)
    if bs is not None and x.shape[0] > bs:
        xp = get_array_module(x)
        chunks = _progress(range(0, x.shape[0], bs), verbose, "ISM")
        out = xp.concatenate([rows(s, s + bs) for s in chunks], axis=0)
    else:
        out = rows(0, x.shape[0])
    return _per_position(out, lengths, short)


# -----------------------------------------------------------------------
# Packed path: kernel deltas from the windows covering each position
# -----------------------------------------------------------------------


def _ism_packed(model, x, lengths=None):
    """ISM from the base kernel's packed windows (any kernel whose SVs pack).

    Kernels with an l-mer table add table weights over the windows covering
    each position (_ism_table). With the l-mer index, a fused pass per
    (position, sequence) reduces each mutant over the SVs (_ism_index).
    Otherwise per-SV kernel values for each mutant are computed in blocks
    that hold at most ``GkmSVM._MAX_SCORE_BYTES`` of them (_ism_kernel).
    """
    xp = get_array_module(x)
    B, _, L = x.shape
    base = model.kernel.base
    W = L - base.l + 1
    S = model.num_support_vectors
    # A table exists only for non-RBF kernels
    table = model._lmer_table_for(xp)
    # Sequences per block: the mutant windows stay near _BLOCK_BYTES, and the
    # [rows, S] kernel values within _MAX_SCORE_BYTES (the reference rows for
    # the index path, those and a block of positions' mutants otherwise)
    rows = _BLOCK_BYTES // (8 * L * W * 4)
    if table is not None:
        fn = _ism_table
    elif model._sv_index() is not None:
        fn = _ism_index
        rows = min(rows, model._MAX_SCORE_BYTES // (8 * S))
    else:
        fn = _ism_kernel
        rows = min(rows, model._MAX_SCORE_BYTES // (6 * L * 8 * S))
    rows = max(1, rows)
    parts = [
        fn(
            model,
            x[s : s + rows],
            table,
            _rows(lengths, s, rows),
        )
        for s in range(0, B, rows)
    ]
    out = parts[0] if len(parts) == 1 else xp.concatenate(parts, axis=0)
    return out.astype(float_dtype(out))


# Mutant windows built per block of sequences
_BLOCK_BYTES = 256 * 2**20


def _ism_table(model, x, table, lengths=None):
    """f(mutant) - f(x) from the l-mer table: [B, 4, L]."""
    xp = get_array_module(x)
    base = model.kernel
    bx, bx_rc = base.pack_both(x, lengths)
    old, new, diag_ref, mut_diag = _mutants(base, x, bx, bx_rc)
    dev = (lambda a: a) if xp is np else xp.asarray
    # Padding (identical in old and new) reads entry 0, which cancels in new - old
    old = dev(np.where(old == PAD_WINDOW, 0, old))
    new = dev(np.where(new == PAD_WINDOW, 0, new))
    L = x.shape[2]
    slot_wt = _slot_weights(base.base, L, lengths)
    ref = model._table_raw(bx, table)  # [B]
    if slot_wt is None:
        k_new = ref + (
            table[new].sum(axis=3) - table[old].sum(axis=2)[None]
        )  # [4, L, B]
    else:
        slot_wt = dev(slot_wt.astype(np.float32 if is_mlx(x) else np.float64))
        k_new = ref + (
            (table[new] * slot_wt).sum(axis=3)
            - (table[old] * slot_wt).sum(axis=2)[None]
        )
    if base.normalize:
        k_new = k_new / xp.sqrt(xp.clip(mut_diag.astype(ref.dtype), 1e-10, None))
        ref = ref / xp.sqrt(xp.clip(diag_ref.astype(ref.dtype), 1e-10, None))
    return (k_new - ref).transpose(2, 0, 1)


def _ism_index(model, x, _table=None, lengths=None):
    """f(mutant) - f(x) through the l-mer index: [B, 4, L].

    One fused pass per (position, sequence) adds the changed windows' hits to
    the reference row and reduces over SVs, so no [mutants, S] block exists
    (on CPU a cache-resident length-S buffer, on CUDA/Metal a scratch slice
    per thread block).
    """
    kernel = model.kernel
    base = kernel.base
    index = model._sv_index()
    xp = get_array_module(x)
    bx, bx_rc = base.pack_both(x, lengths)
    old, new, diag_ref, mut_diag = _mutants(base, x, bx, bx_rc)
    inv_dx, inv_ds = model._inv_norms(
        None if diag_ref is None else to_cpu(diag_ref), x.shape[0]
    )
    inv_dmut = (
        np.ones(new.shape[:3])
        if mut_diag is None
        else 1 / np.sqrt(np.clip(to_cpu(mut_diag).astype(np.float64), 1e-20, None))
    )
    gamma = kernel.gamma if isinstance(kernel, RbfKernel) else np.nan
    coef = model.coefficients
    host = xp is np or not index.mutants_on_device(device_of(bx))
    if host:  # (MLX with a dense index: the CPU pass is faster)
        bx = np.ascontiguousarray(to_cpu(bx))
        coef = to_cpu(coef)
    ref = index.pairwise(bx)  # [B, S], on the device mutants() runs on
    ref_score = index.reduce(bx, inv_dx, inv_ds, coef, gamma) if host else None
    out = index.mutants(
        old,
        new,
        ref,
        ref_score,
        inv_dx,
        inv_dmut,
        inv_ds,
        coef,
        gamma,
        _slot_weights(base, x.shape[2], lengths),
    )
    if host and xp is not np:
        out = xp.asarray(out.astype(np.float32 if is_mlx(x) else np.float64))
    return out.transpose(2, 0, 1)


def _ism_kernel(model, x, _table=None, lengths=None):
    """f(mutant) - f(x) from per-SV kernel values: [B, 4, L]."""
    xp = get_array_module(x)
    B, _, L = x.shape
    kernel = model.kernel
    base = kernel.base
    S = model.num_support_vectors
    coefs = model.coefficients
    bx, bx_rc = base.pack_both(x, lengths)
    old, new, diag_ref, mut_diag = _mutants(base, x, bx, bx_rc)
    dev = (lambda a: a) if xp is np else xp.asarray
    ref_raw = model._sv_raw(bx)  # [B, S]
    if base.normalize:
        diag_sv = model._sv_diag()
        ref_k = ref_raw / xp.clip(
            xp.sqrt(diag_ref[:, None] * diag_sv[None, :]), 1e-10, None
        )
    else:
        ref_k = ref_raw
    ref_score = (kernel.finish(ref_k) * coefs).sum(axis=1)  # [B]
    l = base.l
    slot_wt = _slot_weights(base, L, lengths)
    # Weighted windows go in one per row (a lone window weighs window_weights(1)),
    # then sum with their positions' weights
    per_row = 1 if slot_wt is None else l
    # Positions per block: [5, P, B, S] kernel values (per window when
    # weighted) within _MAX_SCORE_BYTES
    P = max(1, model._MAX_SCORE_BYTES // (5 * B * S * 8 * per_row))
    scores = []
    for p0 in range(0, L, P):
        p1 = min(p0 + P, L)
        n = p1 - p0
        if slot_wt is None:
            old_pw = model._sv_raw(dev(old[p0:p1].reshape(n * B, l))).reshape(n, B, S)
            new_pw = model._sv_raw(dev(new[:, p0:p1].reshape(4 * n * B, l))).reshape(
                4, n, B, S
            )
        else:
            wt = slot_wt[p0:p1] / float(base.window_weights(1)[0])  # [n, 1 or B, l]
            wt = dev(wt.astype(np.float32 if is_mlx(x) else np.float64))
            old_pw = (
                model._sv_raw(dev(old[p0:p1].reshape(n * B * l, 1))).reshape(n, B, l, S)
                * wt[..., None]
            ).sum(axis=2)
            new_pw = (
                model._sv_raw(dev(new[:, p0:p1].reshape(4 * n * B * l, 1))).reshape(
                    4, n, B, l, S
                )
                * wt[None, ..., None]
            ).sum(axis=3)
        k_new = ref_raw + (new_pw - old_pw[None])  # [4, n, B, S]
        if base.normalize:
            k_new = k_new / xp.clip(
                xp.sqrt(mut_diag[:, p0:p1, :, None] * diag_sv), 1e-10, None
            )
        scores.append((kernel.finish(k_new) * coefs).sum(axis=3))  # [4, n, B]
    scores = scores[0] if len(scores) == 1 else xp.concatenate(scores, axis=1)
    return (scores - ref_score).transpose(2, 0, 1)


def _mutants(base, x, bx, bx_rc):
    """Packed windows covering each position before and after each substitution,
    and the self-kernels of the reference and of every mutant.

    Returns old [L, B, l], new [4, L, B, l] (uint32, on CPU), K_raw(x, x) [B]
    and K_raw(mutant, mutant) [4, L, B] on x's device (both None without
    normalization).
    """
    L = x.shape[2]
    l = base.l
    W = L - l + 1
    old, new = _packed_mutations(to_cpu(bx), L, W, l)
    if not base.normalize:
        return old, new, None, None
    new_rc = None
    if bx_rc is not None:
        # In the reverse complement the substitution sits at L-1-p, as base 3-v
        new_rc = _packed_mutations(to_cpu(bx_rc), L, W, l)[1][::-1, ::-1]
    diag = base.raw_diag_packed(bx, bx_rc)
    return old, new, diag, base.raw_diag_mutants(bx, bx_rc, new, new_rc, diag)


def _slot_weights(base, L, lengths=None):
    """[L, R, l] float64 window weights of the windows covering each position,
    as _packed_mutations lays them out (0 for padding), R = 1 when every
    sequence is L long, else one per sequence (from its own length); None
    without window weights."""
    l = base.l
    if base.window_weights(L - l + 1) is None:
        return None
    lengths = np.array([L]) if lengths is None else np.asarray(lengths)
    out = np.zeros((L, len(lengths), l))
    for n in np.unique(lengths):
        W = int(n) - l + 1
        if W < 1:
            continue
        wx = base.window_weights(W)
        rows = lengths == n
        for p in range(int(n)):
            a_s, a_e = max(0, p - l + 1), min(p + 1, W)
            out[p, rows, : a_e - a_s] = wx[a_s:a_e]
    return out


def _packed_mutations(bx, L, W, l):
    """Packed windows covering each position, before and after each substitution.

    Returns old [L, B, l] and new [4, L, B, l] uint32: row (p, b) holds the
    up-to-l windows of sequence b that contain position p, padded with
    PAD_WINDOW (identical in old and new, so padding cancels in new - old;
    PAD past a shorter sequence's windows stays PAD).
    """
    B = bx.shape[0]
    old = np.full((L, B, l), PAD_WINDOW, dtype=np.uint32)
    new = np.full((4, L, B, l), PAD_WINDOW, dtype=np.uint32)
    for p in range(L):
        a_s, a_e = max(0, p - l + 1), min(p + 1, W)
        win = bx[:, a_s:a_e]
        old[p, :, : a_e - a_s] = win
        shift = (2 * (p - np.arange(a_s, a_e))).astype(np.uint32)
        cleared = win & ~(np.uint32(3) << shift)
        pad = win == PAD_WINDOW
        for v in range(4):
            new[v, p, :, : a_e - a_s] = np.where(
                pad, PAD_WINDOW, cleared | (np.uint32(v) << shift)
            )
    return old, new
