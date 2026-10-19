"""In silico mutagenesis for gapped k-mer SVMs.

Window-delta optimization: when a single base changes, only ~l of the
W = L - l + 1 windows are affected. Instead of recomputing the full kernel,
we compute the delta from the affected windows only. Kernels with window
weights (-t 4/-t 5), whose window sets here don't follow sequence positions,
score every mutant instead.
"""

from __future__ import annotations

import numpy as np

from gkmsvm.backend import float_dtype, get_array_module, is_mlx, to_cpu
from gkmsvm.kernels.packed import PAD_WINDOW
from gkmsvm.kernels.rbf import RbfKernel
from gkmsvm.svm import GkmSVM


def ism(
    model: GkmSVM,
    x: np.ndarray,
    *,
    batch_size: int | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Compute score change for every single-base substitution.

    Args:
        model: A GkmSVM model.
        x: [B, 4, L] one-hot encoded reference sequences.
        batch_size: Process inputs in batches of this size.  ``None``
            means process all at once (when ``verbose=False``) or
            one at a time (when ``verbose=True``).
        verbose: Show tqdm progress bar over input batches.

    Returns:
        [B, 4, L] score deltas.
    """
    x = model._match_device(x)
    base = model.kernel.base
    packed = (model._sv_packed() is not None
              and base.window_weights(x.shape[2] - base.l + 1) is None)
    fn = _ism_packed if packed else _ism_mutants

    bs = batch_size if batch_size is not None else (1 if verbose else None)
    if bs is not None and x.shape[0] > bs:
        from tqdm import tqdm
        xp = get_array_module(x)
        chunks = range(0, x.shape[0], bs)
        if verbose:
            chunks = tqdm(chunks, desc="ISM",
                          total=(x.shape[0] + bs - 1) // bs)
        return xp.concatenate([fn(model, x[s:s + bs]) for s in chunks], axis=0)

    return fn(model, x)


# -----------------------------------------------------------------------
# Packed path: kernel deltas from the windows covering each position
# -----------------------------------------------------------------------


def _ism_packed(model, x):
    """ISM from the base kernel's packed windows (any kernel whose SVs pack).

    Kernels with an l-mer table add table weights over the windows covering
    each position (_ism_table). On CPU with the l-mer index, a fused pass per
    (position, sequence) reduces each mutant over the SVs (_ism_index).
    Otherwise per-SV kernel values for each mutant are computed in blocks
    that hold at most ``GkmSVM._MAX_SCORE_BYTES`` of them (_ism_kernel).
    """
    xp = get_array_module(x)
    B, _, L = x.shape
    base = model.kernel.base
    W = L - base.l + 1
    S = model.num_support_vectors
    # ism() checked that x packs; a table exists only for non-RBF kernels
    table = model._lmer_table_for(xp)
    # Sequences per block: the mutant windows stay near _BLOCK_BYTES, and the
    # [rows, S] kernel values within _MAX_SCORE_BYTES (the reference rows for
    # the index path, those and a block of positions' mutants otherwise)
    rows = _BLOCK_BYTES // (8 * L * W * 4)
    if table is not None:
        fn = _ism_table
    elif xp is np and model._sv_index() is not None:
        fn = _ism_index
        rows = min(rows, model._MAX_SCORE_BYTES // (8 * S))
    else:
        fn = _ism_kernel
        rows = min(rows, model._MAX_SCORE_BYTES // (6 * L * 8 * S))
    rows = max(1, rows)
    parts = [fn(model, x[s:s + rows], table) for s in range(0, B, rows)]
    out = parts[0] if len(parts) == 1 else xp.concatenate(parts, axis=0)
    return out.astype(float_dtype(out))


# Mutant windows built per block of sequences
_BLOCK_BYTES = 256 * 2**20


def _ism_table(model, x, table):
    """f(mutant) - f(x) from the l-mer table: [B, 4, L]."""
    xp = get_array_module(x)
    base = model.kernel
    bx, bx_rc = base.pack_both(x)
    old, new, diag_ref, mut_diag = _mutants(base, x, bx, bx_rc)
    dev = (lambda a: a) if xp is np else xp.asarray
    # Padding (identical in old and new) reads entry 0, which cancels in new - old
    old = dev(np.where(old == PAD_WINDOW, 0, old))
    new = dev(np.where(new == PAD_WINDOW, 0, new))
    ref = table[bx].sum(axis=1)  # [B]
    k_new = ref + (table[new].sum(axis=3) - table[old].sum(axis=2)[None])  # [4, L, B]
    if base.normalize:
        k_new = k_new / xp.sqrt(xp.clip(mut_diag.astype(ref.dtype), 1e-10, None))
        ref = ref / xp.sqrt(xp.clip(diag_ref.astype(ref.dtype), 1e-10, None))
    return (k_new - ref).transpose(2, 0, 1)


def _ism_index(model, x, _table=None):
    """f(mutant) - f(x) on CPU through the l-mer index: [B, 4, L].

    One fused pass per (position, sequence) adds the changed windows' hits to
    the reference row and reduces over SVs, so no [mutants, S] block exists.
    """
    kernel = model.kernel
    base = kernel.base
    index = model._sv_index()
    bx, bx_rc = base.pack_both(x)
    old, new, diag_ref, mut_diag = _mutants(base, x, bx, bx_rc)
    inv_dx, inv_ds = model._inv_norms(diag_ref, x.shape[0])
    inv_dmut = (np.ones(new.shape[:3]) if mut_diag is None
                else 1 / np.sqrt(np.clip(np.asarray(mut_diag, dtype=np.float64), 1e-20, None)))
    gamma = kernel.gamma if isinstance(kernel, RbfKernel) else np.nan
    coef = model.coefficients
    ref = index.pairwise(bx)  # [B, S]
    ref_score = index.reduce(bx, inv_dx, inv_ds, coef, gamma)
    out = index.mutants(old, new, ref, ref_score, inv_dx, inv_dmut, inv_ds, coef, gamma)
    return out.transpose(2, 0, 1)


def _ism_kernel(model, x, _table=None):
    """f(mutant) - f(x) from per-SV kernel values: [B, 4, L]."""
    xp = get_array_module(x)
    B, _, L = x.shape
    kernel = model.kernel
    base = kernel.base
    S = model.num_support_vectors
    coefs = model.coefficients
    bx, bx_rc = base.pack_both(x)
    old, new, diag_ref, mut_diag = _mutants(base, x, bx, bx_rc)
    dev = (lambda a: a) if xp is np else xp.asarray
    ref_raw = model._sv_raw(bx)  # [B, S]
    if base.normalize:
        diag_sv = model._sv_diag()
        ref_k = ref_raw / xp.clip(xp.sqrt(diag_ref[:, None] * diag_sv[None, :]), 1e-10, None)
    else:
        ref_k = ref_raw
    ref_score = (kernel.finish(ref_k) * coefs).sum(axis=1)  # [B]
    l = base.l
    # Positions per block: [5, P, B, S] kernel values within _MAX_SCORE_BYTES
    P = max(1, model._MAX_SCORE_BYTES // (5 * B * S * 8))
    scores = []
    for p0 in range(0, L, P):
        p1 = min(p0 + P, L)
        n = p1 - p0
        old_pw = model._sv_raw(dev(old[p0:p1].reshape(n * B, l))).reshape(n, B, S)
        new_pw = model._sv_raw(dev(new[:, p0:p1].reshape(4 * n * B, l))).reshape(4, n, B, S)
        k_new = ref_raw + (new_pw - old_pw[None])  # [4, n, B, S]
        if base.normalize:
            k_new = k_new / xp.clip(xp.sqrt(mut_diag[:, p0:p1, :, None] * diag_sv), 1e-10, None)
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


def _packed_mutations(bx, L, W, l):
    """Packed windows covering each position, before and after each substitution.

    Returns old [L, B, l] and new [4, L, B, l] uint32: row (p, b) holds the
    up-to-l windows of sequence b that contain position p, padded with
    PAD_WINDOW (identical in old and new, so padding cancels in new - old).
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
        for v in range(4):
            new[v, p, :, : a_e - a_s] = cleared | (np.uint32(v) << shift)
    return old, new


# -----------------------------------------------------------------------
# Mutant-scoring path (inputs or SVs the packed path can't represent)
# -----------------------------------------------------------------------

# Mutants scored per model() call: caps the [batch, S] intermediates
_MUTANT_BATCH = 1024


def _ism_mutants(model, x):
    """ISM by scoring all 4L single-base mutants through the model."""
    xp = get_array_module(x)
    B, C, L = x.shape
    x_np = to_cpu(x)
    mut = np.broadcast_to(x_np[:, None, None], (B, 4, L, C, L)).copy()
    pos = np.arange(L)
    mut[:, :, pos, :, pos] = 0
    for v in range(4):
        mut[:, v, pos, v, pos] = 1
    scores = to_cpu(model(mut.reshape(B * 4 * L, C, L), batch_size=_MUTANT_BATCH))
    delta = scores.reshape(B, 4, L) - to_cpu(model(x_np)).reshape(B, 1, 1)
    delta[x_np.astype(bool)] = 0  # the reference base itself
    return xp.asarray(delta.astype(np.float32 if is_mlx(x) else np.float64))
