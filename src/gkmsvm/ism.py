"""In silico mutagenesis for gapped k-mer SVMs.

Window-delta optimization: when a single base changes, only ~l of the
W = L - l + 1 windows are affected. Instead of recomputing the full kernel,
we compute the delta from the affected windows only. Inputs the packed
path can't represent (weighted kernels with N) score every mutant instead.
"""

from __future__ import annotations

import numpy as np

from gkmsvm.backend import get_array_module, to_cpu
from gkmsvm.kernels.packed import PAD_WINDOW
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
    packed = model._sv_packed() is not None and base.packs(x)
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
    """ISM from the base kernel's packed windows (any kernel whose SVs pack)."""
    xp = get_array_module(x)
    B, C, L = x.shape
    kernel = model.kernel
    base = kernel.base
    l = base.l
    W = L - l + 1
    S = model.num_support_vectors
    coefs = model.coefficients

    bx, bx_rc = base.pack_both(x)
    ref_raw = model._sv_raw(bx)  # [B, S]
    old, new = _packed_mutations(to_cpu(bx), L, W, l)
    dev = (lambda a: a) if xp is np else xp.asarray
    old_pw = model._sv_raw(dev(old.reshape(L * B, l))).reshape(L, B, S)
    new_pw = model._sv_raw(dev(new.reshape(4 * L * B, l))).reshape(4, L, B, S)
    k_new = ref_raw + (new_pw - old_pw[None])  # [4, L, B, S]

    if base.normalize:
        diag_sv = model._sv_diag()
        diag_ref = base.raw_diag_packed(bx, bx_rc)
        diag_new = _mutated_diagonals(base, x, xp)
        k_new = k_new / xp.clip(xp.sqrt(diag_new[..., None] * diag_sv), 1e-10, None)
        ref_raw = ref_raw / xp.clip(xp.sqrt(diag_ref[:, None] * diag_sv[None, :]), 1e-10, None)
    scores = (kernel.finish(k_new) * coefs).sum(axis=3)  # [4, L, B]
    ref_score = (kernel.finish(ref_raw) * coefs).sum(axis=1)
    return (scores - ref_score).transpose(2, 0, 1).astype(x.dtype)


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


def _mutated_diagonals(kernel, x, xp):
    """Self-kernel diagonals for all single-base mutations.

    Only 4*L*B sequences — trivially fast even for large models.
    Mutations are built on CPU then moved to device for diagonal computation.
    """
    B, C, L = x.shape
    x_np = x if xp is np else to_cpu(x)
    x_exp = x_np[None, None]  # [1, 1, B, 4, L]
    mut_x = np.broadcast_to(x_exp, (4, L, B, C, L)).copy()
    for p in range(L):
        mut_x[:, p, :, :, p] = 0
        for vi in range(4):
            mut_x[vi, p, :, vi, p] = 1
    mut_flat = xp.asarray(mut_x.reshape(4 * L * B, C, L))
    return kernel._raw_diagonal(mut_flat).reshape(4, L, B)


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
    return xp.asarray(delta.astype(x_np.dtype))
