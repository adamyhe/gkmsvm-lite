"""In silico mutagenesis for gapped k-mer SVMs.

Window-delta optimization: when a single base changes, only ~l of the
W = L - l + 1 windows are affected. Instead of recomputing the full kernel,
we compute the delta from the affected windows only.
Kernels without the packed index path (-t 3/4/5) score every mutant instead.
"""

from __future__ import annotations

import numpy as np

from gkmsvm.backend import get_array_module, to_cpu
from gkmsvm.svm import GkmSVM

_BASES = list(range(4))


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
    kernel = model.kernel
    chunk = model.sv_chunk_size
    fn = _ism_index if hasattr(kernel, "pairwise_from_indices") else _ism_mutants

    bs = batch_size if batch_size is not None else (1 if verbose else None)
    if bs is not None and x.shape[0] > bs:
        from tqdm import tqdm
        xp = get_array_module(x)
        chunks = range(0, x.shape[0], bs)
        if verbose:
            chunks = tqdm(chunks, desc="ISM",
                          total=(x.shape[0] + bs - 1) // bs)
        results = []
        for start in chunks:
            end = min(start + bs, x.shape[0])
            results.append(fn(model, x[start:end], chunk))
        return xp.concatenate(results, axis=0)

    return fn(model, x, chunk)


# -----------------------------------------------------------------------
# Index-based path (int8, ~16x less memory, 4x fewer ops)
# -----------------------------------------------------------------------


def _ism_index(model, x, chunk):
    """ISM via int8 base-index comparison (DirectGkmKernel fast path)."""
    xp = get_array_module(x)
    B, C, L = x.shape
    kernel = model.kernel
    l = kernel.l
    W = L - l + 1
    S = model.num_support_vectors
    do_norm = kernel.normalize
    do_rc = kernel.include_rc
    coefs = model.coefficients
    bias = model.bias

    sv_idx, sv_rc_idx = model._get_sv_index_windows()
    bx = kernel.base_index_windows(x)

    ref_raw = kernel.pairwise_from_indices(bx, sv_idx)
    if do_rc:
        ref_raw = ref_raw + kernel.pairwise_from_indices(bx, sv_rc_idx)

    ref_score = model(x).squeeze(-1)

    if do_norm:
        diag_sv = model._get_sv_diag()

    bx_cpu = bx if xp is np else xp.asnumpy(bx)
    all_old, all_new = _build_idx_mutations(bx_cpu, L, W, l, B)

    all_old = xp.asarray(all_old)
    all_new = xp.asarray(all_new)

    old_flat = all_old.reshape(L * B, l, l)
    new_flat = all_new.reshape(4 * L * B, l, l)

    old_pw = _partial_pairwise_idx(kernel, old_flat, sv_idx, chunk).reshape(L, B, S)
    new_pw = _partial_pairwise_idx(kernel, new_flat, sv_idx, chunk).reshape(4, L, B, S)
    delta_K = new_pw - old_pw[None]

    if do_rc:
        old_pw_rc = _partial_pairwise_idx(kernel, old_flat, sv_rc_idx, chunk).reshape(L, B, S)
        new_pw_rc = _partial_pairwise_idx(kernel, new_flat, sv_rc_idx, chunk).reshape(4, L, B, S)
        delta_K = delta_K + (new_pw_rc - old_pw_rc[None])

    K_raw_new = ref_raw + delta_K

    if not do_norm:
        scores = (K_raw_new * coefs).sum(axis=3) + bias
        return (scores - ref_score).transpose(2, 0, 1).astype(x.dtype)

    diag_new = _mutated_diagonals(kernel, x, xp)

    norm = xp.sqrt(diag_new[:, :, :, None] * diag_sv)
    K_norm = K_raw_new / xp.clip(norm, 1e-10, None)

    scores = (K_norm * coefs).sum(axis=3) + bias
    return (scores - ref_score).transpose(2, 0, 1).astype(x.dtype)


def _build_idx_mutations(bx_cpu, L, W, l, B):
    """Build ref/mutant int8 window tensors for all positions.

    Sentinel -1 pads variable-length affected windows: never matches
    any base (0-3), so table[l] = C(0, k) = 0 contribution.
    """
    all_old = np.full((L, B, l, l), -1, dtype=np.int8)
    all_new = np.full((4, L, B, l, l), -1, dtype=np.int8)

    for p in range(L):
        a_s = max(0, p - l + 1)
        a_e = min(p + 1, W)
        n_a = a_e - a_s
        offsets = [p - i for i in range(a_s, a_e)]

        old_A = bx_cpu[:, a_s:a_e, :]
        all_old[p, :, :n_a, :] = old_A

        muts = _mutate_index_windows(old_A, offsets, _BASES)
        all_new[:, p, :, :n_a, :] = muts

    return all_old, all_new


def _mutate_index_windows(bx_A, offsets, bases):
    """Apply each of 4 base substitutions to affected int8 windows."""
    result = np.broadcast_to(bx_A[None], (4, *bx_A.shape)).copy()
    for a_idx, off in enumerate(offsets):
        for vi, base in enumerate(bases):
            result[vi, :, a_idx, off] = base
    return result


def _partial_pairwise_idx(kernel, bx_A, by, chunk_size):
    """Chunked pairwise from int8 index windows."""
    xp = get_array_module(bx_A)
    S = by.shape[0]
    if chunk_size is None or chunk_size >= S:
        return kernel.pairwise_from_indices(bx_A, by)
    chunks = []
    for s in range(0, S, chunk_size):
        e = min(s + chunk_size, S)
        chunks.append(kernel.pairwise_from_indices(bx_A, by[s:e]))
    return xp.concatenate(chunks, axis=1)


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
    diag = kernel._raw_diagonal(mut_flat)
    return diag.reshape(4, L, B)


# -----------------------------------------------------------------------
# Mutant-scoring path (kernels without pairwise_from_indices: -t 3/4/5)
# -----------------------------------------------------------------------

# Mutants scored per model() call: caps the [batch, S] intermediates
_MUTANT_BATCH = 1024


def _ism_mutants(model, x, chunk):
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
