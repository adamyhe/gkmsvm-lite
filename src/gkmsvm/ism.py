"""In silico mutagenesis for gapped k-mer SVMs.

Window-delta optimization: when a single base changes, only ~l of the
W = L - l + 1 windows are affected. Instead of recomputing the full kernel,
we compute the delta from the affected windows only.
"""

from __future__ import annotations

import numpy as np

from gkmsvm.backend import get_array_module, to_cpu
from gkmsvm.codec import reverse_complement
from gkmsvm.svm import GkmSVM

_COMPLEMENT = [3, 2, 1, 0]
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
    fn = _ism_index if hasattr(kernel, "pairwise_from_indices") else _ism_float

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
# Float-based path (fallback for non-DirectGkmKernel)
# -----------------------------------------------------------------------


def _ism_float(model, x, chunk):
    """ISM via float32 flat windows (fallback for non-DirectGkmKernel)."""
    xp = get_array_module(x)
    B, C, L = x.shape
    kernel = model.kernel
    l = kernel.l
    W = L - l + 1

    wx = kernel.flat_windows(x)
    sv = model.support_sequences
    S = sv.shape[0]
    wy = kernel.flat_windows(sv)
    F = wx.shape[2]

    ref_K_raw = _raw_pairwise_chunked(kernel, x, sv, chunk)
    ref_score = model(x).squeeze(-1)

    do_norm = kernel.normalize
    do_rc = kernel.include_rc

    diag_chunk = chunk if chunk is not None else 1000
    if do_norm:
        diag_sv = kernel._raw_diagonal(sv, chunk_size=diag_chunk)
    if do_rc:
        wy_rc = kernel.flat_windows(reverse_complement(sv))
        wx_rc = kernel.flat_windows(reverse_complement(x))

    coefs = model.coefficients
    bias = model.bias

    wx_cpu = wx if xp is np else xp.asnumpy(wx)
    all_old, all_new, all_wx_full, ranges = _build_float_mutations(
        wx_cpu, L, W, l, B, F, wx_cpu.dtype
    )
    if do_rc and do_norm:
        wx_rc_cpu = wx_rc if xp is np else xp.asnumpy(wx_rc)
        all_wx_rc_full = _build_rc_float_mutations(wx_rc_cpu, ranges, L, W, l, B, F, wx_rc_cpu.dtype)

    all_old = xp.asarray(all_old)
    all_new = xp.asarray(all_new)

    old_flat = all_old.reshape(L * B, l, F)
    new_flat = all_new.reshape(4 * L * B, l, F)

    old_pw = _partial_pairwise(kernel, old_flat, wy, chunk).reshape(L, B, S)
    new_pw = _partial_pairwise(kernel, new_flat, wy, chunk).reshape(4, L, B, S)
    delta_K = new_pw - old_pw[None, :, :, :]

    if do_rc:
        old_pw_rc = _partial_pairwise(kernel, old_flat, wy_rc, chunk).reshape(L, B, S)
        new_pw_rc = _partial_pairwise(kernel, new_flat, wy_rc, chunk).reshape(4, L, B, S)
        delta_K = delta_K + (new_pw_rc - old_pw_rc[None, :, :, :])

    K_raw_new = ref_K_raw + delta_K

    if not do_norm:
        scores = (K_raw_new * coefs).sum(axis=3) + bias
        return (scores - ref_score).transpose(2, 0, 1).astype(x.dtype)

    all_wx_full = xp.asarray(all_wx_full)
    diag_new = _batched_self_kernel(kernel, all_wx_full, L, B, W, F)

    if do_rc:
        all_wx_rc_full = xp.asarray(all_wx_rc_full)
        diag_new = diag_new + _batched_cross_kernel(
            kernel, all_wx_full, all_wx_rc_full, L, B, W, F
        )

    norm = xp.sqrt(diag_new[:, :, :, None] * diag_sv)
    K_norm = K_raw_new / xp.clip(norm, 1e-10, None)

    scores = (K_norm * coefs).sum(axis=3) + bias
    return (scores - ref_score).transpose(2, 0, 1).astype(x.dtype)


def _build_float_mutations(wx_cpu, L, W, l, B, F, dtype):
    """Build ref/mutant flat-window tensors and full mutated windows."""
    all_old = np.zeros((L, B, l, F), dtype=dtype)
    all_new = np.zeros((4, L, B, l, F), dtype=dtype)
    all_wx_full = np.broadcast_to(wx_cpu[None], (4 * L, B, W, F)).copy()
    ranges = []

    for p in range(L):
        a_s = max(0, p - l + 1)
        a_e = min(p + 1, W)
        n_a = a_e - a_s
        offsets = [p - i for i in range(a_s, a_e)]
        ranges.append((a_s, a_e, n_a, offsets))

        old_A = wx_cpu[:, a_s:a_e, :]
        all_old[p, :, :n_a, :] = old_A

        muts = _mutate_float_windows(old_A, offsets, l, _BASES)
        all_new[:, p, :, :n_a, :] = muts

        for vi in range(4):
            all_wx_full[vi * L + p, :, a_s:a_e, :] = muts[vi]

    return all_old, all_new, all_wx_full, ranges


def _build_rc_float_mutations(wx_rc_cpu, ranges, L, W, l, B, F, dtype):
    """Build mutated reverse-complement flat windows for normalization."""
    all_wx_rc_full = np.broadcast_to(wx_rc_cpu[None], (4 * L, B, W, F)).copy()

    for p in range(L):
        rc_p = L - 1 - p
        rc_as = max(0, rc_p - l + 1)
        rc_ae = min(rc_p + 1, W)
        n_rc = rc_ae - rc_as
        if n_rc > 0:
            rc_off = [rc_p - i for i in range(rc_as, rc_ae)]
            rc_A = wx_rc_cpu[:, rc_as:rc_ae, :]
            rc_muts = _mutate_float_windows(rc_A, rc_off, l, _COMPLEMENT)
            for vi in range(4):
                all_wx_rc_full[vi * L + p, :, rc_as:rc_ae, :] = rc_muts[vi]

    return all_wx_rc_full


def _batched_self_kernel(kernel, all_wx_full, L, B, W, F, group_sz=200):
    """Self-kernel diagonals for mutated sequences (float path)."""
    xp = get_array_module(all_wx_full)
    total = 4 * L
    diag = xp.zeros(total * B, dtype=all_wx_full.dtype)

    for g_start in range(0, total, group_sz):
        g_end = min(g_start + group_sz, total)
        chunk = all_wx_full[g_start:g_end]
        chunk_flat = chunk.reshape(-1, W, F)
        sm = xp.matmul(chunk_flat, chunk_flat.transpose(0, 2, 1))
        diag[g_start * B : g_end * B] = kernel._apply_table(sm)

    return diag.reshape(4, L, B)


def _batched_cross_kernel(
    kernel, all_wx_full, all_wx_rc_full, L, B, W, F, group_sz=200
):
    """Cross-kernel diagonals (fwd × RC) for mutated sequences (float path)."""
    xp = get_array_module(all_wx_full)
    total = 4 * L
    diag = xp.zeros(total * B, dtype=all_wx_full.dtype)

    for g_start in range(0, total, group_sz):
        g_end = min(g_start + group_sz, total)
        c1 = all_wx_full[g_start:g_end].reshape(-1, W, F)
        c2 = all_wx_rc_full[g_start:g_end].reshape(-1, W, F)
        cross = xp.matmul(c1, c2.transpose(0, 2, 1))
        diag[g_start * B : g_end * B] = kernel._apply_table(cross)

    return diag.reshape(4, L, B)


def _mutate_float_windows(wx_A, offsets, l, bases):
    """Apply each of 4 base substitutions to affected flat windows."""
    result = np.broadcast_to(wx_A[None], (4, *wx_A.shape)).copy()
    for a_idx, off in enumerate(offsets):
        for c in range(4):
            result[:, :, a_idx, c * l + off] = 0.0
        for vi, base in enumerate(bases):
            result[vi, :, a_idx, base * l + off] = 1.0
    return result


def _partial_pairwise(kernel, wx_A, wy, chunk_size):
    """Chunked pairwise from flat windows."""
    xp = get_array_module(wx_A)
    S = wy.shape[0]
    if chunk_size is None or chunk_size >= S:
        return kernel.pairwise_from_windows(wx_A, wy)
    chunks = []
    for s in range(0, S, chunk_size):
        e = min(s + chunk_size, S)
        chunks.append(kernel.pairwise_from_windows(wx_A, wy[s:e]))
    return xp.concatenate(chunks, axis=1)


def _raw_pairwise_chunked(kernel, x, sv, chunk_size):
    """Full pairwise with optional SV chunking."""
    xp = get_array_module(x)
    S = sv.shape[0]
    if chunk_size is None or chunk_size >= S:
        return kernel._raw_pairwise(x, sv)
    chunks = []
    for s in range(0, S, chunk_size):
        e = min(s + chunk_size, S)
        chunks.append(kernel._raw_pairwise(x, sv[s:e]))
    return xp.concatenate(chunks, axis=1)
