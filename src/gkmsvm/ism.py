"""In silico mutagenesis for gapped k-mer SVMs.

Window-delta optimization: when a single base changes, only ~l of the
W = L - l + 1 windows are affected. Instead of recomputing the full kernel,
we compute the delta from the affected windows only — roughly W/l ≈ 26x
faster than naive rescoring for ENCODE parameters (L=300, l=11).

All mutations are built on CPU (where Python loops are fast) and
transferred to the device in bulk. Pairwise deltas and self-kernel
bmm are batched into single GPU operations.
"""

from __future__ import annotations

import torch

from gkmsvm.codec import reverse_complement
from gkmsvm.svm import GkmSVM

_COMPLEMENT = [3, 2, 1, 0]
_BASES = list(range(4))


def ism(
    model: GkmSVM,
    x: torch.Tensor,
    *,
    sv_chunk_size: int | None = None,
) -> torch.Tensor:
    """Compute score change for every single-base substitution.

    Args:
        model: A GkmSVM model.
        x: [B, 4, L] one-hot encoded reference sequences.
        sv_chunk_size: Chunk size for SV pairwise computation.
            Defaults to model.sv_chunk_size.

    Returns:
        [B, 4, L] score deltas. ``result[b, c, p]`` is
        ``score(mutant) - score(ref)`` where the mutant has base *c*
        at position *p*. Zero when *c* is the reference base.
    """
    B, C, L = x.shape
    kernel = model.kernel
    l = kernel.l
    W = L - l + 1
    device = x.device
    dtype = x.dtype
    chunk = sv_chunk_size if sv_chunk_size is not None else model.sv_chunk_size

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
    bias = model._bias

    # --- Build mutation tensors on CPU (fast Python loops) ---
    wx_cpu = wx.cpu()
    all_old, all_new, all_wx_full, ranges = _build_mutations(
        wx_cpu, L, W, l, B, F, dtype
    )
    if do_rc and do_norm:
        wx_rc_cpu = wx_rc.cpu()
        all_wx_rc_full = _build_rc_mutations(wx_rc_cpu, ranges, L, W, l, B, F, dtype)

    # Transfer to device
    all_old = all_old.to(device)
    all_new = all_new.to(device)

    # --- Batched pairwise delta ---
    old_flat = all_old.reshape(L * B, l, F)
    new_flat = all_new.reshape(4 * L * B, l, F)

    old_pw = _partial_pairwise(kernel, old_flat, wy, chunk).reshape(L, B, S)
    new_pw = _partial_pairwise(kernel, new_flat, wy, chunk).reshape(4, L, B, S)
    delta_K = new_pw - old_pw.unsqueeze(0)

    if do_rc:
        old_pw_rc = _partial_pairwise(kernel, old_flat, wy_rc, chunk).reshape(L, B, S)
        new_pw_rc = _partial_pairwise(kernel, new_flat, wy_rc, chunk).reshape(
            4, L, B, S
        )
        delta_K = delta_K + (new_pw_rc - old_pw_rc.unsqueeze(0))

    K_raw_new = ref_K_raw + delta_K

    if not do_norm:
        scores = (K_raw_new * coefs).sum(dim=3) + bias
        return (scores - ref_score).permute(2, 0, 1).to(dtype)

    # --- Batched self-kernel ---
    all_wx_full = all_wx_full.to(device)
    diag_new = _batched_self_kernel(kernel, all_wx_full, L, B, W, F)

    if do_rc:
        all_wx_rc_full = all_wx_rc_full.to(device)
        diag_new = diag_new + _batched_cross_kernel(
            kernel, all_wx_full, all_wx_rc_full, L, B, W, F
        )

    norm = torch.sqrt(diag_new.unsqueeze(3) * diag_sv)
    K_norm = K_raw_new / torch.clamp(norm, min=1e-10)

    scores = (K_norm * coefs).sum(dim=3) + bias
    return (scores - ref_score).permute(2, 0, 1).to(dtype)


def _build_mutations(wx_cpu, L, W, l, B, F, dtype):
    """Build all mutation tensors on CPU. Returns (all_old, all_new, all_wx_full, ranges)."""
    all_old = torch.zeros(L, B, l, F, dtype=dtype)
    all_new = torch.zeros(4, L, B, l, F, dtype=dtype)
    all_wx_full = wx_cpu.unsqueeze(0).expand(4 * L, -1, -1, -1).clone()
    ranges = []

    for p in range(L):
        a_s = max(0, p - l + 1)
        a_e = min(p + 1, W)
        n_a = a_e - a_s
        offsets = [p - i for i in range(a_s, a_e)]
        ranges.append((a_s, a_e, n_a, offsets))

        old_A = wx_cpu[:, a_s:a_e, :]
        all_old[p, :, :n_a, :] = old_A

        muts = _mutate_windows(old_A, offsets, l, _BASES)
        all_new[:, p, :, :n_a, :] = muts

        for vi in range(4):
            all_wx_full[vi * L + p, :, a_s:a_e, :] = muts[vi]

    return all_old, all_new, all_wx_full, ranges


def _build_rc_mutations(wx_rc_cpu, ranges, L, W, l, B, F, dtype):
    """Build RC mutation tensors on CPU."""
    all_wx_rc_full = wx_rc_cpu.unsqueeze(0).expand(4 * L, -1, -1, -1).clone()

    for p in range(L):
        rc_p = L - 1 - p
        W_seq = W  # same W since same length
        rc_as = max(0, rc_p - l + 1)
        rc_ae = min(rc_p + 1, W_seq)
        n_rc = rc_ae - rc_as
        if n_rc > 0:
            rc_off = [rc_p - i for i in range(rc_as, rc_ae)]
            rc_A = wx_rc_cpu[:, rc_as:rc_ae, :]
            rc_muts = _mutate_windows(rc_A, rc_off, l, _COMPLEMENT)
            for vi in range(4):
                all_wx_rc_full[vi * L + p, :, rc_as:rc_ae, :] = rc_muts[vi]

    return all_wx_rc_full


def _batched_self_kernel(kernel, all_wx_full, L, B, W, F, group_sz=200):
    """Batched diagonal kernel via grouped bmm."""
    device = all_wx_full.device
    dtype = all_wx_full.dtype
    total = 4 * L
    diag = torch.zeros(total * B, device=device, dtype=dtype)

    for g_start in range(0, total, group_sz):
        g_end = min(g_start + group_sz, total)
        chunk = all_wx_full[g_start:g_end]
        chunk_flat = chunk.reshape(-1, W, F)
        sm = torch.bmm(chunk_flat, chunk_flat.transpose(1, 2))
        diag[g_start * B : g_end * B] = kernel._apply_table(sm)

    return diag.reshape(4, L, B)


def _batched_cross_kernel(
    kernel, all_wx_full, all_wx_rc_full, L, B, W, F, group_sz=200
):
    """Batched cross-RC kernel via grouped bmm."""
    device = all_wx_full.device
    dtype = all_wx_full.dtype
    total = 4 * L
    diag = torch.zeros(total * B, device=device, dtype=dtype)

    for g_start in range(0, total, group_sz):
        g_end = min(g_start + group_sz, total)
        c1 = all_wx_full[g_start:g_end].reshape(-1, W, F)
        c2 = all_wx_rc_full[g_start:g_end].reshape(-1, W, F)
        cross = torch.bmm(c1, c2.transpose(1, 2))
        diag[g_start * B : g_end * B] = kernel._apply_table(cross)

    return diag.reshape(4, L, B)


def _mutate_windows(wx_A, offsets, l, bases):
    """Create 4 window variants with different bases at the mutation site."""
    result = wx_A.unsqueeze(0).expand(4, -1, -1, -1).clone()
    for a_idx, off in enumerate(offsets):
        for c in range(4):
            result[:, :, a_idx, c * l + off] = 0.0
        for vi, base in enumerate(bases):
            result[vi, :, a_idx, base * l + off] = 1.0
    return result


def _partial_pairwise(kernel, wx_A, wy, chunk_size):
    """apply_table(einsum(wx_A, wy)), optionally chunked over SVs."""
    S = wy.shape[0]
    B = wx_A.shape[0]
    if chunk_size is None or chunk_size >= S:
        matches = torch.einsum("bif,sjf->bsij", wx_A, wy)
        return kernel._apply_table(matches)
    compute_dtype = torch.float64 if wx_A.device.type == "cpu" else torch.float32
    result = torch.zeros(B, S, device=wx_A.device, dtype=compute_dtype)
    for s in range(0, S, chunk_size):
        e = min(s + chunk_size, S)
        matches = torch.einsum("bif,sjf->bsij", wx_A, wy[s:e])
        result[:, s:e] = kernel._apply_table(matches)
    return result


def _raw_pairwise_chunked(kernel, x, sv, chunk_size):
    """Compute raw pairwise kernel, chunked over SVs."""
    S = sv.shape[0]
    B = x.shape[0]
    if chunk_size is None or chunk_size >= S:
        return kernel._raw_pairwise(x, sv)
    result = torch.zeros(B, S, dtype=x.dtype, device=x.device)
    for s in range(0, S, chunk_size):
        e = min(s + chunk_size, S)
        result[:, s:e] = kernel._raw_pairwise(x, sv[s:e])
    return result
