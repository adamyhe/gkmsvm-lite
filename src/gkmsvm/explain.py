"""GkmExplain: analytical attribution for gapped k-mer SVMs.

Decomposes the SVM decision function into per-position, per-base
importance scores without gradients. Ported from the C implementation
in kundajelab/lsgkm (Shrikumar et al. 2019).

For each pair of L-mers (query window w, SV window s) with m mismatches,
each matching position k within the L-mer contributes
g_weights[m] / (L - m) to the explanation at position w+k.

Modes:
  0 — importance scores (only the actual base gets nonzero values)
  1 — hypothetical importance scores (all 4 bases get values;
      at mismatching positions, the SV's base gets kappa)
"""

from __future__ import annotations

import torch

from gkmsvm.codec import reverse_complement
from gkmsvm.svm import GkmSVM


def gkmexplain(
    model: GkmSVM,
    x: torch.Tensor,
    *,
    mode: int = 0,
    sv_chunk_size: int | None = None,
) -> torch.Tensor:
    """Compute GkmExplain attribution scores.

    Args:
        model: A GkmSVM model.
        x: [B, 4, L] one-hot encoded sequences.
        mode: 0 = importance scores, 1 = hypothetical importance scores.
        sv_chunk_size: Chunk size for SV processing.
            Defaults to model.sv_chunk_size.

    Returns:
        [B, 4, L] attribution scores.
    """
    if mode not in (0, 1):
        raise ValueError(f"mode must be 0 or 1, got {mode}")

    B, C, seqlen = x.shape
    kernel = model.kernel
    l = kernel.l
    d = getattr(kernel, "d", l)
    W = seqlen - l + 1
    device = x.device
    dtype = x.dtype
    chunk = sv_chunk_size if sv_chunk_size is not None else model.sv_chunk_size

    sv = model.support_sequences
    S = sv.shape[0]
    coefs = model.coefficients
    do_norm = kernel.normalize
    do_rc = kernel.include_rc

    compute_dtype = torch.float32 if device.type == "mps" else torch.float64
    alpha_table = _build_alpha_table(kernel._mismatch_table, l, d).to(
        device=device, dtype=compute_dtype
    )
    kappa_table = None
    if mode == 1:
        kappa_table = _build_kappa_table(kernel._mismatch_table, l, d).to(
            device=device, dtype=compute_dtype
        )

    wx = kernel.flat_windows(x)
    wx_4l = wx.reshape(B, W, 4, l)

    if do_norm:
        diag_x = kernel._raw_diagonal(x)
        diag_sv = kernel._raw_diagonal(sv)
        norm = torch.sqrt(diag_x.unsqueeze(1) * diag_sv.unsqueeze(0)).clamp(min=1e-10)

    result = torch.zeros(B, 4, seqlen, dtype=compute_dtype, device=device)

    cs = chunk if chunk is not None else S
    for sv_start in range(0, S, cs):
        sv_end = min(sv_start + cs, S)
        sv_c = sv[sv_start:sv_end]
        S_c = sv_end - sv_start

        persv = _explain_windows(
            wx,
            wx_4l,
            kernel.flat_windows(sv_c),
            x,
            alpha_table,
            kappa_table,
            l,
            mode,
            B,
            W,
            seqlen,
            S_c,
        )

        if do_rc:
            persv += _explain_windows(
                wx,
                wx_4l,
                kernel.flat_windows(reverse_complement(sv_c)),
                x,
                alpha_table,
                kappa_table,
                l,
                mode,
                B,
                W,
                seqlen,
                S_c,
            )

        if do_norm:
            persv = persv / norm[:, sv_start:sv_end].unsqueeze(1).unsqueeze(1)

        result += (persv * coefs[sv_start:sv_end]).sum(dim=-1)

    return result.to(dtype)


def _build_alpha_table(mismatch_table: torch.Tensor, l: int, d: int) -> torch.Tensor:
    """alpha[m] = g_weights[m] / (l - m) for m <= d, else 0."""
    alpha = torch.zeros(l + 2, dtype=torch.float64)
    for m in range(min(d, l) + 1):
        if l - m > 0:
            alpha[m] = mismatch_table[m].item() / (l - m)
    return alpha


def _build_kappa_table(mismatch_table: torch.Tensor, l: int, d: int) -> torch.Tensor:
    """kappa[m] = g_weights[m-1] / (l - m + 1) for 1 <= m <= d, else 0.

    Contribution when a mismatching position hypothetically becomes a match,
    reducing the mismatch count from m to m-1.
    """
    kappa = torch.zeros(l + 2, dtype=torch.float64)
    for m in range(1, min(d, l) + 1):
        if l - m + 1 > 0:
            kappa[m] = mismatch_table[m - 1].item() / (l - m + 1)
    return kappa


def _explain_windows(
    wx,
    wx_4l,
    wy,
    x,
    alpha_table,
    kappa_table,
    l,
    mode,
    B,
    W,
    seqlen,
    S_c,
):
    """Compute per-SV explanation from one set of SV windows (fwd or RC)."""
    device = wx.device
    compute_dtype = alpha_table.dtype

    matches = torch.einsum("bif,sjf->bsij", wx, wy)
    m = (l - matches).round().long().clamp(0, l + 1)
    alpha = alpha_table[m]

    wy_4l = wy.reshape(S_c, -1, 4, l)

    persv = torch.zeros(B, 4, seqlen, S_c, dtype=compute_dtype, device=device)

    if mode == 0:
        _mode0_loop(persv, wx_4l, wy_4l, alpha, x, l, W)
    else:
        kappa = kappa_table[m]
        _mode1_loop(persv, wx_4l, wy_4l, alpha, kappa, l, W)

    return persv


def _mode0_loop(persv, wx_4l, wy_4l, alpha, x, l, W):
    """Mode 0: attribute to matching positions at the query's actual base."""
    dt = persv.dtype
    wx_4l = wx_4l.to(dt)
    wy_4l = wy_4l.to(dt)
    for k in range(l):
        qk = wx_4l[:, :, :, k]
        sk = wy_4l[:, :, :, k]
        pm_k = torch.einsum("bwc,sjc->bswj", qk, sk)
        weighted = (pm_k * alpha).sum(dim=-1)
        x_k = x[:, :, k : k + W].to(dt)
        persv[:, :, k : k + W, :] += x_k.unsqueeze(3) * weighted.permute(
            0, 2, 1
        ).unsqueeze(1)


def _mode1_loop(persv, wx_4l, wy_4l, alpha, kappa, l, W):
    """Mode 1: attribute to the SV's base, with alpha (match) or kappa (mismatch)."""
    dt = persv.dtype
    wx_4l = wx_4l.to(dt)
    wy_4l = wy_4l.to(dt)
    for k in range(l):
        qk = wx_4l[:, :, :, k]
        sk = wy_4l[:, :, :, k]
        pm_k = torch.einsum("bwc,sjc->bswj", qk, sk)
        total_weight = alpha * pm_k + kappa * (1 - pm_k)
        contrib = torch.einsum("bswj,sjh->bhws", total_weight, sk)
        persv[:, :, k : k + W, :] += contrib
