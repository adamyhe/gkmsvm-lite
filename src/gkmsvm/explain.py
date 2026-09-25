"""GkmExplain: analytical attribution for gapped k-mer SVMs.

Decomposes the SVM decision function into per-position, per-base
importance scores without gradients. Ported from the C implementation
in kundajelab/lsgkm (Shrikumar et al. 2019).
"""

from __future__ import annotations

import numpy as np

from gkmsvm.backend import get_array_module
from gkmsvm.codec import reverse_complement
from gkmsvm.svm import GkmSVM


def gkmexplain(
    model: GkmSVM,
    x: np.ndarray,
    *,
    mode: int = 0,
    sv_chunk_size: int | None = None,
) -> np.ndarray:
    """Compute GkmExplain attribution scores.

    Args:
        model: A GkmSVM model.
        x: [B, 4, L] one-hot encoded sequences.
        mode: 0 = importance scores, 1 = hypothetical importance scores.
        sv_chunk_size: Chunk size for SV processing.

    Returns:
        [B, 4, L] attribution scores.
    """
    if mode not in (0, 1):
        raise ValueError(f"mode must be 0 or 1, got {mode}")

    xp = get_array_module(x)
    B, C, seqlen = x.shape
    kernel = model.kernel
    l = kernel.l
    d = getattr(kernel, "d", l)
    W = seqlen - l + 1
    chunk = sv_chunk_size if sv_chunk_size is not None else model.sv_chunk_size

    sv = model.support_sequences
    S = sv.shape[0]
    coefs = model.coefficients
    do_norm = kernel.normalize
    do_rc = kernel.include_rc

    alpha_table = _build_alpha_table(kernel._mismatch_table, l, d)
    alpha_table = xp.asarray(alpha_table)
    kappa_table = None
    if mode == 1:
        kappa_table = xp.asarray(_build_kappa_table(kernel._mismatch_table, l, d))

    wx = kernel.flat_windows(x)
    wx_4l = wx.reshape(B, W, 4, l)

    diag_chunk = chunk if chunk is not None else 1000
    if do_norm:
        diag_x = kernel._raw_diagonal(x)
        diag_sv = kernel._raw_diagonal(sv, chunk_size=diag_chunk)
        norm = xp.clip(xp.sqrt(diag_x[:, None] * diag_sv[None, :]), 1e-10, None)

    result = xp.zeros((B, 4, seqlen), dtype=np.float64)

    cs = chunk if chunk is not None else min(S, 2000)
    for sv_start in range(0, S, cs):
        sv_end = min(sv_start + cs, S)
        sv_c = sv[sv_start:sv_end]
        S_c = sv_end - sv_start

        persv = _explain_windows(
            wx, wx_4l, kernel.flat_windows(sv_c), x,
            alpha_table, kappa_table, l, mode, B, W, seqlen, S_c,
        )

        if do_rc:
            persv = persv + _explain_windows(
                wx, wx_4l, kernel.flat_windows(reverse_complement(sv_c)), x,
                alpha_table, kappa_table, l, mode, B, W, seqlen, S_c,
            )

        if do_norm:
            persv = persv / norm[:, None, None, sv_start:sv_end]

        result = result + (persv * coefs[sv_start:sv_end]).sum(axis=-1)

    return result.astype(x.dtype)


def _build_alpha_table(mismatch_table: np.ndarray, l: int, d: int) -> np.ndarray:
    alpha = np.zeros(l + 2, dtype=np.float64)
    for m in range(min(d, l) + 1):
        if l - m > 0:
            alpha[m] = float(mismatch_table[m]) / (l - m)
    return alpha


def _build_kappa_table(mismatch_table: np.ndarray, l: int, d: int) -> np.ndarray:
    kappa = np.zeros(l + 2, dtype=np.float64)
    for m in range(1, min(d, l) + 1):
        if l - m + 1 > 0:
            kappa[m] = float(mismatch_table[m - 1]) / (l - m + 1)
    return kappa


def _explain_windows(
    wx, wx_4l, wy, x,
    alpha_table, kappa_table, l, mode, B, W, seqlen, S_c,
):
    """Compute per-SV explanation from one set of SV windows (fwd or RC)."""
    xp = get_array_module(wx)

    matches = xp.einsum("bif,sjf->bsij", wx, wy)
    m = xp.clip(xp.rint(l - matches).astype(np.int64), 0, l + 1)
    alpha = alpha_table[m]

    wy_4l = wy.reshape(S_c, -1, 4, l)

    persv = xp.zeros((B, 4, seqlen, S_c), dtype=np.float64)

    if mode == 0:
        _mode0_loop(persv, wx_4l, wy_4l, alpha, x, l, W, xp)
    else:
        kappa = kappa_table[m]
        _mode1_loop(persv, wx_4l, wy_4l, alpha, kappa, l, W, xp)

    return persv


def _mode0_loop(persv, wx_4l, wy_4l, alpha, x, l, W, xp):
    dt = persv.dtype
    wx_4l = wx_4l.astype(dt)
    wy_4l = wy_4l.astype(dt)
    for k in range(l):
        qk = wx_4l[:, :, :, k]
        sk = wy_4l[:, :, :, k]
        pm_k = xp.einsum("bwc,sjc->bswj", qk, sk)
        weighted = (pm_k * alpha).sum(axis=-1)
        x_k = x[:, :, k : k + W].astype(dt)
        persv[:, :, k : k + W, :] += x_k[:, :, :, None] * weighted.transpose(
            0, 2, 1
        )[:, None, :, :]


def _mode1_loop(persv, wx_4l, wy_4l, alpha, kappa, l, W, xp):
    dt = persv.dtype
    wx_4l = wx_4l.astype(dt)
    wy_4l = wy_4l.astype(dt)
    for k in range(l):
        qk = wx_4l[:, :, :, k]
        sk = wy_4l[:, :, :, k]
        pm_k = xp.einsum("bwc,sjc->bswj", qk, sk)
        total_weight = alpha * pm_k + kappa * (1 - pm_k)
        contrib = xp.einsum("bswj,sjh->bhws", total_weight, sk)
        persv[:, :, k : k + W, :] += contrib
