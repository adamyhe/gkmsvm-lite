"""GkmExplain: analytical attribution for gapped k-mer SVMs.

Decomposes the SVM decision function into per-position, per-base
importance scores without gradients. Ported from the C implementation
in kundajelab/lsgkm (Shrikumar et al. 2019).

Uses packed uint32 pre-filtering to skip ~99.88% of window pairs
(those with more mismatches than d), matching the min-matches skip
used in the forward pass. Per-position base identity is extracted
directly from packed representations via bit shifts, eliminating
all float intermediate arrays from the inner loop.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from gkmsvm.backend import get_array_module, is_mlx
from gkmsvm.codec import reverse_complement
from gkmsvm.kernels.direct import (
    _pack_windows_cpu,
    _pack_windows_uint32,
    _packed_mismatches,
)
from gkmsvm.svm import GkmSVM


# ---------------------------------------------------------------------------
# Fused CuPy RawKernel CUDA source — GkmExplain mode 0 (importance)
# ---------------------------------------------------------------------------

_FUSED_EXPLAIN_MODE0_CUDA = r"""
extern "C" __global__
void fused_explain_mode0(
    const unsigned int* __restrict__ bx_packed,
    const unsigned int* __restrict__ by_packed_t,
    const double* __restrict__ alpha_table,
    double* __restrict__ persv,
    const int total_pairs,
    const int S,
    const int Wx,
    const int Wy,
    const int l,
    const int min_matches,
    const int seqlen
) {
    // bx_packed: [B, Wx] uint32 — query packed windows
    // by_packed_t: [Wy, S] uint32 — SV packed windows (transposed for coalescing)
    // persv: [B, 4, seqlen, S] float64 — output (S last for coalesced writes)
    //
    // One thread per (b, s) pair. No atomics — each thread owns its output slice.
    // Per-position base extracted via bit shifts on packed uint32:
    //   base = (packed >> (2*k)) & 3
    // Mode 0: at matching positions, persv[b, base_x, k+i, s] += alpha[mm]

    extern __shared__ char smem[];
    double* s_alpha = (double*)smem;
    unsigned int* s_bx = (unsigned int*)((char*)smem + (l + 2) * sizeof(double));

    for (int i = threadIdx.x; i < l + 2; i += blockDim.x)
        s_alpha[i] = alpha_table[i];

    const int block_start = blockDim.x * blockIdx.x;
    const int b_first = block_start / S;
    const int last_idx = block_start + (int)blockDim.x - 1;
    const int b_last = (last_idx < total_pairs ? last_idx : total_pairs - 1) / S;
    if (b_first == b_last) {
        const unsigned int* src = bx_packed + b_first * Wx;
        for (int i = threadIdx.x; i < Wx; i += blockDim.x)
            s_bx[i] = src[i];
    }
    __syncthreads();

    const int idx = block_start + threadIdx.x;
    if (idx >= total_pairs) return;

    const int b = idx / S;
    const int s = idx % S;
    const bool use_smem = (b_first == b_last);

    for (int i = 0; i < Wx; i++) {
        unsigned int bx_val = use_smem ? s_bx[i] : bx_packed[b * Wx + i];
        for (int j = 0; j < Wy; j++) {
            unsigned int by_val = by_packed_t[j * S + s];
            unsigned int xv = bx_val ^ by_val;
            int mm = __popc(((xv >> 1) | xv) & 0x55555555u);
            if (l - mm >= min_matches) {
                double av = s_alpha[mm];
                for (int k = 0; k < l; k++) {
                    int bx_k = (bx_val >> (2 * k)) & 3;
                    int by_k = (by_val >> (2 * k)) & 3;
                    if (bx_k == by_k) {
                        int pos = k + i;
                        long long off = ((long long)(b * 4 + bx_k) * seqlen + pos) * S + s;
                        persv[off] += av;
                    }
                }
            }
        }
    }
}
"""


# ---------------------------------------------------------------------------
# Fused CuPy RawKernel CUDA source — GkmExplain mode 1 (hypothetical)
# ---------------------------------------------------------------------------

_FUSED_EXPLAIN_MODE1_CUDA = r"""
extern "C" __global__
void fused_explain_mode1(
    const unsigned int* __restrict__ bx_packed,
    const unsigned int* __restrict__ by_packed_t,
    const double* __restrict__ alpha_table,
    const double* __restrict__ kappa_table,
    double* __restrict__ persv,
    const int total_pairs,
    const int S,
    const int Wx,
    const int Wy,
    const int l,
    const int min_matches,
    const int seqlen
) {
    // Mode 1: at every position k, scatter weight to persv[b, base_y, k+i, s]
    //   weight = alpha[mm] if match, kappa[mm] if mismatch

    extern __shared__ char smem[];
    double* s_alpha = (double*)smem;
    double* s_kappa = s_alpha + l + 2;
    unsigned int* s_bx = (unsigned int*)((char*)smem + 2 * (l + 2) * sizeof(double));

    for (int i = threadIdx.x; i < l + 2; i += blockDim.x) {
        s_alpha[i] = alpha_table[i];
        s_kappa[i] = kappa_table[i];
    }

    const int block_start = blockDim.x * blockIdx.x;
    const int b_first = block_start / S;
    const int last_idx = block_start + (int)blockDim.x - 1;
    const int b_last = (last_idx < total_pairs ? last_idx : total_pairs - 1) / S;
    if (b_first == b_last) {
        const unsigned int* src = bx_packed + b_first * Wx;
        for (int i = threadIdx.x; i < Wx; i += blockDim.x)
            s_bx[i] = src[i];
    }
    __syncthreads();

    const int idx = block_start + threadIdx.x;
    if (idx >= total_pairs) return;

    const int b = idx / S;
    const int s = idx % S;
    const bool use_smem = (b_first == b_last);

    for (int i = 0; i < Wx; i++) {
        unsigned int bx_val = use_smem ? s_bx[i] : bx_packed[b * Wx + i];
        for (int j = 0; j < Wy; j++) {
            unsigned int by_val = by_packed_t[j * S + s];
            unsigned int xv = bx_val ^ by_val;
            int mm = __popc(((xv >> 1) | xv) & 0x55555555u);
            if (l - mm >= min_matches) {
                double av = s_alpha[mm];
                double kv = s_kappa[mm];
                for (int k = 0; k < l; k++) {
                    int bx_k = (bx_val >> (2 * k)) & 3;
                    int by_k = (by_val >> (2 * k)) & 3;
                    double w = (bx_k == by_k) ? av : kv;
                    int pos = k + i;
                    long long off = ((long long)(b * 4 + by_k) * seqlen + pos) * S + s;
                    persv[off] += w;
                }
            }
        }
    }
}
"""


# ---------------------------------------------------------------------------
# CuPy kernel cache and getters
# ---------------------------------------------------------------------------

_cupy_explain_m0_kernel = None
_cupy_explain_m1_kernel = None


def _get_explain_mode0_kernel():
    global _cupy_explain_m0_kernel
    import cupy as cp

    if _cupy_explain_m0_kernel is None:
        _cupy_explain_m0_kernel = cp.RawKernel(
            _FUSED_EXPLAIN_MODE0_CUDA,
            "fused_explain_mode0",
            options=("--use_fast_math",),
        )
    return _cupy_explain_m0_kernel


def _get_explain_mode1_kernel():
    global _cupy_explain_m1_kernel
    import cupy as cp

    if _cupy_explain_m1_kernel is None:
        _cupy_explain_m1_kernel = cp.RawKernel(
            _FUSED_EXPLAIN_MODE1_CUDA,
            "fused_explain_mode1",
            options=("--use_fast_math",),
        )
    return _cupy_explain_m1_kernel


# ---------------------------------------------------------------------------
# GPU fused kernel wrapper
# ---------------------------------------------------------------------------


def _explain_windows_gpu_fused(
    wx_4l, wy_4l, alpha_table, kappa_table,
    l, mode, B, W, seqlen, S_c, Wy, min_matches, xp,
):
    """Fused CUDA kernel for GkmExplain on GPU.

    Uses packed uint32 + bit extraction — no float intermediate arrays.
    One thread per (b, s) pair with coalesced SV reads and output writes.
    """
    import cupy as cp

    bx = xp.argmax(wx_4l, axis=2).astype(xp.int8)
    by = xp.argmax(wy_4l, axis=2).astype(xp.int8)
    packed_x = xp.ascontiguousarray(_pack_windows_uint32(bx, xp))
    packed_y_t = xp.ascontiguousarray(_pack_windows_uint32(by, xp).T)

    alpha_gpu = xp.asarray(alpha_table, dtype=xp.float64)
    persv = xp.zeros((B, 4, seqlen, S_c), dtype=xp.float64)

    total = B * S_c
    block = 256
    grid = (total + block - 1) // block

    alpha_bytes = (l + 2) * 8
    bx_bytes = W * 4

    if mode == 0:
        shared = alpha_bytes + bx_bytes
        _get_explain_mode0_kernel()(
            (grid,), (block,),
            (packed_x, packed_y_t, alpha_gpu, persv,
             np.int32(total), np.int32(S_c), np.int32(W), np.int32(Wy),
             np.int32(l), np.int32(min_matches), np.int32(seqlen)),
            shared_mem=shared,
        )
    else:
        kappa_gpu = xp.asarray(kappa_table, dtype=xp.float64)
        kappa_bytes = (l + 2) * 8
        shared = alpha_bytes + kappa_bytes + bx_bytes
        _get_explain_mode1_kernel()(
            (grid,), (block,),
            (packed_x, packed_y_t, alpha_gpu, kappa_gpu, persv,
             np.int32(total), np.int32(S_c), np.int32(W), np.int32(Wy),
             np.int32(l), np.int32(min_matches), np.int32(seqlen)),
            shared_mem=shared,
        )

    return persv


# ---------------------------------------------------------------------------
# Numba-accelerated CPU kernels (packed-only, bit extraction)
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True, fastmath=True)
def _fused_explain_mode0(
    packed_x, packed_y, alpha_table,
    l, min_matches, B, S_c, W, Wy, seqlen,
):
    persv = np.zeros((B, 4, seqlen, S_c), dtype=np.float64)
    for idx in prange(B * S_c):
        b = idx // S_c
        s = idx % S_c
        for i in range(W):
            bx_val = packed_x[b, i]
            for j in range(Wy):
                by_val = packed_y[s, j]
                mm = _packed_mismatches(bx_val, by_val)
                if l - mm >= min_matches:
                    alpha_val = alpha_table[mm]
                    for k in range(l):
                        bx_k = int((bx_val >> np.uint32(2 * k)) & np.uint32(3))
                        by_k = int((by_val >> np.uint32(2 * k)) & np.uint32(3))
                        if bx_k == by_k:
                            persv[b, bx_k, k + i, s] += alpha_val
    return persv


@njit(parallel=True, cache=True, fastmath=True)
def _fused_explain_mode1(
    packed_x, packed_y, alpha_table, kappa_table,
    l, min_matches, B, S_c, W, Wy, seqlen,
):
    persv = np.zeros((B, 4, seqlen, S_c), dtype=np.float64)
    for idx in prange(B * S_c):
        b = idx // S_c
        s = idx % S_c
        for i in range(W):
            bx_val = packed_x[b, i]
            for j in range(Wy):
                by_val = packed_y[s, j]
                mm = _packed_mismatches(bx_val, by_val)
                if l - mm >= min_matches:
                    a_val = alpha_table[mm]
                    k_val = kappa_table[mm]
                    for k in range(l):
                        bx_k = int((bx_val >> np.uint32(2 * k)) & np.uint32(3))
                        by_k = int((by_val >> np.uint32(2 * k)) & np.uint32(3))
                        if bx_k == by_k:
                            weight = a_val
                        else:
                            weight = k_val
                        persv[b, by_k, k + i, s] += weight
    return persv


# ---------------------------------------------------------------------------
# Dense path (fallback for kernels without min-matches skip)
# ---------------------------------------------------------------------------


def _explain_windows_dense(
    wx, wy_4l, x, alpha_table, kappa_table,
    l, mode, B, W, seqlen, S_c, xp,
):
    wy = wy_4l.reshape(S_c, -1, 4 * l)
    matches = xp.einsum("bif,sjf->bsij", wx, wy)
    m = xp.clip(xp.rint(l - matches).astype(np.int64), 0, l + 1)
    alpha = alpha_table[m]

    wx_4l = wx.reshape(B, W, 4, l)
    persv = xp.zeros((B, 4, seqlen, S_c), dtype=np.float64)

    if mode == 0:
        _mode0_loop_dense(persv, wx_4l, wy_4l, alpha, x, l, W, xp)
    else:
        kappa = kappa_table[m]
        _mode1_loop_dense(persv, wx_4l, wy_4l, alpha, kappa, l, W, xp)

    return persv


def _mode0_loop_dense(persv, wx_4l, wy_4l, alpha, x, l, W, xp):
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


def _mode1_loop_dense(persv, wx_4l, wy_4l, alpha, kappa, l, W, xp):
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


# ---------------------------------------------------------------------------
# Dispatch and helpers
# ---------------------------------------------------------------------------


def _explain_min_matches(alpha_table, kappa_table, l):
    """Minimum match count for non-zero GkmExplain contribution.

    Accounts for both alpha (mode 0) and kappa (mode 1) tables, since
    kappa[m] uses mismatch_table[m-1] and can be non-zero one level
    beyond the alpha table.
    """
    max_m = -1
    for m in range(l + 1):
        if alpha_table[m] != 0:
            max_m = m
        if kappa_table is not None and m < len(kappa_table) and kappa_table[m] != 0:
            max_m = max(max_m, m)
    return l - max_m if max_m >= 0 else l + 1


def _explain_windows(
    wx, wx_4l, wy, x, alpha_table, kappa_table,
    l, mode, B, W, seqlen, S_c, min_matches,
):
    """Compute per-SV explanation from one set of SV windows (fwd or RC)."""
    xp = get_array_module(wx)
    wy_4l = wy.reshape(S_c, -1, 4, l)
    Wy = wy_4l.shape[1]

    if min_matches > 0 and xp is np:
        packed_x = _pack_windows_cpu(
            np.argmax(wx_4l, axis=2).astype(np.int8),
        )
        packed_y = _pack_windows_cpu(
            np.argmax(wy_4l, axis=2).astype(np.int8),
        )
        if mode == 0:
            return _fused_explain_mode0(
                packed_x, packed_y, alpha_table,
                l, min_matches, B, S_c, W, Wy, seqlen,
            )
        return _fused_explain_mode1(
            packed_x, packed_y, alpha_table, kappa_table,
            l, min_matches, B, S_c, W, Wy, seqlen,
        )

    if min_matches > 0 and xp is not np:
        return _explain_windows_gpu_fused(
            wx_4l, wy_4l, alpha_table, kappa_table,
            l, mode, B, W, seqlen, S_c, Wy, min_matches, xp,
        )

    return _explain_windows_dense(
        wx, wy_4l, x, alpha_table, kappa_table,
        l, mode, B, W, seqlen, S_c, xp,
    )


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------


def _build_alpha_table(mismatch_table: np.ndarray, l: int, d: int) -> np.ndarray:
    """Per-mismatch importance weights: table[m] / (l - m) for matching positions."""
    alpha = np.zeros(l + 2, dtype=np.float64)
    for m in range(min(d, l) + 1):
        if l - m > 0:
            alpha[m] = float(mismatch_table[m]) / (l - m)
    return alpha


def _build_kappa_table(mismatch_table: np.ndarray, l: int, d: int) -> np.ndarray:
    """Per-mismatch hypothetical weights for mode=1 (counterfactual attribution)."""
    kappa = np.zeros(l + 2, dtype=np.float64)
    for m in range(1, min(d, l) + 1):
        if l - m + 1 > 0:
            kappa[m] = float(mismatch_table[m - 1]) / (l - m + 1)
    return kappa


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def gkmexplain(
    model: GkmSVM,
    x: np.ndarray,
    *,
    mode: int = 0,
    sv_chunk_size: int | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Compute GkmExplain attribution scores.

    Args:
        model: A GkmSVM model.
        x: [B, 4, L] one-hot encoded sequences.
        mode: 0 = importance scores, 1 = hypothetical importance scores.
        sv_chunk_size: Chunk size for SV processing.
        verbose: Show tqdm progress bar over SV chunks.

    Returns:
        [B, 4, L] attribution scores.
    """
    if mode not in (0, 1):
        raise ValueError(f"mode must be 0 or 1, got {mode}")

    if is_mlx(x):
        from gkmsvm.backend import to_cpu, to_mlx
        cpu_model = GkmSVM(
            to_cpu(model.support_sequences),
            to_cpu(model.coefficients),
            model.bias,
            model.kernel_type,
            model._kernel_params,
            sv_chunk_size=model.sv_chunk_size,
        )
        return to_mlx(
            gkmexplain(
                cpu_model, to_cpu(x), mode=mode,
                sv_chunk_size=sv_chunk_size, verbose=verbose,
            )
        )

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

    alpha_table_cpu = _build_alpha_table(kernel._mismatch_table, l, d)
    kappa_table_cpu = (
        _build_kappa_table(kernel._mismatch_table, l, d) if mode == 1 else None
    )
    min_matches = _explain_min_matches(alpha_table_cpu, kappa_table_cpu, l)

    alpha_table = xp.asarray(alpha_table_cpu)
    kappa_table = xp.asarray(kappa_table_cpu) if kappa_table_cpu is not None else None

    wx = kernel.flat_windows(x)
    wx_4l = wx.reshape(B, W, 4, l)

    diag_chunk = chunk if chunk is not None else 1000
    if do_norm:
        diag_x = kernel._raw_diagonal(x)
        diag_sv = kernel._raw_diagonal(sv, chunk_size=diag_chunk)
        norm = xp.clip(xp.sqrt(diag_x[:, None] * diag_sv[None, :]), 1e-10, None)

    result = xp.zeros((B, 4, seqlen), dtype=np.float64)

    cs = chunk if chunk is not None else min(S, 2000)
    sv_iter = range(0, S, cs)
    if verbose:
        from tqdm import tqdm

        sv_iter = tqdm(sv_iter, desc="GkmExplain", total=(S + cs - 1) // cs)
    for sv_start in sv_iter:
        sv_end = min(sv_start + cs, S)
        sv_c = sv[sv_start:sv_end]
        S_c = sv_end - sv_start

        persv = _explain_windows(
            wx, wx_4l, kernel.flat_windows(sv_c), x,
            alpha_table, kappa_table, l, mode, B, W, seqlen, S_c,
            min_matches,
        )

        if do_rc:
            persv = persv + _explain_windows(
                wx, wx_4l, kernel.flat_windows(reverse_complement(sv_c)), x,
                alpha_table, kappa_table, l, mode, B, W, seqlen, S_c,
                min_matches,
            )

        if do_norm:
            persv = persv / norm[:, None, None, sv_start:sv_end]

        result = result + (persv * coefs[sv_start:sv_end]).sum(axis=-1)

    return result.astype(x.dtype)
