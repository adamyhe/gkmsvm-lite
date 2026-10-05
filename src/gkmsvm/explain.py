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

import warnings

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
# Fused CuPy CUDA kernel — reduced output with atomicAdd
# ---------------------------------------------------------------------------
# Folds coefficient multiplication and SV-dimension reduction directly
# into the inner loop, accumulating weighted contributions into a
# [B, 4, seqlen] result array via float64 atomicAdd (CC >= 6.0).
# This eliminates the O(B * 4 * L * S_c) intermediate ``persv`` array
# (~640 MB at typical sizes).  Forward + RC SV windows are concatenated
# along the Wy axis so a single kernel launch processes both.

_FUSED_EXPLAIN_CUDA = r"""
extern "C" __global__
void fused_explain_reduced(
    const unsigned int* __restrict__ bx_packed,
    const unsigned int* __restrict__ by_packed_t,
    const double* __restrict__ alpha_table,
    const double* __restrict__ kappa_table,
    const double* __restrict__ eff_coef,
    double* __restrict__ result,
    const int total_pairs,
    const int S,
    const int Wx,
    const int Wy,
    const int l,
    const int min_matches,
    const int seqlen
) {
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
    const double ec = eff_coef[b * S + s];

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
                    long long off = (long long)(b * 4 + by_k) * seqlen + pos;
                    atomicAdd(&result[off], ec * w);
                }
            }
        }
    }
}
"""


# ---------------------------------------------------------------------------
# CuPy kernel cache and getter
# ---------------------------------------------------------------------------

_cupy_explain_kernel = None


def _get_explain_kernel():
    global _cupy_explain_kernel
    import cupy as cp

    if _cupy_explain_kernel is None:
        _cupy_explain_kernel = cp.RawKernel(
            _FUSED_EXPLAIN_CUDA,
            "fused_explain_reduced",
            options=("--use_fast_math",),
        )
    return _cupy_explain_kernel


_FUSED_PERTURBATION_CUDA = r"""
extern "C" __global__
void fused_perturbation_reduced(
    const unsigned int* __restrict__ bx_packed,
    const unsigned int* __restrict__ by_packed_t,
    const double* __restrict__ gamma_table,
    const double* __restrict__ kappa_table,
    const double* __restrict__ eff_coef,
    double* __restrict__ result,
    const int total_pairs,
    const int S,
    const int Wx,
    const int Wy,
    const int l,
    const int min_matches,
    const int seqlen
) {
    extern __shared__ char smem[];
    double* s_gamma = (double*)smem;
    double* s_kappa = s_gamma + l + 2;
    unsigned int* s_bx = (unsigned int*)((char*)smem + 2 * (l + 2) * sizeof(double));

    for (int i = threadIdx.x; i < l + 2; i += blockDim.x) {
        s_gamma[i] = gamma_table[i];
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
    const double ec = eff_coef[b * S + s];

    for (int i = 0; i < Wx; i++) {
        unsigned int bx_val = use_smem ? s_bx[i] : bx_packed[b * Wx + i];
        for (int j = 0; j < Wy; j++) {
            unsigned int by_val = by_packed_t[j * S + s];
            unsigned int xv = bx_val ^ by_val;
            int mm = __popc(((xv >> 1) | xv) & 0x55555555u);
            if (l - mm >= min_matches) {
                double gv = s_gamma[mm];
                double kv = s_kappa[mm];
                for (int k = 0; k < l; k++) {
                    int bx_k = (bx_val >> (2 * k)) & 3;
                    int by_k = (by_val >> (2 * k)) & 3;
                    int pos = k + i;
                    if (bx_k == by_k) {
                        double w = ec * gv;
                        for (int h = 0; h < 4; h++) {
                            if (h != by_k) {
                                long long off = (long long)(b * 4 + h) * seqlen + pos;
                                atomicAdd(&result[off], w);
                            }
                        }
                    } else {
                        long long off = (long long)(b * 4 + by_k) * seqlen + pos;
                        atomicAdd(&result[off], ec * kv);
                    }
                }
            }
        }
    }
}
"""

_cupy_perturbation_kernel = None


def _get_perturbation_kernel():
    global _cupy_perturbation_kernel
    import cupy as cp

    if _cupy_perturbation_kernel is None:
        _cupy_perturbation_kernel = cp.RawKernel(
            _FUSED_PERTURBATION_CUDA,
            "fused_perturbation_reduced",
            options=("--use_fast_math",),
        )
    return _cupy_perturbation_kernel


# ---------------------------------------------------------------------------
# GPU fused kernel wrappers — reduced output
# ---------------------------------------------------------------------------


def _perturbation_gpu_fused_reduced(
    packed_x, packed_y_t, gamma_table, kappa_table,
    eff_coef, result_view,
    B, S_c, Wx, Wy, l, min_matches, seqlen, xp,
):
    gamma_gpu = xp.asarray(gamma_table, dtype=xp.float64)
    kappa_gpu = xp.asarray(kappa_table, dtype=xp.float64)
    eff_coef_gpu = xp.ascontiguousarray(eff_coef.astype(xp.float64))

    total = B * S_c
    block = 256
    grid = (total + block - 1) // block

    alpha_bytes = (l + 2) * 8
    kappa_bytes = (l + 2) * 8
    bx_bytes = Wx * 4
    shared = alpha_bytes + kappa_bytes + bx_bytes

    _get_perturbation_kernel()(
        (grid,), (block,),
        (packed_x, packed_y_t, gamma_gpu, kappa_gpu, eff_coef_gpu,
         result_view,
         np.int32(total), np.int32(S_c), np.int32(Wx), np.int32(Wy),
         np.int32(l), np.int32(min_matches), np.int32(seqlen)),
        shared_mem=shared,
    )


def _explain_gpu_fused_reduced(
    packed_x, packed_y_t, alpha_table, kappa_table,
    eff_coef, result_view,
    B, S_c, Wx, Wy, l, min_matches, seqlen, xp,
):
    """Fused CUDA kernel for GkmExplain with inline coefficient reduction.

    Accumulates weighted contributions directly into ``result_view`` via
    atomicAdd, eliminating the [B, 4, seqlen, S_c] intermediate array.
    Forward and RC SV windows should already be concatenated along Wy.
    """
    alpha_gpu = xp.asarray(alpha_table, dtype=xp.float64)
    kappa_gpu = xp.asarray(kappa_table, dtype=xp.float64)
    eff_coef_gpu = xp.ascontiguousarray(eff_coef.astype(xp.float64))

    total = B * S_c
    block = 256
    grid = (total + block - 1) // block

    alpha_bytes = (l + 2) * 8
    kappa_bytes = (l + 2) * 8
    bx_bytes = Wx * 4
    shared = alpha_bytes + kappa_bytes + bx_bytes

    _get_explain_kernel()(
        (grid,), (block,),
        (packed_x, packed_y_t, alpha_gpu, kappa_gpu, eff_coef_gpu,
         result_view,
         np.int32(total), np.int32(S_c), np.int32(Wx), np.int32(Wy),
         np.int32(l), np.int32(min_matches), np.int32(seqlen)),
        shared_mem=shared,
    )


# ---------------------------------------------------------------------------
# Numba-accelerated CPU kernel (packed-only, bit extraction)
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True, fastmath=True)
def _fused_explain_reduced(
    packed_x, packed_y, alpha_table, kappa_table, eff_coef,
    l, min_matches, B, S_c, W, Wy, seqlen,
):
    result = np.zeros((B, 4, seqlen), dtype=np.float64)
    for b in prange(B):
        for s in range(S_c):
            ec = eff_coef[b, s]
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
                            result[b, by_k, k + i] += ec * weight
    return result


@njit(parallel=True, cache=True, fastmath=True)
def _fused_perturbation_reduced(
    packed_x, packed_y, gamma_table, kappa_table, eff_coef,
    l, min_matches, B, S_c, W, Wy, seqlen,
):
    """Perturbation effect kernel: discrete kernel-value deltas per mutation."""
    result = np.zeros((B, 4, seqlen), dtype=np.float64)
    for b in prange(B):
        for s in range(S_c):
            ec = eff_coef[b, s]
            for i in range(W):
                bx_val = packed_x[b, i]
                for j in range(Wy):
                    by_val = packed_y[s, j]
                    mm = _packed_mismatches(bx_val, by_val)
                    if l - mm >= min_matches:
                        g_val = gamma_table[mm]
                        k_val = kappa_table[mm]
                        for k in range(l):
                            bx_k = int((bx_val >> np.uint32(2 * k)) & np.uint32(3))
                            by_k = int((by_val >> np.uint32(2 * k)) & np.uint32(3))
                            pos = k + i
                            if bx_k == by_k:
                                w = ec * g_val
                                for h in range(4):
                                    if h != by_k:
                                        result[b, h, pos] += w
                            else:
                                result[b, by_k, pos] += ec * k_val
    return result


# ---------------------------------------------------------------------------
# Dense path (fallback for kernels without min-matches skip)
# ---------------------------------------------------------------------------


def _explain_windows_dense(
    wx, wy_4l, alpha_table, kappa_table,
    l, B, W, seqlen, S_c, xp,
):
    wy = wy_4l.reshape(S_c, -1, 4 * l)
    matches = xp.einsum("bif,sjf->bsij", wx, wy)
    m = xp.clip(xp.rint(l - matches).astype(np.int64), 0, l + 1)
    alpha = alpha_table[m]
    kappa = kappa_table[m]

    wx_4l = wx.reshape(B, W, 4, l)
    persv = xp.zeros((B, 4, seqlen, S_c), dtype=np.float64)

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

    return persv


def _explain_windows_dense_perturbation(
    wx, wy_4l, gamma_table, kappa_table,
    l, B, W, seqlen, S_c, xp,
):
    """Dense perturbation-effect fallback (no min-matches skip)."""
    wy = wy_4l.reshape(S_c, -1, 4 * l)
    matches = xp.einsum("bif,sjf->bsij", wx, wy)
    m = xp.clip(xp.rint(l - matches).astype(np.int64), 0, l + 1)
    gamma_m = gamma_table[m]
    kappa_m = kappa_table[m]

    wx_4l = wx.reshape(B, W, 4, l)
    persv = xp.zeros((B, 4, seqlen, S_c), dtype=np.float64)

    dt = persv.dtype
    wx_4l = wx_4l.astype(dt)
    wy_4l = wy_4l.astype(dt)
    for k in range(l):
        qk = wx_4l[:, :, :, k]
        sk = wy_4l[:, :, :, k]
        pm_k = xp.einsum("bwc,sjc->bswj", qk, sk)

        uniform = gamma_m * pm_k
        correction_weight = kappa_m * (1 - pm_k) - gamma_m * pm_k
        contrib_selective = xp.einsum("bswj,sjh->bhws", correction_weight, sk)

        contrib_uniform = uniform.sum(axis=-1).transpose(0, 2, 1)[:, None, :, :]
        persv[:, :, k : k + W, :] += contrib_uniform + contrib_selective

    return persv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _explain_min_matches(alpha_table, kappa_table, l):
    """Minimum match count for non-zero GkmExplain contribution.

    Accounts for both alpha and kappa tables, since kappa[m] uses
    mismatch_table[m-1] and can be non-zero one level beyond alpha.
    """
    max_m = -1
    for m in range(l + 1):
        if alpha_table[m] != 0:
            max_m = m
        if kappa_table is not None and m < len(kappa_table) and kappa_table[m] != 0:
            max_m = max(max_m, m)
    return l - max_m if max_m >= 0 else l + 1


def _build_alpha_table(mismatch_table: np.ndarray, l: int, d: int) -> np.ndarray:
    """Per-mismatch importance weights: table[m] / (l - m) for matching positions."""
    alpha = np.zeros(l + 2, dtype=np.float64)
    for m in range(min(d, l) + 1):
        if l - m > 0:
            alpha[m] = float(mismatch_table[m]) / (l - m)
    return alpha


def _build_kappa_table(mismatch_table: np.ndarray, l: int, d: int) -> np.ndarray:
    """Per-mismatch hypothetical weights for counterfactual attribution."""
    kappa = np.zeros(l + 2, dtype=np.float64)
    for m in range(1, min(d, l) + 1):
        if l - m + 1 > 0:
            kappa[m] = float(mismatch_table[m - 1]) / (l - m + 1)
    return kappa


def _build_perturbation_tables(
    mismatch_table: np.ndarray, l: int, d: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-mismatch perturbation effect weights (C code's perturbation_eff=1).

    gamma[m] = mismatch_table[m+1] - mismatch_table[m]  (match->mismatch)
    kappa[m] = mismatch_table[m-1] - mismatch_table[m]  (mismatch->match)
    """
    gamma = np.zeros(l + 2, dtype=np.float64)
    kappa = np.zeros(l + 2, dtype=np.float64)
    mt_len = len(mismatch_table)
    for m in range(l + 1):
        gw_now = float(mismatch_table[m]) if m < mt_len else 0.0
        gw_more = float(mismatch_table[m + 1]) if m + 1 < mt_len else 0.0
        gamma[m] = gw_more - gw_now
        if m >= 1:
            gw_fewer = float(mismatch_table[m - 1])
            kappa[m] = gw_fewer - gw_now
    return gamma, kappa


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_MODE_ALIASES: dict[int | str, str] = {
    0: "importance",
    1: "hypothetical",
    2: "perturbation",
    "importance": "importance",
    "hypothetical": "hypothetical",
    "perturbation": "perturbation",
}


def gkmexplain(
    model: GkmSVM,
    x: np.ndarray,
    *,
    mode: int | str = "importance",
    sv_chunk_size: int | None = None,
    batch_size: int = 50,
    verbose: bool = False,
) -> np.ndarray:
    """Compute GkmExplain attribution scores.

    Args:
        model: A GkmSVM model.
        x: [B, 4, L] one-hot encoded sequences.
        mode: Attribution mode (string or legacy int).
            ``"importance"`` (0) — hypothetical * one-hot.  Non-reference
                bases are zero; sum over [B, 4, L] equals score - bias
                (completeness axiom).  Use for per-position importance
                visualization.
            ``"hypothetical"`` (1) — hypothetical importance for all four
                bases at every position.  Use for TF-MoDISco (both
                projected and hypothetical contribution inputs).
            ``"perturbation"`` (2) — discrete kernel-value delta for
                each possible single-base mutation (lsgkm C mode 3,
                ``perturbation_eff=1``).  Use for variant effect
                prediction via ``score_variants(method="gkmexplain")``.
        sv_chunk_size: Chunk size for SV processing.
        batch_size: Number of input sequences to process at a time.
        verbose: Show tqdm progress bar.

    Returns:
        [B, 4, L] attribution scores.
    """
    resolved = _MODE_ALIASES.get(mode)
    if resolved is None:
        raise ValueError(
            f"mode must be 'importance', 'hypothetical', or 'perturbation' "
            f"(or 0/1/2), got {mode!r}"
        )

    if resolved == "importance":
        hyp = gkmexplain(
            model, x, mode="hypothetical", sv_chunk_size=sv_chunk_size,
            batch_size=batch_size, verbose=verbose,
        )
        return hyp * model._match_device(x)

    is_perturbation = resolved == "perturbation"

    x = model._match_device(x)

    if is_mlx(x):
        warnings.warn(
            "GkmExplain does not have an MLX kernel — falling back to CPU. "
            "This will be slower than MLX inference.",
            stacklevel=2,
        )
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
                cpu_model, to_cpu(x), mode=resolved,
                sv_chunk_size=sv_chunk_size, batch_size=batch_size,
                verbose=verbose,
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

    if is_perturbation:
        table_a_cpu, table_b_cpu = _build_perturbation_tables(
            kernel._mismatch_table, l, d,
        )
    else:
        table_a_cpu = _build_alpha_table(kernel._mismatch_table, l, d)
        table_b_cpu = _build_kappa_table(kernel._mismatch_table, l, d)
    min_matches = _explain_min_matches(table_a_cpu, table_b_cpu, l)

    table_a = xp.asarray(table_a_cpu)
    table_b = xp.asarray(table_b_cpu)

    diag_chunk = chunk if chunk is not None else 1000
    if do_norm:
        diag_sv = kernel._raw_diagonal(sv, chunk_size=diag_chunk)

    cs = chunk if chunk is not None else min(S, 2000)
    use_packed = min_matches > 0
    use_gpu = xp is not np

    result = xp.zeros((B, 4, seqlen), dtype=np.float64)

    n_sv_chunks = (S + cs - 1) // cs
    n_seq_batches = (B + batch_size - 1) // batch_size
    total_iters = n_sv_chunks * n_seq_batches
    pbar = None
    if verbose:
        from tqdm import tqdm
        pbar = tqdm(total=total_iters, desc="GkmExplain")

    for sv_start in range(0, S, cs):
        sv_end = min(sv_start + cs, S)
        sv_c = sv[sv_start:sv_end]
        S_c = sv_end - sv_start
        coefs_c = coefs[sv_start:sv_end]

        wy = kernel.flat_windows(sv_c)
        wy_4l = wy.reshape(S_c, -1, 4, l)
        Wy = wy_4l.shape[1]

        if do_rc:
            wy_rc = kernel.flat_windows(reverse_complement(sv_c))
            wy_rc_4l = wy_rc.reshape(S_c, -1, 4, l)

        if use_packed:
            if use_gpu:
                by = xp.argmax(wy_4l, axis=2).astype(xp.int8)
                packed_y = _pack_windows_uint32(by, xp)
                if do_rc:
                    by_rc = xp.argmax(wy_rc_4l, axis=2).astype(xp.int8)
                    packed_y = xp.concatenate(
                        [packed_y, _pack_windows_uint32(by_rc, xp)],
                        axis=1,
                    )
                packed_y_t = xp.ascontiguousarray(packed_y.T)
                Wy_total = packed_y.shape[1]
            else:
                by = np.argmax(wy_4l, axis=2).astype(np.int8)
                packed_y = _pack_windows_cpu(by)
                if do_rc:
                    by_rc = np.argmax(wy_rc_4l, axis=2).astype(np.int8)
                    packed_y = np.concatenate(
                        [packed_y, _pack_windows_cpu(by_rc)], axis=1,
                    )
                Wy_total = packed_y.shape[1]

        for b_start in range(0, B, batch_size):
            b_end = min(b_start + batch_size, B)
            x_b = x[b_start:b_end]
            Bb = b_end - b_start

            wx_b = kernel.flat_windows(x_b)
            wx_4l_b = wx_b.reshape(Bb, W, 4, l)

            if do_norm:
                diag_x_b = kernel._raw_diagonal(x_b)
                norm = xp.clip(
                    xp.sqrt(
                        diag_x_b[:, None]
                        * diag_sv[sv_start:sv_end][None, :]
                    ),
                    1e-10,
                    None,
                )

            if do_norm:
                eff_coef = coefs_c[None, :] / norm
            else:
                eff_coef = xp.broadcast_to(
                    coefs_c[None, :], (Bb, S_c),
                )

            if use_packed and use_gpu:
                bx = xp.argmax(wx_4l_b, axis=2).astype(xp.int8)
                packed_x = xp.ascontiguousarray(
                    _pack_windows_uint32(bx, xp)
                )
                gpu_fn = (
                    _perturbation_gpu_fused_reduced if is_perturbation
                    else _explain_gpu_fused_reduced
                )
                gpu_fn(
                    packed_x, packed_y_t,
                    table_a, table_b,
                    xp.ascontiguousarray(eff_coef),
                    result[b_start:b_end],
                    Bb, S_c, W, Wy_total,
                    l, min_matches, seqlen, xp,
                )
            elif use_packed:
                bx = np.argmax(wx_4l_b, axis=2).astype(np.int8)
                packed_x = _pack_windows_cpu(bx)
                cpu_fn = (
                    _fused_perturbation_reduced if is_perturbation
                    else _fused_explain_reduced
                )
                result[b_start:b_end] += cpu_fn(
                    packed_x, packed_y, table_a, table_b,
                    np.ascontiguousarray(eff_coef.astype(np.float64)),
                    l, min_matches, Bb, S_c, W, Wy_total, seqlen,
                )
            else:
                dense_fn = (
                    _explain_windows_dense_perturbation if is_perturbation
                    else _explain_windows_dense
                )
                persv = dense_fn(
                    wx_b, wy_4l, table_a, table_b,
                    l, Bb, W, seqlen, S_c, xp,
                )
                if do_rc:
                    persv = persv + dense_fn(
                        wx_b, wy_rc_4l, table_a, table_b,
                        l, Bb, W, seqlen, S_c, xp,
                    )
                if do_norm:
                    persv = persv / norm[:, None, None, :]
                result[b_start:b_end] += (persv * coefs_c).sum(axis=-1)

            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    return result.astype(x.dtype)
