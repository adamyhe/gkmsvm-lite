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
from gkmsvm.kernels.direct import _packed_mismatches
from gkmsvm.svm import GkmSVM

# ---------------------------------------------------------------------------
# Fused kernels: coefficient multiplication and SV reduction in the inner loop
# ---------------------------------------------------------------------------
# Each (query, SV) pair accumulates its weighted contributions straight into
# the [B, 4, seqlen] result, so there is no per-SV intermediate. Forward and
# RC SV windows are concatenated along the window axis. With `perturbation`,
# table_a/table_b are the match->mismatch / mismatch->match kernel deltas
# (lsgkm perturbation_eff=1); otherwise they are the alpha/kappa importance
# weights. CUDA accumulates with float64 atomicAdd (CC >= 6.0).

_FUSED_EXPLAIN_CUDA = r"""
extern "C" __global__
void fused_explain(
    const unsigned int* __restrict__ bx_packed,
    const unsigned int* __restrict__ by_packed_t,
    const double* __restrict__ table_a,
    const double* __restrict__ table_b,
    const double* __restrict__ eff_coef,
    double* __restrict__ result,
    const int total_pairs, const int S, const int Wx, const int Wy,
    const int l, const int min_matches, const int seqlen, const int perturbation
) {
    extern __shared__ char smem[];
    double* s_a = (double*)smem;
    double* s_b = s_a + l + 2;
    unsigned int* s_bx = (unsigned int*)((char*)smem + 2 * (l + 2) * sizeof(double));

    for (int i = threadIdx.x; i < l + 2; i += blockDim.x) {
        s_a[i] = table_a[i];
        s_b[i] = table_b[i];
    }

    // Blocks whose pairs all belong to one query row cache its windows
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
            if (l - mm < min_matches) continue;
            double av = ec * s_a[mm];
            double bv = ec * s_b[mm];
            for (int k = 0; k < l; k++) {
                int bx_k = (bx_val >> (2 * k)) & 3;
                int by_k = (by_val >> (2 * k)) & 3;
                double* row = result + (long long)b * 4 * seqlen + k + i;
                if (!perturbation) {
                    atomicAdd(&row[(long long)by_k * seqlen], bx_k == by_k ? av : bv);
                } else if (bx_k == by_k) {
                    for (int h = 0; h < 4; h++)
                        if (h != by_k) atomicAdd(&row[(long long)h * seqlen], av);
                } else {
                    atomicAdd(&row[(long long)by_k * seqlen], bv);
                }
            }
        }
    }
}
"""

_cupy_explain_kernel = None


def _explain_gpu(packed_x, packed_y_t, table_a, table_b, eff_coef, result,
                 S_c, Wy, l, min_matches, perturbation):
    """Launch the CUDA kernel, accumulating into ``result`` [B, 4, seqlen]."""
    import cupy as cp

    global _cupy_explain_kernel
    if _cupy_explain_kernel is None:
        _cupy_explain_kernel = cp.RawKernel(
            _FUSED_EXPLAIN_CUDA, "fused_explain", options=("--use_fast_math",))
    B, Wx = packed_x.shape
    total = B * S_c
    block = 256
    shared = 2 * (l + 2) * 8 + Wx * 4
    _cupy_explain_kernel(
        ((total + block - 1) // block,), (block,),
        (packed_x, packed_y_t, cp.asarray(table_a, dtype=cp.float64),
         cp.asarray(table_b, dtype=cp.float64),
         cp.ascontiguousarray(eff_coef, dtype=cp.float64), result,
         np.int32(total), np.int32(S_c), np.int32(Wx), np.int32(Wy),
         np.int32(l), np.int32(min_matches), np.int32(result.shape[2]),
         np.int32(perturbation)),
        shared_mem=shared,
    )


@njit(parallel=True, cache=True, fastmath=True)
def _explain_cpu(packed_x, packed_y, table_a, table_b, eff_coef, l, min_matches,
                 seqlen, perturbation):
    """[B, 4, seqlen] attributions of packed_x [B, W] against packed_y [S, Wy]."""
    B, W = packed_x.shape
    S_c, Wy = packed_y.shape
    result = np.zeros((B, 4, seqlen), dtype=np.float64)
    for b in prange(B):
        for s in range(S_c):
            ec = eff_coef[b, s]
            for i in range(W):
                bx_val = packed_x[b, i]
                for j in range(Wy):
                    by_val = packed_y[s, j]
                    mm = _packed_mismatches(bx_val, by_val)
                    if l - mm < min_matches:
                        continue
                    av = ec * table_a[mm]
                    bv = ec * table_b[mm]
                    for k in range(l):
                        bx_k = int((bx_val >> np.uint32(2 * k)) & np.uint32(3))
                        by_k = int((by_val >> np.uint32(2 * k)) & np.uint32(3))
                        pos = k + i
                        if not perturbation:
                            result[b, by_k, pos] += av if bx_k == by_k else bv
                        elif bx_k == by_k:
                            for h in range(4):
                                if h != by_k:
                                    result[b, h, pos] += av
                        else:
                            result[b, by_k, pos] += bv
    return result


@njit(parallel=True, cache=True)
def _impact_cpu(k_both, diag_r, diag_sv, norm, coef, gamma):
    """[n] mutation impact from raw K [ref rows; alt rows] against the SVs.

    With norm, both K(ref) and dK are divided by sqrt(K(ref, ref) K(sv, sv)).
    gamma (not NaN) applies the RBF redistribution of mutation_impact().
    """
    n = k_both.shape[0] // 2
    S = k_both.shape[1]
    rbf = not np.isnan(gamma)
    floor = np.exp(-gamma) if rbf else 0.0
    out = np.zeros(n)
    for b in prange(n):
        acc = 0.0
        for s in range(S):
            k = k_both[b, s]
            dk = k_both[n + b, s] - k
            if norm:
                z = max(np.sqrt(diag_r[b] * diag_sv[s]), 1e-10)
                k /= z
                dk /= z
            if rbf:
                if k <= 0:
                    continue
                dk *= (np.exp(gamma * (k - 1.0)) - floor) / k
            acc += coef[s] * dk
        out[b] = acc
    return out


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
        batch_size: Process inputs in batches of this size (default 50).
        verbose: Show tqdm progress bar over input batches.

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
            model, x, mode="hypothetical",
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
                batch_size=batch_size, verbose=verbose,
            )
        )

    kernel = model.kernel
    if getattr(kernel, "_mismatch_table", None) is None:
        raise ValueError(
            f"GkmExplain attribution supports -t 0/1/2 kernels, not {model.kernel_type!r}; "
            "use mutation_impact() or ism()"
        )
    xp = get_array_module(x)
    B, _, seqlen = x.shape
    l = kernel.l
    d = getattr(kernel, "d", l)
    if is_perturbation:
        table_a, table_b = _build_perturbation_tables(kernel._mismatch_table, l, d)
    else:
        table_a = _build_alpha_table(kernel._mismatch_table, l, d)
        table_b = _build_kappa_table(kernel._mismatch_table, l, d)
    min_matches = _explain_min_matches(table_a, table_b, l)

    S = model.num_support_vectors
    coefs = model.coefficients
    diag_sv = model._sv_diag() if kernel.normalize else None
    sv_fwd, sv_rc = model._sv_packed()
    packed_sv = sv_fwd if sv_rc is None else xp.concatenate([sv_fwd, sv_rc], axis=1)
    if xp is not np:
        packed_sv = xp.ascontiguousarray(packed_sv.T)  # [Wy, S] for coalesced reads
    cs = model.sv_chunk_size if model.sv_chunk_size is not None else min(S, 2000)

    result = xp.zeros((B, 4, seqlen), dtype=np.float64)
    starts = range(0, B, batch_size)
    if verbose:
        from tqdm import tqdm
        starts = tqdm(starts, desc="GkmExplain", total=(B + batch_size - 1) // batch_size)

    for b0 in starts:
        x_b = x[b0:b0 + batch_size]
        packed_x, packed_x_rc = kernel.pack_both(x_b)
        packed_x = xp.ascontiguousarray(packed_x)
        diag_x = kernel.raw_diag_packed(packed_x, packed_x_rc) if kernel.normalize else None
        for s0 in range(0, S, cs):
            s1 = min(s0 + cs, S)
            eff_coef = xp.broadcast_to(coefs[None, s0:s1], (x_b.shape[0], s1 - s0))
            if diag_x is not None:
                norm = xp.sqrt(diag_x[:, None] * diag_sv[None, s0:s1])
                eff_coef = eff_coef / xp.clip(norm, 1e-10, None)
            if xp is np:
                result[b0:b0 + batch_size] += _explain_cpu(
                    packed_x, np.ascontiguousarray(packed_sv[s0:s1]), table_a, table_b,
                    np.ascontiguousarray(eff_coef, dtype=np.float64), l, min_matches,
                    seqlen, is_perturbation)
            else:
                _explain_gpu(packed_x, xp.ascontiguousarray(packed_sv[:, s0:s1]),
                             table_a, table_b, eff_coef, result[b0:b0 + batch_size],
                             s1 - s0, packed_sv.shape[0], l, min_matches, is_perturbation)

    return result.astype(x.dtype)


def mutation_impact(
    model: GkmSVM,
    ref: np.ndarray,
    alt: np.ndarray,
    *,
    batch_size: int | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """GkmExplain mutation impact score (lsgkm ``gkmexplain -m 5``).

    Shrikumar et al. 2019, as used for the dsQTL evaluation. Per support
    vector j, with base (pre-RBF) kernel values normalized by the *reference*
    sequence's norm::

        K_j  = K_raw(ref, sv_j) / (|ref| |sv_j|)
        dK_j = (K_raw(alt, sv_j) - K_raw(ref, sv_j)) / (|ref| |sv_j|)

    Linear kernels: ``impact = sum_j coef_j * dK_j``. RBF kernels distribute
    the kernel value above its K=0 floor proportionally::

        impact = sum_j coef_j * (exp(g(K_j - 1)) - exp(-g)) * dK_j / K_j

    (contribution is 0 when K_j <= 0). This is a linearization, so it differs
    from ``score(alt) - score(ref)``.

    Args:
        model: Trained GkmSVM.
        ref: [B, 4, L] reference sequences.
        alt: [B, 4, L] alternate sequences.
        batch_size: Process inputs in batches of this size, raised as in
            ``GkmSVM.__call__``.
        verbose: Show tqdm progress bar over input batches.

    Returns:
        [B, 1] mutation impact scores.
    """
    from gkmsvm.backend import to_cpu

    kernel = model.kernel
    base = kernel.base
    coef = model.coefficients
    packed_svs = model._sv_packed() is not None

    B = ref.shape[0]
    bs = B if batch_size is None else model._min_batch_size(batch_size, table=False)
    starts = range(0, B, bs)
    if verbose:
        from tqdm import tqdm
        starts = tqdm(starts, desc="Mutation impact", total=(B + bs - 1) // bs)

    parts = []
    for s in starts:
        r = model._match_device(ref[s:s + bs])
        a = model._match_device(alt[s:s + bs])
        xp = get_array_module(r)
        n = r.shape[0]
        both = xp.concatenate([r, a])
        if packed_svs and base.packs(both):
            # One call for both alleles (the masked join builds its SV index once)
            bx, bx_rc = base.pack_both(both)
            k_both = model._sv_raw(bx)
            diag_r = base.raw_diag_packed(bx[:n], None if bx_rc is None else bx_rc[:n])
        else:
            k_both = base._raw_pairwise(both, model.support_sequences)
            diag_r = base._raw_diagonal(r)
        gamma = kernel.gamma if base is not kernel else np.nan  # RBF
        if xp is np:
            parts.append(_impact_cpu(
                np.ascontiguousarray(k_both, dtype=np.float64),
                diag_r.astype(np.float64) if base.normalize else np.ones(n),
                model._sv_diag().astype(np.float64) if base.normalize else np.ones(1),
                base.normalize, coef.astype(np.float64), float(gamma))[:, None])
            continue
        k_ref = k_both[:n]
        dk = k_both[n:] - k_ref
        if base.normalize:
            norm = xp.clip(xp.sqrt(diag_r[:, None] * model._sv_diag()[None, :]), 1e-10, None)
            k_ref = k_ref / norm
            dk = dk / norm
        if not np.isnan(gamma):
            pos = k_ref > 0
            safe_k = xp.where(pos, k_ref, 1.0)
            scale = (kernel.finish(k_ref) - np.exp(-gamma)) / safe_k
            dk = xp.where(pos, dk * scale, 0.0)
        parts.append(to_cpu((dk * coef).sum(axis=1, keepdims=True)))

    return np.concatenate(parts, axis=0)
