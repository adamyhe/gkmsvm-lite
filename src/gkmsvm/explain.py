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


# The l-mer path (_explain_lmer_cpu / _explain_index_cpu) on CUDA. Each thread
# takes one query window and a slice of the masks (grid.y), summing its
# contributions in a per-thread [4, l] array, so no atomics run per neighbor.
# Perturbation adds a match's weight to all four bases and subtracts it from
# the matching one. A block's TILE windows then add their arrays into a shared
# [4, TILE + l - 1] tile one offset k at a time (thread j writes position
# j + k, so writes don't collide) and flush it with float64 atomicAdd
# (CC >= 6.0). use_index: V[v] is the sum of w[b, ids[e]] over v's entries.
_EXPLAIN_LMER_TILE = 128
_EXPLAIN_LMER_CUDA = r"""
#define TILE 128
#define MAX_L 12
extern "C" __global__
void explain_lmer(
    const unsigned int* __restrict__ bx, const int W, const int seqlen, const int l,
    const unsigned int* __restrict__ masks, const int* __restrict__ levels,
    const int n_masks, const int mask_chunk,
    const double* __restrict__ V,
    const long long* __restrict__ offsets, const int* __restrict__ ids,
    const double* __restrict__ w, const int S, const int use_index,
    const long long n_lmers, const double* __restrict__ table_a,
    const double* __restrict__ table_b, const int perturbation,
    double* __restrict__ out
) {
    __shared__ double s_acc[4][TILE + MAX_L - 1];
    const int b = blockIdx.z;
    const int i0 = blockIdx.x * TILE;
    const int i = i0 + threadIdx.x;
    for (int j = threadIdx.x; j < 4 * (TILE + MAX_L - 1); j += TILE)
        (&s_acc[0][0])[j] = 0.0;

    double p[4][MAX_L];
    double q[MAX_L];
    #pragma unroll
    for (int k = 0; k < MAX_L; k++) {
        q[k] = 0.0;
        p[0][k] = p[1][k] = p[2][k] = p[3][k] = 0.0;
    }
    if (i < W) {
        const unsigned int u = bx[(long long)b * W + i];
        const double* wb = w + (long long)b * S;
        const int t1 = min(n_masks, (blockIdx.y + 1) * mask_chunk);
        for (int t = blockIdx.y * mask_chunk; t < t1; t++) {
            const unsigned int mask = masks[t];
            const unsigned int v = u ^ mask;
            if ((long long)v >= n_lmers) continue;  // padding window
            double wv;
            if (use_index) {
                wv = 0.0;
                for (long long e = offsets[v]; e < offsets[v + 1]; e++) wv += wb[ids[e]];
            } else {
                wv = V[v];
            }
            if (wv == 0.0) continue;
            const double av = wv * table_a[levels[t]];
            const double bv = wv * table_b[levels[t]];
            #pragma unroll
            for (int k = 0; k < MAX_L; k++) {
                if (k >= l) break;
                const int vk = (v >> (2 * k)) & 3;
                const bool same = ((mask >> (2 * k)) & 3) == 0;
                if (!perturbation) {
                    p[vk][k] += same ? av : bv;
                } else if (same) {
                    q[k] += av;
                    p[vk][k] -= av;
                } else {
                    p[vk][k] += bv;
                }
            }
        }
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < MAX_L; k++) {
        if (k < l) {
            for (int h = 0; h < 4; h++) s_acc[h][threadIdx.x + k] += p[h][k] + q[k];
        }
        __syncthreads();
    }
    double* row = out + (long long)b * 4 * seqlen;
    for (int j = threadIdx.x; j < TILE + l - 1; j += TILE) {
        const int pos = i0 + j;
        if (pos >= seqlen) break;
        for (int h = 0; h < 4; h++)
            if (s_acc[h][j] != 0.0) atomicAdd(&row[(long long)h * seqlen + pos], s_acc[h][j]);
    }
}
"""

_cupy_explain_lmer_kernel = None
# Threads to aim for per launch; mask slices (grid.y) make up for few windows
_EXPLAIN_LMER_THREADS = 1 << 18


def _explain_lmer_gpu(packed_x, inv_dx, V, offsets, ids, w, masks, levels, table_a,
                      table_b, l, seqlen, perturbation):
    """[B, 4, seqlen] on CUDA: _explain_lmer_cpu when V is given, else
    _explain_index_cpu with the CSR (offsets, ids) and SV weights w [B, S]."""
    import cupy as cp

    global _cupy_explain_lmer_kernel
    if _cupy_explain_lmer_kernel is None:
        _cupy_explain_lmer_kernel = cp.RawKernel(_EXPLAIN_LMER_CUDA, "explain_lmer")
    packed_x = cp.ascontiguousarray(packed_x, dtype=cp.uint32)
    B, W = packed_x.shape
    n_masks = len(masks)
    tiles = -(-W // _EXPLAIN_LMER_TILE)
    n_slices = max(1, min(-(-n_masks // 256),
                          -(-_EXPLAIN_LMER_THREADS // (B * tiles * _EXPLAIN_LMER_TILE))))
    chunk = -(-n_masks // n_slices)
    n_slices = -(-n_masks // chunk)
    use_index = V is None
    dummy_f = cp.zeros(1, dtype=cp.float64)
    out = cp.zeros((B, 4, seqlen), dtype=cp.float64)
    for b0 in range(0, B, 65535):  # grid.z limit
        nb = min(65535, B - b0)
        _cupy_explain_lmer_kernel(
            (tiles, n_slices, nb), (_EXPLAIN_LMER_TILE,),
            (packed_x[b0:b0 + nb], np.int32(W), np.int32(seqlen), np.int32(l),
             masks, levels, np.int32(n_masks), np.int32(chunk),
             dummy_f if use_index else V,
             offsets if use_index else cp.zeros(1, dtype=cp.int64),
             ids if use_index else cp.zeros(1, dtype=cp.int32),
             cp.ascontiguousarray(w[b0:b0 + nb]) if use_index else dummy_f,
             np.int32(w.shape[1] if use_index else 0), np.int32(use_index),
             np.int64(4**l), table_a, table_b, np.int32(perturbation), out[b0:b0 + nb]))
    out *= inv_dx[:, None, None]
    return out


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


@njit(inline="always")
def _distribute(acc, i, mask, v, wv, a, b, l, perturbation):
    """Add one (query window i, l-mer v = u ^ mask) pair to acc [4, seqlen], as
    _explain_cpu does for an SV window equal to v with effective coefficient wv;
    a, b are table_a / table_b at the pair's mismatch count."""
    av = wv * a
    bv = wv * b
    for k in range(l):
        v_k = int((v >> np.uint32(2 * k)) & np.uint32(3))
        same = ((mask >> np.uint32(2 * k)) & np.uint32(3)) == 0
        pos = k + i
        if not perturbation:
            acc[v_k, pos] += av if same else bv
        elif same:
            for h in range(4):
                if h != v_k:
                    acc[h, pos] += av
        else:
            acc[v_k, pos] += bv


@njit(parallel=True, cache=True, fastmath=True)
def _explain_lmer_cpu(packed_x, inv_dx, V, masks, levels, table_a, table_b, l, seqlen,
                      perturbation, n_split):
    """[B, 4, seqlen] attributions from the SVs' l-mer weights V (see _explain_lmer).

    Each query window u visits every l-mer v = u ^ mask within the mismatch
    cutoff and adds V[v] times the alpha/kappa (or perturbation) weight of
    their mismatch count to each of its positions, as _explain_cpu does for
    every SV window equal to v. Rows are split into n_split window chunks.
    """
    B, W = packed_x.shape
    n_lmers = V.shape[0]
    part = np.zeros((B * n_split, 4, seqlen))
    step = (W + n_split - 1) // n_split
    for task in prange(B * n_split):
        b, c = task // n_split, task % n_split
        acc = part[task]
        for i in range(c * step, min(W, (c + 1) * step)):
            u = packed_x[b, i]
            for t in range(masks.shape[0]):
                v = u ^ masks[t]
                if np.int64(v) >= n_lmers or V[v] == 0.0:
                    continue
                _distribute(acc, i, masks[t], v, V[v], table_a[levels[t]],
                            table_b[levels[t]], l, perturbation)
    out = part.reshape(B, n_split, 4, seqlen).sum(axis=1)
    for b in range(B):
        out[b] *= inv_dx[b]
    return out


@njit(parallel=True, cache=True, fastmath=True)
def _explain_index_cpu(packed_x, inv_dx, offsets, ids, w, masks, levels, table_a, table_b,
                       l, seqlen, perturbation, n_split):
    """As _explain_lmer_cpu with per-query SV weights w [B, S]: V[v] for row b is
    the sum of w[b, s] over the index's entries for v (RBF models)."""
    B, W = packed_x.shape
    n_lmers = offsets.shape[0] - 1
    part = np.zeros((B * n_split, 4, seqlen))
    step = (W + n_split - 1) // n_split
    for task in prange(B * n_split):
        b, c = task // n_split, task % n_split
        acc = part[task]
        wb = w[b]
        for i in range(c * step, min(W, (c + 1) * step)):
            u = packed_x[b, i]
            for t in range(masks.shape[0]):
                v = u ^ masks[t]
                if np.int64(v) >= n_lmers:
                    continue
                wv = 0.0
                for e in range(offsets[v], offsets[v + 1]):
                    wv += wb[ids[e]]
                if wv == 0.0:
                    continue
                _distribute(acc, i, masks[t], v, wv, table_a[levels[t]],
                            table_b[levels[t]], l, perturbation)
    out = part.reshape(B, n_split, 4, seqlen).sum(axis=1)
    for b in range(B):
        out[b] *= inv_dx[b]
    return out


def _rbf_scale(k_norm, gamma, xp=np):
    """LS-GKM's RBF attribution factor per SV: (exp(gamma (K - 1)) - exp(-gamma)) / K,
    0 where K <= 0, for normalized base kernel values K."""
    pos = k_norm > 0
    safe = xp.where(pos, k_norm, 1.0)
    return xp.where(pos, (xp.exp(gamma * (safe - 1)) - np.exp(-gamma)) / safe, 0.0)


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

    if is_mlx(x):
        # Metal can't accumulate in float64, which the completeness axiom needs
        warnings.warn(
            "GkmExplain does not have an MLX kernel — falling back to CPU. "
            "This will be slower than MLX inference.",
            stacklevel=2,
        )
        from gkmsvm.backend import to_cpu, to_mlx

        # A CPU copy of the model, kept with the model's caches so its packed
        # SVs, norms and l-mer weights are built once
        cpu_model = model._sv_cache.get("cpu_model")
        if cpu_model is None:
            cpu_model = GkmSVM(
                to_cpu(model.support_sequences),
                to_cpu(model.coefficients),
                model.bias,
                model.kernel_type,
                model._kernel_params,
                sv_chunk_size=model.sv_chunk_size,
            )
            cpu_model.use_lmer_index = model.use_lmer_index
            cpu_model.use_lmer_table = model.use_lmer_table
            model._sv_cache["cpu_model"] = cpu_model
        return to_mlx(gkmexplain(cpu_model, to_cpu(x), mode=resolved,
                                 batch_size=batch_size, verbose=verbose).astype(np.float32))

    if resolved == "importance":
        hyp = gkmexplain(
            model, x, mode="hypothetical",
            batch_size=batch_size, verbose=verbose,
        )
        return hyp * model._match_device(x)

    is_perturbation = resolved == "perturbation"

    x = model._match_device(x)


    kernel = model.kernel
    base = kernel.base
    if getattr(base, "_mismatch_table", None) is None:
        raise ValueError(
            f"GkmExplain attribution supports -t 0/1/2/3 kernels, not {model.kernel_type!r}; "
            "use mutation_impact() or ism()"
        )
    # RBF (-t 3): each SV's contributions scale by _rbf_scale of its kernel value
    gamma = kernel.gamma if base is not kernel else None
    xp = get_array_module(x)
    B, _, seqlen = x.shape
    l = base.l
    d = getattr(base, "d", l)
    if is_perturbation:
        table_a, table_b = _build_perturbation_tables(base._mismatch_table, l, d)
    else:
        table_a = _build_alpha_table(base._mismatch_table, l, d)
        table_b = _build_kappa_table(base._mismatch_table, l, d)
    min_matches = _explain_min_matches(table_a, table_b, l)
    lmer = _explain_lmer(model, x, table_a, table_b, l - min_matches, is_perturbation,
                         gamma, batch_size, verbose)
    if lmer is not None:
        return lmer

    S = model.num_support_vectors
    coefs = model.coefficients
    diag_sv = model._sv_diag() if base.normalize else None
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
        packed_x, packed_x_rc = base.pack_both(x_b)
        packed_x = xp.ascontiguousarray(packed_x)
        diag_x = base.raw_diag_packed(packed_x, packed_x_rc) if base.normalize else None
        if gamma is not None:
            k_norm = model._sv_raw(packed_x) / xp.clip(
                xp.sqrt(diag_x[:, None] * diag_sv[None, :]), 1e-10, None)
            rbf = _rbf_scale(k_norm, gamma, xp)
        for s0 in range(0, S, cs):
            s1 = min(s0 + cs, S)
            eff_coef = xp.broadcast_to(coefs[None, s0:s1], (x_b.shape[0], s1 - s0))
            if diag_x is not None:
                norm = xp.sqrt(diag_x[:, None] * diag_sv[None, s0:s1])
                eff_coef = eff_coef / xp.clip(norm, 1e-10, None)
            if gamma is not None:
                eff_coef = eff_coef * rbf[:, s0:s1]
            if xp is np:
                result[b0:b0 + batch_size] += _explain_cpu(
                    packed_x, np.ascontiguousarray(packed_sv[s0:s1]), table_a, table_b,
                    np.ascontiguousarray(eff_coef, dtype=np.float64), l, min_matches,
                    seqlen, is_perturbation)
            else:
                _explain_gpu(packed_x, xp.ascontiguousarray(packed_sv[:, s0:s1]),
                             table_a, table_b, xp.ascontiguousarray(eff_coef, dtype=np.float64),
                             result[b0:b0 + batch_size], s1 - s0, packed_sv.shape[0], l,
                             min_matches, is_perturbation)

    return result


def _explain_lmer(model, x, table_a, table_b, dmax, perturbation, gamma, batch_size,
                  verbose):
    """GkmExplain through the SVs' l-mer weights, or None when it doesn't apply.

    The attribution is linear in the SVs: with normalization each SV window
    v contributes coef_s / sqrt(K(sv, sv) K(x, x)) times a weight that
    depends only on the query window u and v. Summing over SVs first gives
    V[v] = sum_s coef_s / sqrt(K(sv, sv)) * (occurrences of v in sv_s and its
    RC), and a query window then visits only the l-mers within dmax
    mismatches. A call costs nothing per SV; V (4^l float64) is cached.

    RBF models (gamma given) scale each SV by _rbf_scale of its kernel value
    with the query, so V depends on the query: each neighbor sums the
    per-query SV weights over the l-mer index's entries instead (about one
    more index query per call). Runs on CPU; results go back to x's device.
    """
    import numba

    from gkmsvm.backend import to_cpu
    from gkmsvm.kernels import lmer_index

    base = model.kernel.base
    l = base.l
    if (l > lmer_index.MAX_L or dmax < 0
            or lmer_index.neighbor_count(l, dmax) > lmer_index.MAX_NEIGHBORS):
        return None
    coef = to_cpu(model.coefficients).astype(np.float64)
    inv_ds = (1 / np.sqrt(np.clip(to_cpu(model._sv_diag()).astype(np.float64), 1e-20, None))
              if base.normalize else np.ones(len(coef)))
    index = V = None
    if gamma is not None:
        index = model._sv_index()  # its arrays live on CPU whatever the model's device
        if index is None:
            return None
    else:
        c = model._sv_cache
        if "explain_V" not in c:
            V = np.zeros(4**l)
            for packed in model._sv_packed():
                if packed is not None:
                    packed = to_cpu(packed)
                    V += np.bincount(packed.ravel().astype(np.int64),
                                     weights=np.repeat(coef * inv_ds, packed.shape[1]),
                                     minlength=4**l)
            c["explain_V"] = V
        V = c["explain_V"]
    nonzero = np.maximum(np.abs(table_a), np.abs(table_b))
    masks, levels = lmer_index.neighbor_masks(l, nonzero, dmax)

    xp = get_array_module(x)
    if xp is not np and not is_mlx(x):
        return _explain_lmer_cuda(model, x, V, index, coef, inv_ds, masks, levels, table_a,
                                  table_b, gamma, batch_size, verbose, perturbation)
    x_cpu = to_cpu(x)
    B, _, seqlen = x_cpu.shape
    starts = range(0, B, batch_size)
    if verbose:
        from tqdm import tqdm
        starts = tqdm(starts, desc="GkmExplain", total=(B + batch_size - 1) // batch_size)
    parts = []
    for b0 in starts:
        x_b = x_cpu[b0:b0 + batch_size]
        packed_x, packed_x_rc = base.pack_both(x_b)
        packed_x = np.ascontiguousarray(packed_x)
        inv_dx = (1 / np.sqrt(np.clip(base.raw_diag_packed(packed_x, packed_x_rc),
                                      1e-20, None)) if base.normalize
                  else np.ones(x_b.shape[0]))
        W = packed_x.shape[1]
        n_split = max(1, min(W, -(-numba.get_num_threads() // x_b.shape[0])))
        if index is None:
            parts.append(_explain_lmer_cpu(packed_x, inv_dx, V, masks, levels, table_a,
                                           table_b, l, seqlen, perturbation, n_split))
            continue
        k_norm = index.pairwise(packed_x) * inv_dx[:, None] * inv_ds[None, :]
        w = coef * inv_ds * _rbf_scale(k_norm, gamma)
        parts.append(_explain_index_cpu(packed_x, inv_dx, index.offsets, index.ids,
                                        np.ascontiguousarray(w), masks, levels, table_a,
                                        table_b, l, seqlen, perturbation, n_split))
    out = np.concatenate(parts)
    return out if xp is np else xp.asarray(out)


def _explain_lmer_cuda(model, x, V, index, coef, inv_ds, masks, levels, table_a, table_b,
                       gamma, batch_size, verbose, perturbation):
    """_explain_lmer's loop on CUDA (_explain_lmer_gpu); V, the masks and the
    l-mer index's arrays are uploaded once and kept in the model's caches."""
    import cupy as cp

    base = model.kernel.base
    l = base.l
    c = model._sv_cache
    if index is None:
        if "explain_V_cuda" not in c:
            c["explain_V_cuda"] = cp.asarray(V)
        V = c["explain_V_cuda"]
        offsets = ids = None
    else:
        offsets, ids, _, _ = index._arrays("cuda")
        coef_d = cp.asarray(coef)
        inv_ds_d = cp.asarray(inv_ds)
    masks, levels = cp.asarray(masks), cp.asarray(levels)
    table_a = cp.asarray(table_a, dtype=cp.float64)
    table_b = cp.asarray(table_b, dtype=cp.float64)

    B, _, seqlen = x.shape
    starts = range(0, B, batch_size)
    if verbose:
        from tqdm import tqdm
        starts = tqdm(starts, desc="GkmExplain", total=(B + batch_size - 1) // batch_size)
    out = cp.empty((B, 4, seqlen), dtype=cp.float64)
    for b0 in starts:
        x_b = x[b0:b0 + batch_size]
        packed_x, packed_x_rc = base.pack_both(x_b)
        packed_x = cp.ascontiguousarray(packed_x)
        inv_dx = (1 / cp.sqrt(cp.clip(base.raw_diag_packed(packed_x, packed_x_rc),
                                      1e-20, None)) if base.normalize
                  else cp.ones(x_b.shape[0]))
        w = None
        if index is not None:
            k_norm = index.pairwise(packed_x) * inv_dx[:, None] * inv_ds_d[None, :]
            w = coef_d * inv_ds_d * _rbf_scale(k_norm, gamma, cp)
        out[b0:b0 + batch_size] = _explain_lmer_gpu(
            packed_x, inv_dx, V, offsets, ids, w, masks, levels, table_a, table_b, l,
            seqlen, perturbation)
    return out


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
