"""CUDA kernels: the fused per-SV kernel (float64 atomicAdd) and the l-mer path
(16 lanes per query window)."""

from __future__ import annotations

from functools import cache

import numpy as np

from gkmsvm.explain.common import _launch_slices

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
    const int l, const int min_matches, const int seqlen, const int perturbation,
    const double* __restrict__ wx, const int wx_stride, const double* __restrict__ wy,
    const int wy_rs, const int wy_cs, const int weighted
) {
    // Query b's window i weighs wx[b * wx_stride + i] (-t 4/5; wx_stride 0
    // when the rows share one [Wx] row); SV s's window j weighs
    // wy[j * wy_rs + s * wy_cs] (wy [Wy, S], or one [Wy] row shared: wy_cs 0).
    // PAD windows (past a shorter SV's) are skipped
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
        if (bx_val == 0xFFFFFFFFu) continue;
        for (int j = 0; j < Wy; j++) {
            unsigned int by_val = by_packed_t[j * S + s];
            if (by_val == 0xFFFFFFFFu) continue;
            unsigned int xv = bx_val ^ by_val;
            int mm = __popc(((xv >> 1) | xv) & 0x55555555u);
            if (l - mm < min_matches) continue;
            const double ew = weighted
                ? ec * wx[(long long)b * wx_stride + i] * wy[(long long)j * wy_rs + s * wy_cs]
                : ec;
            double av = ew * s_a[mm];
            double bv = ew * s_b[mm];
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


@cache
def _cupy_explain_kernel():
    import cupy as cp

    return cp.RawKernel(
        _FUSED_EXPLAIN_CUDA, "fused_explain", options=("--use_fast_math",)
    )


def _explain_gpu(
    packed_x,
    packed_y_t,
    table_a,
    table_b,
    eff_coef,
    result,
    l,
    min_matches,
    perturbation,
    wx,
    wy,
):
    """Launch the CUDA kernel, accumulating into ``result`` [B, 4, seqlen] from
    packed_y_t [Wy, S_c]; wx [B or 1, Wx], wy [S_c or 1, Wy]: window weights
    ([1, 0] when unweighted)."""
    import cupy as cp

    B, Wx = packed_x.shape
    Wy, S_c = packed_y_t.shape
    total = B * S_c
    block = 256
    shared = 2 * (l + 2) * 8 + Wx * 4
    weighted = wx.shape[1] > 0
    wx_stride, wy_rs, wy_cs = 0, 1, 0
    if not weighted:
        wx = wy = np.zeros(1)
    else:
        wx_stride = Wx if len(wx) > 1 else 0
        if len(wy) > 1:  # per SV: [Wy, S_c], read like packed_y_t
            wy, wy_rs, wy_cs = np.ascontiguousarray(wy.T), S_c, 1
        else:
            wy = wy[0]
    _cupy_explain_kernel()(
        ((total + block - 1) // block,),
        (block,),
        (
            packed_x,
            packed_y_t,
            cp.asarray(table_a, dtype=cp.float64),
            cp.asarray(table_b, dtype=cp.float64),
            cp.ascontiguousarray(eff_coef, dtype=cp.float64),
            result,
            np.int32(total),
            np.int32(S_c),
            np.int32(Wx),
            np.int32(Wy),
            np.int32(l),
            np.int32(min_matches),
            np.int32(result.shape[2]),
            np.int32(perturbation),
            cp.asarray(wx, dtype=cp.float64),
            np.int32(wx_stride),
            cp.asarray(wy, dtype=cp.float64),
            np.int32(wy_rs),
            np.int32(wy_cs),
            np.int32(weighted),
        ),
        shared_mem=shared,
    )


# The l-mer path (_explain_lmer_cpu / _explain_index_cpu) on CUDA. A group of
# GROUP lanes takes one query window and a slice of the masks (grid.y); lane k
# owns position k. Each round the lanes look up GROUP neighbors' weights at
# once (V[v], or with use_index the sum of w[b, ids[e]] over v's entries) and
# shuffle them round the group. A lane keeps four register sums, indexed by
# the mask's 2-bit delta d at its position (the neighbor's base is u_k ^ d),
# picked with selects so each (neighbor, position) costs one float64 add: a
# dynamically indexed [4, l] array spills to local memory. Perturbation adds a
# match's weight to all four bases (q) and subtracts it from u_k. Lanes flush
# with float64 atomicAdd (CC >= 6.0), 4 per window position. With window
# weights (-t 4/5), wx scales each query window and vals the index's entries
# (V already holds the SV windows' weights).
_EXPLAIN_LMER_GROUP = 16
_EXPLAIN_LMER_BLOCK = 128
_EXPLAIN_LMER_CUDA = r"""
#define GROUP 16
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
    double* __restrict__ out, const double* __restrict__ wx, const int wx_stride,
    const unsigned char* __restrict__ vals, const int weighted
) {
    const int b = blockIdx.z;
    const int lane = threadIdx.x % GROUP;
    const int i = blockIdx.x * (blockDim.x / GROUP) + threadIdx.x / GROUP;
    // Groups past the last window still take part in the shuffles
    const bool valid = i < W;
    const unsigned int u = valid ? bx[(long long)b * W + i] : 0u;
    const double* wb = w + (long long)b * S;
    const int shift = 2 * lane;
    const unsigned int uk = (u >> shift) & 3u;

    double r0 = 0.0, r1 = 0.0, r2 = 0.0, r3 = 0.0, q = 0.0;
    const int t_begin = blockIdx.y * mask_chunk;
    const int t_end = min(n_masks, t_begin + mask_chunk);
    for (int t0 = t_begin; t0 < t_end; t0 += GROUP) {
        // Lane j looks up neighbor t0 + j
        const int t = t0 + lane;
        unsigned int my_mask = 0u;
        double my_a = 0.0, my_b = 0.0;
        if (valid && t < t_end) {
            my_mask = masks[t];
            const unsigned int v = u ^ my_mask;
            if ((long long)v < n_lmers) {  // else a padding window
                double wv;
                if (use_index && weighted) {
                    wv = 0.0;
                    for (long long e = offsets[v]; e < offsets[v + 1]; e++)
                        wv += wb[ids[e]] * vals[e];
                } else if (use_index) {
                    wv = 0.0;
                    for (long long e = offsets[v]; e < offsets[v + 1]; e++) wv += wb[ids[e]];
                } else {
                    wv = V[v];
                }
                if (weighted) wv *= wx[(long long)b * wx_stride + i];
                my_a = wv * table_a[levels[t]];
                my_b = wv * table_b[levels[t]];
            }
        }
        const int n = min(GROUP, t_end - t0);
        for (int j = 0; j < n; j++) {
            const unsigned int mask = __shfl_sync(0xffffffffu, my_mask, j, GROUP);
            const double av = __shfl_sync(0xffffffffu, my_a, j, GROUP);
            const double bv = __shfl_sync(0xffffffffu, my_b, j, GROUP);
            if (av == 0.0 && bv == 0.0) continue;
            const unsigned int d = (mask >> shift) & 3u;
            double val;
            if (d != 0u) {
                val = bv;
            } else if (perturbation) {
                q += av;
                val = -av;
            } else {
                val = av;
            }
            double cur = d == 0u ? r0 : d == 1u ? r1 : d == 2u ? r2 : r3;
            cur += val;
            r0 = d == 0u ? cur : r0;
            r1 = d == 1u ? cur : r1;
            r2 = d == 2u ? cur : r2;
            r3 = d == 3u ? cur : r3;
        }
    }
    if (!valid || lane >= l) return;
    double* row = out + (long long)b * 4 * seqlen + i + lane;
    const double r[4] = {r0, r1, r2, r3};
    for (int dd = 0; dd < 4; dd++) {
        const double val = r[dd] + q;
        if (val != 0.0) atomicAdd(&row[(long long)(uk ^ dd) * seqlen], val);
    }
}
"""


@cache
def _cupy_explain_lmer_kernel():
    import cupy as cp

    return cp.RawKernel(_EXPLAIN_LMER_CUDA, "explain_lmer")


def _explain_lmer_gpu(
    packed_x,
    inv_dx,
    V,
    offsets,
    ids,
    w,
    masks,
    levels,
    table_a,
    table_b,
    l,
    seqlen,
    perturbation,
    wx,
    vals,
):
    """[B, 4, seqlen] on CUDA: _explain_lmer_cpu when V is given, else
    _explain_index_cpu with the CSR (offsets, ids) and SV weights w [B, S].
    wx: query window weights [B or 1, W] ([1, 0] when unweighted), vals: the
    entries' (None when unweighted)."""
    import cupy as cp

    packed_x = cp.ascontiguousarray(packed_x, dtype=cp.uint32)
    B, W = packed_x.shape
    n_masks = len(masks)
    per_block = _EXPLAIN_LMER_BLOCK // _EXPLAIN_LMER_GROUP
    blocks = -(-W // per_block)
    n_slices, chunk = _launch_slices(n_masks, 256, blocks * _EXPLAIN_LMER_BLOCK, B)
    use_index = V is None
    dummy_f = cp.zeros(1, dtype=cp.float64)
    weighted = wx.shape[1] > 0
    wx_stride = W if weighted and len(wx) > 1 else 0
    wx = cp.asarray(wx, dtype=cp.float64) if weighted else dummy_f
    vals = vals if vals is not None else cp.zeros(1, dtype=cp.uint8)
    out = cp.zeros((B, 4, seqlen), dtype=cp.float64)
    for b0 in range(0, B, 65535):  # grid.z limit
        nb = min(65535, B - b0)
        _cupy_explain_lmer_kernel()(
            (blocks, n_slices, nb),
            (_EXPLAIN_LMER_BLOCK,),
            (
                packed_x[b0 : b0 + nb],
                np.int32(W),
                np.int32(seqlen),
                np.int32(l),
                masks,
                levels,
                np.int32(n_masks),
                np.int32(chunk),
                dummy_f if use_index else V,
                offsets if use_index else cp.zeros(1, dtype=cp.int64),
                ids if use_index else cp.zeros(1, dtype=cp.int32),
                cp.ascontiguousarray(w[b0 : b0 + nb]) if use_index else dummy_f,
                np.int32(w.shape[1] if use_index else 0),
                np.int32(use_index),
                np.int64(4**l),
                table_a,
                table_b,
                np.int32(perturbation),
                out[b0 : b0 + nb],
                wx[b0:] if wx_stride else wx,
                np.int32(wx_stride),
                vals,
                np.int32(weighted),
            ),
        )
    out *= inv_dx[:, None, None]
    return out
