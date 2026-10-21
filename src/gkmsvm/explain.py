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
from gkmsvm.kernels.direct import _packed_mismatches
from gkmsvm.kernels.packed import PAD_WINDOW

_PAD = PAD_WINDOW
from gkmsvm.svm import (
    GkmSVM,
    _allele_arrays,
    _joint_lengths,
    _nan_rows,
    _per_position,
    _short_rows,
    as_batch,
)

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

_cupy_explain_kernel = None


def _explain_gpu(
    packed_x,
    packed_y_t,
    table_a,
    table_b,
    eff_coef,
    result,
    S_c,
    Wy,
    l,
    min_matches,
    perturbation,
    wx,
    wy,
):
    """Launch the CUDA kernel, accumulating into ``result`` [B, 4, seqlen]; wx
    [B or 1, Wx], wy [S_c or 1, Wy]: window weights ([1, 0] when unweighted)."""
    import cupy as cp

    global _cupy_explain_kernel
    if _cupy_explain_kernel is None:
        _cupy_explain_kernel = cp.RawKernel(
            _FUSED_EXPLAIN_CUDA, "fused_explain", options=("--use_fast_math",)
        )
    B, Wx = packed_x.shape
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
    _cupy_explain_kernel(
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

_cupy_explain_lmer_kernel = None
# Past lmer_index.MAX_NEIGHBORS, the l-mer path is taken when the SVs have at
# least this many windows per neighbor: -t 0 l=11 k=7 without d (143,980 neighbors, 300 bp
# SVs) breaks even near 1,000 SVs on CPU (Apple M4, Xeon) and 2,000 on a Titan
# Xp, and is 3.6-4.7x faster at 10,000 (benchmarks/dev/explain_lmer_gpu.py)
_EXPLAIN_SV_WINDOWS_PER_NEIGHBOR = 6
# Threads to aim for per launch; mask slices (grid.y) make up for few windows
_EXPLAIN_LMER_THREADS = 1 << 18


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
    wx: query window weights [B or 1, W] (row_weights), vals: the entries'
    (both None when unweighted)."""
    import cupy as cp

    global _cupy_explain_lmer_kernel
    if _cupy_explain_lmer_kernel is None:
        _cupy_explain_lmer_kernel = cp.RawKernel(_EXPLAIN_LMER_CUDA, "explain_lmer")
    packed_x = cp.ascontiguousarray(packed_x, dtype=cp.uint32)
    B, W = packed_x.shape
    n_masks = len(masks)
    per_block = _EXPLAIN_LMER_BLOCK // _EXPLAIN_LMER_GROUP
    blocks = -(-W // per_block)
    n_slices = max(
        1,
        min(
            -(-n_masks // 256),
            -(-_EXPLAIN_LMER_THREADS // (B * blocks * _EXPLAIN_LMER_BLOCK)),
        ),
    )
    chunk = -(-n_masks // n_slices)
    n_slices = -(-n_masks // chunk)
    use_index = V is None
    dummy_f = cp.zeros(1, dtype=cp.float64)
    weighted = wx is not None
    wx_stride = W if weighted and len(wx) > 1 else 0
    wx = cp.asarray(wx, dtype=cp.float64) if weighted else dummy_f
    vals = vals if vals is not None else cp.zeros(1, dtype=cp.uint8)
    out = cp.zeros((B, 4, seqlen), dtype=cp.float64)
    for b0 in range(0, B, 65535):  # grid.z limit
        nb = min(65535, B - b0)
        _cupy_explain_lmer_kernel(
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


# The 16 (delta mask, a, b) pairs a 16-lane group holds, passed round with
# simd_shuffle: lane k adds each to its four sums by the pair's 2-bit delta at
# position k (perturbation: a delta-0 pair adds to the other three bases, by
# way of q), then writes them to part [slices, B, W, 4, 16] (_fold_windows)
_EXPLAIN_METAL_ROUND = """        uint n = min(16u, t_end - t0);
        for (uint j = 0; j < n; j++) {
            uint mask = simd_shuffle(my_mask, (ushort)(src + j));
            float av = simd_shuffle(my_a, (ushort)(src + j));
            float bv = simd_shuffle(my_b, (ushort)(src + j));
            if (av == 0.0f && bv == 0.0f) continue;
            uint d = (mask >> shift) & 3u;
            float val;
            if (d != 0u) {
                val = bv;
            } else if (perturbation) {
                q += av;
                val = -av;
            } else {
                val = av;
            }
            r0 += d == 0u ? val : 0.0f;
            r1 += d == 1u ? val : 0.0f;
            r2 += d == 2u ? val : 0.0f;
            r3 += d == 3u ? val : 0.0f;
        }
    }
    if (!valid || lane >= l) return;
    uint o = (((slice * B + b) * W + i) * 4) * 16 + lane;
    part[o] = r0 + q;
    part[o + 16] = r1 + q;
    part[o + 32] = r2 + q;
    part[o + 48] = r3 + q;
"""

# Metal (MLX) version of explain_lmer: the same 16-lane groups, one per query
# window, lane k owning position k and the neighbors passed round with
# simd_shuffle (two groups per 32-wide SIMD group). Metal has no float64, so
# nothing is summed across windows on the GPU: each window writes its four
# float32 sums per position (r[d] + q) to part [slices, B, W, 4, 16], and the
# CPU folds the overlapping windows in float64 (_fold_windows). No atomics.
_EXPLAIN_LMER_METAL = (
    """
    uint tid = thread_position_in_threadgroup.x;
    uint tg = threads_per_threadgroup.x;
    uint lane = tid % 16;
    uint i = threadgroup_position_in_grid.x * (tg / 16) + tid / 16;
    uint slice = threadgroup_position_in_grid.y;
    uint b = threadgroup_position_in_grid.z;
    uint W = (uint)params[0], n_masks = (uint)params[1], chunk = (uint)params[2];
    uint use_index = (uint)params[3], S = (uint)params[4], n_lmers = (uint)params[5];
    uint perturbation = (uint)params[6], weighted = (uint)params[7];
    uint wx_stride = (uint)params[8], B = (uint)params[9], l = (uint)params[10];
    // Groups past the last window still take part in the shuffles
    bool valid = i < W;
    uint u = valid ? bx[b * W + i] : 0u;
    uint shift = 2 * lane;
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f, q = 0.0f;
    uint t_begin = slice * chunk;
    uint t_end = min(n_masks, t_begin + chunk);
    ushort src = (ushort)((tid % 32) & ~15u);  // this group's first lane
    for (uint t0 = t_begin; t0 < t_end; t0 += 16) {
        uint t = t0 + lane;
        uint my_mask = 0u;
        float my_a = 0.0f, my_b = 0.0f;
        if (valid && t < t_end) {
            my_mask = masks[t];
            uint v = u ^ my_mask;
            if (v < n_lmers) {  // else a padding window
                float wv;
                if (use_index) {
                    // compensated sum: the SV weights are signed
                    float hi = 0.0f, lo = 0.0f;
                    for (int e = offsets[v]; e < offsets[v + 1]; e++) {
                        float y = w[b * S + ids[e]] * (weighted ? (float)vals[e] : 1.0f);
                        float t2 = hi + y;
                        lo += (fabs(hi) >= fabs(y)) ? (hi - t2) + y : (y - t2) + hi;
                        hi = t2;
                    }
                    wv = hi + lo;
                } else {
                    wv = V[v];
                }
                if (weighted) wv *= wx[b * wx_stride + i];
                my_a = wv * table_a[levels[t]];
                my_b = wv * table_b[levels[t]];
            }
        }
"""
    + _EXPLAIN_METAL_ROUND
)
_metal_explain_lmer = None
_EXPLAIN_LMER_METAL_THREADS = 128
# Bytes of window partials per launch ([slices, rows, W, 4, 16] float32)
_EXPLAIN_LMER_METAL_BYTES = 256 * 2**20


@njit(parallel=True, cache=True)
def _fold_windows(packed_x, part, inv_dx, l, seqlen):
    """[B, 4, seqlen] float64 from window partials part [slices, B, W, 4, 16]:
    window i's sum for delta d at its position k goes to base u_k ^ d at
    i + k, times inv_dx[b]."""
    n_slices, B, W = part.shape[:3]
    out = np.zeros((B, 4, seqlen))
    for b in prange(B):
        for i in range(W):
            u = packed_x[b, i]
            if u == _PAD:
                continue
            for k in range(l):
                uk = (u >> np.uint32(2 * k)) & np.uint32(3)
                for d in range(4):
                    acc = 0.0
                    for c in range(n_slices):
                        acc += part[c, b, i, d, k]
                    out[b, int(uk ^ np.uint32(d)), i + k] += acc
        out[b] *= inv_dx[b]
    return out


def _explain_lmer_metal(
    packed_x,
    inv_dx,
    V,
    index,
    w,
    masks,
    levels,
    table_a,
    table_b,
    l,
    seqlen,
    perturbation,
    wx,
    cache,
):
    """_explain_lmer_gpu in Metal: [B, 4, seqlen] float64 NumPy. V (float32 on
    the device, cached in `cache`) or, for RBF, the l-mer index and per-query
    SV weights w [B, S]; wx: query window weights ([B or 1, W] float64, [1, 0]
    unweighted)."""
    import mlx.core as mx

    global _metal_explain_lmer
    if _metal_explain_lmer is None:
        _metal_explain_lmer = mx.fast.metal_kernel(
            name="gkm_explain_lmer",
            input_names=[
                "bx",
                "masks",
                "levels",
                "V",
                "offsets",
                "ids",
                "w",
                "vals",
                "wx",
                "table_a",
                "table_b",
                "params",
            ],
            output_names=["part"],
            source=_EXPLAIN_LMER_METAL,
        )
    B, W = packed_x.shape
    n_masks = len(masks)
    key = ("explain_lmer_metal", id(masks))
    if cache.get(key, (None,))[0] is not masks:
        cache[key] = (
            masks,
            mx.array(np.ascontiguousarray(masks, dtype=np.uint32)),
            mx.array(np.ascontiguousarray(levels, dtype=np.int32)),
        )
    _, masks_d, levels_d = cache[key]
    zero_f, zero_i = mx.zeros(1, dtype=mx.float32), mx.zeros(1, dtype=mx.int32)
    use_index = V is None
    if use_index:
        offsets, ids, _, _, vals = index._arrays("mlx")
        w_d = mx.array(np.ascontiguousarray(w, dtype=np.float32))
        V_d, S = zero_f, w.shape[1]
        vals = vals if vals is not None else mx.zeros(1, dtype=mx.uint32)
    else:
        if "explain_V_mlx" not in cache:
            cache["explain_V_mlx"] = mx.array(V.astype(np.float32))
        V_d, S = cache["explain_V_mlx"], 0
        offsets = ids = zero_i
        w_d, vals = zero_f, mx.zeros(1, dtype=mx.uint32)
    weighted = wx.shape[1] > 0
    wx_stride = W if weighted and wx.shape[0] > 1 else 0
    wx_d = mx.array(wx.astype(np.float32)) if weighted else zero_f
    per_tg = _EXPLAIN_LMER_METAL_THREADS // 16
    groups = -(-W // per_tg)
    n_slices = max(
        1,
        min(
            -(-n_masks // 256),
            -(-_EXPLAIN_LMER_THREADS // (B * groups * _EXPLAIN_LMER_METAL_THREADS)),
            max(1, _EXPLAIN_LMER_METAL_BYTES // (B * W * 4 * 16 * 4)),
        ),
    )
    chunk = -(-n_masks // n_slices)
    n_slices = -(-n_masks // chunk)
    params = [
        W,
        n_masks,
        chunk,
        int(use_index),
        S,
        4**l,
        int(perturbation),
        int(weighted),
        wx_stride,
        B,
        l,
    ]
    part = _metal_explain_lmer(
        inputs=[
            mx.array(np.ascontiguousarray(packed_x, dtype=np.uint32)),
            masks_d,
            levels_d,
            V_d,
            offsets,
            ids,
            w_d,
            vals,
            wx_d,
            mx.array(np.asarray(table_a, dtype=np.float32)),
            mx.array(np.asarray(table_b, dtype=np.float32)),
            mx.array(params, dtype=mx.int32),
        ],
        template=[],
        grid=(groups * _EXPLAIN_LMER_METAL_THREADS, n_slices, B),
        threadgroup=(_EXPLAIN_LMER_METAL_THREADS, 1, 1),
        output_shapes=[(n_slices * B * W * 4 * 16,)],
        output_dtypes=[mx.float32],
        init_value=0,
    )[0]
    part = np.array(part).reshape(n_slices, B, W, 4, 16)
    return _fold_windows(np.ascontiguousarray(packed_x), part, inv_dx, l, seqlen)


# The per-SV kernel (_explain_cpu) in Metal: the same 16-lane groups, one per
# query window, each lane taking one SV window per round of a slice of the
# [S_c, Wy] SV windows; the pair's delta u ^ y and its weights go round the
# group as the l-mer path's neighbors do. Most pairs fall below min_matches,
# so only the lanes holding a pair (a ballot) go round.
_EXPLAIN_SV_METAL = """
    uint tid = thread_position_in_threadgroup.x;
    uint tg = threads_per_threadgroup.x;
    uint lane = tid % 16;
    uint i = threadgroup_position_in_grid.x * (tg / 16) + tid / 16;
    uint slice = threadgroup_position_in_grid.y;
    uint b = threadgroup_position_in_grid.z;
    uint W = (uint)params[0], n_items = (uint)params[1], chunk = (uint)params[2];
    uint Wy = (uint)params[3], S = (uint)params[4], perturbation = (uint)params[5];
    uint weighted = (uint)params[6], wx_stride = (uint)params[7], B = (uint)params[8];
    uint l = (uint)params[9], min_matches = (uint)params[10], wy_stride = (uint)params[11];
    // Groups past the last window (or on a PAD one) still take part in the shuffles
    uint u = i < W ? bx[b * W + i] : 0xFFFFFFFFu;
    bool valid = u != 0xFFFFFFFFu;
    uint shift = 2 * lane;
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f, q = 0.0f;
    uint t_begin = slice * chunk;
    uint t_end = min(n_items, t_begin + chunk);
    ushort src = (ushort)((tid % 32) & ~15u);  // this group's first lane
    for (uint t0 = t_begin; t0 < t_end; t0 += 16) {
        uint t = t0 + lane;
        uint my_mask = 0u;
        float my_a = 0.0f, my_b = 0.0f;
        if (valid && t < t_end) {
            uint y = sv[t];
            uint x = u ^ y;
            uint mm = popcount(((x >> 1) | x) & 0x55555555u);
            if (y != 0xFFFFFFFFu && l - mm >= min_matches) {
                uint s = t / Wy;
                float ew = ec[b * S + s];
                if (weighted)
                    ew *= wx[b * wx_stride + i] * wy[s * wy_stride + (t - s * Wy)];
                my_mask = x;
                my_a = ew * table_a[mm];
                my_b = ew * table_b[mm];
            }
        }
        // Only the lanes holding a pair go round (the same set for the 16 lanes)
        ulong vote = (ulong)(simd_vote::vote_t)simd_ballot(my_a != 0.0f || my_b != 0.0f);
        uint pending = (uint)(vote >> src) & 0xFFFFu;
""" + _EXPLAIN_METAL_ROUND.replace(
    """        uint n = min(16u, t_end - t0);
        for (uint j = 0; j < n; j++) {
""",
    """        while (pending != 0u) {
            uint j = ctz(pending);
            pending &= pending - 1u;
""",
)
_metal_explain_sv = None


def _explain_sv_metal(
    packed_x,
    packed_y,
    table_a,
    table_b,
    eff_coef,
    l,
    min_matches,
    seqlen,
    perturbation,
    wx,
    wy,
):
    """_explain_cpu in Metal (arguments the same, on CPU): [B, 4, seqlen]
    float64 NumPy, folded from float32 window partials."""
    import mlx.core as mx

    global _metal_explain_sv
    if _metal_explain_sv is None:
        _metal_explain_sv = mx.fast.metal_kernel(
            name="gkm_explain_sv",
            input_names=["bx", "sv", "ec", "wx", "wy", "table_a", "table_b"]
            + ["params"],
            output_names=["part"],
            source=_EXPLAIN_SV_METAL,
        )
    B, W = packed_x.shape
    S_c, Wy = packed_y.shape
    n_items = S_c * Wy
    weighted = wx.shape[1] > 0
    zero_f = mx.zeros(2, dtype=mx.float32)  # (a 1-element input would be a constant)
    per_tg = _EXPLAIN_LMER_METAL_THREADS // 16
    groups = -(-W // per_tg)
    n_slices = max(
        1,
        min(
            -(-n_items // 1024),
            -(-_EXPLAIN_LMER_THREADS // (B * groups * _EXPLAIN_LMER_METAL_THREADS)),
            max(1, _EXPLAIN_LMER_METAL_BYTES // (B * W * 4 * 16 * 4)),
        ),
    )
    chunk = -(-n_items // n_slices)
    n_slices = -(-n_items // chunk)
    params = [
        W,
        n_items,
        chunk,
        Wy,
        S_c,
        int(perturbation),
        int(weighted),
        W if weighted and wx.shape[0] > 1 else 0,
        B,
        l,
        min_matches,
        Wy if weighted and wy.shape[0] > 1 else 0,
    ]

    def f32(a):
        return mx.array(np.ascontiguousarray(a, dtype=np.float32).ravel())

    part = _metal_explain_sv(
        inputs=[
            mx.array(np.ascontiguousarray(packed_x, dtype=np.uint32).ravel()),
            mx.array(np.ascontiguousarray(packed_y, dtype=np.uint32).ravel()),
            f32(eff_coef),
            f32(wx) if weighted else zero_f,
            f32(wy) if weighted else zero_f,
            f32(table_a),
            f32(table_b),
            mx.array(params, dtype=mx.int32),
        ],
        template=[],
        grid=(groups * _EXPLAIN_LMER_METAL_THREADS, n_slices, B),
        threadgroup=(_EXPLAIN_LMER_METAL_THREADS, 1, 1),
        output_shapes=[(n_slices * B * W * 4 * 16,)],
        output_dtypes=[mx.float32],
        init_value=0,
    )[0]
    part = np.array(part).reshape(n_slices, B, W, 4, 16)
    return _fold_windows(np.ascontiguousarray(packed_x), part, np.ones(B), l, seqlen)


@njit(parallel=True, cache=True, fastmath=True)
def _explain_cpu(
    packed_x,
    packed_y,
    table_a,
    table_b,
    eff_coef,
    l,
    min_matches,
    seqlen,
    perturbation,
    wx,
    wy,
):
    """[B, 4, seqlen] attributions of packed_x [B, W] against packed_y [S, Wy];
    wx [B or 1, W], wy [S or 1, Wy]: window weights ([1, 0] when unweighted).
    PAD windows (past shorter sequences' windows) are skipped."""
    B, W = packed_x.shape
    S_c, Wy = packed_y.shape
    weighted = wx.shape[1] > 0
    result = np.zeros((B, 4, seqlen), dtype=np.float64)
    for b in prange(B):
        wxb = wx[min(np.int64(b), wx.shape[0] - 1)]
        for s in range(S_c):
            ec = eff_coef[b, s]
            wys = wy[min(np.int64(s), wy.shape[0] - 1)]
            for i in range(W):
                bx_val = packed_x[b, i]
                if bx_val == _PAD:
                    continue
                for j in range(Wy):
                    by_val = packed_y[s, j]
                    if by_val == _PAD:
                        continue
                    mm = _packed_mismatches(bx_val, by_val)
                    if l - mm < min_matches:
                        continue
                    ew = ec * wxb[i] * wys[j] if weighted else ec
                    av = ew * table_a[mm]
                    bv = ew * table_b[mm]
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
def _explain_lmer_cpu(
    packed_x,
    inv_dx,
    V,
    masks,
    levels,
    table_a,
    table_b,
    l,
    seqlen,
    perturbation,
    n_split,
    wx,
):
    """[B, 4, seqlen] attributions from the SVs' l-mer weights V (see _explain_lmer).

    Each query window u visits every l-mer v = u ^ mask within the mismatch
    cutoff and adds V[v] times the alpha/kappa (or perturbation) weight of
    their mismatch count to each of its positions, as _explain_cpu does for
    every SV window equal to v. Rows are split into n_split window chunks.
    wx: the query windows' weights ([B or 1, W], [1, 0] when unweighted; V
    holds the SVs'). PAD query windows have no neighbors (u ^ mask >= 4^l).
    """
    B, W = packed_x.shape
    n_lmers = V.shape[0]
    weighted = wx.shape[1] > 0
    part = np.zeros((B * n_split, 4, seqlen))
    step = (W + n_split - 1) // n_split
    for task in prange(B * n_split):
        b, c = task // n_split, task % n_split
        acc = part[task]
        wxb = wx[min(np.int64(b), wx.shape[0] - 1)]
        for i in range(c * step, min(W, (c + 1) * step)):
            u = packed_x[b, i]
            a = wxb[i] if weighted else 1.0
            for t in range(masks.shape[0]):
                v = u ^ masks[t]
                if np.int64(v) >= n_lmers or V[v] == 0.0:
                    continue
                _distribute(
                    acc,
                    i,
                    masks[t],
                    v,
                    a * V[v],
                    table_a[levels[t]],
                    table_b[levels[t]],
                    l,
                    perturbation,
                )
    out = part.reshape(B, n_split, 4, seqlen).sum(axis=1)
    for b in range(B):
        out[b] *= inv_dx[b]
    return out


@njit(parallel=True, cache=True, fastmath=True)
def _explain_index_cpu(
    packed_x,
    inv_dx,
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
    n_split,
    wx,
    vals,
):
    """As _explain_lmer_cpu with per-query SV weights w [B, S]: V[v] for row b is
    the sum of w[b, s] over the index's entries for v (RBF models), times each
    entry's window weight in vals (empty when unweighted)."""
    B, W = packed_x.shape
    n_lmers = offsets.shape[0] - 1
    weighted = vals.shape[0] > 0
    part = np.zeros((B * n_split, 4, seqlen))
    step = (W + n_split - 1) // n_split
    for task in prange(B * n_split):
        b, c = task // n_split, task % n_split
        acc = part[task]
        wb = w[b]
        wxb = wx[min(np.int64(b), wx.shape[0] - 1)]
        for i in range(c * step, min(W, (c + 1) * step)):
            u = packed_x[b, i]
            a = wxb[i] if weighted else 1.0
            for t in range(masks.shape[0]):
                v = u ^ masks[t]
                if np.int64(v) >= n_lmers:
                    continue
                wv = 0.0
                if weighted:
                    for e in range(offsets[v], offsets[v + 1]):
                        wv += wb[ids[e]] * vals[e]
                else:
                    for e in range(offsets[v], offsets[v + 1]):
                        wv += wb[ids[e]]
                if wv == 0.0:
                    continue
                _distribute(
                    acc,
                    i,
                    masks[t],
                    v,
                    a * wv,
                    table_a[levels[t]],
                    table_b[levels[t]],
                    l,
                    perturbation,
                )
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
    mismatch_table: np.ndarray,
    l: int,
    d: int,
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
    x,
    *,
    mode: int | str = "importance",
    lengths=None,
    batch_size: int = 50,
    verbose: bool = False,
) -> np.ndarray:
    """Compute GkmExplain attribution scores.

    Args:
        model: A GkmSVM model.
        x: [B, 4, L] one-hot encoded sequences, or a list of [4, L_i] arrays
            or DNA strings of different lengths.
        mode: Attribution mode (string or legacy int).
            ``"importance"`` (0) — hypothetical * one-hot.  Non-reference
                bases are zero; sum over [B, 4, L] equals score - bias
                (completeness axiom).  Use for per-position importance
                visualization.
            ``"hypothetical"`` (1) — hypothetical importance for all four
                bases at every position.  Use for TF-MoDISco (both
                projected and hypothetical contribution inputs).
            ``"perturbation"`` (2) — discrete kernel-value delta for
                each possible single-base mutation (lsgkm C mode 4:
                ``perturbation_eff=1`` over every pair it changes).  Use
                for per-position mutation maps; variant scores
                (``score_variants(method="gkmexplain")``) come from
                ``mutation_impact()``.
        lengths: [B] lengths of x's rows when they differ (x padded to the
            longest).
        batch_size: Process inputs in batches of this size (default 50).
        verbose: Show tqdm progress bar over input batches.

    Returns:
        [B, 4, L] float64 attribution scores on the model's device (NumPy for
        a model on MLX, which explains on CPU); 0 past each sequence's
        length, NaN for sequences shorter than l (with a warning).
    """
    resolved = _MODE_ALIASES.get(mode)
    if resolved is None:
        raise ValueError(
            f"mode must be 'importance', 'hypothetical', or 'perturbation' "
            f"(or 0/1/2), got {mode!r}"
        )
    x, lengths = as_batch(x, lengths)
    x = model._match_device(x)
    short = _short_rows(lengths, model.kernel.base.l, "attributions")

    if is_mlx(x):
        # The l-mer path and the per-SV kernel run in Metal (window partials
        # folded in float64 on the CPU), the explain table on CPU, from a CPU
        # copy of the model; the result is NumPy float64
        from gkmsvm.backend import to_cpu

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
                sv_lengths=model.sv_lengths,
                sv_chunk_size=model.sv_chunk_size,
            )
            model._sv_cache["cpu_model"] = cpu_model
        # (checked on every call, as on the model itself)
        cpu_model.use_lmer_index = model.use_lmer_index
        cpu_model.use_lmer_table = model.use_lmer_table
        cpu_model.use_explain_table = model.use_explain_table
        out = _gkmexplain(
            cpu_model, to_cpu(x), lengths, resolved, batch_size, verbose, metal=True
        )
        return _per_position(out, lengths, short)
    out = _gkmexplain(model, x, lengths, resolved, batch_size, verbose)
    return _per_position(out, lengths, short)


def _gkmexplain(model, x, lengths, resolved, batch_size, verbose, metal=False):
    """gkmexplain on x on the model's device (not MLX), lengths None or [B].
    metal: x and the model are a CPU copy of an MLX model's, and the l-mer
    path and the per-SV kernel run in Metal."""
    if resolved == "importance":
        hyp = _gkmexplain(model, x, lengths, "hypothetical", batch_size, verbose, metal)
        return hyp * _n_as_a(x)

    is_perturbation = resolved == "perturbation"

    kernel = model.kernel
    base = kernel.base
    if getattr(base, "_mismatch_table", None) is None:
        raise ValueError(
            f"GkmExplain attribution needs a gapped k-mer kernel, not {model.kernel_type!r}; "
            "use mutation_impact() or ism()"
        )
    # RBF (-t 3/-t 5): each SV's contributions scale by _rbf_scale of its kernel value
    gamma = kernel.gamma if base is not kernel else None
    xp = get_array_module(x)
    B, _, seqlen = x.shape
    l = base.l
    d = getattr(base, "d", None)
    d = l if d is None else d  # -t 0 without d counts every pair
    if is_perturbation:
        table_a, table_b = _build_perturbation_tables(base._mismatch_table, l, d)
    else:
        table_a = _build_alpha_table(base._mismatch_table, l, d)
        table_b = _build_kappa_table(base._mismatch_table, l, d)
    min_matches = _explain_min_matches(table_a, table_b, l)
    A = _explain_table(
        model,
        B,
        "cpu" if xp is np else "cuda",
        table_a,
        table_b,
        is_perturbation,
        gamma,
    )
    if A is not None:
        return _explain_with_table(
            model, x, A, is_perturbation, batch_size, verbose, lengths
        )
    lmer = _explain_lmer(
        model,
        x,
        lengths,
        table_a,
        table_b,
        l - min_matches,
        is_perturbation,
        gamma,
        batch_size,
        verbose,
        metal,
    )
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
    # [S or 1, Wy] SV window weights, forward then RC (an RC window carries its
    # forward weight); [1, 0] when unweighted
    wy = base.row_weights(sv_fwd)
    if wy is None:
        wy = np.zeros((1, 0))
    else:
        if sv_rc is not None:
            wy_rc = base.row_weights(sv_rc, rc=True)
            n = max(len(wy), len(wy_rc))
            wy = np.concatenate(
                [
                    np.broadcast_to(wy, (n, wy.shape[1])),
                    np.broadcast_to(wy_rc, (n, wy_rc.shape[1])),
                ],
                axis=1,
            )
        wy = np.ascontiguousarray(wy, dtype=np.float64)

    result = xp.zeros((B, 4, seqlen), dtype=np.float64)
    starts = range(0, B, batch_size)
    if verbose:
        from tqdm import tqdm

        starts = tqdm(
            starts, desc="GkmExplain", total=(B + batch_size - 1) // batch_size
        )

    for b0 in starts:
        x_b = x[b0 : b0 + batch_size]
        packed_x, packed_x_rc = base.pack_both(x_b, _rows(lengths, b0, batch_size))
        packed_x = xp.ascontiguousarray(packed_x)
        diag_x = base.raw_diag_packed(packed_x, packed_x_rc) if base.normalize else None
        wx = _query_wts(base, packed_x)
        if gamma is not None:
            k_norm = model._sv_raw(packed_x) / xp.clip(
                xp.sqrt(diag_x[:, None] * diag_sv[None, :]), 1e-10, None
            )
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
                fn = _explain_sv_metal if metal else _explain_cpu
                result[b0 : b0 + batch_size] += fn(
                    packed_x,
                    np.ascontiguousarray(packed_sv[s0:s1]),
                    table_a,
                    table_b,
                    np.ascontiguousarray(eff_coef, dtype=np.float64),
                    l,
                    min_matches,
                    seqlen,
                    is_perturbation,
                    wx,
                    wy[s0:s1] if len(wy) > 1 else wy,
                )
            else:
                _explain_gpu(
                    packed_x,
                    xp.ascontiguousarray(packed_sv[:, s0:s1]),
                    table_a,
                    table_b,
                    xp.ascontiguousarray(eff_coef, dtype=np.float64),
                    result[b0 : b0 + batch_size],
                    s1 - s0,
                    packed_sv.shape[0],
                    l,
                    min_matches,
                    is_perturbation,
                    wx,
                    wy[s0:s1] if len(wy) > 1 else wy,
                )

    return result


def _n_as_a(x):
    """One-hot x with N (all-zero) positions set to A: N packs as A, so its
    windows score as A and its importance goes there, as in lsgkm (and the
    importance then sums to score - bias)."""
    xp = get_array_module(x)
    n = (x.sum(axis=1, keepdims=True) == 0).astype(x.dtype)
    return xp.concatenate([x[:, :1] + n, x[:, 1:]], axis=1)


def _query_wts(base, packed):
    """Window weights of packed query rows as float64 [R or 1, W] on CPU
    (row_weights), [1, 0] when every window counts once."""
    w = base.row_weights(packed)
    return np.zeros((1, 0)) if w is None else w.astype(np.float64)


def _rows(lengths, start, n):
    """lengths[start:start + n], or None."""
    return None if lengths is None else lengths[start : start + n]


def _sv_weights(model):
    """(coef, 1 / sqrt(K(sv, sv))) on CPU in float64; ones without normalization."""
    from gkmsvm.backend import to_cpu

    coef = to_cpu(model.coefficients).astype(np.float64)
    inv_ds = (
        1 / np.sqrt(np.clip(to_cpu(model._sv_diag()).astype(np.float64), 1e-20, None))
        if model.kernel.base.normalize
        else np.ones(len(coef))
    )
    return coef, inv_ds


def _explain_V(model):
    """V[v] = sum_s coef_s / sqrt(K(sv, sv)) * (occurrences of l-mer v in sv_s and
    its RC, each counted with its window weight), 4^l float64 on CPU (cached)."""
    from gkmsvm.backend import to_cpu

    c = model._sv_cache
    if "explain_V" not in c:
        coef, inv_ds = _sv_weights(model)
        base = model.kernel.base
        V = np.zeros(4**base.l)
        for rc, packed in enumerate(model._sv_packed()):
            if packed is not None:
                packed = to_cpu(packed)
                # Each SV's window weights (an RC window carries its forward
                # one); PAD past shorter SVs' windows is left out
                wy = base.row_weights(packed, rc=bool(rc))
                w = (coef * inv_ds)[:, None]
                w = np.broadcast_to(w if wy is None else w * wy, packed.shape).ravel()
                keys = packed.ravel().astype(np.int64)
                keep = keys < len(V)
                V += np.bincount(keys[keep], weights=w[keep], minlength=len(V))
        c["explain_V"] = V
    return c["explain_V"]


# ---------------------------------------------------------------------------
# Explain table: every window's attribution, precomputed for non-RBF kernels
# ---------------------------------------------------------------------------
# A pair (query window u, SV l-mer v) adds V[v] times a weight that depends
# only on its mismatch count m and, at each position k, on the 2-bit delta
# d = (u ^ v)_k (the SV's base is u_k ^ d): alpha(m) when d == 0, else kappa(m)
# (or the perturbation deltas). So window u's attribution is
#
#     A[u, d, k] = sum_v V[v] G_kd(u ^ v),   G_kd(z) = [z_k == d] t_d(m(z)),
#
# an XOR convolution over the 4^l l-mers. With the Walsh-Hadamard transform H
# (H H = 4^l I), A[:, d, k] = H(H(V) H(G_kd)) / 4^l, and H(G_kd) is H(G_0d)
# with digits 0 and k swapped, so a table costs 1 + 4 + 4l transforms. A query
# window then reads its [4, l] row. Exact up to rounding; nothing per SV and
# no neighbor cutoff, so -t 0 works too. RBF models have no table (their SV
# weights depend on the query).

# 4^12 x 4 x 12 float64 = 6.4 GB
_EXPLAIN_TABLE_MAX_L = 12
# Largest table "auto" builds: [4^11, 4, 11] float64 is 1.5 GB
_EXPLAIN_TABLE_BYTES = 2 * 2**30
# "auto" builds a table once a call explains as many sequences as it takes to
# pay for it: building costs ~2e-8 s per entry of 4^l x l (0.7-1.4 s at l=11
# on an M4), against ~1.5 ms per 300 bp sequence for the CPU l-mer path and
# ~0.5 ms for CUDA's (-t 2; -t 0 and perturbation mode are slower)
_EXPLAIN_TABLE_BUILD_S = 2e-8
_EXPLAIN_SEQ_S = {"cpu": 1.5e-3, "cuda": 0.5e-3}


@njit(parallel=True, cache=True)
def _wht4(a, l):
    """In-place unnormalized Walsh-Hadamard transform of a [4^l], one radix-4
    pass per 2-bit digit."""
    n = a.shape[0]
    h = 1
    for _ in range(l):
        for j in prange(n // 4):
            i = (j // h) * 4 * h + j % h
            a0, a1, a2, a3 = a[i], a[i + h], a[i + 2 * h], a[i + 3 * h]
            s01, d01, s23, d23 = a0 + a1, a0 - a1, a2 + a3, a2 - a3
            a[i] = s01 + s23
            a[i + h] = d01 + d23
            a[i + 2 * h] = s01 - s23
            a[i + 3 * h] = d01 - d23
        h *= 4


@njit(parallel=True, cache=True)
def _wht4_rows(a, l):
    """_wht4 down the rows of a [4^l, C], C columns at once."""
    n, C = a.shape
    h = 1
    for _ in range(l):
        for j in prange(n // 4):
            i = (j // h) * 4 * h + j % h
            for c in range(C):
                a0, a1, a2, a3 = a[i, c], a[i + h, c], a[i + 2 * h, c], a[i + 3 * h, c]
                s01, d01, s23, d23 = a0 + a1, a0 - a1, a2 + a3, a2 - a3
                a[i, c] = s01 + s23
                a[i + h, c] = d01 + d23
                a[i + 2 * h, c] = s01 - s23
                a[i + 3 * h, c] = d01 - d23
        h *= 4


@njit(parallel=True, cache=True)
def _fill_products(fv, fg, l, out):
    """out[z, d, k] = fv[z] * fg[d, z with digits 0 and k swapped] / 4^l."""
    n = fv.shape[0]
    for z in prange(n):
        z0 = z & 3
        f = fv[z] / n
        for k in range(l):
            sk = np.int64(2 * k)
            zk = (z >> sk) & 3
            w = (z & ~(np.int64(3) << sk) & ~np.int64(3)) | (z0 << sk) | zk
            for d in range(4):
                out[z, d, k] = f * fg[d, w]


def _build_explain_table(V, table_a, table_b, l):
    """A [4^l, 4, l] float64 (see above) from V and the tables t_0 = table_a,
    t_1..3 = table_b at mismatch count m."""
    n = 4**l
    fv = V.astype(np.float64)  # copy
    _wht4(fv, l)
    z = np.arange(n, dtype=np.int64)
    m = np.zeros(n, dtype=np.int64)
    for k in range(l):
        m += ((z >> (2 * k)) & 3) != 0
    z0 = z & 3
    fg = np.empty((4, n))
    for d in range(4):
        fg[d] = np.where(z0 == d, (table_a if d == 0 else table_b)[m], 0.0)
        _wht4(fg[d], l)
    del z, m, z0
    # The 4l inverse transforms run together down A's rows
    A = np.empty((n, 4, l))
    _fill_products(fv, fg, l, A)
    del fv, fg
    _wht4_rows(A.reshape(n, 4 * l), l)
    return A


@njit(parallel=True, cache=True, fastmath=True)
def _explain_table_cpu(packed_x, inv_dx, A, l, seqlen, perturbation, wx):
    """[B, 4, seqlen] attributions: each window u adds A[u, d, k] to base u_k ^ d
    at position i + k (perturbation: d == 0 goes to the other three bases),
    times its window weight wx[b, i] (wx [B or 1, W], [1, 0] when unweighted).
    PAD (past a shorter sequence's windows) is no l-mer and adds nothing."""
    B, W = packed_x.shape
    n_lmers = A.shape[0]
    weighted = wx.shape[1] > 0
    out = np.zeros((B, 4, seqlen))
    for b in prange(B):
        acc = out[b]
        wxb = wx[min(np.int64(b), wx.shape[0] - 1)]
        for i in range(W):
            u = packed_x[b, i]
            if np.int64(u) >= n_lmers:
                continue
            a = wxb[i] if weighted else 1.0
            for k in range(l):
                uk = int((u >> np.uint32(2 * k)) & np.uint32(3))
                pos = i + k
                if perturbation:
                    a0 = a * A[u, 0, k]
                    for h in range(4):
                        if h != uk:
                            acc[h, pos] += a0
                else:
                    acc[uk, pos] += a * A[u, 0, k]
                for d in range(1, 4):
                    acc[uk ^ d, pos] += a * A[u, d, k]
        acc *= inv_dx[b]
    return out


def _explain_table(model, n_rows, device, table_a, table_b, perturbation, gamma):
    """The cached explain table for this mode, built if model.use_explain_table
    calls for it, or None."""
    use = model.use_explain_table
    base = model.kernel.base
    l = base.l
    if use is False or gamma is not None or l > _EXPLAIN_TABLE_MAX_L:
        return None
    tables = model._explain_tables
    if perturbation not in tables:
        build_s = _EXPLAIN_TABLE_BUILD_S * 4**l * l
        if use == "auto" and (
            4**l * 4 * l * 8 > _EXPLAIN_TABLE_BYTES
            or n_rows * _EXPLAIN_SEQ_S[device] < build_s
        ):
            return None
        tables[perturbation] = _build_explain_table(
            _explain_V(model), table_a, table_b, l
        )
    return tables[perturbation]


def _explain_with_table(model, x, A, perturbation, batch_size, verbose, lengths=None):
    """GkmExplain from the explain table, on CPU; results go back to x's device."""
    from gkmsvm.backend import to_cpu

    base = model.kernel.base
    xp = get_array_module(x)
    x_cpu = to_cpu(x)
    B, _, seqlen = x_cpu.shape
    starts = range(0, B, batch_size)
    if verbose:
        from tqdm import tqdm

        starts = tqdm(
            starts, desc="GkmExplain", total=(B + batch_size - 1) // batch_size
        )
    parts = []
    for b0 in starts:
        x_b = x_cpu[b0 : b0 + batch_size]
        packed_x, packed_x_rc = base.pack_both(x_b, _rows(lengths, b0, batch_size))
        inv_dx = (
            1
            / np.sqrt(np.clip(base.raw_diag_packed(packed_x, packed_x_rc), 1e-20, None))
            if base.normalize
            else np.ones(x_b.shape[0])
        )
        parts.append(
            _explain_table_cpu(
                np.ascontiguousarray(packed_x),
                inv_dx,
                A,
                base.l,
                seqlen,
                perturbation,
                _query_wts(base, packed_x),
            )
        )
    out = np.concatenate(parts)
    return out if xp is np else xp.asarray(out)


def _explain_lmer(
    model,
    x,
    lengths,
    table_a,
    table_b,
    dmax,
    perturbation,
    gamma,
    batch_size,
    verbose,
    metal=False,
):
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
    more index query per call). Runs on CPU (in Metal with metal, x on CPU);
    results go back to x's device.
    """
    import numba

    from gkmsvm.backend import to_cpu
    from gkmsvm.kernels import lmer_index

    base = model.kernel.base
    l = base.l
    if not model.use_lmer_index or l > lmer_index.MAX_L or dmax < 0:
        return None
    n_neighbors = lmer_index.neighbor_count(l, dmax)
    if n_neighbors > lmer_index.MAX_NEIGHBORS:
        # A query window visits n_neighbors l-mers here and every SV window
        # in the per-SV kernel
        packed = model._sv_packed()
        sv_windows = (
            0 if packed is None else sum(p.size for p in packed if p is not None)
        )
        if n_neighbors * _EXPLAIN_SV_WINDOWS_PER_NEIGHBOR > sv_windows:
            return None
    coef, inv_ds = _sv_weights(model)
    index = V = None
    if gamma is not None:
        index = model._sv_index()  # its arrays live on CPU whatever the model's device
        if index is None:
            return None
    else:
        V = _explain_V(model)
    nonzero = np.maximum(np.abs(table_a), np.abs(table_b))
    masks, levels = lmer_index.neighbor_masks(l, nonzero, dmax)

    xp = get_array_module(x)
    if xp is not np and not is_mlx(x):
        return _explain_lmer_cuda(
            model,
            x,
            lengths,
            V,
            index,
            coef,
            inv_ds,
            masks,
            levels,
            table_a,
            table_b,
            gamma,
            batch_size,
            verbose,
            perturbation,
        )
    x_cpu = to_cpu(x)
    B, _, seqlen = x_cpu.shape
    starts = range(0, B, batch_size)
    if verbose:
        from tqdm import tqdm

        starts = tqdm(
            starts, desc="GkmExplain", total=(B + batch_size - 1) // batch_size
        )
    parts = []
    for b0 in starts:
        x_b = x_cpu[b0 : b0 + batch_size]
        packed_x, packed_x_rc = base.pack_both(x_b, _rows(lengths, b0, batch_size))
        packed_x = np.ascontiguousarray(packed_x)
        inv_dx = (
            1
            / np.sqrt(np.clip(base.raw_diag_packed(packed_x, packed_x_rc), 1e-20, None))
            if base.normalize
            else np.ones(x_b.shape[0])
        )
        W = packed_x.shape[1]
        n_split = max(1, min(W, -(-numba.get_num_threads() // x_b.shape[0])))
        wx = _query_wts(base, packed_x)
        if metal:
            w = None
            if index is not None:
                # The CPU query: exact in float64 too, and faster than the
                # Metal one on large SV sets (contended atomics)
                k_norm = index.pairwise(packed_x) * inv_dx[:, None] * inv_ds[None, :]
                w = coef * inv_ds * _rbf_scale(k_norm, gamma)
            parts.append(
                _explain_lmer_metal(
                    packed_x,
                    inv_dx,
                    V,
                    index,
                    w,
                    masks,
                    levels,
                    table_a,
                    table_b,
                    l,
                    seqlen,
                    perturbation,
                    wx,
                    model._sv_cache,
                )
            )
            continue
        if index is None:
            parts.append(
                _explain_lmer_cpu(
                    packed_x,
                    inv_dx,
                    V,
                    masks,
                    levels,
                    table_a,
                    table_b,
                    l,
                    seqlen,
                    perturbation,
                    n_split,
                    wx,
                )
            )
            continue
        k_norm = index.pairwise(packed_x) * inv_dx[:, None] * inv_ds[None, :]
        w = coef * inv_ds * _rbf_scale(k_norm, gamma)
        parts.append(
            _explain_index_cpu(
                packed_x,
                inv_dx,
                index.offsets,
                index.ids,
                np.ascontiguousarray(w),
                masks,
                levels,
                table_a,
                table_b,
                l,
                seqlen,
                perturbation,
                n_split,
                wx,
                index.vals,
            )
        )
    out = np.concatenate(parts)
    return out if xp is np else xp.asarray(out)


def _explain_lmer_cuda(
    model,
    x,
    lengths,
    V,
    index,
    coef,
    inv_ds,
    masks,
    levels,
    table_a,
    table_b,
    gamma,
    batch_size,
    verbose,
    perturbation,
):
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
        offsets = ids = vals = None
    else:
        offsets, ids, _, _, vals = index._arrays("cuda")
        coef_d = cp.asarray(coef)
        inv_ds_d = cp.asarray(inv_ds)
    masks, levels = cp.asarray(masks), cp.asarray(levels)
    table_a = cp.asarray(table_a, dtype=cp.float64)
    table_b = cp.asarray(table_b, dtype=cp.float64)

    B, _, seqlen = x.shape
    starts = range(0, B, batch_size)
    if verbose:
        from tqdm import tqdm

        starts = tqdm(
            starts, desc="GkmExplain", total=(B + batch_size - 1) // batch_size
        )
    out = cp.empty((B, 4, seqlen), dtype=cp.float64)
    for b0 in starts:
        x_b = x[b0 : b0 + batch_size]
        packed_x, packed_x_rc = base.pack_both(x_b, _rows(lengths, b0, batch_size))
        packed_x = cp.ascontiguousarray(packed_x)
        inv_dx = (
            1
            / cp.sqrt(cp.clip(base.raw_diag_packed(packed_x, packed_x_rc), 1e-20, None))
            if base.normalize
            else cp.ones(x_b.shape[0])
        )
        w = None
        if index is not None:
            k_norm = index.pairwise(packed_x) * inv_dx[:, None] * inv_ds_d[None, :]
            w = coef_d * inv_ds_d * _rbf_scale(k_norm, gamma, cp)
        wx = base.row_weights(packed_x)
        out[b0 : b0 + batch_size] = _explain_lmer_gpu(
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
        )
    return out


def mutation_impact(
    model: GkmSVM,
    ref,
    alt,
    *,
    ref_lengths=None,
    alt_lengths=None,
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
        ref: [B, 4, L] reference sequences, or a list of [4, L_i] arrays or
            DNA strings of different lengths.
        alt: [B, 4, L'] alternate sequences, as ref.
        ref_lengths, alt_lengths: [B] lengths of the rows of padded ref and
            alt arrays when they differ.
        batch_size: Process inputs in batches of this size, raised as in
            ``GkmSVM.__call__``.
        verbose: Show tqdm progress bar over input batches.

    Returns:
        [B, 1] mutation impact scores; NaN when an allele is shorter than l
        (with a warning).
    """
    from gkmsvm.backend import to_cpu

    kernel = model.kernel
    base = kernel.base
    coef = model.coefficients
    packed_svs = model._sv_packed() is not None
    ref, alt, ref_lengths, alt_lengths = _allele_arrays(
        ref, alt, ref_lengths, alt_lengths
    )
    L = ref.shape[2]
    short = None
    if ref_lengths is not None or alt_lengths is not None:
        if not packed_svs:
            raise ValueError("sequences of different lengths need a packed kernel")
        both_lengths = _joint_lengths(
            L, (ref.shape[0], ref_lengths), (ref.shape[0], alt_lengths)
        )
        short = _short_rows(both_lengths, base.l, "mutation impacts")
        if short is not None:
            short = short.reshape(2, -1).any(axis=0)

    B = ref.shape[0]
    bs = B if batch_size is None else model._min_batch_size(batch_size)
    starts = range(0, B, bs)
    if verbose:
        from tqdm import tqdm

        starts = tqdm(starts, desc="Mutation impact", total=(B + bs - 1) // bs)

    parts = []
    for s in starts:
        r = model._match_device(ref[s : s + bs])
        a = model._match_device(alt[s : s + bs])
        xp = get_array_module(r)
        n = r.shape[0]
        both = xp.concatenate([r, a])
        if packed_svs:
            # One call for both alleles (the masked join builds its SV index once)
            bx, bx_rc = base.pack_both(
                both,
                _joint_lengths(
                    L, (n, _rows(ref_lengths, s, bs)), (n, _rows(alt_lengths, s, bs))
                ),
            )
            diag_r = base.raw_diag_packed(bx[:n], None if bx_rc is None else bx_rc[:n])
            table = model._lmer_table_for(xp)
            if table is not None:
                # Non-RBF: linear in the SVs, so sum_s coef_s dK_raw / sqrt(K(sv, sv))
                # is the l-mer table's alt minus ref (the table holds the SV norms)
                raw = model._table_raw(bx, table)
                dk = raw[n:] - raw[:n]
                if base.normalize:
                    dk = dk / xp.sqrt(xp.clip(diag_r.astype(dk.dtype), 1e-10, None))
                parts.append(to_cpu(dk)[:, None])
                continue
            k_both = model._sv_raw(bx)
        else:
            k_both = base._raw_pairwise(both, model.support_sequences)
            diag_r = base._raw_diagonal(r)
        gamma = kernel.gamma if base is not kernel else np.nan  # RBF
        if xp is np:
            parts.append(
                _impact_cpu(
                    np.ascontiguousarray(k_both, dtype=np.float64),
                    diag_r.astype(np.float64) if base.normalize else np.ones(n),
                    model._sv_diag().astype(np.float64)
                    if base.normalize
                    else np.ones(1),
                    base.normalize,
                    coef.astype(np.float64),
                    float(gamma),
                )[:, None]
            )
            continue
        k_ref = k_both[:n]
        dk = k_both[n:] - k_ref
        if base.normalize:
            norm = xp.clip(
                xp.sqrt(diag_r[:, None] * model._sv_diag()[None, :]), 1e-10, None
            )
            k_ref = k_ref / norm
            dk = dk / norm
        if not np.isnan(gamma):
            pos = k_ref > 0
            safe_k = xp.where(pos, k_ref, 1.0)
            scale = (kernel.finish(k_ref) - np.exp(-gamma)) / safe_k
            dk = xp.where(pos, dk * scale, 0.0)
        parts.append(to_cpu((dk * coef).sum(axis=1, keepdims=True)))

    return _nan_rows(np.concatenate(parts, axis=0), short)
