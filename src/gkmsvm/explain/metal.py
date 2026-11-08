"""Metal kernels (MLX): the l-mer path and the per-SV kernel in 16-lane groups.
Metal has no float64, so each window writes float32 partial sums that are
folded in float64 on the CPU."""

from __future__ import annotations

from functools import cache

import numpy as np
from numba import njit, prange

from gkmsvm.explain.common import _launch_slices
from gkmsvm.kernels.packed import PAD_WINDOW

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
_EXPLAIN_LMER_METAL_THREADS = 128
# Bytes of window partials per launch ([slices, rows, W, 4, 16] float32)
_EXPLAIN_LMER_METAL_BYTES = 256 * 2**20


@cache
def _metal_explain_lmer():
    import mlx.core as mx

    return mx.fast.metal_kernel(
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


def _metal_slices(n, per_slice, B, W):
    """_launch_slices for the Metal kernels' grid (W windows per row, a
    16-lane group each), with the window partials capped at
    _EXPLAIN_LMER_METAL_BYTES; also returns the threadgroups per row."""
    groups = -(-W // (_EXPLAIN_LMER_METAL_THREADS // 16))
    n_slices, chunk = _launch_slices(
        n,
        per_slice,
        groups * _EXPLAIN_LMER_METAL_THREADS,
        B,
        max(1, _EXPLAIN_LMER_METAL_BYTES // (B * W * 4 * 16 * 4)),
    )
    return n_slices, chunk, groups


def _metal_fold(
    kernel, inputs, packed_x, n_slices, groups, inv_dx, l, seqlen, defer=False
):
    """Launch a Metal explain kernel over (groups, n_slices, B) threadgroups and
    fold its window partials: [B, 4, seqlen] float64 NumPy. defer: start the
    kernel (mx.async_eval) and return a function that waits for it and folds,
    so the caller can work on the CPU meanwhile."""
    import mlx.core as mx

    B, W = packed_x.shape
    part = kernel(
        inputs=inputs,
        template=[],
        grid=(groups * _EXPLAIN_LMER_METAL_THREADS, n_slices, B),
        threadgroup=(_EXPLAIN_LMER_METAL_THREADS, 1, 1),
        output_shapes=[(n_slices * B * W * 4 * 16,)],
        output_dtypes=[mx.float32],
        init_value=0,
    )[0]
    packed_x = np.ascontiguousarray(packed_x)

    def finish():
        p = np.array(part).reshape(n_slices, B, W, 4, 16)
        return _fold_windows(packed_x, p, inv_dx, l, seqlen)

    if defer:
        mx.async_eval(part)
        return finish
    return finish()


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
            if u == PAD_WINDOW:
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
    defer=False,
):
    """_explain_lmer_gpu in Metal: [B, 4, seqlen] float64 NumPy. V (float32 on
    the device, cached in `cache`) or, for RBF, the l-mer index and per-query
    SV weights w [B, S]; wx: query window weights ([B or 1, W] float64, [1, 0]
    unweighted). defer: as for _metal_fold."""
    import mlx.core as mx

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
    n_slices, chunk, groups = _metal_slices(n_masks, 256, B, W)
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
    inputs = [
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
    ]
    return _metal_fold(
        _metal_explain_lmer(),
        inputs,
        packed_x,
        n_slices,
        groups,
        inv_dx,
        l,
        seqlen,
        defer,
    )


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


@cache
def _metal_explain_sv():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gkm_explain_sv",
        input_names=["bx", "sv", "ec", "wx", "wy", "table_a", "table_b", "params"],
        output_names=["part"],
        source=_EXPLAIN_SV_METAL,
    )


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

    B, W = packed_x.shape
    S_c, Wy = packed_y.shape
    n_items = S_c * Wy
    weighted = wx.shape[1] > 0
    zero_f = mx.zeros(2, dtype=mx.float32)  # (a 1-element input would be a constant)
    n_slices, chunk, groups = _metal_slices(n_items, 1024, B, W)
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

    inputs = [
        mx.array(np.ascontiguousarray(packed_x, dtype=np.uint32).ravel()),
        mx.array(np.ascontiguousarray(packed_y, dtype=np.uint32).ravel()),
        f32(eff_coef),
        f32(wx) if weighted else zero_f,
        f32(wy) if weighted else zero_f,
        f32(table_a),
        f32(table_b),
        mx.array(params, dtype=mx.int32),
    ]
    return _metal_fold(
        _metal_explain_sv(), inputs, packed_x, n_slices, groups, np.ones(B), l, seqlen
    )
