"""CUDA and Metal sources for the l-mer index (kernels/lmer_index.py) and their
compiled kernels: query, fused ISM, and on Metal the fused reduce and
mutation impact. Launch sizes and limits stay in lmer_index.py."""

from __future__ import annotations

from functools import cache

_CUDA_SRC = r"""
extern "C" __global__ void lmer_index_query(
    const unsigned int* __restrict__ bx, int W, long long n_items,
    const long long* __restrict__ offsets, const int* __restrict__ ids,
    const unsigned int* __restrict__ masks, const double* __restrict__ weights,
    int n_masks, long long n_lmers, int S, double* __restrict__ out,
    const unsigned char* __restrict__ wx, int wx_stride,
    const unsigned char* __restrict__ vals, int weighted, long long diag_offset)
{
    // grid.y = query row; each thread takes one (window, neighbor) item.
    // wx [rows or 1, W], vals: window weights of the queries (wx_stride 0 when
    // the rows share one) and entries (-t 4/5, weighted = 1). diag_offset >= 0:
    // row b only gets columns >= diag_offset + b; each l-mer's entries are
    // sorted by column, so its scan runs down from the last and stops below lo
    long long it = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (it >= n_items) return;
    int b = blockIdx.y;
    int i = (int)(it / n_masks), t = (int)(it - (long long)i * n_masks);
    long long v = (long long)(bx[(long long)b * W + i] ^ masks[t]);
    if (v >= n_lmers) return;  // padding window
    double wt = weights[t];
    double* row = out + (long long)b * S;
    int lo = diag_offset >= 0 ? (int)(diag_offset + b) : 0;
    if (weighted) wt *= wx[(long long)b * wx_stride + i];
    for (long long e = offsets[v + 1] - 1; e >= offsets[v]; e--) {
        int s = ids[e];
        if (s < lo) break;
        atomicAdd(&row[s], weighted ? wt * vals[e] : wt);
    }
}
"""

# Metal has no float64: count hits per mismatch level in int32 and combine
# levels in float64 on the CPU, so results are exact. One threadgroup of
# _METAL_QUERY_THREADS per query row, its threads striding over the row's
# (window, neighbor) items (consecutive threads take consecutive neighbors of
# a window, which the sorted masks keep close in the index). A thread per item
# over many small threadgroups had every resident group adding into the same
# row's counters, and lost to the CPU above ~2.5 index entries per l-mer.
_METAL_QUERY_HEAD = """
    uint W = (uint)params[0], n_masks = (uint)params[1], S = (uint)params[2];
    uint n_lmers = (uint)params[3], B = (uint)params[4], split = (uint)params[5];
    // split threadgroups per row (few rows: spread each over the GPU's cores)
    uint b = threadgroup_position_in_grid.x / split;
    uint part = threadgroup_position_in_grid.x - b * split;
    uint tid = thread_position_in_threadgroup.x, nt = threads_per_threadgroup.x;
    uint n_items = W * n_masks;
    for (uint it = part * nt + tid; it < n_items; it += nt * split) {
        uint i = it / n_masks, t = it - i * n_masks;
        uint v = bx[b * W + i] ^ masks[t];
        if (v >= n_lmers) continue;
        uint base = ((uint)levels[t] * B + b) * S;
"""
_METAL_SRC = (
    _METAL_QUERY_HEAD
    + """
        for (int e = offsets[v]; e < offsets[v + 1]; e++)
            atomic_fetch_add_explicit(&counts[base + ids[e]], 1, memory_order_relaxed);
    }
"""
)
# With window weights (-t 4/5): each hit adds wx[b, i] * vals[e], exact in 64
# bits as a low uint32 word plus a carry word (bumped by the thread whose add
# wraps the low word)
_METAL_SRC_W = (
    _METAL_QUERY_HEAD
    + """
        uint wx_stride = (uint)params[6];
        uint a = (uint)wx[b * wx_stride + i];
        for (int e = offsets[v]; e < offsets[v + 1]; e++) {
            uint c = a * (uint)vals[e];
            uint old = atomic_fetch_add_explicit(&lo[base + ids[e]], c,
                                                 memory_order_relaxed);
            if (old + c < old)
                atomic_fetch_add_explicit(&hi[base + ids[e]], 1u, memory_order_relaxed);
        }
    }
"""
)
# ISM through the index on the GPU (LmerIndex.mutants): one thread block per
# (position, sequence). Its threads scatter the hits of the windows covering
# the position, before (slot 0) and after each substitution (slots 1-4), into
# a [5, S] scratch slice, then reduce sum_s coef_s (f(mutant) - f(reference))
# over the SVs, so no [mutants, S] block leaves the GPU. old [L, B, l], new
# [4, L, B, l]; slot_wt [L, R, l] the windows' weights (weighted only).
_MUTANTS_BODY = """
    long long it = item0 + GROUP;  // p * B + b
    int p = (int)(it / B), b = (int)(it % B);
    long long LB = (long long)L * B, sbase = (long long)GROUP * 5 * S;
    long long o = it * l;  // old[o + j]: window j before
    REAL dx = inv_dx[b];
    bool skip[4];  // the reference base: its windows and norm
    for (int v = 0; v < 4; v++)
        skip[v] = nw[(v * LB + it) * l] == old[o] && inv_dmut[v * LB + it] == dx;
    long long sw = ((long long)p * R + min(b, R - 1)) * l;  // slot_wt[sw + j]
    for (int slot = 0; slot < 5 * l; slot++) {
        int tgt = slot / l, j = slot - tgt * l;
        if (tgt > 0 && skip[tgt - 1]) continue;
        unsigned int u = tgt == 0 ? old[o + j] : nw[((tgt - 1) * LB + it) * l + j];
        if (u >= (unsigned int)n_lmers) continue;  // padding
        REAL a = weighted ? slot_wt[sw + j] : (REAL)1;
        long long dst = sbase + (long long)tgt * S;
        for (int t = TID; t < n_masks; t += NT) {
            unsigned int x = u ^ masks[t];
            REAL wt = a * weights[t];
            for (OFF e = offsets[x]; e < offsets[x + 1]; e++)
                SCR_ADD(dst + ids[e], weighted ? wt * (REAL)vals[e] : wt);
        }
    }
    BARRIER;
    REAL acc[4] = {0, 0, 0, 0};
    REAL sm[4];
    for (int v = 0; v < 4; v++) sm[v] = inv_dmut[v * LB + it];
    for (int s = TID; s < S; s += NT) {
        REAL r = ref[(long long)b * S + s], r0 = SCR(sbase + s), ds = inv_ds[s];
        REAL kref = r * dx * ds;
        REAL fref = rbf ? exp(gamma * kref) : kref;
        for (int v = 0; v < 4; v++) {
            if (skip[v]) continue;
            REAL k = (r + SCR(sbase + (long long)(1 + v) * S + s) - r0) * sm[v] * ds;
            acc[v] += coef[s] * ((rbf ? exp(gamma * k) : k) - fref);
        }
    }
"""

_MUTANTS_CUDA = (
    r"""
#define SCR_ADD(i, x) atomicAdd(&scratch[i], x)
#define SCR(i) scratch[i]
#define BARRIER __syncthreads()
extern "C" __global__ void lmer_index_mutants(
    const unsigned int* __restrict__ old, const unsigned int* __restrict__ nw,
    const double* __restrict__ ref, const long long* __restrict__ offsets,
    const int* __restrict__ ids, const unsigned char* __restrict__ vals,
    const unsigned int* __restrict__ masks, const double* __restrict__ weights,
    const double* __restrict__ slot_wt, const double* __restrict__ inv_dx,
    const double* __restrict__ inv_dmut, const double* __restrict__ inv_ds,
    const double* __restrict__ coef, double* __restrict__ scratch,
    double* __restrict__ out, long long item0, int L, int B, int l, int S,
    int n_masks, long long n_lmers, int R, int weighted, int rbf, double gamma,
    double floor_)
{
    __shared__ double red[4][128];
"""
    + _MUTANTS_BODY.replace("GROUP", "blockIdx.x")
    .replace("REAL", "double")
    .replace("OFF", "long long")
    .replace("TID", "threadIdx.x")
    .replace("NT", "blockDim.x")
    + r"""
    for (int v = 0; v < 4; v++) red[v][threadIdx.x] = acc[v];
    __syncthreads();
    for (int h = blockDim.x / 2; h > 0; h >>= 1) {
        if (threadIdx.x < h)
            for (int v = 0; v < 4; v++) red[v][threadIdx.x] += red[v][threadIdx.x + h];
        __syncthreads();
    }
    if (threadIdx.x < 4) {
        int v = threadIdx.x;
        out[v * LB + it] = skip[v] ? 0.0 : (rbf ? floor_ : 1.0) * red[v][0];
    }
}
"""
)

_METAL_MUTANTS_HEADER = """
#define SCR_ADD(i, x) atomic_fetch_add_explicit(&scratch[i], x, memory_order_relaxed)
#define SCR(i) atomic_load_explicit(&scratch[i], memory_order_relaxed)
#define BARRIER threadgroup_barrier(mem_flags::mem_device)
"""

_METAL_MUTANTS = (
    """
    long item0 = params[0];
    int L = params[1], B = params[2], l = params[3], S = params[4];
    int n_masks = params[5], n_lmers = params[6], R = params[7];
    int weighted = params[8], rbf = params[9];
    float gamma = fparams[0], floor_ = fparams[1];
    uint tid = thread_position_in_threadgroup.x;
    uint nt = threads_per_threadgroup.x;
    threadgroup float red[4][32];
"""
    + _MUTANTS_BODY.replace("GROUP", "threadgroup_position_in_grid.x")
    .replace("REAL", "float")
    .replace("OFF", "int")
    .replace("TID", "tid")
    .replace("NT", "nt")
    .replace("long long", "long")
    + """
    for (int v = 0; v < 4; v++) {
        float a = simd_sum(acc[v]);
        if (thread_index_in_simdgroup == 0) red[v][simdgroup_index_in_threadgroup] = a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 4) {
        float t = 0.0f;
        for (uint g = 0; g < (nt + 31) / 32; g++) t += red[tid][g];
        float val = skip[tid] ? 0.0f : (rbf ? floor_ : 1.0f) * t;
        atomic_store_explicit(&out[tid * LB + it], val, memory_order_relaxed);
    }
"""
)

# Kernel scoring through the index in Metal (LmerIndex.reduce on MLX): the
# fused counterpart of _query_reduce_cpu. A fixed set of threadgroups (_METAL_
# REDUCE_GROUPS) each owns a length-S float32 scratch slice and walks query
# rows: its threads scatter a row's hits into the slice (as _METAL_SRC, the
# pair weight added directly, so no per-level counts), then reduce sum_s coef_s
# f(K[s] inv_dx inv_ds[s]) over it, zeroing it for the next row. No [rows, S]
# block leaves the GPU, and the live scratch (groups x S) stays small enough
# for the caches. The float32 atomics make sums order-dependent at the float32
# level, as in _METAL_MUTANTS.
_METAL_REDUCE = """
    uint g = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x, nt = threads_per_threadgroup.x;
    uint W = (uint)params[0], n_masks = (uint)params[1], S = (uint)params[2];
    uint n_lmers = (uint)params[3], B = (uint)params[4], G = (uint)params[5];
    int weighted = params[6], rbf = params[8];
    uint wx_stride = (uint)params[7];
    float gamma = fparams[0], floor_ = fparams[1];
    threadgroup float red[32];
    uint sbase = g * S, n_items = W * n_masks;
    for (uint b = g; b < B; b += G) {
        for (uint it = tid; it < n_items; it += nt) {
            uint i = it / n_masks, t = it - i * n_masks;
            uint v = bx[b * W + i] ^ masks[t];
            if (v >= n_lmers) continue;  // padding window
            float wt = weights[t];
            if (weighted) wt *= (float)wx[b * wx_stride + i];
            for (int e = offsets[v]; e < offsets[v + 1]; e++)
                atomic_fetch_add_explicit(&scratch[sbase + ids[e]],
                    weighted ? wt * (float)vals[e] : wt, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_device);
        float dx = inv_dx[b], acc = 0.0f;
        for (uint s = tid; s < S; s += nt) {
            float k = atomic_load_explicit(&scratch[sbase + s], memory_order_relaxed);
            if (k != 0.0f)
                atomic_store_explicit(&scratch[sbase + s], 0.0f, memory_order_relaxed);
            k *= dx * inv_ds[s];
            acc += coef[s] * (rbf ? exp(gamma * k) : k);
        }
        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) red[simdgroup_index_in_threadgroup] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float total = 0.0f;
            for (uint q = 0; q < (nt + 31) / 32; q++) total += red[q];
            atomic_store_explicit(&out[b], (rbf ? floor_ : 1.0f) * total,
                                  memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
"""

# Mutation impact (explain.mutation_impact) in Metal, fused as _METAL_REDUCE and
# as the CPU's _query_impact_cpu: each group's scratch holds three length-S sums
# for one variant: ref's windows where the alleles agree, ref's where they
# differ (all of them past an indel), and alt's where they differ. Each window
# is queried once per allele that differs, so a SNP costs little more than one
# query, and K_raw(ref) = the first two, dK = the third minus the second.
# Then sum_s coef_s dK_s z_s, z_s = inv_dr inv_ds[s], with dK scaled by
# (exp(gamma (k - 1)) - exp(-gamma)) / k of the normalized k (0 when k <= 0)
# under RBF.
_METAL_IMPACT = """
    uint g = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x, nt = threads_per_threadgroup.x;
    uint W = (uint)params[0], n_masks = (uint)params[1], S = (uint)params[2];
    uint n_lmers = (uint)params[3], B = (uint)params[4], G = (uint)params[5];
    int weighted = params[6], rbf = params[8];
    uint wx_stride = (uint)params[7];
    float gamma = fparams[0], floor_ = fparams[1];
    threadgroup float red[32];
    uint sbase = g * 3 * S, n_items = W * n_masks;
    for (uint b = g; b < B; b += G) {
        for (uint it = tid; it < n_items; it += nt) {
            uint i = it / n_masks, t = it - i * n_masks;
            uint r = bxr[b * W + i], a = bxa[b * W + i];
            float ar = weighted ? (float)wxr[b * wx_stride + i] : 1.0f;
            float aa = weighted ? (float)wxa[b * wx_stride + i] : 1.0f;
            bool diff = r != a || ar != aa;
            float wt = weights[t];
            for (uint pass = 0; pass < (diff ? 2u : 1u); pass++) {
                uint v = (pass ? a : r) ^ masks[t];
                if (v >= n_lmers) continue;  // padding window
                float w = wt * (pass ? aa : ar);
                uint base = sbase + (pass ? 2 * S : diff ? S : 0);
                for (int e = offsets[v]; e < offsets[v + 1]; e++)
                    atomic_fetch_add_explicit(&scratch[base + ids[e]],
                        weighted ? w * (float)vals[e] : w, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
        float dr = inv_dr[b], acc = 0.0f;
        for (uint s = tid; s < S; s += nt) {
            float same = atomic_load_explicit(&scratch[sbase + s],
                                              memory_order_relaxed);
            float rd = atomic_load_explicit(&scratch[sbase + S + s],
                                            memory_order_relaxed);
            float ad = atomic_load_explicit(&scratch[sbase + 2 * S + s],
                                            memory_order_relaxed);
            if (same != 0.0f)
                atomic_store_explicit(&scratch[sbase + s], 0.0f, memory_order_relaxed);
            if (rd != 0.0f || ad != 0.0f) {
                atomic_store_explicit(&scratch[sbase + S + s], 0.0f,
                                      memory_order_relaxed);
                atomic_store_explicit(&scratch[sbase + 2 * S + s], 0.0f,
                                      memory_order_relaxed);
                float k = same + rd, dk = ad - rd, z = dr * inv_ds[s];
                dk *= z;
                if (rbf) {
                    k *= z;
                    dk = k > 0.0f ? dk * (exp(gamma * (k - 1.0f)) - floor_) / k : 0.0f;
                }
                acc += coef[s] * dk;
            }
        }
        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) red[simdgroup_index_in_threadgroup] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float total = 0.0f;
            for (uint q = 0; q < (nt + 31) / 32; q++) total += red[q];
            atomic_store_explicit(&out[b], total, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
"""


@cache
def _cuda_fn(source, name):
    import cupy as cp

    return cp.RawKernel(source, name)


@cache
def _metal_query(weighted: bool):
    """The Metal query kernel (_METAL_SRC, or _METAL_SRC_W with window weights)."""
    import mlx.core as mx

    if weighted:
        return mx.fast.metal_kernel(
            name="gkm_lmer_index_query_w",
            input_names=["bx", "wx", "offsets", "ids", "vals", "masks", "levels"]
            + ["params"],
            output_names=["lo", "hi"],
            source=_METAL_SRC_W,
            atomic_outputs=True,
        )
    return mx.fast.metal_kernel(
        name="gkm_lmer_index_query",
        input_names=["bx", "offsets", "ids", "masks", "levels", "params"],
        output_names=["counts"],
        source=_METAL_SRC,
        atomic_outputs=True,
    )


@cache
def _metal_reduce():
    """The fused Metal query + reduction (_METAL_REDUCE)."""
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gkm_lmer_index_reduce",
        input_names=["bx", "wx", "offsets", "ids", "vals", "masks", "weights"]
        + ["inv_dx", "inv_ds", "coef", "params", "fparams"],
        output_names=["scratch", "out"],
        source=_METAL_REDUCE,
        atomic_outputs=True,
    )


@cache
def _metal_impact():
    """The fused Metal mutation impact (_METAL_IMPACT)."""
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gkm_lmer_index_impact",
        input_names=["bxr", "bxa", "wxr", "wxa", "offsets", "ids", "vals", "masks"]
        + ["weights", "inv_dr", "inv_ds", "coef", "params", "fparams"],
        output_names=["scratch", "out"],
        source=_METAL_IMPACT,
        atomic_outputs=True,
    )


@cache
def _metal_mutants():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gkm_lmer_index_mutants",
        input_names=["old", "nw", "ref", "offsets", "ids", "vals", "masks"]
        + ["weights", "slot_wt", "inv_dx", "inv_dmut", "inv_ds", "coef"]
        + ["params", "fparams"],
        output_names=["scratch", "out"],
        source=_METAL_MUTANTS,
        header=_METAL_MUTANTS_HEADER,
        atomic_outputs=True,
    )
