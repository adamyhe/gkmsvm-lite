"""Cached l-mer index over a fixed set of sequences (a model's SVs).

The masked join (kernels/masked.py) indexes the column sequences on every
call, which costs as much for 16 queries as for 1,000. When the columns are
fixed, index them once instead: CSR over all 4^l l-mers, row v listing the
column sequence of every window (fwd and RC) equal to v. A query window w
then visits each l-mer within dmax = l - min_matches mismatches (w XOR a
precomputed mask) and adds table[m] to every column in that row. Each
matching window pair is visited once, and a call's cost depends only on the
queries.

Applies to the count-table kernels (-t 0-6; the RBF kernels through their base),
where a neighbor's weight depends on its mismatch count. With window weights
(-t 4/-t 5) every CSR entry also holds its window's weight, and a query
window's hits are scaled by its own. The neighborhood
sum_{m <= dmax} C(l, m) 3^m must be at most MAX_NEIGHBORS (e.g. 4,984 for
l=11 d=3), or DENSE_MAX_NEIGHBORS when the columns are dense enough that
lookups return many hits (see applies()), with l <= MAX_L.
"""

from __future__ import annotations

from itertools import combinations, product

import numpy as np
from numba import njit, prange

from gkmsvm.backend import to_cpu

# Neighbors per query window above which the masked join is used instead
MAX_NEIGHBORS = 8192
# Larger neighborhoods (e.g. l=11 d=4, 25,774) when each neighbor
# lookup returns many hits: the index is then bound by the matching window
# pairs, as any exact per-column method is. Measured on 300 bp queries,
# wgkmrbf: tie with the join at 0.6-2 column windows per l-mer (4k-16k SVs),
# 15 vs 4 seq/s at 10 (72k SVs); SMO columns (one query row each, ~0.3 per
# l-mer at N=2,000) were 4-8x slower than the join, so they keep MAX_NEIGHBORS
DENSE_MAX_NEIGHBORS = 32768
DENSE_WINDOWS_PER_LMER = 4
# 4^12 + 1 int64 offsets = 128 MB
MAX_L = 12


def neighbor_count(l: int, dmax: int) -> int:
    from math import comb

    return sum(comb(l, m) * 3**m for m in range(dmax + 1))


def neighbor_masks(l: int, table: np.ndarray, dmax: int):
    """XOR masks for every l-mer within dmax mismatches, and their mismatch counts.

    Sorted by mask, so a window's neighbors u ^ mask share high bits in runs and
    their offsets lookups stay near each other (1.5x faster queries at 25,774
    neighbors and 72k SVs).
    """
    masks, levels = [], []
    for m in range(dmax + 1):
        if table[m] == 0:
            continue
        for pos in combinations(range(l), m):
            for deltas in product((1, 2, 3), repeat=m):
                x = 0
                for p, dlt in zip(pos, deltas):
                    x |= dlt << (2 * p)
                masks.append(x)
                levels.append(m)
    masks, levels = np.array(masks, dtype=np.uint32), np.array(levels, dtype=np.int32)
    order = np.argsort(masks, kind="stable")
    return masks[order], levels[order]


def _dmax(kernel) -> int | None:
    """Most mismatches with a non-zero pair weight, or None for kernels without an index."""
    if getattr(kernel, "_mismatch_table", None) is not None:
        return kernel.l - kernel._min_matches
    return None


def applies(kernel, n_windows: int | None = None) -> bool:
    """Whether the index can serve this (base) kernel.

    n_windows: windows to be indexed (columns x window count, fwd + RC); with
    at least DENSE_WINDOWS_PER_LMER per l-mer, neighborhoods up to
    DENSE_MAX_NEIGHBORS qualify.
    """
    dmax = _dmax(kernel)
    if dmax is None or kernel.l > MAX_L:
        return False
    count = neighbor_count(kernel.l, dmax)
    dense = n_windows is not None and n_windows >= DENSE_WINDOWS_PER_LMER * 4**kernel.l
    return count <= (DENSE_MAX_NEIGHBORS if dense else MAX_NEIGHBORS)


def index_neighbors(kernel):
    """XOR masks of every l-mer within the kernel's cutoff and their pair
    weights (by mismatch count)."""
    table = np.asarray(kernel._mismatch_table, dtype=np.float64)
    masks, levels = neighbor_masks(kernel.l, table, _dmax(kernel))
    return masks, table[levels]


@njit(cache=True)
def _counting_sort(keys, ids, offsets):
    """ids reordered by key (stable), given the key offsets."""
    fill = offsets[:-1].copy()
    out = np.empty_like(ids)
    for i in range(keys.shape[0]):
        k = keys[i]
        out[fill[k]] = ids[i]
        fill[k] += 1
    return out


@njit(inline="always")
def _add_window(acc, w, scale, offsets, ids, vals, masks, weights, n_lmers):
    """acc[s] += scale * K_raw(window w, s); vals: the entries' window weights
    (empty: all 1)."""
    weighted = vals.shape[0] > 0
    if np.int64(w) >= n_lmers:  # padding window
        return
    for t in range(masks.shape[0]):
        v = np.int64(w ^ masks[t])
        wt = scale * weights[t]
        if weighted:
            for e in range(offsets[v], offsets[v + 1]):
                acc[ids[e]] += wt * vals[e]
        else:
            for e in range(offsets[v], offsets[v + 1]):
                acc[ids[e]] += wt


@njit(parallel=True, cache=True)
def _query_cpu(bx, offsets, ids, vals, masks, weights, n_lmers, S, n_split, wx):
    """Each row's windows in n_split chunks (so few rows still use every
    thread), each accumulating into its own row of partial sums. wx: the
    query windows' weights, [B or 1, W] ([1, 0]: all 1)."""
    B, W = bx.shape
    part = np.zeros((B * n_split, S), dtype=np.float64)
    step = (W + n_split - 1) // n_split
    for task in prange(B * n_split):
        b, c = task // n_split, task % n_split
        acc = part[task]
        wb = wx[min(np.int64(b), wx.shape[0] - 1)]
        for i in range(c * step, min(W, (c + 1) * step)):
            scale = wb[i] if wb.shape[0] > 0 else 1.0
            _add_window(
                acc, bx[b, i], scale, offsets, ids, vals, masks, weights, n_lmers
            )
    if n_split == 1:
        return part
    return part.reshape(B, n_split, S).sum(axis=1)


@njit(parallel=True, cache=True, fastmath={"nsz", "arcp", "contract", "reassoc"})
def _query_reduce_cpu(
    bx,
    offsets,
    ids,
    vals,
    masks,
    weights,
    n_lmers,
    wx,
    inv_dx,
    inv_ds,
    coef,
    gamma,
    n_tasks,
):
    """[B]: sum_s coef_s f(K_raw[b, s] * inv_dx[b] * inv_ds[s]), without the [B, S] block.

    f(k) = exp(gamma (k - 1)) when gamma is not NaN (RBF), else k. Each task
    accumulates one row's hits in a length-S buffer that stays in cache and is
    zeroed as it is reduced.
    """
    B, W = bx.shape
    S = inv_ds.shape[0]
    rbf = not np.isnan(gamma)
    floor = np.exp(-gamma) if rbf else 1.0
    out = np.zeros(B)
    for task in prange(n_tasks):
        acc = np.zeros(S)
        for b in range(task, B, n_tasks):
            wb = wx[min(np.int64(b), wx.shape[0] - 1)]
            for i in range(W):
                a = wb[i] if wb.shape[0] > 0 else 1.0
                _add_window(
                    acc, bx[b, i], a, offsets, ids, vals, masks, weights, n_lmers
                )
            scale = inv_dx[b]
            total = 0.0
            if rbf:
                g = gamma * scale
                for j in range(S):
                    total += coef[j] * np.exp(g * acc[j] * inv_ds[j])
                    acc[j] = 0.0
                total *= floor
            else:
                for j in range(S):
                    total += coef[j] * acc[j] * inv_ds[j]
                    acc[j] = 0.0
                total *= scale
            out[b] = total
    return out


@njit(parallel=True, cache=True, fastmath={"nsz", "arcp", "contract", "reassoc"})
def _query_impact_cpu(
    bx,
    offsets,
    ids,
    vals,
    masks,
    weights,
    n_lmers,
    wx,
    inv_dr,
    inv_ds,
    coef,
    gamma,
    n_tasks,
):
    """[n] mutation impact (explain.mutation_impact) of rows bx = [ref; alt].

    Each task accumulates K_raw(ref) and K_raw(alt) - K_raw(ref) in two length-S
    buffers; the difference visits only window slots where the alleles differ
    (all of them past an indel), so a SNP costs about one query. With gamma
    (not NaN) each SV's dK scales by (exp(gamma (k - 1)) - exp(-gamma)) / k of
    its normalized k (0 when k <= 0), else by 1.
    """
    W = bx.shape[1]
    n = bx.shape[0] // 2
    S = inv_ds.shape[0]
    rbf = not np.isnan(gamma)
    floor = np.exp(-gamma) if rbf else 0.0
    out = np.zeros(n)
    for task in prange(n_tasks):
        acc = np.zeros(S)
        delta = np.zeros(S)
        for b in range(task, n, n_tasks):
            wr = wx[min(np.int64(b), wx.shape[0] - 1)]
            wa = wx[min(np.int64(n + b), wx.shape[0] - 1)]
            for i in range(W):
                r = bx[b, i]
                a = bx[n + b, i]
                ar = wr[i] if wr.shape[0] > 0 else 1.0
                aa = wa[i] if wa.shape[0] > 0 else 1.0
                _add_window(acc, r, ar, offsets, ids, vals, masks, weights, n_lmers)
                if r != a or ar != aa:
                    _add_window(
                        delta, r, -ar, offsets, ids, vals, masks, weights, n_lmers
                    )
                    _add_window(
                        delta, a, aa, offsets, ids, vals, masks, weights, n_lmers
                    )
            scale = inv_dr[b]
            total = 0.0
            for j in range(S):
                dk = delta[j]
                if dk != 0.0:
                    z = scale * inv_ds[j]
                    dk *= z
                    if rbf:
                        k = acc[j] * z
                        dk = (
                            dk * (np.exp(gamma * (k - 1.0)) - floor) / k
                            if k > 0
                            else 0.0
                        )
                    total += coef[j] * dk
                    delta[j] = 0.0
                acc[j] = 0.0
            out[b] = total
    return out


@njit(inline="always")
def _add_hits(acc, row, sign, rowwt, offsets, ids, vals, masks, weights, n_lmers):
    """acc[s] += sign * K_raw(row, s) for the packed windows in row; rowwt: their
    window weights (empty: all 1)."""
    for i in range(row.shape[0]):
        scale = sign * rowwt[i] if rowwt.shape[0] > 0 else sign
        _add_window(acc, row[i], scale, offsets, ids, vals, masks, weights, n_lmers)


@njit(parallel=True, cache=True, fastmath={"nsz", "arcp", "contract", "reassoc"})
def _query_mutants_cpu(
    old,
    new,
    ref,
    ref_score,
    offsets,
    ids,
    vals,
    masks,
    weights,
    n_lmers,
    slot_wt,
    inv_dx,
    inv_dmut,
    inv_ds,
    coef,
    gamma,
    n_tasks,
):
    """[4, L, B] ISM deltas: f(mutant) - ref_score[b], with
    f = sum_s coef_s f(K_raw(mutant, s) inv_dmut inv_ds[s]) as in _query_reduce_cpu.

    K_raw(mutant, s) = ref[b, s] + K_raw(new[v, p, b], s) - K_raw(old[p, b], s): old
    [L, B, l] and new [4, L, B, l] hold the windows covering position p. A base
    with the reference's windows and norm (the reference base) gets 0; an N
    replaced by A keeps the forward windows (N packs as A) but not the norm.
    slot_wt [L, 1 or B, l]: the window weights of those windows (0 for padding;
    [L, 1, 0] when unweighted).
    """
    L, B = old.shape[0], old.shape[1]
    S = inv_ds.shape[0]
    rbf = not np.isnan(gamma)
    floor = np.exp(-gamma) if rbf else 1.0
    out = np.zeros((4, L, B))
    for task in prange(n_tasks):
        d_old = np.zeros(S)
        tmp = np.empty(S)
        for item in range(task, L * B, n_tasks):
            p, b = item // B, item % B
            rowwt = slot_wt[p, min(np.int64(b), slot_wt.shape[1] - 1)]
            _add_hits(
                d_old,
                old[p, b],
                -1.0,
                rowwt,
                offsets,
                ids,
                vals,
                masks,
                weights,
                n_lmers,
            )
            for v in range(4):
                # Window 0 always covers p
                if new[v, p, b, 0] == old[p, b, 0] and inv_dmut[v, p, b] == inv_dx[b]:
                    continue
                tmp[:] = d_old
                _add_hits(
                    tmp,
                    new[v, p, b],
                    1.0,
                    rowwt,
                    offsets,
                    ids,
                    vals,
                    masks,
                    weights,
                    n_lmers,
                )
                scale = inv_dmut[v, p, b]
                total = 0.0
                if rbf:
                    g = gamma * scale
                    for j in range(S):
                        total += coef[j] * np.exp(g * (ref[b, j] + tmp[j]) * inv_ds[j])
                    total *= floor
                else:
                    for j in range(S):
                        total += coef[j] * (ref[b, j] + tmp[j]) * inv_ds[j]
                    total *= scale
                out[v, p, b] = total - ref_score[b]
            d_old[:] = 0.0
    return out


_CUDA_SRC = r"""
extern "C" __global__ void lmer_index_query(
    const unsigned int* __restrict__ bx, int W, long long n_items,
    const long long* __restrict__ offsets, const int* __restrict__ ids,
    const unsigned int* __restrict__ masks, const double* __restrict__ weights,
    int n_masks, long long n_lmers, int S, double* __restrict__ out,
    const unsigned char* __restrict__ wx, int wx_stride,
    const unsigned char* __restrict__ vals, int weighted)
{
    // grid.y = query row; each thread takes one (window, neighbor) item.
    // wx [rows or 1, W], vals: window weights of the queries (wx_stride 0 when
    // the rows share one) and entries (-t 4/5, weighted = 1)
    long long it = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (it >= n_items) return;
    int b = blockIdx.y;
    int i = (int)(it / n_masks), t = (int)(it - (long long)i * n_masks);
    long long v = (long long)(bx[(long long)b * W + i] ^ masks[t]);
    if (v >= n_lmers) return;  // padding window
    double wt = weights[t];
    double* row = out + (long long)b * S;
    if (weighted) {
        wt *= wx[(long long)b * wx_stride + i];
        for (long long e = offsets[v]; e < offsets[v + 1]; e++)
            atomicAdd(&row[ids[e]], wt * vals[e]);
        return;
    }
    for (long long e = offsets[v]; e < offsets[v + 1]; e++) atomicAdd(&row[ids[e]], wt);
}
"""

# Metal has no float64: count hits per mismatch level in int32 and combine
# levels in float64 on the CPU, so results are exact
_METAL_SRC = """
    uint it = thread_position_in_grid.x;
    uint b = thread_position_in_grid.y;
    uint W = (uint)params[0], n_masks = (uint)params[1], S = (uint)params[2];
    uint n_lmers = (uint)params[3], B = (uint)params[4];
    if (it >= W * n_masks) return;
    uint i = it / n_masks, t = it - i * n_masks;
    uint v = bx[b * W + i] ^ masks[t];
    if (v >= n_lmers) return;
    uint base = ((uint)levels[t] * B + b) * S;
    for (int e = offsets[v]; e < offsets[v + 1]; e++)
        atomic_fetch_add_explicit(&counts[base + ids[e]], 1, memory_order_relaxed);
"""
# With window weights (-t 4/5): each hit adds wx[b, i] * vals[e], exact in 64
# bits as a low uint32 word plus a carry word (bumped by the thread whose add
# wraps the low word)
_METAL_SRC_W = """
    uint it = thread_position_in_grid.x;
    uint b = thread_position_in_grid.y;
    uint W = (uint)params[0], n_masks = (uint)params[1], S = (uint)params[2];
    uint n_lmers = (uint)params[3], B = (uint)params[4], wx_stride = (uint)params[5];
    if (it >= W * n_masks) return;
    uint i = it / n_masks, t = it - i * n_masks;
    uint v = bx[b * W + i] ^ masks[t];
    if (v >= n_lmers) return;
    uint a = (uint)wx[b * wx_stride + i];
    uint base = ((uint)levels[t] * B + b) * S;
    for (int e = offsets[v]; e < offsets[v + 1]; e++) {
        uint c = a * (uint)vals[e];
        uint old = atomic_fetch_add_explicit(&lo[base + ids[e]], c, memory_order_relaxed);
        if (old + c < old)
            atomic_fetch_add_explicit(&hi[base + ids[e]], 1u, memory_order_relaxed);
    }
"""
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
_MUTANTS_THREADS = 128
# Scratch per launch: [items, 5, S]
_MUTANTS_SCRATCH_BYTES = 256 * 2**20
# Above this many index entries per l-mer, MLX runs ISM's index pass on CPU:
# the scatter is device-memory float atomics, which an M4 does at ~1G/s (RBF,
# 4 x 300 bp: 2.5x the CPU at 1.1 entries per l-mer, 1.65x at 4.4, 0.83x at
# 8.9; the dsQTL -t 3 model has 18)
_METAL_MUTANTS_MAX_DENSITY = 6.0

_cuda_fn = None
_cuda_mutants = None
_metal_fn = None
_metal_fn_w = None
_metal_mutants = None
# Rows per Metal launch: caps the int32 [levels, rows, S] counts
_METAL_COUNT_BYTES = 256 * 2**20
# Above this many distinct pair weights, MLX queries on CPU
_METAL_MAX_LEVELS = 16
# ... and above this many index entries per l-mer: the [levels, rows, S]
# counts' scattered atomics lose to the CPU (200 x 300 bp queries, l = 10, on
# an M4: 2.1x the CPU at 0.56 entries per l-mer, 1.08x at 2.2, 0.72x at 4.4,
# 0.52x at 17.8, the dsQTL -t 3 model's)
_METAL_QUERY_MAX_DENSITY = 2.5


class LmerIndex:
    """l-mer -> column index for fixed packed columns (fwd + optional RC)."""

    def __init__(self, kernel, by, by_rc=None):
        l = kernel.l
        self.S = by.shape[0]
        self.n_lmers = 4**l
        masks, weights = index_neighbors(kernel)
        keep = weights != 0
        self.masks, self.weights = masks[keep], weights[keep]
        # Metal counts hits per distinct weight (level) and combines them on the CPU
        self.table, self.levels = np.unique(self.weights, return_inverse=True)
        self.levels = self.levels.astype(np.int32)
        blocks = [to_cpu(by)] + ([to_cpu(by_rc)] if by_rc is not None else [])
        keys = np.concatenate([b.ravel() for b in blocks]).astype(np.int64)
        W = blocks[0].shape[1]
        ids = np.tile(np.repeat(np.arange(self.S, dtype=np.int32), W), len(blocks))
        # Window weights (-t 4/5): each entry's, and the query windows' (row_weights)
        self.row_weights = kernel.row_weights
        vals = None
        if (
            kernel.window_weights(W) is not None
        ):  # an RC window carries its forward weight
            vals = np.concatenate(
                [
                    np.broadcast_to(kernel.row_weights(b, rc=k == 1), b.shape).ravel()
                    for k, b in enumerate(blocks)
                ]
            )
        keep = keys < self.n_lmers  # PAD past shorter sequences' windows
        if not keep.all():
            keys, ids = keys[keep], ids[keep]
            vals = None if vals is None else vals[keep]
        counts = np.bincount(keys, minlength=self.n_lmers)
        self.offsets = np.zeros(self.n_lmers + 1, dtype=np.int64)
        np.cumsum(counts, out=self.offsets[1:])
        self.ids = _counting_sort(keys, ids, self.offsets)
        self.vals = np.zeros(0, dtype=np.uint8)
        if vals is not None:
            self.vals = _counting_sort(keys, np.ascontiguousarray(vals), self.offsets)
        self._dev = {}

    def _wx(self, bx):
        """Query rows' window weights as float64 [B or 1, W] ([1, 0] when
        unweighted)."""
        wx = self.row_weights(bx)
        return np.zeros((1, 0)) if wx is None else wx.astype(np.float64)

    def mutants_on_device(self, dev: str) -> bool:
        """Whether mutants() runs on dev (CUDA, MLX) rather than on CPU."""
        if dev == "mlx":
            return self.offsets[-1] <= _METAL_MUTANTS_MAX_DENSITY * self.n_lmers
        return dev == "cuda"

    def queries_on_device(self, dev: str) -> bool:
        """Whether pairwise() queries on dev (CUDA, MLX) rather than on CPU: MLX
        queries on CPU above _METAL_MAX_LEVELS distinct weights or
        _METAL_QUERY_MAX_DENSITY entries per l-mer."""
        if dev == "mlx":
            return (
                len(self.table) <= _METAL_MAX_LEVELS
                and self.offsets[-1] <= _METAL_QUERY_MAX_DENSITY * self.n_lmers
            )
        return dev == "cuda"

    def nbytes(self) -> int:
        return self.offsets.nbytes + self.ids.nbytes + self.vals.nbytes

    def pairwise(self, bx):
        """K_raw [B, S] of packed query rows bx [B, W] against the columns.

        Rows may contain padding windows (any value >= 4^l), which add nothing.
        CPU/CUDA return float64, MLX float32, on bx's device.
        """
        from gkmsvm.kernels.packed import device_of

        dev = device_of(bx)
        if dev == "cpu":
            import numba

            bx = np.ascontiguousarray(bx)
            n_split = min(bx.shape[1], -(-numba.get_num_threads() // bx.shape[0]))
            return _query_cpu(
                bx,
                self.offsets,
                self.ids,
                self.vals,
                self.masks,
                self.weights,
                self.n_lmers,
                self.S,
                max(1, n_split),
                self._wx(bx),
            )
        if dev == "cuda":
            return self._pairwise_cuda(bx)
        return self._pairwise_mlx(bx)

    def reduce(self, bx, inv_dx, inv_ds, coef, gamma=np.nan):
        """sum_s coef_s f(K_raw(row, s) inv_dx[row] inv_ds[s]) [B] for CPU rows bx.

        f is exp(gamma (k - 1)) when gamma is given (RBF), else identity.
        The same as reducing pairwise(bx), without materializing it.
        """
        import numba

        bx = np.ascontiguousarray(bx)
        return _query_reduce_cpu(
            bx,
            self.offsets,
            self.ids,
            self.vals,
            self.masks,
            self.weights,
            self.n_lmers,
            self._wx(bx),
            np.ascontiguousarray(inv_dx, dtype=np.float64),
            np.ascontiguousarray(inv_ds, dtype=np.float64),
            np.ascontiguousarray(coef, dtype=np.float64),
            float(gamma),
            max(1, min(bx.shape[0], 4 * numba.get_num_threads())),
        )

    def impact(self, bx, inv_dr, inv_ds, coef, gamma=np.nan):
        """Mutation impact [n] of CPU rows bx = [ref (n rows); alt (n rows)]
        (see _query_impact_cpu); inv_dr: 1 / sqrt(K_raw(ref, ref)) [n]."""
        import numba

        bx = np.ascontiguousarray(bx)
        return _query_impact_cpu(
            bx,
            self.offsets,
            self.ids,
            self.vals,
            self.masks,
            self.weights,
            self.n_lmers,
            self._wx(bx),
            np.ascontiguousarray(inv_dr, dtype=np.float64),
            np.ascontiguousarray(inv_ds, dtype=np.float64),
            np.ascontiguousarray(coef, dtype=np.float64),
            float(gamma),
            max(1, min(bx.shape[0] // 2, 4 * numba.get_num_threads())),
        )

    def mutants(
        self,
        old,
        new,
        ref,
        ref_score,
        inv_dx,
        inv_dmut,
        inv_ds,
        coef,
        gamma=np.nan,
        slot_wt=None,
    ):
        """ISM deltas [4, L, B] (see _query_mutants_cpu), without [rows, S] blocks.

        slot_wt [L, 1 or B, l]: with window weights, those of the windows in
        old/new (window sets, not sequences, so their weights can't follow
        from length; ism._slot_weights). ref [B, S] on the device to run on
        (CUDA, MLX: _MUTANTS_BODY; ref_score unused), the rest on CPU.
        """
        import numba

        from gkmsvm.kernels.packed import device_of

        L, B = old.shape[:2]
        if len(self.vals) and slot_wt is None:
            raise ValueError("weighted windows need slot_wt")
        if device_of(ref) != "cpu":
            return self._mutants_gpu(
                old, new, ref, inv_dx, inv_dmut, inv_ds, coef, gamma, slot_wt
            )

        def f64(a):
            return np.ascontiguousarray(a, dtype=np.float64)

        return _query_mutants_cpu(
            np.ascontiguousarray(old),
            np.ascontiguousarray(new),
            f64(ref),
            f64(ref_score),
            self.offsets,
            self.ids,
            self.vals,
            self.masks,
            self.weights,
            self.n_lmers,
            np.zeros((L, 1, 0)) if slot_wt is None else f64(slot_wt),
            f64(inv_dx),
            f64(inv_dmut),
            f64(inv_ds),
            f64(coef),
            float(gamma),
            max(1, min(L * B, 4 * numba.get_num_threads())),
        )

    def _arrays(self, dev):
        if dev not in self._dev:
            if dev == "cuda":
                import cupy as cp

                self._dev[dev] = (
                    cp.asarray(self.offsets),
                    cp.asarray(self.ids),
                    cp.asarray(self.masks),
                    cp.asarray(self.weights),
                    cp.asarray(self.vals) if len(self.vals) else None,
                )
            else:
                import mlx.core as mx

                self._dev[dev] = (
                    mx.array(self.offsets.astype(np.int32)),
                    mx.array(self.ids),
                    mx.array(self.masks),
                    mx.array(self.levels),
                    mx.array(self.vals.astype(np.uint32)) if len(self.vals) else None,
                )
        return self._dev[dev]

    def _pairwise_cuda(self, bx):
        import cupy as cp

        global _cuda_fn
        if _cuda_fn is None:
            _cuda_fn = cp.RawKernel(_CUDA_SRC, "lmer_index_query")
        offsets, ids, masks, weights, vals = self._arrays("cuda")
        bx = cp.ascontiguousarray(bx, dtype=cp.uint32)
        B, W = bx.shape
        weighted = vals is not None
        wx_stride = 0
        if weighted:
            wx_np = np.ascontiguousarray(self.row_weights(bx))
            wx = cp.asarray(wx_np)
            wx_stride = W if wx_np.shape[0] > 1 else 0
        else:
            wx = vals = cp.zeros(1, dtype=cp.uint8)
        out = cp.zeros((B, self.S), dtype=cp.float64)
        n_items = W * len(self.masks)
        block = 256
        for b0 in range(0, B, 65535):  # grid.y limit
            nb = min(65535, B - b0)
            _cuda_fn(
                ((n_items + block - 1) // block, nb),
                (block,),
                (
                    bx[b0 : b0 + nb],
                    np.int32(W),
                    np.int64(n_items),
                    offsets,
                    ids,
                    masks,
                    weights,
                    np.int32(len(self.masks)),
                    np.int64(self.n_lmers),
                    np.int32(self.S),
                    out[b0 : b0 + nb],
                    wx[b0:] if wx_stride else wx,
                    np.int32(wx_stride),
                    vals,
                    np.int32(weighted),
                ),
            )
        return out

    def _pairwise_mlx(self, bx, host=False):
        """pairwise() in Metal; host: return the float64 NumPy result
        (combined from exact counts) instead of float32 on the device."""
        import mlx.core as mx

        global _metal_fn, _metal_fn_w
        if _metal_fn is None:
            _metal_fn = mx.fast.metal_kernel(
                name="gkm_lmer_index_query",
                input_names=["bx", "offsets", "ids", "masks", "levels", "params"],
                output_names=["counts"],
                source=_METAL_SRC,
                atomic_outputs=True,
            )
            _metal_fn_w = mx.fast.metal_kernel(
                name="gkm_lmer_index_query_w",
                input_names=[
                    "bx",
                    "wx",
                    "offsets",
                    "ids",
                    "vals",
                    "masks",
                    "levels",
                    "params",
                ],
                output_names=["lo", "hi"],
                source=_METAL_SRC_W,
                atomic_outputs=True,
            )
        if not self.queries_on_device("mlx"):
            # One int32 [rows, S] block per level; with many levels or a dense
            # index, query on CPU, which shares memory with the GPU
            out = self.pairwise(np.ascontiguousarray(to_cpu(bx)))
            return out if host else mx.array(out.astype(np.float32))
        if self.offsets[-1] >= 2**31:
            raise ValueError("l-mer index too large for the Metal kernel")
        offsets, ids, masks, levels, vals = self._arrays("mlx")
        bx_np = np.ascontiguousarray(to_cpu(bx), dtype=np.uint32)
        B, W = bx_np.shape
        weighted = vals is not None
        if weighted:  # the query windows' weights (row_weights)
            wx_np = np.ascontiguousarray(self.row_weights(bx_np), dtype=np.uint32)
            wx_stride = W if wx_np.shape[0] > 1 else 0
        n_lev = len(self.table)
        words = 2 if weighted else 1
        rows = max(1, _METAL_COUNT_BYTES // (4 * words * n_lev * self.S))
        out = np.zeros((B, self.S), dtype=np.float64)
        n_items = W * len(self.masks)
        for b0 in range(0, B, rows):
            nb = min(rows, B - b0)
            launch = dict(
                template=[],
                grid=((n_items + 255) // 256 * 256, nb, 1),
                threadgroup=(256, 1, 1),
                init_value=0,
            )
            q = mx.array(bx_np[b0 : b0 + nb].ravel())
            if weighted:
                params = mx.array(
                    [W, len(self.masks), self.S, self.n_lmers, nb, wx_stride],
                    dtype=mx.int32,
                )
                wq = wx_np[b0 : b0 + nb] if wx_stride else wx_np
                lo, hi = _metal_fn_w(
                    inputs=[q, mx.array(wq.ravel()), offsets, ids, vals, masks]
                    + [levels, params],
                    output_shapes=[(n_lev * nb * self.S,)] * 2,
                    output_dtypes=[mx.uint32, mx.uint32],
                    **launch,
                )
                counts = (np.array(hi).astype(np.uint64) << np.uint64(32)) | np.array(
                    lo
                ).astype(np.uint64)
            else:
                params = mx.array(
                    [W, len(self.masks), self.S, self.n_lmers, nb], dtype=mx.int32
                )
                counts = _metal_fn(
                    inputs=[q, offsets, ids, masks, levels, params],
                    output_shapes=[(n_lev * nb * self.S,)],
                    output_dtypes=[mx.int32],
                    **launch,
                )[0]
                counts = np.array(counts)
            counts = counts.reshape(n_lev, nb, self.S)
            for m in range(n_lev):
                out[b0 : b0 + nb] += self.table[m] * counts[m].astype(np.float64)
        return out if host else mx.array(out.astype(np.float32))

    def _mutants_gpu(
        self, old, new, ref, inv_dx, inv_dmut, inv_ds, coef, gamma, slot_wt
    ):
        """mutants() on ref's device (CUDA or MLX): [4, L, B] there."""
        from gkmsvm.kernels.packed import device_of

        global _cuda_mutants, _metal_mutants
        L, B, l = old.shape
        S = self.S
        weighted = len(self.vals) > 0
        rbf = not np.isnan(gamma)
        R = 1 if slot_wt is None else slot_wt.shape[1]
        # (a 1-element MLX input would be a constant, not a device pointer)
        slot_wt = np.zeros(2) if slot_wt is None else slot_wt
        n_items = L * B
        per_launch = max(1, _MUTANTS_SCRATCH_BYTES // (5 * S * 8))
        ints = (L, B, l, S, len(self.masks), self.n_lmers, R, int(weighted), int(rbf))
        floats = (float(gamma) if rbf else 0.0, float(np.exp(-gamma)) if rbf else 1.0)
        if device_of(ref) == "cuda":
            import cupy as cp

            if _cuda_mutants is None:
                _cuda_mutants = cp.RawKernel(_MUTANTS_CUDA, "lmer_index_mutants")
            offsets, ids, masks, weights, vals = self._arrays("cuda")
            if vals is None:
                vals = cp.zeros(1, dtype=cp.uint8)

            def f64(a):
                return cp.ascontiguousarray(cp.asarray(a, dtype=cp.float64))

            arrays = (
                cp.asarray(np.ascontiguousarray(old, dtype=np.uint32)),
                cp.asarray(np.ascontiguousarray(new, dtype=np.uint32)),
                f64(ref),
                offsets,
                ids,
                vals,
                masks,
                weights,
                f64(slot_wt),
                f64(inv_dx),
                f64(inv_dmut),
                f64(inv_ds),
                f64(coef),
            )
            out = cp.empty(4 * n_items, dtype=cp.float64)
            scratch = cp.empty(min(per_launch, n_items) * 5 * S, dtype=cp.float64)
            for i0 in range(0, n_items, per_launch):
                n = min(per_launch, n_items - i0)
                scratch[: n * 5 * S] = 0
                _cuda_mutants(
                    (n,),
                    (_MUTANTS_THREADS,),
                    arrays
                    + (scratch, out, np.int64(i0))
                    + tuple(np.int32(a) for a in ints[:5])
                    + (np.int64(ints[5]),)
                    + tuple(np.int32(a) for a in ints[6:])
                    + tuple(np.float64(a) for a in floats),
                )
            return out.reshape(4, L, B)
        import mlx.core as mx

        if self.offsets[-1] >= 2**31:
            raise ValueError("l-mer index too large for the Metal kernel")
        if _metal_mutants is None:
            _metal_mutants = mx.fast.metal_kernel(
                name="gkm_lmer_index_mutants",
                input_names=["old", "nw", "ref", "offsets", "ids", "vals", "masks"]
                + ["weights", "slot_wt", "inv_dx", "inv_dmut", "inv_ds", "coef"]
                + ["params", "fparams"],
                output_names=["scratch", "out"],
                source=_METAL_MUTANTS,
                header=_METAL_MUTANTS_HEADER,
                atomic_outputs=True,
            )
        offsets, ids, masks, _, vals = self._arrays("mlx")
        if "mlx_weights" not in self._dev:
            self._dev["mlx_weights"] = mx.array(self.weights.astype(np.float32))
        vals = mx.zeros(2, dtype=mx.uint32) if vals is None else vals

        def f32(a):
            return mx.array(np.asarray(to_cpu(a), dtype=np.float32))

        inputs = [
            mx.array(np.ascontiguousarray(old, dtype=np.uint32)),
            mx.array(np.ascontiguousarray(new, dtype=np.uint32)),
            ref.astype(mx.float32),
            offsets,
            ids,
            vals,
            masks,
            self._dev["mlx_weights"],
            f32(slot_wt),
            f32(inv_dx),
            f32(inv_dmut),
            f32(inv_ds),
            f32(coef),
        ]
        parts = []
        for i0 in range(0, n_items, per_launch):
            n = min(per_launch, n_items - i0)
            _, part = _metal_mutants(
                inputs=inputs
                + [mx.array((i0,) + ints, dtype=mx.int64).astype(mx.int32)]
                + [mx.array(floats, dtype=mx.float32)],
                template=[],
                grid=(n * _MUTANTS_THREADS, 1, 1),
                threadgroup=(_MUTANTS_THREADS, 1, 1),
                output_shapes=[(n * 5 * S,), (4 * n_items,)],
                output_dtypes=[mx.float32, mx.float32],
                init_value=0,
            )
            parts.append(part)
        # Each launch writes only its items (the rest stay 0)
        out = parts[0]
        for part in parts[1:]:
            out = out + part
        return out.reshape(4, L, B)
