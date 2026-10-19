"""Cached l-mer index over a fixed set of sequences (a model's SVs).

The masked join (kernels/masked.py) indexes the column sequences on every
call, which costs as much for 16 queries as for 1,000. When the columns are
fixed, index them once instead: CSR over all 4^l l-mers, row v listing the
column sequence of every window (fwd and RC) equal to v. A query window w
then visits each l-mer within dmax = l - min_matches mismatches (w XOR a
precomputed mask) and adds table[m] to every column in that row. Each
matching window pair is visited once, and a call's cost depends only on the
queries.

Applies to the count-table kernels (-t 0-5; -t 3/-t 5 through their base),
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
    for t in range(masks.shape[0]):
        v = np.int64(w ^ masks[t])
        if v >= n_lmers:  # padding window
            continue
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
    query windows' weights (empty: all 1)."""
    B, W = bx.shape
    part = np.zeros((B * n_split, S), dtype=np.float64)
    step = (W + n_split - 1) // n_split
    for task in prange(B * n_split):
        b, c = task // n_split, task % n_split
        acc = part[task]
        for i in range(c * step, min(W, (c + 1) * step)):
            scale = wx[i] if wx.shape[0] > 0 else 1.0
            _add_window(acc, bx[b, i], scale, offsets, ids, vals, masks, weights, n_lmers)
    if n_split == 1:
        return part
    return part.reshape(B, n_split, S).sum(axis=1)


@njit(parallel=True, cache=True, fastmath={"nsz", "arcp", "contract", "reassoc"})
def _query_reduce_cpu(bx, offsets, ids, vals, masks, weights, n_lmers, wx, inv_dx, inv_ds,
                      coef, gamma, n_tasks):
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
            for i in range(W):
                a = wx[i] if wx.shape[0] > 0 else 1.0
                _add_window(acc, bx[b, i], a, offsets, ids, vals, masks, weights, n_lmers)
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


@njit(inline="always")
def _add_hits(acc, row, sign, offsets, ids, masks, weights, n_lmers):
    """acc[s] += sign * K_raw(row, s) for the packed windows in row."""
    for i in range(row.shape[0]):
        w = row[i]
        for t in range(masks.shape[0]):
            v = np.int64(w ^ masks[t])
            if v >= n_lmers:  # padding window
                continue
            wt = sign * weights[t]
            for e in range(offsets[v], offsets[v + 1]):
                acc[ids[e]] += wt


@njit(parallel=True, cache=True, fastmath={"nsz", "arcp", "contract", "reassoc"})
def _query_mutants_cpu(old, new, ref, ref_score, offsets, ids, masks, weights, n_lmers,
                       inv_dx, inv_dmut, inv_ds, coef, gamma, n_tasks):
    """[4, L, B] ISM deltas: f(mutant) - ref_score[b], with
    f = sum_s coef_s f(K_raw(mutant, s) inv_dmut inv_ds[s]) as in _query_reduce_cpu.

    K_raw(mutant, s) = ref[b, s] + K_raw(new[v, p, b], s) - K_raw(old[p, b], s): old
    [L, B, l] and new [4, L, B, l] hold the windows covering position p. A base
    with the reference's windows and norm (the reference base) gets 0; an N
    replaced by A keeps the forward windows (N packs as A) but not the norm.
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
            _add_hits(d_old, old[p, b], -1.0, offsets, ids, masks, weights, n_lmers)
            for v in range(4):
                # Window 0 always covers p
                if new[v, p, b, 0] == old[p, b, 0] and inv_dmut[v, p, b] == inv_dx[b]:
                    continue
                tmp[:] = d_old
                _add_hits(tmp, new[v, p, b], 1.0, offsets, ids, masks, weights, n_lmers)
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
    const unsigned char* __restrict__ wx, const unsigned char* __restrict__ vals,
    int weighted)
{
    // grid.y = query row; each thread takes one (window, neighbor) item.
    // wx, vals: window weights of the queries and entries (-t 4/5, weighted = 1)
    long long it = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (it >= n_items) return;
    int b = blockIdx.y;
    int i = (int)(it / n_masks), t = (int)(it - (long long)i * n_masks);
    long long v = (long long)(bx[(long long)b * W + i] ^ masks[t]);
    if (v >= n_lmers) return;  // padding window
    double wt = weights[t];
    double* row = out + (long long)b * S;
    if (weighted) {
        wt *= wx[i];
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
_cuda_fn = None
_metal_fn = None
# Rows per Metal launch: caps the int32 [levels, rows, S] counts
_METAL_COUNT_BYTES = 256 * 2**20
# Above this many distinct pair weights, MLX queries on CPU
_METAL_MAX_LEVELS = 16


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
        counts = np.bincount(keys, minlength=self.n_lmers)
        self.offsets = np.zeros(self.n_lmers + 1, dtype=np.int64)
        np.cumsum(counts, out=self.offsets[1:])
        self.ids = _counting_sort(keys, ids, self.offsets)
        # Window weights (-t 4/5): each entry's, and the query windows' by length
        self.window_weights = kernel.window_weights
        wy = kernel.window_weights(W)
        self.vals = np.zeros(0, dtype=np.uint8)
        if wy is not None:  # an RC window carries its forward weight
            vals = np.concatenate([np.tile(w, self.S) for w in (wy, wy[::-1])][:len(blocks)])
            self.vals = _counting_sort(keys, vals, self.offsets)
        self._dev = {}

    def _wx(self, W):
        """Query window weights as float64 (empty when unweighted)."""
        wx = self.window_weights(W)
        return np.zeros(0) if wx is None else wx.astype(np.float64)

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
            return _query_cpu(bx, self.offsets, self.ids, self.vals, self.masks, self.weights,
                              self.n_lmers, self.S, max(1, n_split), self._wx(bx.shape[1]))
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
        return _query_reduce_cpu(bx, self.offsets, self.ids, self.vals, self.masks, self.weights,
                                 self.n_lmers, self._wx(bx.shape[1]),
                                 np.ascontiguousarray(inv_dx, dtype=np.float64),
                                 np.ascontiguousarray(inv_ds, dtype=np.float64),
                                 np.ascontiguousarray(coef, dtype=np.float64), float(gamma),
                                 max(1, min(bx.shape[0], 4 * numba.get_num_threads())))

    def mutants(self, old, new, ref, ref_score, inv_dx, inv_dmut, inv_ds, coef, gamma=np.nan):
        """ISM deltas [4, L, B] on CPU (see _query_mutants_cpu), without [rows, S] blocks.

        Not for weighted windows (old/new are window sets, not sequences).
        """
        import numba

        if len(self.vals):
            raise NotImplementedError("ISM scores weighted kernels' mutants through model()")

        L, B = old.shape[:2]
        def f64(a):
            return np.ascontiguousarray(a, dtype=np.float64)

        return _query_mutants_cpu(
            np.ascontiguousarray(old), np.ascontiguousarray(new), f64(ref), f64(ref_score),
            self.offsets, self.ids, self.masks, self.weights, self.n_lmers, f64(inv_dx),
            f64(inv_dmut),
            f64(inv_ds), f64(coef), float(gamma),
            max(1, min(L * B, 4 * numba.get_num_threads())))

    def _arrays(self, dev):
        if dev not in self._dev:
            if dev == "cuda":
                import cupy as cp

                self._dev[dev] = (cp.asarray(self.offsets), cp.asarray(self.ids),
                                  cp.asarray(self.masks), cp.asarray(self.weights),
                                  cp.asarray(self.vals) if len(self.vals) else None)
            else:
                import mlx.core as mx

                self._dev[dev] = (mx.array(self.offsets.astype(np.int32)), mx.array(self.ids),
                                  mx.array(self.masks), mx.array(self.levels))
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
        if weighted:
            wx = cp.asarray(self.window_weights(W))
        else:
            wx = vals = cp.zeros(1, dtype=cp.uint8)
        out = cp.zeros((B, self.S), dtype=cp.float64)
        n_items = W * len(self.masks)
        block = 256
        for b0 in range(0, B, 65535):  # grid.y limit
            nb = min(65535, B - b0)
            _cuda_fn(((n_items + block - 1) // block, nb), (block,),
                     (bx[b0:b0 + nb], np.int32(W), np.int64(n_items), offsets, ids, masks,
                      weights, np.int32(len(self.masks)), np.int64(self.n_lmers),
                      np.int32(self.S), out[b0:b0 + nb], wx, vals, np.int32(weighted)))
        return out

    def _pairwise_mlx(self, bx):
        import mlx.core as mx

        global _metal_fn
        if _metal_fn is None:
            _metal_fn = mx.fast.metal_kernel(
                name="gkm_lmer_index_query",
                input_names=["bx", "offsets", "ids", "masks", "levels", "params"],
                output_names=["counts"], source=_METAL_SRC, atomic_outputs=True)
        if len(self.table) > _METAL_MAX_LEVELS or len(self.vals):
            # One int32 [rows, S] block per level; many levels, or window
            # weights (counts no longer per level), query on CPU, which shares
            # memory with the GPU
            return mx.array(self.pairwise(np.asarray(to_cpu(bx))).astype(np.float32))
        if self.offsets[-1] >= 2**31:
            raise ValueError("l-mer index too large for the Metal kernel")
        offsets, ids, masks, levels = self._arrays("mlx")
        bx_np = np.ascontiguousarray(to_cpu(bx), dtype=np.uint32)
        B, W = bx_np.shape
        n_lev = len(self.table)
        rows = max(1, _METAL_COUNT_BYTES // (4 * n_lev * self.S))
        out = np.zeros((B, self.S), dtype=np.float64)
        n_items = W * len(self.masks)
        for b0 in range(0, B, rows):
            nb = min(rows, B - b0)
            params = mx.array([W, len(self.masks), self.S, self.n_lmers, nb], dtype=mx.int32)
            counts = _metal_fn(
                inputs=[mx.array(bx_np[b0:b0 + nb].ravel()), offsets, ids, masks, levels, params],
                template=[], grid=((n_items + 255) // 256 * 256, nb, 1), threadgroup=(256, 1, 1),
                output_shapes=[(n_lev * nb * self.S,)], output_dtypes=[mx.int32],
                init_value=0)[0]
            counts = np.array(counts).reshape(n_lev, nb, self.S)
            for m in range(n_lev):
                out[b0:b0 + nb] += self.table[m] * counts[m]
        return mx.array(out.astype(np.float32))
