"""Cached l-mer index over a fixed set of sequences (a model's SVs).

The masked join (kernels/masked.py) indexes the column sequences on every
call, which costs as much for 16 queries as for 1,000. When the columns are
fixed, index them once instead: CSR over all 4^l l-mers, row v listing the
column sequence of every window (fwd and RC) equal to v. A query window w
then visits each l-mer within dmax = l - min_matches mismatches (w XOR a
precomputed mask) and adds table[m] to every column in that row. Each
matching window pair is visited once, and a call's cost depends only on the
queries.

Applies to the count-table kernels (-t 0/1/2) whose neighborhood
sum_{m <= dmax} C(l, m) 3^m is at most MAX_NEIGHBORS (e.g. 4,984 for l=11
d=3), with l <= MAX_L.
"""

from __future__ import annotations

from itertools import combinations, product

import numpy as np
from numba import njit, prange

from gkmsvm.backend import to_cpu

# Neighbors per query window above which the masked join is used instead
MAX_NEIGHBORS = 8192
# 4^12 + 1 int64 offsets = 128 MB
MAX_L = 12


def neighbor_count(l: int, dmax: int) -> int:
    from math import comb

    return sum(comb(l, m) * 3**m for m in range(dmax + 1))


def neighbor_masks(l: int, table: np.ndarray, dmax: int):
    """XOR masks for every l-mer within dmax mismatches, and their mismatch counts."""
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
    return np.array(masks, dtype=np.uint32), np.array(levels, dtype=np.int32)


def applies(kernel) -> bool:
    """Whether the index can serve this (base) kernel."""
    table = getattr(kernel, "_mismatch_table", None)
    if table is None or kernel.l > MAX_L:
        return False
    return neighbor_count(kernel.l, kernel.l - kernel._min_matches) <= MAX_NEIGHBORS


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


@njit(parallel=True, cache=True)
def _query_cpu(bx, offsets, ids, masks, weights, n_lmers, S, n_split):
    """Each row's windows in n_split chunks (so few rows still use every
    thread), each accumulating into its own row of partial sums."""
    B, W = bx.shape
    part = np.zeros((B * n_split, S), dtype=np.float64)
    step = (W + n_split - 1) // n_split
    for task in prange(B * n_split):
        b, c = task // n_split, task % n_split
        acc = part[task]
        for i in range(c * step, min(W, (c + 1) * step)):
            w = bx[b, i]
            for t in range(masks.shape[0]):
                v = np.int64(w ^ masks[t])
                if v >= n_lmers:  # padding window
                    continue
                wt = weights[t]
                for e in range(offsets[v], offsets[v + 1]):
                    acc[ids[e]] += wt
    if n_split == 1:
        return part
    return part.reshape(B, n_split, S).sum(axis=1)


@njit(parallel=True, cache=True, fastmath={"nsz", "arcp", "contract", "reassoc"})
def _query_reduce_cpu(bx, offsets, ids, masks, weights, n_lmers, inv_dx, inv_ds, coef,
                      gamma, n_tasks):
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
                w = bx[b, i]
                for t in range(masks.shape[0]):
                    v = np.int64(w ^ masks[t])
                    if v >= n_lmers:  # padding window
                        continue
                    wt = weights[t]
                    for e in range(offsets[v], offsets[v + 1]):
                        acc[ids[e]] += wt
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


_CUDA_SRC = r"""
extern "C" __global__ void lmer_index_query(
    const unsigned int* __restrict__ bx, int W, long long n_items,
    const long long* __restrict__ offsets, const int* __restrict__ ids,
    const unsigned int* __restrict__ masks, const double* __restrict__ weights,
    int n_masks, long long n_lmers, int S, double* __restrict__ out)
{
    // grid.y = query row; each thread takes one (window, neighbor) item
    long long it = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (it >= n_items) return;
    int b = blockIdx.y;
    int i = (int)(it / n_masks), t = (int)(it - (long long)i * n_masks);
    long long v = (long long)(bx[(long long)b * W + i] ^ masks[t]);
    if (v >= n_lmers) return;  // padding window
    double wt = weights[t];
    double* row = out + (long long)b * S;
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


class LmerIndex:
    """l-mer -> column index for fixed packed columns (fwd + optional RC)."""

    def __init__(self, kernel, by, by_rc=None):
        l = kernel.l
        self.S = by.shape[0]
        self.n_lmers = 4**l
        table = np.asarray(kernel._mismatch_table, dtype=np.float64)
        self.masks, self.levels = neighbor_masks(l, table, l - kernel._min_matches)
        self.weights = table[self.levels]
        self.level_values = np.unique(self.levels)
        self.table = table
        blocks = [to_cpu(by)] + ([to_cpu(by_rc)] if by_rc is not None else [])
        keys = np.concatenate([b.ravel() for b in blocks]).astype(np.int64)
        W = blocks[0].shape[1]
        ids = np.tile(np.repeat(np.arange(self.S, dtype=np.int32), W), len(blocks))
        counts = np.bincount(keys, minlength=self.n_lmers)
        self.offsets = np.zeros(self.n_lmers + 1, dtype=np.int64)
        np.cumsum(counts, out=self.offsets[1:])
        self.ids = _counting_sort(keys, ids, self.offsets)
        self._dev = {}

    def nbytes(self) -> int:
        return self.offsets.nbytes + self.ids.nbytes

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
            return _query_cpu(bx, self.offsets, self.ids, self.masks, self.weights,
                              self.n_lmers, self.S, max(1, n_split))
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
        return _query_reduce_cpu(bx, self.offsets, self.ids, self.masks, self.weights,
                                 self.n_lmers, np.ascontiguousarray(inv_dx, dtype=np.float64),
                                 np.ascontiguousarray(inv_ds, dtype=np.float64),
                                 np.ascontiguousarray(coef, dtype=np.float64), float(gamma),
                                 max(1, min(bx.shape[0], 4 * numba.get_num_threads())))

    def _arrays(self, dev):
        if dev not in self._dev:
            if dev == "cuda":
                import cupy as cp

                self._dev[dev] = (cp.asarray(self.offsets), cp.asarray(self.ids),
                                  cp.asarray(self.masks), cp.asarray(self.weights))
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
        offsets, ids, masks, weights = self._arrays("cuda")
        bx = cp.ascontiguousarray(bx, dtype=cp.uint32)
        B, W = bx.shape
        out = cp.zeros((B, self.S), dtype=cp.float64)
        n_items = W * len(self.masks)
        block = 256
        for b0 in range(0, B, 65535):  # grid.y limit
            nb = min(65535, B - b0)
            _cuda_fn(((n_items + block - 1) // block, nb), (block,),
                     (bx[b0:b0 + nb], np.int32(W), np.int64(n_items), offsets, ids, masks,
                      weights, np.int32(len(self.masks)), np.int64(self.n_lmers),
                      np.int32(self.S), out[b0:b0 + nb]))
        return out

    def _pairwise_mlx(self, bx):
        import mlx.core as mx

        global _metal_fn
        if _metal_fn is None:
            _metal_fn = mx.fast.metal_kernel(
                name="gkm_lmer_index_query",
                input_names=["bx", "offsets", "ids", "masks", "levels", "params"],
                output_names=["counts"], source=_METAL_SRC, atomic_outputs=True)
        if self.offsets[-1] >= 2**31:
            raise ValueError("l-mer index too large for the Metal kernel")
        offsets, ids, masks, levels = self._arrays("mlx")
        bx_np = np.ascontiguousarray(to_cpu(bx), dtype=np.uint32)
        B, W = bx_np.shape
        n_lev = int(self.levels.max()) + 1
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
            for m in self.level_values:
                out[b0:b0 + nb] += self.table[m] * counts[m]
        return mx.array(out.astype(np.float32))
