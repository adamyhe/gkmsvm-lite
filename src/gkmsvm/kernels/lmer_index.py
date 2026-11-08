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

from gkmsvm.backend import to_cpu
from gkmsvm.kernels.direct import row_stride
from gkmsvm.kernels.lmer_index_cpu import (
    _counting_sort,
    _query_cpu,
    _query_impact_cpu,
    _query_mutants_cpu,
    _query_reduce_cpu,
)
from gkmsvm.kernels.lmer_index_gpu import (
    _CUDA_SRC,
    _MUTANTS_CUDA,
    _cuda_fn,
    _metal_impact,
    _metal_mutants,
    _metal_query,
    _metal_reduce,
)

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


_MUTANTS_THREADS = 128
# Metal: 1024 threads per (position, sequence). Each threadgroup's [5, S]
# scratch is device memory its float atomics go through; with 128-thread
# groups, the resident ones' scratch outgrew the caches and the atomics went to
# DRAM (0.5x the CPU at 18 index entries per l-mer, so MLX used to run ISM's
# index pass on CPU above 6). Larger groups keep the same occupancy with fewer
# scratch slices live. RBF ISM, 8 x 300 bp on an M4 against the CPU's 10
# threads: 4.1x at 1.1 entries per l-mer, 3.0x at 4.4, 2.75x at 18 (the
# dsQTL -t 3 model; 128 threads: 3.2x, 1.7x, 0.52x)
_METAL_MUTANTS_THREADS = 1024
# Scratch per launch: [items, 5, S]
_MUTANTS_SCRATCH_BYTES = 256 * 2**20
# Threadgroups (each with its own length-S scratch slice) for _METAL_REDUCE
_METAL_REDUCE_GROUPS = 128
# (window, neighbor) items per _METAL_REDUCE launch, so no launch runs long
_METAL_REDUCE_ITEMS = 2**28

# Rows per Metal launch: caps the int32 [levels, rows, S] counts
_METAL_COUNT_BYTES = 256 * 2**20
# Above this many distinct pair weights, MLX queries on CPU (one int32 [rows, S]
# count block per weight)
_METAL_MAX_LEVELS = 16
# Threadgroups per launch that a few rows are split over (each row gets
# ceil(this / rows) groups), so a lone row (an SMO column) uses the whole GPU
# instead of one core
_METAL_QUERY_MIN_GROUPS = 40
# Threads per query row (_METAL_SRC). 256 x 300 bp queries, l = 10, on an M4
# against the CPU's 10 threads: 3.9x at 1.1 entries per l-mer, 3.2x at 4.4,
# 1.45x at 18 (the dsQTL -t 3 model); a thread per item, in groups of 256, was
# 1.7x, 0.76x and 0.45x
_METAL_QUERY_THREADS = 1024


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
        # Entries in column order (each column's fwd then RC windows), so the
        # stable sort leaves every l-mer's entries sorted by column
        keys = np.hstack(blocks).ravel().astype(np.int64)
        W = blocks[0].shape[1]
        ids = np.repeat(np.arange(self.S, dtype=np.int32), W * len(blocks))
        # Window weights (-t 4/5): each entry's, and the query windows' (row_weights)
        self.row_weights = kernel.row_weights
        vals = None
        if (
            kernel.window_weights(W) is not None
        ):  # an RC window carries its forward weight
            vals = np.hstack(
                [
                    np.broadcast_to(kernel.row_weights(b, rc=k == 1), b.shape)
                    for k, b in enumerate(blocks)
                ]
            ).ravel()
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
        return dev in ("cuda", "mlx")

    def impact_on_device(self, dev: str) -> bool:
        """Whether impact() runs on dev's rows: CPU, and MLX in the fused Metal
        pass (as reduce_on_device). CUDA queries pairwise() instead."""
        return dev == "cpu" or self.reduce_on_device(dev)

    def queries_on_device(self, dev: str) -> bool:
        """Whether pairwise() queries on dev (CUDA, MLX) rather than on CPU: MLX
        queries on CPU above _METAL_MAX_LEVELS distinct weights."""
        if dev == "mlx":
            return len(self.table) <= _METAL_MAX_LEVELS
        return dev == "cuda"

    @property
    def _csr(self):
        """The index arguments of the numba queries (see _add_window)."""
        return (
            self.offsets,
            self.ids,
            self.vals,
            self.masks,
            self.weights,
            self.n_lmers,
        )

    def nbytes(self) -> int:
        return self.offsets.nbytes + self.ids.nbytes + self.vals.nbytes

    def pairwise(self, bx, diag_offset=None):
        """K_raw [B, S] of packed query rows bx [B, W] against the columns.

        Rows may contain padding windows (any value >= 4^l), which add nothing.
        CPU/CUDA return float64, MLX float32, on bx's device. diag_offset (CPU,
        CUDA): row b is column diag_offset + b of a self-Gram, and only its
        columns from there on are computed (the rest are 0).
        """
        from gkmsvm.kernels.packed import device_of

        dev = device_of(bx)
        diag_offset = -1 if diag_offset is None else diag_offset
        if dev == "cpu":
            import numba

            bx = np.ascontiguousarray(bx)
            n_split = min(bx.shape[1], -(-numba.get_num_threads() // bx.shape[0]))
            return _query_cpu(
                bx,
                *self._csr,
                self.S,
                max(1, n_split),
                self._wx(bx),
                diag_offset,
            )
        if dev == "cuda":
            return self._pairwise_cuda(bx, diag_offset)
        if diag_offset >= 0:
            raise ValueError("diag_offset is not supported on MLX")
        return self._pairwise_mlx(bx)

    def reduce_on_device(self, dev: str) -> bool:
        """Whether reduce() runs on dev rather than needing CPU rows: MLX
        (the fused Metal pass) when the index fits its int32 offsets."""
        return dev == "mlx" and self.offsets[-1] < 2**31

    def reduce(self, bx, inv_dx, inv_ds, coef, gamma=np.nan):
        """sum_s coef_s f(K_raw(row, s) inv_dx[row] inv_ds[s]) [B] for rows bx.

        f is exp(gamma (k - 1)) when gamma is given (RBF), else identity.
        The same as reducing pairwise(bx), without materializing it. CPU rows
        give float64 NumPy; MLX rows (see reduce_on_device) float32 MLX.
        """
        import numba

        from gkmsvm.kernels.packed import device_of

        if device_of(bx) == "mlx":
            return self._reduce_mlx(bx, inv_dx, inv_ds, coef, gamma)
        bx = np.ascontiguousarray(bx)
        return _query_reduce_cpu(
            bx,
            *self._csr,
            self._wx(bx),
            np.ascontiguousarray(inv_dx, dtype=np.float64),
            np.ascontiguousarray(inv_ds, dtype=np.float64),
            np.ascontiguousarray(coef, dtype=np.float64),
            float(gamma),
            max(1, min(bx.shape[0], 4 * numba.get_num_threads())),
        )

    def impact(self, bx, inv_dr, inv_ds, coef, gamma=np.nan):
        """Mutation impact [n] of rows bx = [ref (n rows); alt (n rows)]
        (see _query_impact_cpu); inv_dr: 1 / sqrt(K_raw(ref, ref)) [n]. CPU rows
        give float64 NumPy; MLX rows (see impact_on_device) float32 MLX."""
        import numba

        from gkmsvm.kernels.packed import device_of

        if device_of(bx) == "mlx":
            return self._impact_mlx(bx, inv_dr, inv_ds, coef, gamma)
        bx = np.ascontiguousarray(bx)
        return _query_impact_cpu(
            bx,
            *self._csr,
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
            *self._csr,
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

    def _pairwise_cuda(self, bx, diag_offset=-1):
        import cupy as cp

        offsets, ids, masks, weights, vals = self._arrays("cuda")
        bx = cp.ascontiguousarray(bx, dtype=cp.uint32)
        B, W = bx.shape
        weighted = vals is not None
        wx_stride = 0
        if weighted:
            wx_np = np.ascontiguousarray(self.row_weights(bx))
            wx = cp.asarray(wx_np)
            wx_stride = row_stride(wx_np, W)
        else:
            wx = vals = cp.zeros(1, dtype=cp.uint8)
        out = cp.zeros((B, self.S), dtype=cp.float64)
        n_items = W * len(self.masks)
        block = 256
        for b0 in range(0, B, 65535):  # grid.y limit
            nb = min(65535, B - b0)
            _cuda_fn(_CUDA_SRC, "lmer_index_query")(
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
                    np.int64(diag_offset + b0 if diag_offset >= 0 else -1),
                ),
            )
        return out

    def _pairwise_mlx(self, bx):
        """pairwise() in Metal: float32 on the device, combined from exact counts."""
        import mlx.core as mx

        if not self.queries_on_device("mlx"):
            # One int32 [rows, S] block per level; with many levels, query on
            # CPU, which shares memory with the GPU
            out = self.pairwise(np.ascontiguousarray(to_cpu(bx)))
            return mx.array(out.astype(np.float32))
        if self.offsets[-1] >= 2**31:
            raise ValueError("l-mer index too large for the Metal kernel")
        offsets, ids, masks, levels, vals = self._arrays("mlx")
        bx_np = np.ascontiguousarray(to_cpu(bx), dtype=np.uint32)
        B, W = bx_np.shape
        weighted = vals is not None
        query = _metal_query(weighted)
        if weighted:  # the query windows' weights (row_weights)
            wx_np = np.ascontiguousarray(self.row_weights(bx_np), dtype=np.uint32)
            wx_stride = row_stride(wx_np, W)
        n_lev = len(self.table)
        words = 2 if weighted else 1
        rows = max(1, _METAL_COUNT_BYTES // (4 * words * n_lev * self.S))
        out = np.zeros((B, self.S), dtype=np.float64)
        for b0 in range(0, B, rows):
            nb = min(rows, B - b0)
            split = max(1, -(-_METAL_QUERY_MIN_GROUPS // nb))
            launch = dict(
                template=[],
                grid=(nb * split * _METAL_QUERY_THREADS, 1, 1),
                threadgroup=(_METAL_QUERY_THREADS, 1, 1),
                init_value=0,
            )
            q = mx.array(bx_np[b0 : b0 + nb].ravel())
            if weighted:
                params = mx.array(
                    [W, len(self.masks), self.S, self.n_lmers, nb, split, wx_stride],
                    dtype=mx.int32,
                )
                wq = wx_np[b0 : b0 + nb] if wx_stride else wx_np
                lo, hi = query(
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
                    [W, len(self.masks), self.S, self.n_lmers, nb, split],
                    dtype=mx.int32,
                )
                counts = query(
                    inputs=[q, offsets, ids, masks, levels, params],
                    output_shapes=[(n_lev * nb * self.S,)],
                    output_dtypes=[mx.int32],
                    **launch,
                )[0]
                counts = np.array(counts)
            counts = counts.reshape(n_lev, nb, self.S)
            for m in range(n_lev):
                out[b0 : b0 + nb] += self.table[m] * counts[m].astype(np.float64)
        return mx.array(out.astype(np.float32))

    def _reduce_mlx(self, bx, inv_dx, inv_ds, coef, gamma):
        """reduce() in Metal (_METAL_REDUCE): float32 [B] on the device."""
        import mlx.core as mx

        if not self.reduce_on_device("mlx"):
            raise ValueError("l-mer index too large for the Metal kernel")
        offsets, ids, masks, _, vals = self._arrays("mlx")
        if "mlx_weights" not in self._dev:
            self._dev["mlx_weights"] = mx.array(self.weights.astype(np.float32))
        bx_np = np.ascontiguousarray(to_cpu(bx), dtype=np.uint32)
        B, W = bx_np.shape
        weighted = vals is not None
        wx_stride = 0
        # (a 1-element MLX input would be a constant, not a device pointer)
        wx_np = np.zeros(2, dtype=np.uint32)
        if weighted:
            wx_np = np.ascontiguousarray(self.row_weights(bx_np), dtype=np.uint32)
            wx_stride = row_stride(wx_np, W)
        vals = mx.zeros(2, dtype=mx.uint32) if vals is None else vals
        rbf = not np.isnan(gamma)
        fparams = mx.array(
            [float(gamma) if rbf else 0.0, float(np.exp(-gamma)) if rbf else 1.0],
            dtype=mx.float32,
        )

        def f32(a):
            return mx.array(np.ascontiguousarray(to_cpu(a), dtype=np.float32))

        inv_dx, inv_ds, coef = f32(inv_dx), f32(inv_ds), f32(coef)
        rows = max(1, _METAL_REDUCE_ITEMS // max(1, W * len(self.masks)))
        parts = []
        for b0 in range(0, B, rows):
            nb = min(rows, B - b0)
            G = min(nb, _METAL_REDUCE_GROUPS)
            wq = wx_np[b0 : b0 + nb] if wx_stride else wx_np
            params = [W, len(self.masks), self.S, self.n_lmers, nb, G]
            params += [int(weighted), wx_stride, int(rbf)]
            _, out = _metal_reduce()(
                inputs=[
                    mx.array(bx_np[b0 : b0 + nb].ravel()),
                    mx.array(np.ascontiguousarray(wq).ravel()),
                    offsets,
                    ids,
                    vals,
                    masks,
                    self._dev["mlx_weights"],
                    inv_dx[b0 : b0 + nb],
                    inv_ds,
                    coef,
                    mx.array(params, dtype=mx.int32),
                    fparams,
                ],
                template=[],
                grid=(G * _METAL_QUERY_THREADS, 1, 1),
                threadgroup=(_METAL_QUERY_THREADS, 1, 1),
                output_shapes=[(G * self.S,), (nb,)],
                output_dtypes=[mx.float32, mx.float32],
                init_value=0,
            )
            parts.append(out)
        return parts[0] if len(parts) == 1 else mx.concatenate(parts)

    def _impact_mlx(self, bx, inv_dr, inv_ds, coef, gamma):
        """impact() in Metal (_METAL_IMPACT): float32 [n] on the device."""
        import mlx.core as mx

        if not self.reduce_on_device("mlx"):
            raise ValueError("l-mer index too large for the Metal kernel")
        offsets, ids, masks, _, vals = self._arrays("mlx")
        if "mlx_weights" not in self._dev:
            self._dev["mlx_weights"] = mx.array(self.weights.astype(np.float32))
        bx_np = np.ascontiguousarray(to_cpu(bx), dtype=np.uint32)
        n, W = bx_np.shape[0] // 2, bx_np.shape[1]
        weighted = vals is not None
        wx_stride = 0
        wxr = wxa = np.zeros(2, dtype=np.uint32)  # (see _reduce_mlx)
        if weighted:
            wx_np = np.ascontiguousarray(self.row_weights(bx_np), dtype=np.uint32)
            wx_stride = row_stride(wx_np, W)
            wxr, wxa = (wx_np[:n], wx_np[n:]) if wx_stride else (wx_np, wx_np)
        vals = mx.zeros(2, dtype=mx.uint32) if vals is None else vals
        rbf = not np.isnan(gamma)
        fparams = mx.array(
            [float(gamma) if rbf else 0.0, float(np.exp(-gamma)) if rbf else 0.0],
            dtype=mx.float32,
        )

        def f32(a):
            return mx.array(np.ascontiguousarray(to_cpu(a), dtype=np.float32))

        inv_dr, inv_ds, coef = f32(inv_dr), f32(inv_ds), f32(coef)
        rows = max(1, _METAL_REDUCE_ITEMS // max(1, W * len(self.masks)))
        parts = []
        for b0 in range(0, n, rows):
            nb = min(rows, n - b0)
            G = min(nb, _METAL_REDUCE_GROUPS)
            params = [W, len(self.masks), self.S, self.n_lmers, nb, G]
            params += [int(weighted), wx_stride, int(rbf)]

            def rows_of(a, start, stride=W):
                return mx.array(a[start : start + nb].ravel() if stride else a)

            _, out = _metal_impact()(
                inputs=[
                    rows_of(bx_np, b0),
                    rows_of(bx_np, n + b0),
                    rows_of(wxr, b0, wx_stride),
                    rows_of(wxa, b0, wx_stride),
                    offsets,
                    ids,
                    vals,
                    masks,
                    self._dev["mlx_weights"],
                    inv_dr[b0 : b0 + nb],
                    inv_ds,
                    coef,
                    mx.array(params, dtype=mx.int32),
                    fparams,
                ],
                template=[],
                grid=(G * _METAL_QUERY_THREADS, 1, 1),
                threadgroup=(_METAL_QUERY_THREADS, 1, 1),
                output_shapes=[(3 * G * self.S,), (nb,)],
                output_dtypes=[mx.float32, mx.float32],
                init_value=0,
            )
            parts.append(out)
        return parts[0] if len(parts) == 1 else mx.concatenate(parts)

    def _mutants_gpu(
        self, old, new, ref, inv_dx, inv_dmut, inv_ds, coef, gamma, slot_wt
    ):
        """mutants() on ref's device (CUDA or MLX): [4, L, B] there."""
        from gkmsvm.kernels.packed import device_of

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
                _cuda_fn(_MUTANTS_CUDA, "lmer_index_mutants")(
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
            _, part = _metal_mutants()(
                inputs=inputs
                + [mx.array((i0,) + ints, dtype=mx.int64).astype(mx.int32)]
                + [mx.array(floats, dtype=mx.float32)],
                template=[],
                grid=(n * _METAL_MUTANTS_THREADS, 1, 1),
                threadgroup=(_METAL_MUTANTS_THREADS, 1, 1),
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
