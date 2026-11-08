"""The masked join in Metal (MLX; kernels/masked.py): the column index is built
by histogram and scatter kernels, and the join writes int32 counts (two
uint32 words with window weights) that the CPU combines in float64."""

from __future__ import annotations

from functools import cache

import numpy as np

from gkmsvm.kernels.masked import _MASK_CHUNK, _column_weights, mask_runs
from gkmsvm.kernels.primitives import row_stride

# ---------------------------------------------------------------------------
# Metal (MLX)
# ---------------------------------------------------------------------------

# Key of packed window w under one mask's R runs (shift, bitmask,
# destination; see mask_runs). A template: MLX passes small inputs (runs) in
# the constant address space, larger ones in device
_METAL_MASKED_KEY_HEADER = """
template <typename P>
inline uint masked_key(uint w, P runs, int R) {
    uint key = 0;
    for (int r = 0; r < R; r++)
        key |= ((w >> runs[3 * r]) & runs[3 * r + 1]) << runs[3 * r + 2];
    return key;
}
"""

# Same structure as masked_join (CUDA). Metal has no float64, so each launch
# writes exact int32 counts per (query, column) and the caller combines
# levels in float64 on the CPU (unified memory makes that copy cheap).
_METAL_MASKED_JOIN = """
    uint b = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint tg = threads_per_threadgroup.x;
    int Wx = params[0], M = params[1], R = params[2], S = params[3], lanes = params[4];
    uint K1 = (uint)params[5];
    uint n_win = (uint)params[6];
    threadgroup atomic_int counts[TILE];
    for (int s = tid; s < S; s += tg)
        atomic_store_explicit(&counts[s], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Each lane group takes one (mask, window) item and scans its bucket.
    // Measured on an M4 this
    // beats loading each window once and looping masks inside (0.60 s vs
    // 0.65 s at 2000x2000).
    int group = tid / lanes, lane = tid % lanes, n_groups = tg / lanes;
    for (int it = group; it < M * Wx; it += n_groups) {
        int m = it / Wx;
        int i = it - m * Wx;
        uint w = bx[b * Wx + i];
        if (w != 0xFFFFFFFFu) {  // PAD past a shorter sequence's windows
            uint key = masked_key(w, runs + m * R * 3, R);
            uint base = (uint)m * K1;
            int hi = offsets[base + key + 1];
            uint cbase = (uint)m * n_win;
            for (int t = offsets[base + key] + lane; t < hi; t += lanes)
                atomic_fetch_add_explicit(&counts[colseq[cbase + t]], 1, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = tid; s < S; s += tg)
        out[b * S + s] = atomic_load_explicit(&counts[s], memory_order_relaxed);
"""

# With window weights (-t 4/5): counts[s] += wx[b, i] * the entry's weight
# (packed in colseq's top bits by _METAL_SCATTER_W), exact in 64 bits: a low
# uint32 word and a carry word, which the adding thread bumps when its add
# wraps the low word (atomic_fetch_add returns the word before its own add)
_METAL_MASKED_JOIN_W = """
    uint b = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint tg = threads_per_threadgroup.x;
    int Wx = params[0], M = params[1], R = params[2], S = params[3], lanes = params[4];
    uint K1 = (uint)params[5];
    uint n_win = (uint)params[6];
    uint wx_stride = (uint)params[7];
    threadgroup atomic_uint lo[TILE];
    threadgroup atomic_uint hi[TILE];
    for (int s = tid; s < S; s += tg) {
        atomic_store_explicit(&lo[s], 0u, memory_order_relaxed);
        atomic_store_explicit(&hi[s], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int group = tid / lanes, lane = tid % lanes, n_groups = tg / lanes;
    for (int it = group; it < M * Wx; it += n_groups) {
        int m = it / Wx;
        int i = it - m * Wx;
        uint w = bx[b * Wx + i];
        if (w != 0xFFFFFFFFu) {  // PAD past a shorter sequence's windows
            uint key = masked_key(w, runs + m * R * 3, R);
            uint a = (uint)wx[b * wx_stride + i];
            uint base = (uint)m * K1;
            int hi_t = offsets[base + key + 1];
            uint cbase = (uint)m * n_win;
            for (int t = offsets[base + key] + lane; t < hi_t; t += lanes) {
                uint e = (uint)colseq[cbase + t];
                uint s = e & 0xFFFFu, v = a * (e >> 16);
                uint old = atomic_fetch_add_explicit(&lo[s], v, memory_order_relaxed);
                if (old + v < old)
                    atomic_fetch_add_explicit(&hi[s], 1u, memory_order_relaxed);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = tid; s < S; s += tg) {
        out_lo[b * S + s] = atomic_load_explicit(&lo[s], memory_order_relaxed);
        out_hi[b * S + s] = atomic_load_explicit(&hi[s], memory_order_relaxed);
    }
"""

_METAL_HIST = """
    uint idx = thread_position_in_grid.x;
    uint M = (uint)params[0], R = (uint)params[1], K1 = (uint)params[2], n_win = (uint)params[3];
    if (idx >= M * n_win) return;
    uint m = idx / n_win, w = idx - m * n_win;
    uint v = by[w];
    if (v == 0xFFFFFFFFu) return;  // PAD
    uint key = masked_key(v, runs + m * R * 3, (int)R);
    atomic_fetch_add_explicit(&counts[m * K1 + key + 1], 1, memory_order_relaxed);
"""

# slot = bucket start + atomic counter; colseq is atomic only because MLX makes
# every output of an atomic kernel atomic
_METAL_SCATTER = """
    uint idx = thread_position_in_grid.x;
    uint M = (uint)params[0], R = (uint)params[1], K1 = (uint)params[2], n_win = (uint)params[3];
    uint Wy = (uint)params[4];
    if (idx >= M * n_win) return;
    uint m = idx / n_win, w = idx - m * n_win;
    uint v = by[w];
    if (v == 0xFFFFFFFFu) return;  // PAD
    uint key = masked_key(v, runs + m * R * 3, (int)R);
    int slot = offsets[m * K1 + key]
             + atomic_fetch_add_explicit(&fill[m * (K1 - 1) + key], 1, memory_order_relaxed);
    // per_window: seq_of_row holds one entry per window (weighted), else per row
    uint e = (uint)params[5] ? (uint)w : w / Wy;
    atomic_store_explicit(&colseq[m * n_win + slot], seq_of_row[e], memory_order_relaxed);
"""


@cache
def _metal_index_kernels():
    import mlx.core as mx

    hist = mx.fast.metal_kernel(
        name="gkm_masked_hist",
        input_names=["by", "runs", "params"],
        output_names=["counts"],
        source=_METAL_HIST,
        header=_METAL_MASKED_KEY_HEADER,
        atomic_outputs=True,
    )
    scatter = mx.fast.metal_kernel(
        name="gkm_masked_scatter",
        input_names=["by", "seq_of_row", "runs", "offsets", "params"],
        output_names=["fill", "colseq"],
        source=_METAL_SCATTER,
        header=_METAL_MASKED_KEY_HEADER,
        atomic_outputs=True,
    )
    return hist, scatter


def _metal_index(by_mx, seq_of_row_mx, chunk, j, Wy):
    """Column index (offsets [M*(K+1)], colseq [M*n_win]) built in Metal, and
    the chunk's mask_runs. With window weights, seq_of_row_mx holds each
    column row's sequence id in its low 16 bits and, per window, the weight
    in the top bits ([n_win]; see masked_pairwise_mlx), and colseq gets
    those entries."""
    import mlx.core as mx

    hist, scatter = _metal_index_kernels()
    runs_np = mask_runs(chunk)
    M, R = runs_np.shape[0], runs_np.shape[1]
    K = 1 << (2 * j)
    n_win = by_mx.shape[0]
    pos = mx.array(runs_np.ravel())
    per_window = int(seq_of_row_mx.shape[0] == n_win)  # (with Wy = 1 both agree)
    params = mx.array([M, R, K + 1, n_win, Wy, per_window], dtype=mx.int32)
    threads = 256
    grid = ((M * n_win + threads - 1) // threads) * threads
    counts = hist(
        inputs=[by_mx, pos, params],
        template=[],
        grid=(grid, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(M * (K + 1),)],
        output_dtypes=[mx.int32],
        init_value=0,
    )[0]
    offsets = mx.cumsum(counts.reshape(M, K + 1), axis=1).reshape(-1)
    _, colseq = scatter(
        inputs=[by_mx, seq_of_row_mx, pos, offsets, params],
        template=[],
        grid=(grid, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(M * K,), (M * n_win,)],
        output_dtypes=[mx.int32, mx.int32],
        init_value=0,
    )
    return offsets, colseq, runs_np


# Columns per launch: the threadgroup counter row (32 KB limit) is TILE int32
# (two uint32 words per column with window weights)
_METAL_COL_TILE = 4096
_METAL_COL_TILE_W = 2048
_METAL_LANES = 16
_METAL_THREADS = 256


@cache
def _metal_join(weighted: bool):
    import mlx.core as mx

    if weighted:
        return mx.fast.metal_kernel(
            name="gkm_masked_join_w",
            input_names=["bx", "wx", "runs", "offsets", "colseq", "params"],
            output_names=["out_lo", "out_hi"],
            source=_METAL_MASKED_JOIN_W,
            header=_METAL_MASKED_KEY_HEADER,
        )
    return mx.fast.metal_kernel(
        name="gkm_masked_join",
        input_names=["bx", "runs", "offsets", "colseq", "params"],
        output_names=["out"],
        source=_METAL_MASKED_JOIN,
        header=_METAL_MASKED_KEY_HEADER,
    )


def masked_pairwise_mlx(
    bx_packed, by_packed, plan, by_rc_packed=None, wx=None, wy=None, wy_rc=None
) -> np.ndarray:
    """Metal version of masked_pairwise(); the column index is built in Metal too.

    Inputs may be NumPy or MLX uint32 arrays; wx, wy, wy_rc: row_weights as
    in masked_pairwise. Returns [B, S] float64 NumPy, combined exactly from
    per-level integer counts (64-bit with window weights).
    """
    import mlx.core as mx

    weighted = wx is not None
    join = _metal_join(weighted)
    tile = _METAL_COL_TILE_W if weighted else _METAL_COL_TILE
    bx = np.ascontiguousarray(np.asarray(bx_packed), dtype=np.uint32)
    by_all = np.asarray(by_packed, dtype=np.uint32)
    by_rc_all = (
        None if by_rc_packed is None else np.asarray(by_rc_packed, dtype=np.uint32)
    )
    B, Wx = bx.shape
    S_total, Wy = by_all.shape
    bx_mx = mx.array(bx.ravel())
    result = np.zeros((B, S_total), dtype=np.float64)
    if weighted:
        wx_np = np.ascontiguousarray(wx, dtype=np.uint32)
        wx_mx = mx.array(wx_np.ravel())
        wx_stride = row_stride(wx_np, Wx)
        blocks_w = [wy] + ([wy_rc] if by_rc_all is not None else [])

    for c0 in range(0, S_total, tile):
        c1 = min(c0 + tile, S_total)
        S = c1 - c0
        rows = [by_all[c0:c1]] + ([by_rc_all[c0:c1]] if by_rc_all is not None else [])
        by = np.ascontiguousarray(np.concatenate(rows))
        seq_of_row = np.tile(np.arange(S, dtype=np.int32), len(rows))
        n_win = by.shape[0] * Wy
        if weighted:  # one entry per window: sequence id | weight << 16
            tile_w = [w if w.shape[0] == 1 else w[c0:c1] for w in blocks_w]
            wby = _column_weights(tile_w, S, interleave=False).astype(np.int32)
            seq_of_row = (np.repeat(seq_of_row, Wy) | (wby.ravel() << 16)).astype(
                np.int32
            )
        by_mx, seq_mx = mx.array(by.ravel()), mx.array(seq_of_row)
        for j, coef, positions in plan:
            K1 = (1 << (2 * j)) + 1
            level = None
            for m0 in range(0, len(positions), _MASK_CHUNK):
                chunk = positions[m0 : m0 + _MASK_CHUNK]
                offsets, colseq, runs_np = _metal_index(by_mx, seq_mx, chunk, j, Wy)
                params = mx.array(
                    [
                        Wx,
                        runs_np.shape[0],
                        runs_np.shape[1],
                        S,
                        _METAL_LANES,
                        K1,
                        n_win,
                    ],
                    dtype=mx.int32,
                )
                runs_mx = mx.array(runs_np.ravel())
                if weighted:
                    params = mx.concatenate(
                        [params, mx.array([wx_stride], dtype=mx.int32)]
                    )
                    lo, hi = join(
                        inputs=[bx_mx, wx_mx, runs_mx, offsets, colseq, params],
                        template=[("TILE", tile)],
                        grid=(B * _METAL_THREADS, 1, 1),
                        threadgroup=(_METAL_THREADS, 1, 1),
                        output_shapes=[(B * S,), (B * S,)],
                        output_dtypes=[mx.uint32, mx.uint32],
                    )
                    counts = (
                        np.array(hi).astype(np.uint64) << np.uint64(32)
                    ) | np.array(lo).astype(np.uint64)
                else:
                    counts = join(
                        inputs=[bx_mx, runs_mx, offsets, colseq, params],
                        template=[("TILE", tile)],
                        grid=(B * _METAL_THREADS, 1, 1),
                        threadgroup=(_METAL_THREADS, 1, 1),
                        output_shapes=[(B * S,)],
                        output_dtypes=[mx.int32],
                    )[0]
                level = counts if level is None else level + counts
            result[:, c0:c1] += coef * np.array(level, dtype=np.float64).reshape(B, S)
    return result
