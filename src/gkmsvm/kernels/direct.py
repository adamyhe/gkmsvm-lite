from __future__ import annotations

from math import comb

import numpy as np
from numba import njit, prange

from gkmsvm.backend import get_array_module
from gkmsvm.codec import reverse_complement
from gkmsvm.kernels.base import GkmKernel


# ---------------------------------------------------------------------------
# Numba-accelerated kernels
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _fused_pairwise_numba(wx, wy, table):
    """Fused match-count + table-lookup + sum using float dot products."""
    B = wx.shape[0]
    Wx = wx.shape[1]
    F = wx.shape[2]
    S = wy.shape[0]
    Wy = wy.shape[1]
    l = table.shape[0] - 1
    result = np.empty((B, S), dtype=np.float64)
    for idx in prange(B * S):
        b = idx // S
        s = idx % S
        acc = 0.0
        for i in range(Wx):
            for j in range(Wy):
                dot = 0.0
                for f in range(F):
                    dot += wx[b, i, f] * wy[s, j, f]
                m = l - int(dot + 0.5)
                if m < 0:
                    m = 0
                elif m > l:
                    m = l
                acc += table[m]
        result[b, s] = acc
    return result


@njit(parallel=True, cache=True)
def _fused_pairwise_idx_numba(bx, by, table):
    """Fused pairwise using int8 base-index comparison (4x fewer ops)."""
    B = bx.shape[0]
    Wx = bx.shape[1]
    l = bx.shape[2]
    S = by.shape[0]
    Wy = by.shape[1]
    result = np.empty((B, S), dtype=np.float64)
    for idx in prange(B * S):
        b = idx // S
        s = idx % S
        acc = 0.0
        for i in range(Wx):
            for j in range(Wy):
                matches = 0
                for k in range(l):
                    if bx[b, i, k] == by[s, j, k]:
                        matches += 1
                acc += table[l - matches]
        result[b, s] = acc
    return result


@njit(parallel=True, cache=True)
def _fused_diagonal_idx_numba(bx, table):
    """Self-kernel diagonal from base index windows."""
    B = bx.shape[0]
    W = bx.shape[1]
    l = bx.shape[2]
    result = np.empty(B, dtype=np.float64)
    for b in prange(B):
        acc = 0.0
        for i in range(W):
            for j in range(W):
                matches = 0
                for k in range(l):
                    if bx[b, i, k] == bx[b, j, k]:
                        matches += 1
                acc += table[l - matches]
        result[b] = acc
    return result


@njit(parallel=True, cache=True)
def _fused_cross_diagonal_idx_numba(bx, bx_rc, table):
    """Cross-kernel diagonal (fwd vs RC) from base index windows."""
    B = bx.shape[0]
    W = bx.shape[1]
    l = bx.shape[2]
    W_rc = bx_rc.shape[1]
    result = np.empty(B, dtype=np.float64)
    for b in prange(B):
        acc = 0.0
        for i in range(W):
            for j in range(W_rc):
                matches = 0
                for k in range(l):
                    if bx[b, i, k] == bx_rc[b, j, k]:
                        matches += 1
                acc += table[l - matches]
        result[b] = acc
    return result


# ---------------------------------------------------------------------------
# CuPy CUDA kernels
# ---------------------------------------------------------------------------

_FUSED_PAIRWISE_CUDA = r"""
extern "C" __global__
void fused_pairwise(
    const float* __restrict__ wx,
    const float* __restrict__ wy,
    const double* __restrict__ table,
    double* __restrict__ result,
    const int total_pairs,
    const int S,
    const int Wx,
    const int Wy,
    const int F,
    const int l
) {
    extern __shared__ double s_table[];
    for (int i = threadIdx.x; i <= l; i += blockDim.x) {
        s_table[i] = table[i];
    }
    __syncthreads();

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total_pairs) return;

    const int b = idx / S;
    const int s = idx % S;

    double acc = 0.0;
    for (int i = 0; i < Wx; i++) {
        const float* wx_row = wx + (b * Wx + i) * F;
        for (int j = 0; j < Wy; j++) {
            const float* wy_row = wy + (s * Wy + j) * F;
            float dot = 0.0f;
            for (int f = 0; f < F; f++) {
                dot += wx_row[f] * wy_row[f];
            }
            int m = l - __float2int_rn(dot);
            if (m < 0) m = 0;
            if (m > l) m = l;
            acc += s_table[m];
        }
    }
    result[idx] = acc;
}
"""

_FUSED_PAIRWISE_IDX_CUDA = r"""
extern "C" __global__
void fused_pairwise_idx(
    const signed char* __restrict__ bx,
    const signed char* __restrict__ by_t,
    const double* __restrict__ table,
    double* __restrict__ result,
    const int total_pairs,
    const int S,
    const int Wx,
    const int Wy,
    const int l
) {
    // by_t layout: [l, Wy, S] — adjacent threads (consecutive s) read
    // adjacent bytes, giving coalesced 32B memory transactions.
    extern __shared__ double s_table[];
    for (int i = threadIdx.x; i <= l; i += blockDim.x) {
        s_table[i] = table[i];
    }
    __syncthreads();

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total_pairs) return;

    const int b = idx / S;
    const int s = idx % S;

    const int WyS = Wy * S;
    double acc = 0.0;
    for (int i = 0; i < Wx; i++) {
        const signed char* bx_row = bx + (b * Wx + i) * l;
        for (int j = 0; j < Wy; j++) {
            int matches = 0;
            for (int k = 0; k < l; k++) {
                matches += (bx_row[k] == by_t[k * WyS + j * S + s]);
            }
            acc += s_table[l - matches];
        }
    }
    result[idx] = acc;
}
"""

_cupy_fused_kernel = None
_cupy_fused_idx_kernel = None


def _fused_pairwise_gpu(wx, wy, table):
    """Launch fused pairwise CUDA kernel (float path)."""
    global _cupy_fused_kernel
    import cupy as cp

    if _cupy_fused_kernel is None:
        _cupy_fused_kernel = cp.RawKernel(
            _FUSED_PAIRWISE_CUDA, "fused_pairwise",
            options=("--use_fast_math",),
        )

    B, Wx, F = wx.shape
    S, Wy, _ = wy.shape
    l = table.shape[0] - 1

    wx_f32 = cp.ascontiguousarray(wx, dtype=cp.float32)
    wy_f32 = cp.ascontiguousarray(wy, dtype=cp.float32)
    table_gpu = cp.asarray(table, dtype=cp.float64)
    result = cp.empty(B * S, dtype=cp.float64)

    total = B * S
    block = 256
    grid = (total + block - 1) // block

    _cupy_fused_kernel(
        (grid,), (block,),
        (wx_f32, wy_f32, table_gpu, result,
         np.int32(total), np.int32(S), np.int32(Wx), np.int32(Wy),
         np.int32(F), np.int32(l)),
        shared_mem=(l + 1) * 8,
    )

    return result.reshape(B, S)


def _fused_pairwise_idx_gpu(bx, by, table):
    """Launch fused pairwise CUDA kernel (int8 index path).

    Transposes by from [S, Wy, l] to [l, Wy, S] so adjacent threads
    (consecutive s values) read adjacent bytes — coalesced access.
    """
    global _cupy_fused_idx_kernel
    import cupy as cp

    if _cupy_fused_idx_kernel is None:
        _cupy_fused_idx_kernel = cp.RawKernel(
            _FUSED_PAIRWISE_IDX_CUDA, "fused_pairwise_idx",
            options=("--use_fast_math",),
        )

    B, Wx, l = bx.shape
    S, Wy, _ = by.shape

    bx_i8 = cp.ascontiguousarray(bx, dtype=cp.int8)
    by_t = cp.ascontiguousarray(by.transpose(2, 1, 0), dtype=cp.int8)
    table_gpu = cp.asarray(table, dtype=cp.float64)
    result = cp.empty(B * S, dtype=cp.float64)

    total = B * S
    block = 256
    grid = (total + block - 1) // block

    _cupy_fused_idx_kernel(
        (grid,), (block,),
        (bx_i8, by_t, table_gpu, result,
         np.int32(total), np.int32(S), np.int32(Wx), np.int32(Wy),
         np.int32(l)),
        shared_mem=(l + 1) * 8,
    )

    return result.reshape(B, S)


# ---------------------------------------------------------------------------
# Kernel class
# ---------------------------------------------------------------------------

class DirectGkmKernel(GkmKernel):
    """Direct gapped k-mer kernel (LS-GKM -t 0 / gkm_cnt).

    For window length l and k informative positions, two windows differing
    in m positions share C(l-m, k) gapped k-mer features. The kernel sums
    these shared features over all pairs of windows from the two sequences.
    """

    def __init__(
        self, l: int, k: int, *, normalize: bool = True, include_rc: bool = True
    ):
        super().__init__(l, k, normalize=normalize, include_rc=include_rc)
        self._mismatch_table = np.array(
            [comb(l - m, k) if l - m >= k else 0 for m in range(l + 1)],
            dtype=np.float64,
        )

    def flat_windows(self, x: np.ndarray) -> np.ndarray:
        """Extract length-l windows and flatten channels for matmul.

        Args:
            x: [B, 4, L] one-hot array.

        Returns:
            [B, W, 4*l] array where W = L - l + 1.
        """
        xp = get_array_module(x)
        B, C, L = x.shape
        if L < self.l:
            raise ValueError(
                f"Sequence length {L} is shorter than window length {self.l}"
            )
        W = L - self.l + 1
        strides = x.strides
        shape = (B, C, W, self.l)
        new_strides = (strides[0], strides[1], strides[2], strides[2])
        wx = xp.lib.stride_tricks.as_strided(x, shape=shape, strides=new_strides)
        wx = xp.ascontiguousarray(wx.transpose(0, 2, 1, 3))  # [B, W, 4, l]
        return wx.reshape(B, W, C * self.l)

    def base_index_windows(self, x: np.ndarray) -> np.ndarray:
        """Extract base indices for sliding windows.

        Args:
            x: [B, 4, L] one-hot array.

        Returns:
            [B, W, l] int8 array of base indices (0-3).
        """
        xp = get_array_module(x)
        B, C, L = x.shape
        if L < self.l:
            raise ValueError(
                f"Sequence length {L} is shorter than window length {self.l}"
            )
        W = L - self.l + 1
        base_idx = xp.argmax(x, axis=1).astype(np.int8)  # [B, L]
        strides = base_idx.strides
        shape = (B, W, self.l)
        new_strides = (strides[0], strides[1], strides[1])
        bw = xp.lib.stride_tricks.as_strided(base_idx, shape=shape, strides=new_strides)
        return xp.ascontiguousarray(bw)

    def _apply_table(self, matches: np.ndarray) -> np.ndarray:
        """Look up weight table from match counts, sum over window dims."""
        xp = get_array_module(matches)
        table = xp.asarray(self._mismatch_table)
        mismatches = xp.clip(xp.rint(self.l - matches).astype(np.int64), 0, self.l)
        return table[mismatches].sum(axis=(-2, -1))

    def pairwise_from_windows(
        self, wx: np.ndarray, wy: np.ndarray
    ) -> np.ndarray:
        """Raw kernel from pre-extracted flat windows (no RC, no normalization).

        Args:
            wx: [B, Wx, F] flat windows from query sequences.
            wy: [S, Wy, F] flat windows from support sequences.

        Returns:
            [B, S] raw kernel values.
        """
        xp = get_array_module(wx)
        if xp is np:
            return _fused_pairwise_numba(
                np.ascontiguousarray(wx),
                np.ascontiguousarray(wy),
                self._mismatch_table,
            )
        return _fused_pairwise_gpu(wx, wy, self._mismatch_table)

    def pairwise_from_indices(
        self, bx: np.ndarray, by: np.ndarray
    ) -> np.ndarray:
        """Raw kernel from base-index windows (4x fewer ops than float path).

        Args:
            bx: [B, Wx, l] int8 base indices from query sequences.
            by: [S, Wy, l] int8 base indices from support sequences.

        Returns:
            [B, S] raw kernel values.
        """
        xp = get_array_module(bx)
        if xp is np:
            return _fused_pairwise_idx_numba(
                np.ascontiguousarray(bx),
                np.ascontiguousarray(by),
                self._mismatch_table,
            )
        return _fused_pairwise_idx_gpu(bx, by, self._mismatch_table)

    def diagonal_from_windows(self, wx: np.ndarray) -> np.ndarray:
        """Raw self-kernel from pre-extracted flat windows (no RC, no normalization).

        Args:
            wx: [B, W, F] flat windows.

        Returns:
            [B] raw self-kernel values.
        """
        xp = get_array_module(wx)
        self_matches = xp.matmul(wx, wx.transpose(0, 2, 1))
        return self._apply_table(self_matches)

    def _raw_pairwise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        wx = self.flat_windows(x)
        wy = self.flat_windows(y)
        result = self.pairwise_from_windows(wx, wy)

        if self.include_rc:
            wy_rc = self.flat_windows(reverse_complement(y))
            result = result + self.pairwise_from_windows(wx, wy_rc)

        return result.astype(x.dtype)

    def _raw_diagonal(
        self, x: np.ndarray, *, chunk_size: int | None = None
    ) -> np.ndarray:
        xp = get_array_module(x)
        B = x.shape[0]

        if chunk_size is not None and chunk_size < B:
            result = xp.empty(B, dtype=x.dtype)
            for start in range(0, B, chunk_size):
                end = min(start + chunk_size, B)
                result[start:end] = self._raw_diagonal(x[start:end])
            return result

        if xp is np:
            bx = self.base_index_windows(x)
            result = _fused_diagonal_idx_numba(bx, self._mismatch_table)
            if self.include_rc:
                bx_rc = self.base_index_windows(reverse_complement(x))
                result = result + _fused_cross_diagonal_idx_numba(
                    bx, bx_rc, self._mismatch_table
                )
            return result.astype(x.dtype)

        wx = self.flat_windows(x)
        result = self.diagonal_from_windows(wx)
        if self.include_rc:
            wx_rc = self.flat_windows(reverse_complement(x))
            rc_matches = xp.matmul(wx, wx_rc.transpose(0, 2, 1))
            result = result + self._apply_table(rc_matches)
        return result.astype(x.dtype)
