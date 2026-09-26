from __future__ import annotations

from math import comb

import numpy as np
from numba import njit, prange

from gkmsvm.backend import get_array_module, is_mlx
from gkmsvm.codec import reverse_complement
from gkmsvm.kernels.base import GkmKernel


# ---------------------------------------------------------------------------
# Numba-accelerated kernels
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True, fastmath=True)
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


@njit(inline="always")
def _popcount(x):
    """Hamming weight (number of set bits) for uint32."""
    x = np.uint32(x - ((x >> 1) & np.uint32(0x55555555)))
    x = np.uint32((x & np.uint32(0x33333333)) + ((x >> 2) & np.uint32(0x33333333)))
    x = np.uint32((x + (x >> 4)) & np.uint32(0x0F0F0F0F))
    return int(np.uint32(x * np.uint32(0x01010101)) >> 24)


@njit(inline="always")
def _packed_mismatches(a, b):
    """Count mismatching 2-bit positions between two packed uint32 values."""
    x = a ^ b
    return _popcount(((x >> 1) | x) & np.uint32(0x55555555))


@njit(parallel=True, cache=True, fastmath=True)
def _fused_pairwise_packed_numba(bx, by, table, l, min_matches):
    """Fused pairwise using packed uint32 comparison with min-matches skip."""
    B = bx.shape[0]
    Wx = bx.shape[1]
    S = by.shape[0]
    Wy = by.shape[1]
    result = np.empty((B, S), dtype=np.float64)
    for idx in prange(B * S):
        b = idx // S
        s = idx % S
        acc = 0.0
        for i in range(Wx):
            bx_val = bx[b, i]
            for j in range(Wy):
                matches = l - _packed_mismatches(bx_val, by[s, j])
                if matches >= min_matches:
                    acc += table[l - matches]
        result[b, s] = acc
    return result


@njit(parallel=True, cache=True, fastmath=True)
def _fused_diagonal_packed_numba(bx, table, l, min_matches):
    """Self-kernel diagonal from packed uint32 windows."""
    B = bx.shape[0]
    W = bx.shape[1]
    result = np.empty(B, dtype=np.float64)
    for b in prange(B):
        acc = 0.0
        for i in range(W):
            bx_val = bx[b, i]
            for j in range(W):
                matches = l - _packed_mismatches(bx_val, bx[b, j])
                if matches >= min_matches:
                    acc += table[l - matches]
        result[b] = acc
    return result


@njit(parallel=True, cache=True, fastmath=True)
def _fused_cross_diagonal_packed_numba(bx, bx_rc, table, l, min_matches):
    """Cross-kernel diagonal (fwd vs RC) from packed uint32 windows."""
    B = bx.shape[0]
    W = bx.shape[1]
    W_rc = bx_rc.shape[1]
    result = np.empty(B, dtype=np.float64)
    for b in prange(B):
        acc = 0.0
        for i in range(W):
            bx_val = bx[b, i]
            for j in range(W_rc):
                matches = l - _packed_mismatches(bx_val, bx_rc[b, j])
                if matches >= min_matches:
                    acc += table[l - matches]
        result[b] = acc
    return result


def _pack_windows_cpu(bw):
    """Pack int8 base-index windows [N, W, l] → [N, W] uint32 on CPU."""
    N, W, l = bw.shape
    packed = np.zeros((N, W), dtype=np.uint32)
    bw_u32 = bw.astype(np.uint32)
    for k in range(l):
        packed |= bw_u32[:, :, k] << (2 * k)
    return packed


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
    const unsigned int* __restrict__ bx_packed,
    const unsigned int* __restrict__ by_packed,
    const double* __restrict__ table,
    double* __restrict__ result,
    const int total_pairs,
    const int S,
    const int Wx,
    const int Wy,
    const int l,
    const int min_matches
) {
    // bx_packed: [B, Wx] uint32 — each element packs l bases (2 bits each).
    // by_packed: [Wy, S] uint32 — adjacent threads read adjacent uint32s
    //   (coalesced 128B transactions, replaces 11 strided byte reads).
    // Match count via XOR + popcount on 2-bit fields.
    extern __shared__ char smem[];
    double* s_table = (double*)smem;
    unsigned int* s_bx = (unsigned int*)((double*)smem + l + 1);

    for (int i = threadIdx.x; i <= l; i += blockDim.x) {
        s_table[i] = table[i];
    }

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // Cooperatively load query windows into shared memory when all
    // threads in the block share the same query index.
    const int block_start = blockDim.x * blockIdx.x;
    const int b_first = block_start / S;
    const int b_last = (block_start + blockDim.x - 1) / S;
    if (b_first == b_last) {
        const unsigned int* bx_src = bx_packed + b_first * Wx;
        for (int i = threadIdx.x; i < Wx; i += blockDim.x) {
            s_bx[i] = bx_src[i];
        }
    }
    __syncthreads();

    if (idx >= total_pairs) return;

    const int b = idx / S;
    const int s = idx % S;

    const bool use_shared = (b_first == b_last);

    double acc = 0.0;
    for (int i = 0; i < Wx; i++) {
        unsigned int bx_val = use_shared ? s_bx[i]
                                         : bx_packed[b * Wx + i];
        for (int j = 0; j < Wy; j++) {
            unsigned int by_val = by_packed[j * S + s];
            unsigned int x = bx_val ^ by_val;
            unsigned int mismatches = ((x >> 1) | x) & 0x55555555u;
            int mismatch_count = __popc(mismatches);
            int matches = l - mismatch_count;
            if (matches >= min_matches) {
                acc += s_table[l - matches];
            }
        }
    }
    result[idx] = acc;
}
"""

_cupy_fused_kernel = None
_cupy_fused_idx_kernel = None


def _fused_pairwise_gpu(wx, wy, table):
    """Fused pairwise CUDA kernel (float dot-product path)."""
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


def _pack_windows_uint32(bw, xp):
    """Pack int8 base-index windows [N, W, l] → [N, W] uint32 (2 bits per base)."""
    N, W, l = bw.shape
    packed = xp.zeros((N, W), dtype=xp.uint32)
    bw_u32 = bw.astype(xp.uint32)
    for k in range(l):
        packed |= bw_u32[:, :, k] << (2 * k)
    return packed


def _fused_pairwise_idx_gpu(bx, by, table, min_matches, *, by_packed_t=None):
    """Fused pairwise CUDA kernel (packed uint32 path, coalesced SV access).

    Args:
        bx: [B, Wx, l] int8 query windows.
        by: [S, Wy, l] int8 SV windows (ignored if by_packed_t is given).
        table: [l+1] mismatch weight table.
        min_matches: minimum match count for non-zero contribution.
        by_packed_t: optional pre-packed [Wy, S] uint32 SV windows.
    """
    global _cupy_fused_idx_kernel
    import cupy as cp

    if _cupy_fused_idx_kernel is None:
        _cupy_fused_idx_kernel = cp.RawKernel(
            _FUSED_PAIRWISE_IDX_CUDA, "fused_pairwise_idx",
            options=("--use_fast_math",),
        )

    B, Wx, l = bx.shape
    if by_packed_t is not None:
        Wy, S = by_packed_t.shape
    else:
        S, Wy, _ = by.shape
        by_packed_t = cp.ascontiguousarray(
            _pack_windows_uint32(cp.asarray(by), cp).T
        )

    bx_packed = cp.ascontiguousarray(_pack_windows_uint32(cp.asarray(bx), cp))
    table_gpu = cp.asarray(table, dtype=cp.float64)
    result = cp.empty(B * S, dtype=cp.float64)

    total = B * S
    block = 256
    grid = (total + block - 1) // block
    table_bytes = (l + 1) * 8
    bx_cache_bytes = Wx * 4
    shared_mem = table_bytes + bx_cache_bytes

    _cupy_fused_idx_kernel(
        (grid,), (block,),
        (bx_packed, by_packed_t, table_gpu, result,
         np.int32(total), np.int32(S), np.int32(Wx), np.int32(Wy),
         np.int32(l), np.int32(min_matches)),
        shared_mem=shared_mem,
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
        self._min_matches = self._get_min_matches()

    def _get_min_matches(self) -> int:
        """Minimum match count that yields a non-zero table entry."""
        for m in range(len(self._mismatch_table) - 1, -1, -1):
            if self._mismatch_table[m] != 0.0:
                return self.l - m
        return self.l + 1

    def flat_windows(self, x: np.ndarray) -> np.ndarray:
        """Extract length-l sliding windows, flattened across channels.

        Args:
            x: [B, 4, L] one-hot encoded sequences.

        Returns:
            [B, W, 4*l] where W = L - l + 1.
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
            x: [B, 4, L] one-hot encoded sequences.

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
        base_idx = xp.argmax(x, axis=1).astype(xp.int8)  # [B, L]
        strides = base_idx.strides
        shape = (B, W, self.l)
        new_strides = (strides[0], strides[1], strides[1])
        bw = xp.lib.stride_tricks.as_strided(base_idx, shape=shape, strides=new_strides)
        return xp.ascontiguousarray(bw)

    def _apply_table(self, matches: np.ndarray) -> np.ndarray:
        """Look up mismatch table from match counts and sum over window dims."""
        xp = get_array_module(matches)
        table = xp.asarray(self._mismatch_table)
        mismatches = xp.clip(xp.rint(self.l - matches).astype(xp.int64), 0, self.l)
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
        if is_mlx(wx):
            matches = xp.einsum("bif,sjf->bsij", wx, wy)
            return self._apply_table(matches)
        return _fused_pairwise_gpu(wx, wy, self._mismatch_table)

    def pairwise_from_indices(
        self, bx: np.ndarray, by: np.ndarray, *, by_packed_t=None
    ) -> np.ndarray:
        """Raw kernel from base-index windows (no RC, no normalization).

        Args:
            bx: [B, Wx, l] int8 base indices from query sequences.
            by: [S, Wy, l] int8 base indices from support sequences.
            by_packed_t: pre-packed SV windows. CPU: [S, Wy] uint32.
                GPU: [Wy, S] uint32 (transposed for coalesced access).

        Returns:
            [B, S] raw kernel values.
        """
        xp = get_array_module(bx)
        if xp is np:
            bx_packed = _pack_windows_cpu(np.ascontiguousarray(bx))
            by_packed = (
                by_packed_t if by_packed_t is not None
                else _pack_windows_cpu(np.ascontiguousarray(by))
            )
            return _fused_pairwise_packed_numba(
                bx_packed,
                by_packed,
                self._mismatch_table,
                self.l,
                self._min_matches,
            )
        if is_mlx(bx):
            xp = get_array_module(bx)
            B, Wx, l_dim = bx.shape
            S, Wy, _ = by.shape
            bases = xp.arange(4).reshape(1, 1, 1, 4)
            bx_oh = (bx[:, :, :, None] == bases).astype(xp.float32)
            by_oh = (by[:, :, :, None] == bases).astype(xp.float32)
            return self.pairwise_from_windows(
                bx_oh.reshape(B, Wx, 4 * l_dim),
                by_oh.reshape(S, Wy, 4 * l_dim),
            )
        return _fused_pairwise_idx_gpu(
            bx, by, self._mismatch_table, self._min_matches,
            by_packed_t=by_packed_t,
        )

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
            chunks = []
            for start in range(0, B, chunk_size):
                end = min(start + chunk_size, B)
                chunks.append(self._raw_diagonal(x[start:end]))
            return xp.concatenate(chunks)

        if xp is np:
            bx_packed = _pack_windows_cpu(self.base_index_windows(x))
            mm = self._min_matches
            result = _fused_diagonal_packed_numba(
                bx_packed, self._mismatch_table, self.l, mm
            )
            if self.include_rc:
                bx_rc_packed = _pack_windows_cpu(
                    self.base_index_windows(reverse_complement(x))
                )
                result = result + _fused_cross_diagonal_packed_numba(
                    bx_packed, bx_rc_packed, self._mismatch_table, self.l, mm
                )
            return result.astype(x.dtype)

        wx = self.flat_windows(x)
        result = self.diagonal_from_windows(wx)
        if self.include_rc:
            wx_rc = self.flat_windows(reverse_complement(x))
            rc_matches = xp.matmul(wx, wx_rc.transpose(0, 2, 1))
            result = result + self._apply_table(rc_matches)
        return result.astype(x.dtype)
