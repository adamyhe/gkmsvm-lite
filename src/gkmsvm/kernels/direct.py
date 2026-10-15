from __future__ import annotations

from math import comb

import numpy as np
from numba import njit, prange

from gkmsvm.kernels.packed import PackedGkmKernel

# ---------------------------------------------------------------------------
# Numba-accelerated kernels
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CuPy CUDA kernels
# ---------------------------------------------------------------------------

_FUSED_PAIRWISE_IDX_CUDA = r"""
extern "C" __global__
void fused_pairwise_idx(
    const unsigned int* __restrict__ bx_packed,
    const unsigned int* __restrict__ by_packed,
    const double* __restrict__ table,
    double* __restrict__ result,
    const long long total_pairs,
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

    const long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;

    // Cooperatively load query windows into shared memory when all
    // threads in the block share the same query index.
    const long long block_start = (long long)blockDim.x * blockIdx.x;
    const long long b_first = block_start / S;
    const long long b_last = (block_start + blockDim.x - 1) / S;
    if (b_first == b_last) {
        const unsigned int* bx_src = bx_packed + b_first * Wx;
        for (int i = threadIdx.x; i < Wx; i += blockDim.x) {
            s_bx[i] = bx_src[i];
        }
    }
    __syncthreads();

    if (idx >= total_pairs) return;

    const long long b = idx / S;
    const int s = (int)(idx % S);

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

_FUSED_CROSS_DIAGONAL_CUDA = r"""
extern "C" __global__
void fused_cross_diagonal(
    const unsigned int* __restrict__ bx_packed,
    const unsigned int* __restrict__ bx_rc_packed,
    const double* __restrict__ table,
    double* __restrict__ result,
    const int B,
    const int W,
    const int W_rc,
    const int l,
    const int min_matches
) {
    extern __shared__ double s_table[];
    for (int i = threadIdx.x; i <= l; i += blockDim.x) {
        s_table[i] = table[i];
    }
    __syncthreads();

    const int b = blockDim.x * blockIdx.x + threadIdx.x;
    if (b >= B) return;

    double acc = 0.0;
    for (int i = 0; i < W; i++) {
        unsigned int bx_val = bx_packed[b * W + i];
        for (int j = 0; j < W_rc; j++) {
            unsigned int x = bx_val ^ bx_rc_packed[b * W_rc + j];
            unsigned int mismatches = ((x >> 1) | x) & 0x55555555u;
            int mismatch_count = __popc(mismatches);
            int matches = l - mismatch_count;
            if (matches >= min_matches) {
                acc += s_table[l - matches];
            }
        }
    }
    result[b] = acc;
}
"""

_cupy_fused_idx_kernel = None
_cupy_fused_cross_diag_kernel = None


def _fused_cross_diagonal_packed_gpu(bx_packed, bx_rc_packed, table, l, min_matches):
    """Cross-kernel diagonal (fwd x RC) via CUDA RawKernel."""
    global _cupy_fused_cross_diag_kernel
    import cupy as cp

    if _cupy_fused_cross_diag_kernel is None:
        _cupy_fused_cross_diag_kernel = cp.RawKernel(
            _FUSED_CROSS_DIAGONAL_CUDA,
            "fused_cross_diagonal",
            options=("--use_fast_math",),
        )

    B, W = bx_packed.shape
    _, W_rc = bx_rc_packed.shape
    table_gpu = cp.asarray(table, dtype=cp.float64)
    result = cp.empty(B, dtype=cp.float64)

    block = 256
    grid = (B + block - 1) // block

    _cupy_fused_cross_diag_kernel(
        (grid,),
        (block,),
        (bx_packed, bx_rc_packed, table_gpu, result,
         np.int32(B), np.int32(W), np.int32(W_rc),
         np.int32(l), np.int32(min_matches)),
        shared_mem=(l + 1) * 8,
    )
    return result


# ---------------------------------------------------------------------------
# MLX Metal kernels — fused XOR + popcount with per-thread early exit
# ---------------------------------------------------------------------------

_METAL_PAIRWISE_SOURCE = """
    uint idx = thread_position_in_grid.x;
    uint total_pairs = (uint)params[0];
    uint S = (uint)params[1];
    uint Wx = (uint)params[2];
    uint Wy = (uint)params[3];
    uint l = (uint)params[4];
    uint min_matches = (uint)params[5];

    if (idx >= total_pairs) return;

    uint b = idx / S;
    uint s = idx % S;

    float acc = 0.0f;
    for (uint i = 0; i < Wx; i++) {
        uint32_t bx_val = bx_packed[b * Wx + i];
        for (uint j = 0; j < Wy; j++) {
            uint32_t xored = bx_val ^ by_packed[s * Wy + j];
            uint32_t mask = ((xored >> 1) | xored) & 0x55555555u;
            uint mm = popcount(mask);
            uint matches = l - mm;
            if (matches >= min_matches) {
                acc += table[l - matches];
            }
        }
    }
    result[b * S + s] = acc;
"""

_METAL_CROSS_DIAGONAL_SOURCE = """
    uint idx = thread_position_in_grid.x;
    uint B = (uint)params[0];
    uint W = (uint)params[1];
    uint W_rc = (uint)params[2];
    uint l = (uint)params[3];
    uint min_matches = (uint)params[4];

    if (idx >= B) return;

    float acc = 0.0f;
    for (uint i = 0; i < W; i++) {
        uint32_t bx_val = bx_packed[idx * W + i];
        for (uint j = 0; j < W_rc; j++) {
            uint32_t xored = bx_val ^ bx_rc_packed[idx * W_rc + j];
            uint32_t mask = ((xored >> 1) | xored) & 0x55555555u;
            uint mm = popcount(mask);
            uint matches = l - mm;
            if (matches >= min_matches) {
                acc += table[l - matches];
            }
        }
    }
    result[idx] = acc;
"""

_mlx_pairwise_kernel = None
_mlx_cross_diagonal_kernel = None


def _get_mlx_pairwise_kernel():
    global _mlx_pairwise_kernel
    if _mlx_pairwise_kernel is None:
        import mlx.core as mx
        _mlx_pairwise_kernel = mx.fast.metal_kernel(
            name="gkm_fused_pairwise",
            input_names=["bx_packed", "by_packed", "table", "params"],
            output_names=["result"],
            source=_METAL_PAIRWISE_SOURCE,
        )
    return _mlx_pairwise_kernel


def _get_mlx_cross_diagonal_kernel():
    global _mlx_cross_diagonal_kernel
    if _mlx_cross_diagonal_kernel is None:
        import mlx.core as mx
        _mlx_cross_diagonal_kernel = mx.fast.metal_kernel(
            name="gkm_fused_cross_diagonal",
            input_names=["bx_packed", "bx_rc_packed", "table", "params"],
            output_names=["result"],
            source=_METAL_CROSS_DIAGONAL_SOURCE,
        )
    return _mlx_cross_diagonal_kernel


def _fused_pairwise_packed_mlx(bx_packed, by_packed, table, l, min_matches):
    """Fused pairwise via Metal kernel — per-thread XOR+popcount with early exit."""
    import mlx.core as mx
    B, Wx = bx_packed.shape
    S, Wy = by_packed.shape
    total = B * S
    table_mlx = mx.array(np.asarray(table, dtype=np.float32))
    params = mx.array([total, S, Wx, Wy, l, min_matches], dtype=mx.int32)
    block = min(total, 256)
    grid = ((total + block - 1) // block) * block
    result = _get_mlx_pairwise_kernel()(
        inputs=[bx_packed, by_packed, table_mlx, params],
        template=[],
        grid=(grid, 1, 1),
        threadgroup=(block, 1, 1),
        output_shapes=[(total,)],
        output_dtypes=[mx.float32],
    )[0]
    return result.reshape(B, S)


def _fused_cross_diagonal_packed_mlx(bx_packed, bx_rc_packed, table, l, min_matches):
    """Cross-kernel diagonal (fwd × RC) via Metal kernel."""
    import mlx.core as mx
    B, W = bx_packed.shape
    _, W_rc = bx_rc_packed.shape
    table_mlx = mx.array(np.asarray(table, dtype=np.float32))
    params = mx.array([B, W, W_rc, l, min_matches], dtype=mx.int32)
    block = min(B, 256)
    grid = ((B + block - 1) // block) * block
    return _get_mlx_cross_diagonal_kernel()(
        inputs=[bx_packed, bx_rc_packed, table_mlx, params],
        template=[],
        grid=(grid, 1, 1),
        threadgroup=(block, 1, 1),
        output_shapes=[(B,)],
        output_dtypes=[mx.float32],
    )[0]


def _fused_pairwise_packed_gpu(bx_packed, by_packed, table, l, min_matches):
    """Fused pairwise CUDA kernel: bx [B, Wx], by [S, Wy] packed uint32.

    The kernel reads columns as [Wy, S] (adjacent threads, adjacent SVs), so
    by is transposed per call.
    """
    global _cupy_fused_idx_kernel
    import cupy as cp

    if _cupy_fused_idx_kernel is None:
        _cupy_fused_idx_kernel = cp.RawKernel(
            _FUSED_PAIRWISE_IDX_CUDA,
            "fused_pairwise_idx",
            options=("--use_fast_math",),
        )

    bx_packed = cp.ascontiguousarray(bx_packed)
    B, Wx = bx_packed.shape
    S, Wy = by_packed.shape
    by_t = cp.ascontiguousarray(by_packed.T)
    table_gpu = cp.asarray(table, dtype=cp.float64)
    result = cp.empty(B * S, dtype=cp.float64)

    total = B * S
    block = 256
    grid = (total + block - 1) // block
    shared_mem = (l + 1) * 8 + Wx * 4

    _cupy_fused_idx_kernel(
        (grid,),
        (block,),
        (bx_packed, by_t, table_gpu, result, np.int64(total), np.int32(S),
         np.int32(Wx), np.int32(Wy), np.int32(l), np.int32(min_matches)),
        shared_mem=shared_mem,
    )
    return result.reshape(B, S)


# ---------------------------------------------------------------------------
# Kernel class
# ---------------------------------------------------------------------------


class DirectGkmKernel(PackedGkmKernel):
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

    def _join_plan(self):
        """Mask plan for this kernel's table, or None without a mismatch cutoff."""
        from gkmsvm.kernels.masked import mask_plan

        return mask_plan(tuple(self._mismatch_table), self.l, self._min_matches)

    def _get_min_matches(self) -> int:
        """Minimum match count that yields a non-zero table entry."""
        for m in range(len(self._mismatch_table) - 1, -1, -1):
            if self._mismatch_table[m] != 0.0:
                return self.l - m
        return self.l + 1


    _HAS_BRUTE = True

    def _brute(self, bx, by, dev):
        args = (self._mismatch_table, self.l, self._min_matches)
        if dev == "cpu":
            return _fused_pairwise_packed_numba(
                np.ascontiguousarray(bx), np.ascontiguousarray(by), *args)
        if dev == "cuda":
            return _fused_pairwise_packed_gpu(bx, by, *args)
        return _fused_pairwise_packed_mlx(bx, by, *args)

    def _brute_cross_diag(self, bx, by, dev):
        args = (self._mismatch_table, self.l, self._min_matches)
        if dev == "cpu":
            return _fused_cross_diagonal_packed_numba(
                np.ascontiguousarray(bx), np.ascontiguousarray(by), *args)
        if dev == "cuda":
            import cupy as cp

            return _fused_cross_diagonal_packed_gpu(
                cp.ascontiguousarray(bx), cp.ascontiguousarray(by), *args)
        return _fused_cross_diagonal_packed_mlx(bx, by, *args)
