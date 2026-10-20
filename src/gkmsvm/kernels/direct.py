from __future__ import annotations

from math import comb

import numpy as np
from numba import njit, prange

from gkmsvm.kernels.packed import PAD_WINDOW, PackedGkmKernel, device_of

_PAD = PAD_WINDOW

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


@njit(cache=True)
def _spans(packed):
    """[R, 2] (start, end) of each row's non-PAD windows (one contiguous run:
    first in forward rows, last in RC rows)."""
    R, W = packed.shape
    out = np.zeros((R, 2), dtype=np.int64)
    for r in range(R):
        start = 0
        while start < W and packed[r, start] == _PAD:
            start += 1
        end = start
        while end < W and packed[r, end] != _PAD:
            end += 1
        out[r, 0] = start
        out[r, 1] = end
    return out


@njit(cache=True)
def _counts(packed):
    """Number of windows before the first PAD in each row (forward rows)."""
    R, W = packed.shape
    out = np.empty(R, dtype=np.int64)
    for r in range(R):
        n = 0
        while n < W and packed[r, n] != _PAD:
            n += 1
        out[r] = n
    return out


@njit(parallel=True, cache=True, fastmath=True)
def _fused_pairwise_packed_numba(bx, by, table, l, min_matches):
    """Fused pairwise using packed uint32 comparison with min-matches skip;
    PAD windows (rows of shorter sequences) are skipped."""
    B = bx.shape[0]
    S = by.shape[0]
    sx, sy = _spans(bx), _spans(by)
    result = np.empty((B, S), dtype=np.float64)
    for idx in prange(B * S):
        b = idx // S
        s = idx % S
        acc = 0.0
        for i in range(sx[b, 0], sx[b, 1]):
            bx_val = bx[b, i]
            for j in range(sy[s, 0], sy[s, 1]):
                matches = l - _packed_mismatches(bx_val, by[s, j])
                if matches >= min_matches:
                    acc += table[l - matches]
        result[b, s] = acc
    return result


@njit(inline="always")
def _rc_window(w, l):
    """Packed reverse complement of one packed l-mer."""
    out = np.uint32(0)
    for k in range(l):
        out |= (np.uint32(3) - ((w >> np.uint32(2 * k)) & np.uint32(3))) << np.uint32(
            2 * (l - 1 - k)
        )
    return out


@njit(inline="always")
def _g(u, v, table):
    """Pair weight table[mismatch count]. Branch-free: zero past the cutoff."""
    return table[_packed_mismatches(u, v)]


@njit(inline="always")
def _block_sum(a, c, symmetric, table):
    """sum over i, j of g(a[i], c[j]); for a symmetric matrix only pairs j >= i are visited.

    Rows go in blocks of 4 so that each pass over c serves four rows (a loop
    pass, not a pair, is the cost on short rows).
    """
    W = a.shape[0]
    diag = 0.0  # j == i, symmetric only
    off = 0.0  # j > i when symmetric, every j otherwise
    i0 = 0
    while i0 < W:
        n = min(4, W - i0)
        if n == 4 and not symmetric:
            u0, u1, u2, u3 = a[i0], a[i0 + 1], a[i0 + 2], a[i0 + 3]
            for j in range(W):
                v = c[j]
                off += (
                    _g(u0, v, table)
                    + _g(u1, v, table)
                    + _g(u2, v, table)
                    + _g(u3, v, table)
                )
        elif n == 4:
            u0, u1, u2, u3 = a[i0], a[i0 + 1], a[i0 + 2], a[i0 + 3]
            diag += (
                _g(u0, c[i0], table)
                + _g(u1, c[i0 + 1], table)
                + _g(u2, c[i0 + 2], table)
                + _g(u3, c[i0 + 3], table)
            )
            off += (
                _g(u0, c[i0 + 1], table)
                + _g(u0, c[i0 + 2], table)
                + _g(u0, c[i0 + 3], table)
                + _g(u1, c[i0 + 2], table)
                + _g(u1, c[i0 + 3], table)
                + _g(u2, c[i0 + 3], table)
            )
            for j in range(i0 + 4, W):
                v = c[j]
                off += (
                    _g(u0, v, table)
                    + _g(u1, v, table)
                    + _g(u2, v, table)
                    + _g(u3, v, table)
                )
        else:
            for i in range(i0, W):
                for j in range(i if symmetric else 0, W):
                    if symmetric and j == i:
                        diag += _g(a[i], c[j], table)
                    else:
                        off += _g(a[i], c[j], table)
        i0 += n
    return diag + 2.0 * off if symmetric else off


@njit(parallel=True, cache=True, fastmath=True)
def _self_diagonal_packed_numba(bx, by, table, l):
    """K_raw(x, x) [B] from packed windows; by is bx, or bx followed by its RC windows.

    Both blocks are symmetric, so each unordered pair is visited once: the
    forward block as (i, j), the RC block as (i, W-1-j), since its column j is
    rc(window W-1-j). A row whose RC block isn't the reverse complement of its
    forward one visits every RC pair. A padded row (a shorter sequence) is
    its first n windows, and its RC block reversed, whose pads come last too.
    """
    B, W = bx.shape
    has_rc = by.shape[1] > W
    counts = _counts(bx)
    result = np.empty(B, dtype=np.float64)
    for b in prange(B):
        n = counts[b]
        fwd = bx[b, :n]
        acc = _block_sum(fwd, fwd, True, table)
        if has_rc:
            # RC windows in forward order: rev[k] = rc(fwd[k]) when paired
            rev = by[b, W:][::-1][:n].copy()
            paired = True
            for k in range(n):
                if rev[k] != _rc_window(fwd[k], l):
                    paired = False
                    break
            acc += _block_sum(fwd, rev, paired, table)
        result[b] = acc
    return result


@njit(inline="always")
def _block_delta(f, nf, a_s, a_e, c, nc, c_s, c_e, table):
    """sum over i, j of g(f'_i, c'_j) - g(f_i, c_j), where f' is f with windows
    a_s..a_e replaced by nf and c' is c with windows c_s..c_e replaced by nc."""
    d = 0.0
    for i in range(a_s, a_e):  # changed rows, every column
        u, u_new = f[i], nf[i - a_s]
        for j in range(c.shape[0]):
            cj = c[j]
            cj_new = nc[j - c_s] if c_s <= j < c_e else cj
            d += _g(u_new, cj_new, table) - _g(u, cj, table)
    for i in range(f.shape[0]):  # unchanged rows, changed columns
        if a_s <= i < a_e:
            continue
        u = f[i]
        for j in range(c_s, c_e):
            d += _g(u, nc[j - c_s], table) - _g(u, c[j], table)
    return d


@njit(parallel=True, cache=True, fastmath=True)
def _mutant_diag_numba(fwd, rc, new, new_rc, diag, table, l):
    """K_raw(mutant, mutant) [4, L, B]: the reference's plus the pairs touching a
    changed window (see PackedGkmKernel.raw_diag_mutants; rc has 0 columns
    without RC). A padded row is its first n windows (its RC row's last n);
    a position past them changes nothing."""
    B, W = fwd.shape
    L = new.shape[1]
    has_rc = rc.shape[1] > 0
    counts = _counts(fwd)
    out = np.empty((4, L, B))
    for item in prange(L * B):
        p, b = item // B, item % B
        n = counts[b]
        a_s, a_e = max(0, p - l + 1), min(p + 1, n)
        if a_s >= a_e:
            for v in range(4):
                out[v, p, b] = diag[b]
            continue
        f = fwd[b, :n]
        # RC windows r_s..r_e change; the row's own are W-n.. (new_rc from r_s)
        q = L - 1 - p
        r_s, r_e = max(0, q - l + 1), min(q + 1, W)
        o = W - n
        c_s = max(r_s, o)
        for v in range(4):
            nf = new[v, p, b]
            d = _block_delta(f, nf, a_s, a_e, f, nf, a_s, a_e, table)
            if has_rc:
                d += _block_delta(
                    f,
                    nf,
                    a_s,
                    a_e,
                    rc[b, o:],
                    new_rc[v, p, b, c_s - r_s :],
                    c_s - o,
                    r_e - o,
                    table,
                )
            out[v, p, b] = diag[b] + d
    return out


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

    // PAD windows (0xFFFFFFFF, past a shorter sequence's) are skipped
    double acc = 0.0;
    for (int i = 0; i < Wx; i++) {
        unsigned int bx_val = use_shared ? s_bx[i]
                                         : bx_packed[b * Wx + i];
        if (bx_val == 0xFFFFFFFFu) continue;
        for (int j = 0; j < Wy; j++) {
            unsigned int by_val = by_packed[j * S + s];
            if (by_val == 0xFFFFFFFFu) continue;
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

    // PAD windows (0xFFFFFFFF, past a shorter sequence's) are skipped
    double acc = 0.0;
    for (int i = 0; i < W; i++) {
        unsigned int bx_val = bx_packed[b * W + i];
        if (bx_val == 0xFFFFFFFFu) continue;
        for (int j = 0; j < W_rc; j++) {
            unsigned int y = bx_rc_packed[b * W_rc + j];
            if (y == 0xFFFFFFFFu) continue;
            unsigned int x = bx_val ^ y;
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
        (
            bx_packed,
            bx_rc_packed,
            table_gpu,
            result,
            np.int32(B),
            np.int32(W),
            np.int32(W_rc),
            np.int32(l),
            np.int32(min_matches),
        ),
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

    // PAD windows (0xFFFFFFFF, past a shorter sequence's) are skipped
    float acc = 0.0f;
    for (uint i = 0; i < Wx; i++) {
        uint32_t bx_val = bx_packed[b * Wx + i];
        if (bx_val == 0xFFFFFFFFu) continue;
        for (uint j = 0; j < Wy; j++) {
            uint32_t by_val = by_packed[s * Wy + j];
            if (by_val == 0xFFFFFFFFu) continue;
            uint32_t xored = bx_val ^ by_val;
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

    // PAD windows (0xFFFFFFFF, past a shorter sequence's) are skipped
    float acc = 0.0f;
    for (uint i = 0; i < W; i++) {
        uint32_t bx_val = bx_packed[idx * W + i];
        if (bx_val == 0xFFFFFFFFu) continue;
        for (uint j = 0; j < W_rc; j++) {
            uint32_t y = bx_rc_packed[idx * W_rc + j];
            if (y == 0xFFFFFFFFu) continue;
            uint32_t xored = bx_val ^ y;
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
        (
            bx_packed,
            by_t,
            table_gpu,
            result,
            np.int64(total),
            np.int32(S),
            np.int32(Wx),
            np.int32(Wy),
            np.int32(l),
            np.int32(min_matches),
        ),
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

    d: pairs with more than d mismatches count 0, as in LS-GKM, whose
    search visits only pairs within d mismatches for every kernel type
    (default 3, which drops m = 4 at l = 11, k = 7). None counts every
    pair (the full gapped k-mer count).
    """

    def __init__(
        self,
        l: int,
        k: int,
        *,
        d: int | None = None,
        normalize: bool = True,
        include_rc: bool = True,
    ):
        super().__init__(l, k, normalize=normalize, include_rc=include_rc)
        self.d = d
        self._mismatch_table = np.array(
            [
                comb(l - m, k) if l - m >= k and (d is None or m <= d) else 0
                for m in range(l + 1)
            ],
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
                np.ascontiguousarray(bx), np.ascontiguousarray(by), *args
            )
        if dev == "cuda":
            return _fused_pairwise_packed_gpu(bx, by, *args)
        return _fused_pairwise_packed_mlx(bx, by, *args)

    def raw_diag_mutants(self, bx, bx_rc, new, new_rc, diag):
        """See PackedGkmKernel.raw_diag_mutants; on CPU only pairs touching a
        changed window are visited (about 4 l W per mutant instead of W^2)."""
        if device_of(bx) != "cpu":
            return super().raw_diag_mutants(bx, bx_rc, new, new_rc, diag)
        B = bx.shape[0]
        rc = (
            np.zeros((B, 0), np.uint32)
            if bx_rc is None
            else np.ascontiguousarray(bx_rc)
        )
        new_rc = np.zeros((4, 1, 1, 1), np.uint32) if new_rc is None else new_rc
        return _mutant_diag_numba(
            np.ascontiguousarray(bx),
            rc,
            np.ascontiguousarray(new),
            np.ascontiguousarray(new_rc),
            np.asarray(diag, dtype=np.float64),
            self._mismatch_table,
            self.l,
        )

    def _brute_cross_diag(self, bx, by, dev):
        args = (self._mismatch_table, self.l, self._min_matches)
        if dev == "cpu":
            return _self_diagonal_packed_numba(
                np.ascontiguousarray(bx), np.ascontiguousarray(by), *args[:2]
            )
        if dev == "cuda":
            import cupy as cp

            return _fused_cross_diagonal_packed_gpu(
                cp.ascontiguousarray(bx), cp.ascontiguousarray(by), *args
            )
        return _fused_cross_diagonal_packed_mlx(bx, by, *args)
