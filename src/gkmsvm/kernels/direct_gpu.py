"""CUDA and Metal versions of direct.py's loops: brute-force pairs, cross
self-kernels, and mutant self-kernels (one thread block per base, position
and sequence)."""

from __future__ import annotations

from functools import cache

import numpy as np

from gkmsvm.kernels.packed import PAD_WINDOW, device_of
from gkmsvm.kernels.primitives import row_stride

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


@cache
def _cuda_kernel(source, name):
    import cupy as cp

    return cp.RawKernel(source, name, options=("--use_fast_math",))


def _fused_cross_diagonal_packed_gpu(bx_packed, bx_rc_packed, table, l, min_matches):
    """Cross-kernel diagonal (fwd x RC) via CUDA RawKernel."""
    import cupy as cp

    B, W = bx_packed.shape
    _, W_rc = bx_rc_packed.shape
    table_gpu = cp.asarray(table, dtype=cp.float64)
    result = cp.empty(B, dtype=cp.float64)

    block = 256
    grid = (B + block - 1) // block

    _cuda_kernel(_FUSED_CROSS_DIAGONAL_CUDA, "fused_cross_diagonal")(
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
# MLX Metal kernels — fused XOR + popcount, one thread per output
# ---------------------------------------------------------------------------

# params: total, S, Wx, Wy, l, min_matches
_METAL_PAIRWISE_SOURCE = """
    uint idx = thread_position_in_grid.x;
    uint total = (uint)params[0];
    uint S = (uint)params[1];
    uint Wx = (uint)params[2];
    uint Wy = (uint)params[3];
    uint l = (uint)params[4];
    uint min_matches = (uint)params[5];

    if (idx >= total) return;

    // DIAG: row b against row b of by (self-kernels), else column c
    uint b = DIAG ? idx : idx / S;
    uint c = DIAG ? idx : idx % S;

    // PAD windows (0xFFFFFFFF, past a shorter sequence's) are skipped
    float acc = 0.0f;
    for (uint i = 0; i < Wx; i++) {
        uint32_t bx_val = bx_packed[b * Wx + i];
        if (bx_val == 0xFFFFFFFFu) continue;
        for (uint j = 0; j < Wy; j++) {
            uint32_t by_val = by_packed[c * Wy + j];
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
    result[idx] = acc;
"""


@cache
def _mlx_pairwise_kernel(diag: bool):
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gkm_fused_cross_diagonal" if diag else "gkm_fused_pairwise",
        input_names=["bx_packed", "by_packed", "table", "params"],
        output_names=["result"],
        source=_METAL_PAIRWISE_SOURCE.replace("DIAG", "true" if diag else "false"),
    )


def _fused_pairwise_packed_mlx(bx_packed, by_packed, table, l, min_matches, diag=False):
    """Fused pairwise [B, S] via Metal kernel, one thread per pair (diag: [B],
    row b against row b of by_packed)."""
    import mlx.core as mx

    B, Wx = bx_packed.shape
    S, Wy = by_packed.shape
    total = B if diag else B * S
    table_mlx = mx.array(np.asarray(table, dtype=np.float32))
    params = mx.array([total, S, Wx, Wy, l, min_matches], dtype=mx.int32)
    block = min(total, 256)
    grid = ((total + block - 1) // block) * block
    result = _mlx_pairwise_kernel(diag)(
        inputs=[bx_packed, by_packed, table_mlx, params],
        template=[],
        grid=(grid, 1, 1),
        threadgroup=(block, 1, 1),
        output_shapes=[(total,)],
        output_dtypes=[mx.float32],
    )[0]
    return result if diag else result.reshape(B, S)


# ---------------------------------------------------------------------------
# CUDA and Metal: mutant self-kernels (_mutant_diag_numba on the GPU). One
# thread block per (base, position, sequence); its threads stride over the
# pairs touching a changed window: changed rows x every column, then
# unchanged rows x changed columns, for the forward block and the RC block.
# Window weights are [rows or 1, W] with a row stride (0 when shared; all
# ones for the unweighted kernels).
# ---------------------------------------------------------------------------

_MUTANT_DIAG_BODY = """
    long long item = BLOCK_ID;  // (v * L + p) * B + b
    int b = (int)(item % B), p = (int)((item / B) % L);
    int n = counts[b];
    int a_s = max(0, p - l + 1), a_e = min(p + 1, n);
    const REAL_W* wb = w + (long long)b * w_stride;
    const unsigned int* f = fwd + (long long)b * W;
    const unsigned int* nf = new_f + item * l - a_s;  // nf[i]: window i after
    bool same = true;  // the reference base: nothing changes
    for (int i = a_s; i < a_e; i++) same = same && nf[i] == f[i];
    // RC block: the row's windows are W-n.., column j weighs forward window n-1-j
    int q = L - 1 - p, r_s = max(0, q - l + 1), r_e = min(q + 1, W);
    int o = W - n;
    const unsigned int* c = rc + (long long)b * W + o;
    const unsigned int* nc = new_rc + item * l + (o - r_s);
    ACC d = 0;
    for (int blk = 0; !same && blk < (has_rc ? 2 : 1); blk++) {
        bool fw = blk == 0;
        const unsigned int* cc = fw ? f : c;
        const unsigned int* ncc = fw ? nf : nc;
        int c_s = fw ? a_s : max(r_s, o) - o, c_e = fw ? a_e : r_e - o;
        // changed rows, every column
        for (int i = a_s; i < a_e; i++) {
            unsigned int u = f[i], u2 = nf[i];
            ACC row = 0;
            for (int j = TID; j < n; j += NT) {
                unsigned int cj = cc[j];
                unsigned int c2 = (j >= c_s && j < c_e) ? ncc[j] : cj;
                ACC wj = fw ? wb[j] : wb[n - 1 - j];
                row += wj * ((ACC)table[n_mismatch(u2, c2)] - (ACC)table[n_mismatch(u, cj)]);
            }
            d += (ACC)wb[i] * row;
        }
        // unchanged rows, changed columns
        for (int i = TID; i < n; i += NT) {
            if (i >= a_s && i < a_e) continue;
            unsigned int u = f[i];
            ACC row = 0;
            for (int j = c_s; j < c_e; j++) {
                ACC wj = fw ? wb[j] : wb[n - 1 - j];
                row += wj * ((ACC)table[n_mismatch(u, ncc[j])] - (ACC)table[n_mismatch(u, cc[j])]);
            }
            d += (ACC)wb[i] * row;
        }
    }
"""

_MUTANT_DIAG_CUDA = (
    r"""
__device__ __forceinline__ int n_mismatch(unsigned int a, unsigned int b)
{
    unsigned int x = a ^ b;
    return __popc(((x >> 1) | x) & 0x55555555u);
}

extern "C" __global__ void mutant_diag(
    const unsigned int* __restrict__ fwd, const unsigned int* __restrict__ rc,
    const unsigned int* __restrict__ new_f, const unsigned int* __restrict__ new_rc,
    const int* __restrict__ counts, const double* __restrict__ w, int w_stride,
    const double* __restrict__ table, const double* __restrict__ diag,
    double* __restrict__ out, int B, int L, int W, int l, int has_rc)
{
    __shared__ double partial[256];
"""
    + _MUTANT_DIAG_BODY.replace("BLOCK_ID", "blockIdx.x")
    .replace("REAL_W", "double")
    .replace("ACC", "double")
    .replace("TID", "threadIdx.x")
    .replace("NT", "blockDim.x")
    + r"""
    partial[threadIdx.x] = d;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[item] = diag[b] + partial[0];
}
"""
)

_METAL_MUTANT_DIAG = (
    """
    int B = params[0], L = params[1], W = params[2], l = params[3];
    int has_rc = params[4], w_stride = params[5];
    uint tid = thread_position_in_threadgroup.x;
    uint nt = threads_per_threadgroup.x;
    threadgroup float partial[32];
"""
    + _MUTANT_DIAG_BODY.replace("BLOCK_ID", "threadgroup_position_in_grid.x")
    .replace("const REAL_W*", "const device float*")
    .replace("const unsigned int*", "const device uint*")
    .replace("ACC", "float")
    .replace("TID", "tid")
    .replace("NT", "nt")
    .replace("long long", "long")
    + """
    d = simd_sum(d);
    if (thread_index_in_simdgroup == 0) partial[simdgroup_index_in_threadgroup] = d;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0.0f;
        for (uint g = 0; g < (nt + 31) / 32; g++) s += partial[g];
        out[item] = diag[b] + s;
    }
"""
)

_METAL_MUTANT_DIAG_HEADER = """
inline int n_mismatch(uint a, uint b) {
    uint x = a ^ b;
    return popcount(((x >> 1) | x) & 0x55555555u);
}
"""

_MUTANT_DIAG_THREADS = 128


@cache
def _cuda_mutant_diag():
    import cupy as cp

    return cp.RawKernel(_MUTANT_DIAG_CUDA, "mutant_diag")


@cache
def _metal_mutant_diag():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gkm_mutant_diag",
        input_names=["fwd", "rc", "new_f", "new_rc", "counts", "w", "table"]
        + ["diag", "params"],
        output_names=["out"],
        source=_METAL_MUTANT_DIAG,
        header=_METAL_MUTANT_DIAG_HEADER,
    )


def _mutant_diag_device(bx, bx_rc, new, new_rc, diag, w, table, l):
    """K_raw(mutant, mutant) [4, L, B] on bx's device (CUDA or MLX); the
    arguments as for _mutant_diag_numba, on bx's device or on CPU. w: [B or 1,
    W] window weights, or None (all 1)."""
    dev = device_of(bx)
    B, W = bx.shape
    L = new.shape[1]
    has_rc = bx_rc is not None
    w = np.ones((1, W)) if w is None else w
    params = (B, L, W, l, int(has_rc), row_stride(w, W))
    new = np.ascontiguousarray(new, dtype=np.uint32)
    new_rc = new if new_rc is None else np.ascontiguousarray(new_rc, dtype=np.uint32)
    if dev == "cuda":
        import cupy as cp

        fwd = cp.ascontiguousarray(bx, dtype=cp.uint32)
        rc = fwd if bx_rc is None else cp.ascontiguousarray(bx_rc, dtype=cp.uint32)
        out = cp.empty(4 * L * B, dtype=cp.float64)
        _cuda_mutant_diag()(
            (4 * L * B,),
            (_MUTANT_DIAG_THREADS,),
            (fwd, rc, cp.asarray(new), cp.asarray(new_rc))
            + ((fwd != PAD_WINDOW).sum(axis=1, dtype=cp.int32),)
            + (cp.asarray(w, dtype=cp.float64), np.int32(params[5]))
            + (cp.asarray(table, dtype=cp.float64),)
            + (cp.asarray(diag, dtype=cp.float64), out)
            + tuple(np.int32(a) for a in params[:5]),
        )
        return out.reshape(4, L, B)
    import mlx.core as mx

    fwd = mx.array(bx).astype(mx.uint32)
    rc = fwd if bx_rc is None else mx.array(bx_rc).astype(mx.uint32)
    out = _metal_mutant_diag()(
        inputs=[fwd, rc, mx.array(new), mx.array(new_rc)]
        + [(fwd != PAD_WINDOW).astype(mx.int32).sum(axis=1)]
        + [mx.array(np.asarray(w, dtype=np.float32))]
        + [mx.array(np.asarray(table, dtype=np.float32))]
        + [mx.array(diag).astype(mx.float32), mx.array(params, dtype=mx.int32)],
        template=[],
        grid=(4 * L * B * _MUTANT_DIAG_THREADS, 1, 1),
        threadgroup=(_MUTANT_DIAG_THREADS, 1, 1),
        output_shapes=[(4 * L * B,)],
        output_dtypes=[mx.float32],
    )[0]
    return out.reshape(4, L, B)


def _fused_pairwise_packed_gpu(bx_packed, by_packed, table, l, min_matches):
    """Fused pairwise CUDA kernel: bx [B, Wx], by [S, Wy] packed uint32.

    The kernel reads columns as [Wy, S] (adjacent threads, adjacent SVs), so
    by is transposed per call.
    """
    import cupy as cp

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

    _cuda_kernel(_FUSED_PAIRWISE_IDX_CUDA, "fused_pairwise_idx")(
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
