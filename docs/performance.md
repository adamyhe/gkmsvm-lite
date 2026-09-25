# Performance

## CPU backend

Fused Numba kernels (`@njit(parallel=True, fastmath=True)`) combine match-count, table-lookup, and accumulation into a single parallel kernel. This avoids materializing the full `[B, S, W, W]` match tensor.

Two kernel paths:
- **Float path** (`pairwise_from_windows`): flattened one-hot dot products. Used by GkmExplain and weighted kernels.
- **Int8 index path** (`pairwise_from_indices`): base-index comparison. 4x fewer ops, 16x less memory. Used by default forward pass and ISM.

Numba's threading layer is pinned to `workqueue` on import to avoid OpenMP conflicts.

## GPU backend

CuPy RawKernel CUDA code with:
- **Coalesced memory access**: SV int8 windows are transposed from `[S, Wy, l]` to `[l, Wy, S]` so adjacent threads (consecutive `s` values) read adjacent bytes.
- **Shared-memory table caching**: the mismatch weight table is loaded into shared memory once per thread block.
- **`--use_fast_math`**: enables fast math intrinsics and implicit loop unrolling.

GPU uses float32 accumulation for the float path. The int8 path accumulates in float64 (same as CPU). Score differences between CPU and GPU are <3e-3 due to the float32 diagonal computation on GPU.

## Chunked SV inference

`GkmSVM(sv_chunk_size=N)` chunks pairwise kernel computation over support vectors to bound memory. GkmExplain defaults to 2000-SV chunks to prevent GPU OOM and improve CPU cache locality.

## ISM window-delta optimization

When a single base changes, only ~l of the W = L - l + 1 windows are affected. ISM computes the delta from affected windows only, rather than recomputing the full kernel. Variable-length affected windows are padded with sentinel value -1 (never matches any base, so table[l] = C(0,k) = 0).

## Throughput (ENCODE ENCFF579AOX, 72K SVs, 300bp, RTX 3080)

| Implementation | seq/s |
|---|---|
| gkmsvm-lite GPU (CuPy) | ~48 |
| gkmsvm-lite CPU (Numba) | ~15 |
| LS-GKM C (`gkmpredict`) | ~1.25 |

## Bottleneck: gkm kernel vs RBF kernel

The gapped k-mer kernel is inherently more expensive than a standard RBF kernel on pre-extracted features:

| | RBF on features | gkm string kernel |
|---|---|---|
| Kernel eval per (query, SV) | D multiply-adds | W^2 x l match comparisons |
| Typical dimensions | D = 360 | W = 290, l = 11 |
| FLOPs per SV | ~720 | ~5,000,000 |

This is inherent to the kernel function, not an implementation gap.
