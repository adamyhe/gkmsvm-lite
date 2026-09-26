# Performance

## CPU backend

Fused Numba kernels (`@njit(parallel=True, fastmath=True)`) combine match-count, table-lookup, and accumulation into a single parallel kernel. This avoids materializing the full `[B, S, W, W]` match tensor.

Two kernel paths:
- **Float path** (`pairwise_from_windows`): flattened one-hot dot products. Used by GkmExplain and weighted kernels.
- **Packed uint32 path** (`pairwise_from_indices`): base indices packed to uint32, match counting via XOR + popcount. ~3x faster than the previous int8 loop. Used by default forward pass and ISM.

Numba's threading layer is pinned to `workqueue` on import to avoid OpenMP conflicts.

## GPU backend

CuPy RawKernel CUDA code with:
- **Coalesced memory access**: SV int8 windows are transposed from `[S, Wy, l]` to `[l, Wy, S]` so adjacent threads (consecutive `s` values) read adjacent bytes.
- **Shared-memory table + bx caching**: the mismatch weight table and query sequence windows are loaded into shared memory once per thread block. For S >> blockDim (typical), all threads in a block share the same query index, eliminating redundant global reads of query data.
- **Min-matches skip**: window pairs below `min_matches` (derived from mismatch table sparsity) skip the table lookup entirely. For esttrunc l=11 k=7 d=3, 99.88% of random window pairs are skipped.
- **`--use_fast_math`**: enables fast math intrinsics and implicit loop unrolling.

GPU uses float32 accumulation for the float path. The int8 path accumulates in float64 (same as CPU). Score differences between CPU and GPU are <3e-3 due to the float32 diagonal computation on GPU.

## Chunked SV inference

`GkmSVM(sv_chunk_size=N)` chunks pairwise kernel computation over support vectors to bound memory. GkmExplain defaults to 2000-SV chunks to prevent GPU OOM and improve CPU cache locality.

## ISM window-delta optimization

When a single base changes, only ~l of the W = L - l + 1 windows are affected. ISM computes the delta from affected windows only, rather than recomputing the full kernel. Variable-length affected windows are padded with sentinel value -1 (never matches any base, so table[l] = C(0,k) = 0).

## Throughput (ENCODE ENCFF579AOX, 72K SVs, RTX 3080)

| Sequence length | GPU (CuPy) | CPU (Numba) | LS-GKM C |
|---|---|---|---|
| 19bp (dsQTL variants) | 336.6 seq/s | 27.7 seq/s | — |
| 300bp (ENCODE peaks) | ~10 seq/s | ~0.6 seq/s | ~1.25 seq/s |

GPU throughput is measured steady-state (packed SV windows cached on both CPU and GPU). First call includes a one-time packing step. CPU uses all available cores via Numba parallel threading. LS-GKM C is single-threaded.

## Bottleneck: gkm kernel vs RBF kernel

The gapped k-mer kernel is inherently more expensive than a standard RBF kernel on pre-extracted features:

| | RBF on features | gkm string kernel |
|---|---|---|
| Kernel eval per (query, SV) | D multiply-adds | W^2 x l match comparisons |
| Typical dimensions | D = 360 | W = 290, l = 11 |
| FLOPs per SV | ~720 | ~5,000,000 |

This is inherent to the kernel function, not an implementation gap.
