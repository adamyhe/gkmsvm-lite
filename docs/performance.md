# Performance

## Kernel paths

Two kernel computation paths, selected automatically:

- **Packed uint32 path** (`pairwise_from_indices`): Base indices packed to uint32 (2 bits/base), match counting via XOR + popcount. Used by default forward pass and ISM. ~3x faster than int8 on CPU.
- **Float path** (`pairwise_from_windows`): Flattened one-hot dot products. Used by GkmExplain and weighted kernels (`-t 4`/`-t 5`).

## CPU backend

Fused Numba kernels (`@njit(parallel=True, fastmath=True)`) combine match-count, table-lookup, and accumulation into a single parallel kernel, avoiding materialization of the full `[B, S, W, W]` match tensor. Numba's threading layer is pinned to `workqueue` on import to avoid OpenMP conflicts.

Packed SV windows are cached on the model after first call, eliminating redundant packing on repeated scoring.

## NVIDIA GPU backend (CuPy)

CuPy RawKernel CUDA code with:

- **Packed uint32 windows**: Each l-mer packed into a uint32. SV array goes from `[l, Wy, S]` int8 (229 MB for 72K SVs at 300bp) to `[Wy, S]` uint32 (84 MB) — 2.7x memory reduction with better cache locality.
- **Coalesced memory access**: SV windows laid out so adjacent CUDA threads read adjacent memory.
- **Shared-memory caching**: Query sequence packed windows and the mismatch weight table are loaded into shared memory once per thread block. For S >> blockDim (typical), all threads in a block share the same query data.
- **Min-matches skip**: Window pairs below `min_matches` (derived from mismatch table sparsity) skip the table lookup entirely. For esttrunc l=11 k=7 d=3, 99.88% of random window pairs are skipped.
- **`--use_fast_math`**: Enables fast math intrinsics.

GPU uses float32 accumulation for the float path. Score differences between CPU and GPU are <3e-3 due to float32 diagonal computation on GPU.

## Apple Silicon GPU backend (MLX)

Custom Metal shaders via `mx.fast.metal_kernel` with the same algorithmic approach as the CUDA path:

- **Packed uint32 XOR + popcount**: Each l-mer packed into a uint32. Metal's hardware `popcount()` counts mismatches in a single instruction.
- **Per-thread early exit**: `min_matches` threshold skips window pairs that contribute nothing, same as the CUDA path.
- **No intermediate materialization**: Each Metal thread accumulates its own (batch, SV) result — no `[B, S, Wx, Wy]` tensor.
- **`mx.as_strided` sliding windows**: Native MLX strided view replaces the Python-loop concatenation that was the initial performance bottleneck.

MLX arrays lack some NumPy features (`.strides`, `.copy()`, fancy indexing). These are handled by:
- `get_strides()` helper returning zeros for MLX (the MLX shim's `_mlx_sliding_windows` computes strides from shape).
- `xp.ascontiguousarray()` in place of `.copy()`.
- `to_cpu(X)[indices]` guard in training code (transfers to numpy before fancy indexing).

DeltaSVM auto-chunks batch dimension when intermediates would exceed 256 MB, preventing memory thrashing on Apple Silicon's unified memory.

## Chunked SV inference

`GkmSVM(sv_chunk_size=N)` chunks pairwise kernel computation over support vectors to bound memory. Recommended chunk sizes for single-sequence scoring at 300bp:

| GPU VRAM | `sv_chunk_size` | Peak memory |
|---|---|---|
| 10 GB | 5000 | ~3.4 GB |
| 16+ GB | 10000 | ~6.8 GB |
| CPU | 5000 or None | N/A |

For batch scoring, reduce proportionally (`5000 / batch_size`). GkmExplain defaults to 2000-SV chunks.

## ISM window-delta optimization

When a single base changes, only ~l of the W = L - l + 1 windows are affected. ISM computes the delta from affected windows only, rather than recomputing the full kernel.

## Throughput benchmarks

### NVIDIA GPU (ENCFF579AOX, 72K SVs)

All benchmarks use the ENCODE ENCFF579AOX model (72,145 SVs, esttrunc l=11 k=7 d=3). GPU: RTX 3080. CPU: Numba parallel threading on all available cores. Steady-state throughput (packed SV windows cached).

### Best throughput per sequence length

| Sequence length | GPU (seq/s) | CPU (seq/s) | GPU/CPU speedup |
|---|---|---|---|
| 19bp (dsQTL variants) | 342.9 | 52.8 | 6.5x |
| 50bp | 74.6 | 12.9 | 5.8x |
| 100bp | 33.0 | 5.9 | 5.6x |
| 200bp | 15.9 | 3.0 | 5.3x |
| 300bp (ENCODE peaks) | 10.4 | 1.9 | 5.5x |

GPU throughput is flat across batch sizes (compute-bound). CPU benefits slightly from larger batches at short sequences.

### Full sweep: GPU (CuPy)

| Length | Batch 1 | Batch 4 | Batch 16 | Batch 64 | Batch 256 |
|---|---|---|---|---|---|
| 19bp | 288.8 | 325.4 | 335.0 | 316.9 | **342.9** |
| 50bp | 71.9 | 70.9 | **74.6** | 74.1 | 72.8 |
| 100bp | 30.8 | 31.3 | 31.7 | **33.0** | 32.6 |
| 200bp | 15.2 | 14.8 | 15.1 | 15.8 | **15.9** |
| 300bp | 10.1 | 10.0 | 10.3 | **10.4** | 10.3 |

### Full sweep: CPU (Numba)

| Length | Batch 1 | Batch 4 | Batch 16 | Batch 64 | Batch 256 |
|---|---|---|---|---|---|
| 19bp | 35.5 | 44.6 | 49.0 | 50.1 | **52.8** |
| 50bp | 10.3 | 10.8 | 11.5 | 12.3 | **12.9** |
| 100bp | 4.7 | 5.0 | 5.3 | 5.7 | **5.9** |
| 200bp | 2.9 | 2.9 | **3.0** | 3.0 | 2.7 |
| 300bp | 1.9 | **1.9** | 1.9 | 1.8 | 1.9 |

Values in **bold** are the best batch size per sequence length. All values are seq/s.

### Comparison with LS-GKM C

LS-GKM C (`gkmpredict`) is single-threaded. At 300bp with 72K SVs:

| Implementation | seq/s |
|---|---|
| gkmsvm-lite GPU | 10.4 |
| gkmsvm-lite CPU | 1.9 |
| LS-GKM C (single thread) | ~1.25 |

## Bottleneck analysis

The gapped k-mer kernel is inherently more expensive than a standard RBF kernel on pre-extracted features:

| | RBF on features | gkm string kernel |
|---|---|---|
| Kernel eval per (query, SV) | D multiply-adds | W² × l match comparisons |
| Typical dimensions | D = 360 | W = 290, l = 11 |
| FLOPs per SV | ~720 | ~5,000,000 |

Throughput scales as O(W² × S) per query where W = seq_len - l + 1 and S = num SVs. This is inherent to the kernel function, not an implementation gap.

## Batch size recommendations

- **GPU**: Batch size has minimal impact on throughput (GPU is compute-bound). Use batch_size=16-64 to save VRAM without sacrificing speed.
- **CPU**: Larger batches help at short sequences (up to ~1.5x at 19bp). At 200bp+, batch size doesn't matter.

### Apple Silicon GPU (Nanog, 8.8K SVs)

Benchmarks use the Nanog reference model (8,873 SVs, esttrunc l=11 k=7 d=3) on an M1 MacBook Pro (16 GB unified memory). Custom Metal kernels via `mx.fast.metal_kernel`.

| Config | CPU (seq/s) | MLX (seq/s) | Speedup |
|---|---|---|---|
| batch=1, 19bp | 215 | 387 | 1.8x |
| batch=1, 50bp | 46 | 126 | 2.7x |
| batch=4, 50bp | 48 | 277 | 5.8x |
| batch=16, 50bp | 49 | 308 | 6.3x |
| batch=1, 100bp | 20 | 61 | 3.1x |
| batch=4, 100bp | 22 | 129 | 5.9x |
| batch=16, 100bp | 23 | 140 | 6.1x |

MLX throughput scales well with batch size (GPU saturation). At batch=16 the speedup plateaus at ~6x.

DeltaSVM (linear k-mer scoring) gets 1.7-2.5x on MLX. The gather-heavy combinatorial path doesn't benefit from GPU as much as the fused XOR+popcount kernel path.

## Reproducing benchmarks

```bash
python benchmarks/throughput_sweep.py --device cpu
python benchmarks/throughput_sweep.py --device cuda
python benchmarks/throughput_sweep.py --device both   # side-by-side with speedup table
```

Requires the ENCODE ENCFF579AOX fixture in `tests/fixtures/`.
