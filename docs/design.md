# Design decisions

These decisions are load-bearing — do not deviate without discussion.

## Inference-first

Ship an LS-GKM model importer and predictor before building training. Match `gkmpredict` output within floating-point tolerance. Oracle scores are validated against Dongwon-Lee/lsgkm built from source.

## Kernel modes are distinct

`-t 0` (direct gapped k-mer / `gkm_cnt`) and `-t 2` (truncated estimated l-mer / `gkm_esttrunc`, the LS-GKM default) use different math. The binomial coefficient table `C(l-m, k)` for `-t 0` does NOT apply to `-t 2`. The `-t 2` weight table is ported from `calc_gkm_kernel_lmerest_wt` in `libsvm_gkm.c`.

`-t 1` (`gkm_estfull`) is supported via `EstTruncGkmKernel(truncate=False)`.

`-t 3` (`gkmrbf`) wraps the unnormalized `-t 2` kernel with an RBF transform: `K_rbf(x,y) = exp(-gamma * (K(x,x) + K(y,y) - 2*K(x,y)))`. Self-similarity is always 1.

`-t 4` (`wgkm`) uses center-weighted position importance. Positions near the l-mer center get weight 1.0; edges decay as `2^(-d/H)`. Uses per-position match computation with an elementary symmetric polynomial DP — exact position weighting, not a mismatch-count average.

`-t 5` (`wgkmrbf`) combines center-weighted base with RBF distance transform.

## Score formula

`score(x) = Σ dual_coef_i × K(x, support_i) + bias`

- Coefficients are signed (positive and negative class)
- `bias = -rho` (LIBSVM sign convention)
- LS-GKM's `rho` is the NEGATIVE bias

## Kernel normalization

Normalize by self-similarity: `K_norm(x,y) = K(x,y) / sqrt(K(x,x) * K(y,y))`. Unequal-length sequences have different norms. Normalization is on by default.

## Reverse-complement equivalence

On by default (`norc=0` in LS-GKM). Scores must be invariant under RC. When enabled, kernel computes K(x,y) + K(x, RC(y)).

## Kernel naming

Each kernel mode has three identifiers: LS-GKM `-t N` integer, an internal name (used in model files), and a descriptive alias. `resolve_kernel_type()` accepts any of these and returns the canonical internal name. `GkmSVM` resolves on construction — `model.kernel_type` is always the internal name.

## Kernel computation

Two computation paths, selected automatically:

**Packed uint32 path** (default forward pass, ISM): Each l-mer window is packed into a uint32 (2 bits per base). Match counting uses XOR + popcount on the packed representation instead of per-base comparisons. See "Packed uint32 comparison" below for details.

**Float one-hot path** (weighted kernels): Window match counts via matmul on flattened one-hot windows (`[B, W, 4*l]`). Both NumPy and CuPy use the same code path since CuPy mirrors NumPy's fancy indexing.

**GkmExplain sparse path**: Uses packed uint32 pre-filtering to identify the ~0.12% of window pairs with ≤ d mismatches (same min-matches skip as the forward pass), then decomposes only those pairs per-position. Per-position base identity is extracted directly from packed uint32 via bit shifts (`(packed >> 2k) & 3`), eliminating all float intermediate arrays. Mode 0 is computed as `mode_1 * one_hot_input` — only mode 1 has a dedicated kernel. The inner loop fuses normalization, coefficient multiplication, and SV-dimension reduction, accumulating directly into a `[B, 4, L]` result array (no per-SV intermediate). Forward and RC SV windows are concatenated along the Wy axis for a single kernel launch per chunk. CPU: fused Numba `@njit(parallel=True)` kernel with `prange(B)` — each thread owns its `[4, L]` result slice (L1-resident, ~6.4 KB). GPU: fused CuPy RawKernel with one thread per (b, s) pair — coalesced SV reads via transposed `[Wy, S]` layout, shared-memory caching of weight tables and query packed windows, float64 `atomicAdd` into the result array (CC >= 6.0). Falls back to the dense float path for kernels without a min-matches threshold (weighted kernels).

## Fused pairwise kernels

CPU: Numba `@njit(parallel=True, fastmath=True)` fuses match-count + table-lookup + sum into a single parallel kernel, avoiding materialization of the full `[B, S, W, W]` match tensor.

NVIDIA GPU: CuPy RawKernel with coalesced memory access. Uses `--use_fast_math` and shared-memory caching of both the mismatch weight table and query packed windows.

Apple GPU: Custom Metal shaders via `mx.fast.metal_kernel`. Each thread handles one (batch, SV) pair, looping over all window pairs with XOR+popcount and `min_matches` early exit. Three kernel variants: pairwise (forward pass), diagonal (self-kernel), and cross-diagonal (forward × RC). Kernels are lazily compiled on first use.

## Chunked SV inference

`GkmSVM(sv_chunk_size=N)` chunks pairwise kernel computation over support vectors, bounding memory for large models. ENCODE ENCFF579AOX has 72,145 SVs — without chunking at 300bp, the matches tensor alone would be hundreds of GB. Recommended chunk sizes for single-sequence scoring at 300bp:

- 10 GB GPU: `sv_chunk_size=5000` (peak ~3.4 GB for matches + histogram)
- 16+ GB GPU: `sv_chunk_size=10000`
- CPU: `sv_chunk_size=5000` or `None` for small models

For batch scoring, reduce proportionally (`5000 / batch_size`).

## Training

Two solver backends, selected automatically based on available memory:

**Precomputed Gram + sklearn** (`solver="libsvm"`, default for small N): Computes the full N×N kernel matrix via tiled Gram computation, then calls scikit-learn's LIBSVM-backed `SVC(kernel='precomputed')` / `SVR(kernel='precomputed')`. Supports both C-SVC (`train_gkmsvm`, `-s 0`) and epsilon-SVR (`train_gkmsvr`, `-s 3`). Fast — LIBSVM's SMO is highly optimized — but requires O(N²) memory.

**Column-cached SMO** (`solver="smo"`, large N): WSS3 working set selection (Fan et al. 2005) with shrinking and LRU-cached kernel columns. Memory is O(cache_size × N) instead of O(N²). Currently supports C-SVC only. A C extension (`_csmo`) handles the optimization loop; kernel columns are computed by `KernelColumnCache` using the fastest available backend (Numba CPU, CuPy GPU, or MLX). Falls back to a pure-Python serial solver when the C extension is unavailable.

**Nyström approximation** (`solver="nystrom"`): Low-rank kernel approximation using landmark points. Faster than both Gram and SMO for large datasets, but produces an approximate solution. Useful when exact training is too slow and approximate is acceptable.

`solver="auto"` estimates whether N²×8 bytes fits in 75% of available RAM (CPU) or VRAM (GPU). Falls back to SMO when it doesn't.

The scoring formula `Σ coef_i × K(x, sv_i) + bias` is identical for SVC and SVR — only the dual coefficients differ (α_i × y_i for SVC, α_i* − α_i for SVR). The `GkmSVM` model class is shared.

## No dense Gram matrix at scale

LS-GKM exists because the full N×N kernel matrix doesn't fit in memory at scale (50k examples ≈ 10 GB, 90k ≈ 32 GB). Use block/column evaluation with chunked SV inference.

## The `d` parameter

`d` limits mismatch depth in LS-GKM's tree-based kernel evaluation. In our brute-force window comparison, this is implemented by zeroing weight table entries for m > d. Without this cutoff, scores diverge ~1.5% from `gkmpredict`.

## LS-GKM model format

Header key-value pairs until `SV` marker, then `<signed_coef> <DNA_sequence>` per line. Key fields: `svm_type`, `kernel_type`, `L`, `k`, `d`, `norc`, `rho`, `nr_class`, `total_sv`. Auto-detects gzip. Binary classification only (nr_class=2).

## Classic gkmSVM format (C)

Original gkmSVM C implementation uses OPPOSITE sign convention: `bias = +rho`. LIBSVM-style `key value` headers followed by `SV` marker. Supports both embedded SVs (single file: `coef sequence` per line) and two-file format (model + FASTA). Integer kernel types (0-5) are mapped to string names. Load via `load_classic_model(model_path, svseq_path=...)`.

## R gkmSVM format

R gkmSVM package (Ghandi et al. 2016, kernlab-based) uses `#`-prefixed headers (`#rho`, `#nsv`, `#npos`, `#nneg`, `#L`, `#k`, `#d`). Two sub-formats:

- **Unified `.gkmmodel`**: headers then FASTA entries where the header line is `>seq_id\tcoefficient`.
- **Legacy two-file**: `_svalpha.out` (tab-separated `seq_id\tcoef`) plus `_svseq.fa`.

Same bias convention as C gkmSVM: `bias = +rho`. Load via `load_r_gkmsvm_model(path, svseq_path=...)`. Auto-detects `_svseq.fa` companion file when loading `_svalpha.out`.

## SV diagonal cache

The SVM caches the support-vector self-kernel diagonal (`_raw_diagonal(sv)`) after first computation, avoiding a chunked recomputation on every call.

## NumPy + CuPy + MLX (no PyTorch)

gkm-SVMs are not differentiable — autograd provides no value. The array operations (matmul, einsum, fancy indexing) are identical in NumPy and CuPy, so a single codebase handles both CPU and NVIDIA GPU via `gkmsvm.backend.get_array_module()`.

CPU: NumPy arrays + Numba `@njit(parallel=True)` fused pairwise kernels.
NVIDIA GPU: CuPy arrays (optional `[gpu]` extra) + CuPy RawKernel CUDA code. `model.cuda()` moves data to GPU.
Apple GPU: MLX arrays (optional `[mlx]` extra) + custom Metal shaders via `mx.fast.metal_kernel`. `model.mlx()` moves data to Apple Silicon GPU. MLX arrays lack `.strides`, `.copy()`, and NumPy-style fancy indexing — an `_MlxShim` wrapper and `get_strides()` helper provide compatibility. Packing uses CPU (via `to_cpu()` round-trip) since Numba isn't available on Apple GPU; the Metal kernel handles the expensive pairwise computation.

tangermeme interop is vendored — only `extract_loci` (pyfaidx) and FASTA I/O are needed. ledidi requires differentiable models and does not work with gkm-SVMs.

## Packed uint32 comparison

Each l-mer window is packed into a uint32 (2 bits per base, 22 bits for l=11). Match counting uses XOR + popcount on 2-bit fields instead of l individual byte comparisons. This replaces 11 strided memory reads with a single contiguous uint32 read per window pair. On GPU, the SV array goes from `[l, Wy, S]` int8 (229MB for 72K SVs at 300bp) to `[Wy, S]` uint32 (84MB) — 2.8x memory reduction plus drastically better cache locality. On CPU, ~3x faster than the int8 loop. SV packed windows are cached on the model for both CPU and GPU, avoiding redundant packing on repeated calls.

## Mismatch table sparsity

The mismatch table has zero entries for high-mismatch counts. For esttrunc l=11 k=7 d=3, only m=0..3 are non-zero (min_matches=8). For random DNA, P(match)=0.25, so P(>=8 matches out of 11) ≈ 0.12% — 99.88% of window pairs contribute nothing. Both CPU and GPU paths skip the table lookup for pairs below min_matches. On GPU, shared memory caches the query sequence's packed windows so all threads in a block share a single load from global memory.

## Epsilon-SVR

`train_gkmsvr()` trains an epsilon-SVR for continuous-valued prediction (e.g. quantitative chromatin accessibility from lsgkm-svr). The epsilon parameter controls the tube width — errors within ±epsilon are not penalized. Uses LIBSVM's `-s 3` solver with precomputed Gram matrix. SMO SVR (2N dual variables, different box constraints) is deferred — the Gram path handles typical regression dataset sizes.

## References

- Ghandi M, Lee D, Mohammad-Noori M, Beer MA. Enhanced regulatory sequence prediction using gapped k-mer features. *PLoS Comput Biol* 10(7):e1003711 (2014). doi:10.1371/journal.pcbi.1003711
- Ghandi M, Mohammad-Noori M, Ghareghani N, Lee D, Garraway L, Beer MA. gkmSVM: an R package for gapped-kmer SVM. *Bioinformatics* 32(14):2205–2207 (2016). doi:10.1093/bioinformatics/btw203
- Lee D. LS-GKM: a new gkm-SVM for large-scale datasets. *Bioinformatics* 32(14):2196–2198 (2016). doi:10.1093/bioinformatics/btw142
- Shrikumar A, Prakash E, Kundaje A. GkmExplain: fast and accurate interpretation of nonlinear gapped k-mer SVMs. *Bioinformatics* 35(14):i173–i182 (2019). doi:10.1093/bioinformatics/btz322
- Lee D, Gorkin DU, Baker M, Strober BJ, Asoni AL, McCallion AS, Beer MA. A method to predict the impact of regulatory variants from DNA sequence. *Nat Genet* 47(8):955–961 (2015). doi:10.1038/ng.3331
- Chang CC, Lin CJ. LIBSVM: a library for support vector machines. *ACM Trans Intell Syst Technol* 2(3):1–27 (2011). doi:10.1145/1961189.1961199
- Dongwon-Lee/lsgkm: https://github.com/Dongwon-Lee/lsgkm
- kundajelab/lsgkm (with GkmExplain): https://github.com/kundajelab/lsgkm
