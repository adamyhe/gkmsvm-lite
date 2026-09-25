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

## Kernel computation: matmul + table lookup

Window match counts are computed via matmul on flattened one-hot windows (`[B, W, 4*l]`), avoiding a 6D broadcast intermediate that is 440x larger. The `_apply_table` step uses eager gather: `table[mismatches].sum()`. Both NumPy (CPU) and CuPy (GPU) use the same code path since CuPy mirrors NumPy's fancy indexing.

## Chunked SV inference

`GkmSVM(sv_chunk_size=N)` chunks pairwise kernel computation over support vectors, bounding memory for large models. ENCODE ENCFF579AOX has 72,145 SVs — without chunking at 300bp, the matches tensor alone would be hundreds of GB. Recommended chunk sizes for single-sequence scoring at 300bp:

- 10 GB GPU: `sv_chunk_size=5000` (peak ~3.4 GB for matches + histogram)
- 16+ GB GPU: `sv_chunk_size=10000`
- CPU: `sv_chunk_size=5000` or `None` for small models

For batch scoring, reduce proportionally (`5000 / batch_size`).

## No dense Gram matrix

LS-GKM exists because the full N×N kernel matrix doesn't fit in memory at scale (50k examples ≈ 10 GB, 90k ≈ 32 GB). Use block/column evaluation with chunked SV inference.

## The `d` parameter

`d` limits mismatch depth in LS-GKM's tree-based kernel evaluation. In our brute-force window comparison, this is implemented by zeroing weight table entries for m > d. Without this cutoff, scores diverge ~1.5% from `gkmpredict`.

## LS-GKM model format

Header key-value pairs until `SV` marker, then `<signed_coef> <DNA_sequence>` per line. Key fields: `svm_type`, `kernel_type`, `L`, `k`, `d`, `norc`, `rho`, `nr_class`, `total_sv`. Auto-detects gzip. Binary classification only (nr_class=2).

## Classic gkmSVM format

Original gkmSVM (Ghandi et al. 2014) uses OPPOSITE sign convention: `bias = +rho`. Supports both embedded SVs (single file) and two-file format (model + FASTA). Integer kernel types (0-5) are mapped to string names. Load via `load_classic_model(model_path, svseq_path=...)`.

## SV diagonal cache

The SVM caches the support-vector self-kernel diagonal (`_raw_diagonal(sv)`) after first computation, avoiding a chunked recomputation on every call.

## NumPy + CuPy (no PyTorch)

gkm-SVMs are not differentiable — autograd provides no value. The array operations (matmul, einsum, fancy indexing) are identical in NumPy and CuPy, so a single codebase handles both CPU and GPU via `gkmsvm.backend.get_array_module()`.

CPU: NumPy arrays + Numba `@njit(parallel=True)` for reference kernels.
GPU: CuPy arrays (optional `[gpu]` extra). `model.cuda()` moves data to GPU.

tangermeme interop is vendored — only `extract_loci` (pyfaidx) and FASTA I/O are needed. ledidi requires differentiable models and does not work with gkm-SVMs.

## References

- Ghandi et al., "Enhanced regulatory sequence prediction using gapped k-mer features" (2014)
- Lee, "LS-GKM: a new gkm-SVM for large-scale datasets" (2016)
- Shrikumar et al., "GkmExplain: fast and accurate interpretation of nonlinear gapped k-mer SVMs" (2019)
- Dongwon-Lee/lsgkm: https://github.com/Dongwon-Lee/lsgkm
- kundajelab/lsgkm (with GkmExplain): https://github.com/kundajelab/lsgkm
