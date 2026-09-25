# Design decisions

These decisions are load-bearing — do not deviate without discussion.

## Inference-first

Ship an LS-GKM model importer and predictor before building training. Match `gkmpredict` output within floating-point tolerance. Oracle scores are validated against Dongwon-Lee/lsgkm built from source.

## Kernel modes are distinct

`-t 0` (direct gapped k-mer / `gkm_cnt`) and `-t 2` (truncated estimated l-mer / `gkm_esttrunc`, the LS-GKM default) use different math. The binomial coefficient table `C(l-m, k)` for `-t 0` does NOT apply to `-t 2`. The `-t 2` weight table is ported from `calc_gkm_kernel_lmerest_wt` in `libsvm_gkm.c`.

`-t 1` (`gkm_estfull`) is supported via `EstTruncGkmKernel(truncate=False)`.

## Score formula

`score(x) = Σ dual_coef_i × K(x, support_i) + bias`

- Coefficients are signed (positive and negative class)
- `bias = -rho` (LIBSVM sign convention)
- LS-GKM's `rho` is the NEGATIVE bias

## Kernel normalization

Normalize by self-similarity: `K_norm(x,y) = K(x,y) / sqrt(K(x,x) * K(y,y))`. Unequal-length sequences have different norms. Normalization is on by default.

## Reverse-complement equivalence

On by default (`norc=0` in LS-GKM). Scores must be invariant under RC. When enabled, kernel computes K(x,y) + K(x, RC(y)).

## No dense Gram matrix

LS-GKM exists because the full N×N kernel matrix doesn't fit in memory at scale (50k examples ≈ 10 GB, 90k ≈ 32 GB). Use block/column evaluation with chunked SV inference.

## The `d` parameter

`d` limits mismatch depth in LS-GKM's tree-based kernel evaluation. In our brute-force window comparison, this is implemented by zeroing weight table entries for m > d. Without this cutoff, scores diverge ~1.5% from `gkmpredict`.

## LS-GKM model format

Header key-value pairs until `SV` marker, then `<signed_coef> <DNA_sequence>` per line. Key fields: `svm_type`, `kernel_type`, `L`, `k`, `d`, `norc`, `rho`, `nr_class`, `total_sv`. Auto-detects gzip. Binary classification only (nr_class=2).

Original gkmSVM format (`.gkmmodel`) uses OPPOSITE sign convention for bias — not yet implemented.

## PyTorch over cuML/CuPy

PyTorch chosen for compatibility with S2F ecosystem (tangermeme, Cherimoya, ledidi, scverse). All sequence manipulation and variant effect scoring uses tangermeme utilities.

## References

- Ghandi et al., "Enhanced regulatory sequence prediction using gapped k-mer features" (2014)
- Lee, "LS-GKM: a new gkm-SVM for large-scale datasets" (2016)
- Shrikumar et al., "GkmExplain: fast and accurate interpretation of nonlinear gapped k-mer SVMs" (2019)
- Dongwon-Lee/lsgkm: https://github.com/Dongwon-Lee/lsgkm
- kundajelab/lsgkm (with GkmExplain): https://github.com/kundajelab/lsgkm
