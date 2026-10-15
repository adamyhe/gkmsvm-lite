# Performance

## Which path a call takes

Every kernel works on windows packed 2 bits per base into uint32 (`PackedGkmKernel.pack`); RBF kernels (`-t 3`, `-t 5`) go through their base kernel and apply `exp(γ(K - 1))` at the end.

| Work | Path | Cost |
|---|---|---|
| Scoring `-t 0/1/2/4` (l <= 12, 500+ SVs) | Exact l-mer weight table (`lmer_table.py`): one lookup per window | Independent of the number of SVs once built |
| Scoring `-t 1/2/3` (d-cutoff, neighborhood <= 8,192 l-mers) | Cached l-mer index over the SVs (`kernels/lmer_index.py`): each query window visits its d-mismatch neighborhood | Proportional to the matching (window, SV) pairs; nothing per SV |
| Other scoring, Gram tiles, SMO columns for `-t 0/4/5` | Masked join (`kernels/masked.py`) when both sides have 16+ sequences, else brute force (XOR + popcount over all window pairs) | Masked join indexes one side per call |
| SMO columns for `-t 1/2/3` | l-mer index over the training set | One query row per column |
| GkmExplain attribution | Fused packed kernel (CPU, CUDA), reducing over SVs in the inner loop | All window pairs within the mismatch cutoff |
| Inputs with N for `-t 4/5` | Per-position one-hot match path | Slow; chunked to `_MATCH_BYTES` |

The model caches packed SV windows, their self-kernels, the l-mer table and the l-mer index per device (`GkmSVM._sv_cache`); the first call pays for building them (0.55 s and 81 MB for the l-mer index over 32,715 SVs at l=10).

CPU kernels are Numba `@njit(parallel=True)`; the threading layer is pinned to `workqueue` on import. On CPU, kernel scoring normalizes, applies the RBF and reduces over SVs in one parallel pass (`svm._normalized_dot`, and `explain._impact_cpu` for mutation impact), so the only [rows, S] array is the raw kernel. CUDA kernels are CuPy RawKernels; Metal kernels use `mx.fast.metal_kernel`. Both GPU index queries count hits per mismatch level in int32 and combine levels in float64, so they are exact; Metal results are returned as float32.

## Measurements

Variant scoring on the GkmExplain paper's `-t 3` dsQTL model (gkmrbf l=10 k=6 d=3, 32,715 SVs) for the 28,309 dsQTL test variants (51 bp windows), `benchmarks/timing_variants.py --no-lsgkm`. Server with an NVIDIA Titan Xp; compute time excludes model loading and JIT. "Before" is the code before the l-mer index (`cba8064`).

| Method | Device | Before | Now | Peak RSS before → now |
|---|---|---|---|---|
| Kernel delta, `score_variants(method="kernel")` | CPU | 1137 s | 155 s | 2916 → 1537 MB |
| | CUDA | 86.9 s | 20.2 s | 1557 → 1557 MB (GPU pool 2139 → 786 MB) |
| Mutation impact, `method="gkmexplain"` | CPU | 570 s | 168 s | 5190 → 1537 MB |
| | CUDA | 43.0 s | 19.2 s | 1558 → 1557 MB (GPU pool 5388 → 786 MB) |
| `to_deltasvm(query_norm=True)`, all 4^10 10-mers | CPU | ~1 h (est.) | 113 s | → 1631 MB |
| | CUDA | | 43.7 s | → 1557 MB |

The two variant methods cost the same: both are one index query on [ref; alt] against the SVs, followed by a different reduction. `to_deltasvm` on an RBF model is a one-time conversion and is the most expensive of the three, because every one of the 1M l-mers still produces a full row of 32,715 kernel values; afterwards a `DeltaSVM` scores a variant with a few table lookups.

Training the `-t 2` l=10 k=6 d=3 model on GM12878 DHSs (300 bp), `benchmarks/training_memory.py --no-lsgkm` with a 2,048-column SMO cache, on the same server's GPU:

| N | libsvm (precomputed Gram) | SMO | Nyström | SMO peak vs libsvm |
|---|---|---|---|---|
| 1,000 | 1.7 s | 0.9 s | 2.0 s | 647 vs 717 MB |
| 2,000 | 2.0 s | 1.3 s | 2.1 s | 672 vs 765 MB |
| 5,000 | 3.4 s | 5.7 s | 2.9 s | 763 vs 921 MB |
| 10,000 | 8.4 s | 22.7 s | 5.1 s | 899 vs 1698 MB |

SMO and libsvm give identical models (same SVs, AUROC, held-out scores r = 1.00000). With a cache smaller than N, SMO recomputes evicted columns, so it trades time for O(cache × N) memory. Nyström computes fewer kernel values but still builds a dense N × N approximate Gram, so it saves time, not memory.

On an Apple M4 (CPU), `-t 3` SMO at N=1,000 takes 1.2 s and `-t 5` SMO at N=600 takes 1.4 s; both previously computed one column at a time through the one-hot path (80 s and 2.7 h). Biases and SV counts match the earlier code.

Comparisons with LS-GKM are the paper figures in `benchmarks/` (`make -C benchmarks fig1b fig1c s2 s3`); see [benchmarks/README.md](../benchmarks/README.md).

## Memory

- Kernel scoring holds [rows, S] kernel values in blocks of at most `GkmSVM._MAX_SCORE_BYTES` (512 MB), whatever `batch_size` says. `batch_size` bounds the inputs moved to the device at once.
- Without the l-mer table or index, every masked-join call indexes all SV windows, so `batch_size` is raised until a call fills ~256 MB of [batch, S] (`GkmSVM._min_batch_size`).
- GkmExplain processes inputs in batches (default 50) and SVs in chunks of `model.sv_chunk_size` (default 2,000).
- Training with `solver="libsvm"` holds the N × N float64 Gram in host RAM; `solver="auto"` switches to SMO when it would exceed 75% of available RAM or `max_gram_gb`.

## Reproducing

```bash
python benchmarks/timing_variants.py --device cuda --no-lsgkm --n-variants 500
python benchmarks/training_memory.py --device cuda --no-lsgkm --sizes 1000,2000,5000,10000
make -C benchmarks s2      # throughput sweep, benchmarks/results/throughput.tsv
```
