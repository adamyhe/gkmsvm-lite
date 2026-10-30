# Performance

## Which path a call takes

Every kernel works on windows packed 2 bits per base into uint32 (`PackedGkmKernel.pack`); RBF kernels (`-t 3`, `-t 5`) go through their base kernel and apply `exp(γ(K - 1))` at the end.

| Work | Path | Cost |
|---|---|---|
| Scoring `-t 0/1/2/4` (l <= 12, 500+ SVs) | Exact l-mer weight table (`lmer_table.py`): one lookup per window | Independent of the number of SVs once built |
| Scoring, ISM, mutation impact `-t 1-5` (d-cutoff, neighborhood <= 8,192 l-mers, or <= 32,768 for large SV sets) | Cached l-mer index over the SVs (`kernels/lmer_index.py`): each query window visits its d-mismatch neighborhood (CPU, CUDA, Metal) | Proportional to the matching (window, SV) pairs; nothing per SV |
| Gram tiles and SMO columns `-t 1-5` (neighborhood <= 8,192) | l-mer index over the training set (Gram: not on MLX); a self-Gram row block computes only its columns from the diagonal on | One query row per SMO column |
| Other scoring, Gram tiles, SMO columns (`-t 0`, MLX Grams) | Masked join (`kernels/masked.py`) when both sides have 16+ sequences, else brute force (XOR + popcount over all window pairs) | Masked join indexes one side per call (the forward columns only, probed with both strands of the rows, when the columns are larger) |
| GkmExplain attribution `-t 0/1/2/4`, many sequences | Explain table (`explain._build_explain_table`, [4^l, 4, l], ~1 s at l=11): W row lookups per sequence, on CPU | 0.035 ms per 300 bp sequence on an M4 |
| GkmExplain attribution (l <= 12, d-cutoff neighborhood <= 8,192) | SVs' l-mer weights V (4^l, cached; RBF: summed over the index entries per query): each query window visits its d-mismatch neighborhood (CPU, CUDA, Metal) | Nothing per SV; 3.5 ms per 300 bp sequence for ENCFF579AOX (72,145 SVs) on an M4 CPU (0.9 ms in Metal), against 1.7 s for the pair loop |
| Other GkmExplain attribution | Fused packed kernel (CPU, CUDA, Metal), reducing over SVs in the inner loop | All window pairs within the mismatch cutoff |

The model caches packed SV windows, their self-kernels, the l-mer table and the l-mer index per device (`GkmSVM._sv_cache`); the first call pays for building them (0.55 s and 81 MB for the l-mer index over 32,715 SVs at l=10).

CPU kernels are Numba `@njit(parallel=True)`; the threading layer is pinned to `workqueue` on import. On CPU, kernel scoring through the l-mer index fuses the query with the reduction over SVs (`LmerIndex.reduce`): each row's kernel values accumulate in a cache-resident buffer and are normalized, passed through the RBF and summed at once, so no [rows, S] array exists. Other paths reduce the raw block in one parallel pass (`svm._normalized_dot`, and `explain._impact_cpu` for mutation impact). `to_deltasvm` on an RBF model scores only the canonical half of the l-mers (an l-mer and its reverse complement score the same with RC on). CUDA kernels are CuPy RawKernels; Metal kernels use `mx.fast.metal_kernel`. The CUDA index query accumulates float64 with `atomicAdd`; Metal has no float64, so it counts hits per mismatch level in int32 and combines levels in float64 on the CPU (exact; results are returned as float32). Above `_METAL_QUERY_MAX_DENSITY` (2.5) index entries per l-mer MLX queries on CPU instead, and ISM above `_METAL_MUTANTS_MAX_DENSITY` (6): the scattered atomics lose to the CPU there (dsQTL `-t 3` model, 500 × 300 bp: prediction 11.7 s -> 5.3 s, CPU 4.9 s). On CUDA the cost is the scattered atomics themselves (about 7×10¹⁰ for `to_deltasvm` on the model below), and int32 counting measured no faster.

## Measurements

Variant scoring on the GkmExplain paper's `-t 3` dsQTL model (gkmrbf l=10 k=6 d=3, 32,715 SVs, negset 1) for the 28,309 dsQTL test variants (51 bp windows): panel 1C, `benchmarks/results/timing_variants.tsv`, on Sherlock (CPU: a whole AMD EPYC 9384X node, 16 threads unless noted; GPU: an NVIDIA L40S). gkmsvm-lite compute time, excluding model loading and JIT; LS-GKM and mLS-GKM whole-process wall time.

| Method | gkmsvm-lite CPU | gkmsvm-lite CPU, 1 thread | gkmsvm-lite L40S | LS-GKM | mLS-GKM |
|---|---|---|---|---|---|
| Kernel delta (ISM), `score_variants(method="kernel")` | 40.2 s | 475 s | 5.95 s | 445 s (1 thread: 1230 s) | 115 s |
| Mutation impact, `method="gkmexplain"` | 27.6 s | 332 s | 5.42 s | 1540 s (1 thread) | 176 s |
| deltaSVM table, `to_deltasvm(query_norm=True)` | 20.0 s | 273 s | 5.84 s | 460 s (1 thread) | 45.3 s |

gkmsvm-lite's deltaSVM table scores all 4^10 10-mers; LS-GKM and mLS-GKM score the 524,800 canonical ones (`nrkmers.py`). Results agree with LS-GKM's to 1e-5 (ISM), 5e-6 (mutation impact) and 5e-6 (deltaSVM weights). `results/a100/` and `results/p100/` hold the GPU parts on an A100 and a P100.

Earlier measurement, on a server with an NVIDIA Titan Xp (`benchmarks/timing_variants.py --no-lsgkm`, compute time). "Before" is the code before the l-mer index (`cba8064`), "after" the code at `40c7c5f`:

| Method | Device | Before | After | Peak RSS before → after |
|---|---|---|---|---|
| Kernel delta, `score_variants(method="kernel")` | CPU | 1137 s | 155 s | 2916 → 1537 MB |
| | CUDA | 86.9 s | 20.2 s | 1557 → 1557 MB (GPU pool 2139 → 786 MB) |
| Mutation impact, `method="gkmexplain"` | CPU | 570 s | 168 s | 5190 → 1537 MB |
| | CUDA | 43.0 s | 19.2 s | 1558 → 1557 MB (GPU pool 5388 → 786 MB) |
| `to_deltasvm(query_norm=True)`, all 4^10 10-mers | CPU | ~1 h (est.) | 113 s | → 1631 MB |
| | CUDA | | 43.7 s | → 1557 MB |

There, the two variant methods cost the same: both were one index query on [ref; alt] against the SVs, followed by a different reduction. RBF mutation impact now runs one fused CPU pass per variant in which the alt row visits only the windows that differ from ref (`LmerIndex.impact`; 500 variants on CPU: 9.8 s -> 5.1 s), while kernel delta still scores [alt; ref] through the index, fused with the reduction over SVs (`LmerIndex.reduce`). `to_deltasvm` on an RBF model is a one-time conversion, because every one of the 1M l-mers still produces a full row of 32,715 kernel values; afterwards a `DeltaSVM` scores a variant with a few table lookups (28,309 variants in 0.03 s on CPU).

Training the `-t 2` l=10 k=6 d=3 model on GM12878 DHSs (300 bp) with a 2,048-column SMO cache on the L40S: panel S3, `benchmarks/results/training_memory.tsv` (training time; peak host RSS of the process):

| N | libsvm (precomputed Gram) | SMO | Nyström | SMO peak vs libsvm |
|---|---|---|---|---|
| 1,000 | 0.12 s | 0.13 s | 0.34 s | 440 vs 512 MB |
| 2,000 | 0.31 s | 0.30 s | 0.45 s | 446 vs 582 MB |
| 5,000 | 1.34 s | 2.47 s | 1.03 s | 472 vs 1082 MB |
| 10,000 | 4.12 s | 11.0 s | 2.62 s | 600 vs 1849 MB |
| 20,000 | 13.2 s | 37.0 s | 8.09 s | 823 vs 3845 MB |
| 30,000 | 25.9 s | 77.0 s | 16.6 s | 1065 vs 7747 MB |
| 43,768 | 48.4 s | 150 s | 34.6 s | 1324 vs 15519 MB |

SMO and libsvm give the same models to the solvers' tolerance (SV counts within 5 of each other, held-out scores r >= 0.99999 against the exact model). With a cache smaller than N, SMO recomputes evicted columns, so it trades time for O(cache × N) memory. Nyström computes fewer kernel values but still builds a dense N × N approximate Gram, so it saves time, not memory, and its held-out scores drift from the exact model's as N grows (r = 0.93 at 43,768).

On an Apple M4 (CPU), `-t 3` SMO at N=1,000 takes 1.2 s and `-t 5` SMO at N=600 takes 1.4 s; both previously computed one column at a time through the one-hot path (80 s and 2.7 h). Biases and SV counts match the earlier code.

Comparisons with LS-GKM are the paper figures in `benchmarks/` (`make -C benchmarks fig1b fig1c s2 s3`); see [benchmarks/README.md](../benchmarks/README.md).

## Memory

- Kernel scoring holds [rows, S] kernel values in blocks of at most `GkmSVM._MAX_SCORE_BYTES` (512 MB), whatever `batch_size` says. `batch_size` bounds the inputs moved to the device at once.
- Without the l-mer table or index, every masked-join call indexes all SV windows, so `batch_size` is raised until a call fills ~256 MB of [batch, S] (`GkmSVM._min_batch_size`).
- GkmExplain processes inputs in batches (default 50); its per-SV kernels take SVs in chunks of `model.sv_chunk_size` (default 2,000). The explain table is [4^l, 4, l] float64 in host memory (1.4 GiB at l=11), built only when it fits `_EXPLAIN_TABLE_BYTES` (2 GiB) under `use_explain_table="auto"`; the l-mer path's V is 4^l float64.
- Training with `solver="libsvm"` holds the N × N float64 Gram in host RAM; `solver="auto"` switches to SMO when it would exceed 75% of available RAM or `max_gram_gb`. Available RAM is capped at the process's cgroup limit (a Slurm job's `--mem`).

## Reproducing

```bash
python benchmarks/timing_variants.py --device cuda --no-lsgkm --n-variants 500
python benchmarks/training_memory.py --device cuda --no-lsgkm --sizes 1000,2000,5000,10000
make -C benchmarks s2      # throughput vs number of SVs, benchmarks/results/throughput.tsv
```
