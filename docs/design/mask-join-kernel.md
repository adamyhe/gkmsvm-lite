# Masked-join gkm kernel: handoff

Status: implemented on `main`. The masked join serves Gram tiles and SMO columns for `-t 0` to `-t 5` on CPU, CUDA and MLX (Metal join with int32 counts); the exact l-mer table serves prediction and `to_deltasvm` for `-t 0/1/2/4`; a cached l-mer index serves scoring, ISM and mutation impact against a model's SVs, and SMO columns over the training set, for `-t 1`-`-t 5`. Open: CUDA join tuning from a hardware profile, and the optimizations listed near the end. The sections below keep the design rationale and the measurements that motivated each step.

## Goal

Make gkmsvm-lite's kernel computation faster than LS-GKM at every training size, on CPU and GPU, with results that match the current kernel to floating-point rounding. Nothing is being published soon, so the aim is the fastest exact kernel, not a quick fix.

## Why the current kernel loses at large N

Every packed path (`_fused_pairwise_packed_numba`, the CUDA and Metal equivalents) compares every window of x with every window of y: an XOR plus a popcount per pair, then a table lookup if the pair is within `dmax = l - min_matches` mismatches. About 0.1% of pairs contribute. The min-matches check skips the addition but not the comparison, so the cost per sequence pair is W² (~84k at 300 bp), and a Gram matrix costs O(N² W²).

LS-GKM computes the same kernel with a k-mer tree that abandons a branch beyond `d` mismatches and shares prefix work across all sequences. Measured on one server (Xeon E5-2620 v4, 16 threads for both; Titan Xp; `-t 2 l=11 k=7 d=3`, 300 bp, default solvers):

| N | LS-GKM | lite CPU | lite GPU | LS-GKM / GPU |
|---|---|---|---|---|
| 500 | 3.4 s | 3.6 s | 0.2 s | 17× |
| 1,000 | 8.7 s | 14.2 s | 0.6 s | 14× |
| 2,000 | 22.3 s | 56.8 s | 1.8 s | 12× |
| 5,000 | 75.4 s | 353 s | 8.9 s | 8.5× |

Fitted time ∝ N^b between 2k and 5k: LS-GKM b = 1.33, lite GPU b = 1.74 (heading to 2), lite CPU b = 1.99. Extrapolated to the full GM12878 set (~45k), the GPU's lead falls to 2–3.5×, and the CPU is ~10× slower than LS-GKM. Diagonal-tile symmetry (already on `main`) and a pigeonhole filter (1.5–2.2×; the prototype script is in git history) shift the curve but don't change its slope.

## The identity

For every kernel with a packed path, the weight table h(m) is zero beyond some `dmax` (checked: `-t 0` l11k7 nonzero for m ≤ 4, `-t 1` and `-t 2` l11k7d3 for m ≤ 3). Let j_t = l − dmax + t for t = 0..dmax. The (dmax+1)×(dmax+1) system

    h(m) = Σ_t a_t · C(l − m, j_t),   m = 0..dmax

is triangular (C(l−m, j_t) = 0 when m > dmax − t), so it has a unique solution a_t. For m > dmax, every C(l − m, j_t) is zero, matching h(m) = 0, so the identity holds for all m.

C(l − m, j) is the number of size-j position subsets S on which two l-mers with m mismatches agree. Summing over window pairs:

    K_raw(x, y) = Σ_t a_t · Σ_{|S| = j_t} ⟨Φ_S(x), Φ_S(y) + Φ_S(rc y)⟩

where Φ_S(x) is the histogram of x's windows restricted to positions S (a vector over 4^j masked keys). Each inner product is an exact-match join on masked keys: no mismatch counting, no verification, and pairs beyond `dmax` never match any mask. Normalization and the RBF `exp` are applied afterwards, unchanged.

| Setting | Mask sizes j_t | Masks per size | Total masks | a_t | Cancellation factor |
|---|---|---|---|---|---|
| `-t 2` l11 k7 d3 | 8, 9, 10, 11 | 165, 55, 11, 1 | 232 | 1.77e-3, −2.65e-3, 7.09e-3, −4.82e-3 | 2.4 |
| `-t 2` l10 k6 d3 | 7, 8, 9, 10 | 120, 45, 10, 1 | 176 | 2.78e-3, −7.28e-3, 2.12e-2, −4.84e-2 | 5.4 |
| `-t 0` l11 k7 | 7 | 330 | 330 | 1 | 1 |

The cancellation factor is max over m of Σ_t |a_t| C(l−m, j_t) / |h(m)|. The per-level join counts are integers, so the only rounding is in the final weighted sum, and a factor of 2–5 costs nothing visible in float64. `-t 0` is the plain gapped k-mer count: one level, a = 1.

The math is derived here, not taken from LS-GKM's source. Keep it that way: LS-GKM is GPL v3, and gkmsvm-lite is MIT. Don't read or translate LS-GKM's code for this work.

## Measured on real sequences

A scipy.sparse prototype (since removed; see git history) implemented the identity (one sparse product per mask) and compared it with the brute-force kernel. 200 GM12878 sequences (100 DHS, 100 null set 1), 300 bp, with RC:

| Setting | max \|ΔK\| / max \|K\| | Brute-force window pairs | Join hits | Ratio |
|---|---|---|---|---|
| l11 k7 d3 | 1.3e-15 | 6.73e9 | 8.17e7 | 82× fewer |
| l10 k6 d3 | 5.4e-16 | 6.77e9 | 1.68e8 | 40× fewer |

A "hit" is one (window, window, mask) match, i.e. one accumulation. The scipy prototype is already within 2× of the Numba brute force at N=200 despite Python overhead per mask, so it's a correctness oracle, not the implementation.

Hits are ~10× the expected number of contributing window pairs for random sequence (8.0e6 at l11). Two reasons: real DNA has more near-duplicate l-mers than random, and a pair with m mismatches hits Σ_t C(l−m, j_t) masks (232 for an identical pair, 1 for m = 3). The second is attackable (see Optimizations).

Both brute force and joins are O(N²) asymptotically: hits per sequence pair don't depend on N. The win is the constant, 40–80× fewer operations before any optimization. LS-GKM's b = 1.33 is a small-N effect; its cost must also approach N² because every contributing pair adds to the output. So beating it by a constant factor at large N is realistic, and the current slope disadvantage goes away.

## Algorithm

Work tile by tile, like `compute_gram` does now: a block of A query sequences against a block of B column sequences (B windows plus their RC). For each mask S:

1. **Keys.** For each window, gather its bases at S into a 2j-bit key. A size-(l−r) mask is "delete r positions", so the key is r+1 contiguous bit segments shifted together: `(w & m0) | ((w >> 2) & m1) | ((w >> 4) & m2) | ...`, O(r) ops instead of O(j). Precompute the segment masks per S.
2. **Group the column side.** Counting sort of the B (+RC) windows by key into CSR (offsets over 4^j keys, column-sequence ids). 4^8 = 65,536 and 4^9 = 262,144 buckets are fine for counting sort; for j = 10 and 11 (12 masks in total), sort or hash instead of allocating 1–4M offsets.
3. **Join.** For each query window, look up its key's bucket and add a_t into `tile[query_seq, col_seq]` for each entry.
4. **Combine.** Accumulate a_t-weighted hits straight into a float64 tile, or count per level in int32 and combine at the end, which keeps the per-level sums exact. Benchmark both.

Then normalize with the diagonal and, for `-t 3`, apply `exp(γ(K − 1))`, exactly as `compute_gram` does now.

Diagonal tiles keep the symmetry trick: process only query < column, plus the diagonal, and mirror. The self-kernel diagonal K(x, x) can stay brute force (O(N W²), small) or use the same join with A = B = one sequence.

### CPU (Numba)

- Parallelize over (tile, mask) or over query windows within a tile. Each thread needs its own tile accumulator, or the masks must be partitioned so that two threads never write the same entry. Per-thread tiles of 256×256 float64 are 512 KB each and fit in L2. Benchmark tile size.
- Per tile, the column-side CSR for all 232 masks is recomputed. Sort cost per tile pair ≈ masks × (A + 2B) × W. Estimate this against the hit count before choosing the tile size: larger tiles amortize sorting, smaller ones keep the accumulator in cache.

### GPU (CuPy)

- Keys: one thread per window per mask, trivially parallel.
- Grouping: CuPy's sort (CUB radix) on (key, seq) pairs, or counting sort with atomics for j ≤ 9.
- Join: for each key, the hits are the outer product of its query-side and column-side seq lists. Assign key groups to warps; accumulate with float64 `atomicAdd` into a tile in global memory, or into shared memory for small tiles. Float64 atomics need CC ≥ 6.0, already required.
- Heavy groups (poly-A and other low-complexity keys) make the outer products very uneven. Detect groups above a threshold and handle them separately, e.g. as a small dense GEMM of indicator matrices, so that one warp doesn't serialize the kernel.

## Prediction: exact l-mer table for non-RBF kernels

Prediction has the same problem as training: lite's cost is B × S × W², linear in the number of support vectors, while LS-GKM's stays roughly flat. Same server, 1,000 test sequences, 16 threads:

| SVs | LS-GKM | lite CPU | lite GPU |
|---|---|---|---|
| 500 | 61.8 s | 19.1 s | 0.34 s |
| 997 | 61.9 s | 27.9 s | 0.66 s |
| 1,968 | 62.2 s | 54.8 s | 1.09 s |
| 4,597 | 62.9 s | 127.9 s | 5.35 s |

For `-t 0`, `-t 1` and `-t 2` the score is linear in the query's windows except for the query's own normalization:

    f(x) = Σ_i w(x_i) / √K_raw(x, x) + b,   w(u) = Σ_sv (α_sv / √K_raw(sv, sv)) · K_raw(u, sv)

where u ranges over l-mers and K_raw(u, sv) already includes the SV's reverse complement. This is the deltaSVM table without its per-l-mer norm, plus the exact query norm. Checked on the 10-SV ENCODE fixture (`-t 2` l11 k7, 40 sequences): `to_deltasvm(query_norm=False)` minus its bias, divided by √K(x,x), plus b, matches `model(x)` to 1.7e-8, which is consistent with the float32 table. Store the table in float64 (32 MB at l = 11) to make it exact to rounding.

Per query this costs W lookups plus the self-kernel (one SV's worth of work), independent of S: ~4,600× less work than now at 4,597 SVs, ~70,000× for a 70k-SV ENCODE model.

Building the table is the cost. `to_deltasvm` scores all 4^l l-mers against every SV (4.3 s for 10 SVs, so hours for thousands). Instead, scatter from the SVs: for each SV window v (and its RC), add α_sv/√K_sv · h(m) to w(u) for every u within dmax mismatches of v. That's S × W × 2 × 4,984 updates at l11 d3 (1.3e10 for 4,597 SVs), and each SV's neighborhood can be enumerated directly, with no search. Parallelize over SVs with per-thread tables, or partition the table by key.

`-t 3` applies exp(γ(K_norm − 1)) per SV, so it doesn't decompose into a table and needs the masked join. Weighted kernels are out of scope as above.

This is the simplest, largest win in this doc. Do it before the masked join's phase 2.

## Scoring: cached l-mer index for RBF and per-SV kernel values

The l-mer table sums over SVs, so it can't serve `-t 3` prediction (exp per SV), ISM or `mutation_impact`, which need K(x, sv) for each SV. The masked join rebuilds its column index on every call, about 6 s for the 32,715-SV dsQTL `-t 3` model, however few queries a call has. When the columns are a model's SVs, `kernels/lmer_index.py` indexes them once instead: CSR over all 4^l l-mers, row v listing the SV of every window (fwd and RC) equal to v. A query window w visits each l-mer within dmax mismatches (w XOR one of sum_{m<=dmax} C(l,m) 3^m precomputed masks: 3,676 for l=10 d=3) and adds h(m) to every SV in that row. Each matching pair is visited once.

Measured against the masked join on that model (M4 CPU, one call):

| Queries | 51 bp | 300 bp |
|---|---|---|
| 16 | 0.03 s vs 4.5 s | 0.25 s vs 4.8 s |
| 64 | 0.10 s vs 4.6 s | 0.72 s vs 5.0 s |
| 1024 | 1.6 s vs 5.7 s | 10.7 s vs 13.7 s |

It applies to count-table kernels with at most `MAX_NEIGHBORS` (8,192) neighbors and l <= 12. `-t 0` has no d cutoff below l - k (about 20k–32k neighbors at l = 10–11) and keeps the masked join. CPU and CUDA accumulate float64 directly; Metal counts hits per mismatch level in int32 (it has no float64) and combines levels in float64 on the CPU. Per-level int32 counting on CUDA measured slower on a Titan Xp (48.7 s vs 43.7 s for `to_deltasvm` on the 32,715-SV `-t 3` model, with a larger GPU pool): the cost is the ~7×10¹⁰ scattered atomics, whatever their type. A faster CUDA query would need shared-memory accumulation over SV tiles.

SMO columns use the same index built over the training set: a column K(i, :) is one query row, so a cache smaller than N no longer degrades to single brute-force columns.

## Where it plugs in

| Entry point | Path today | File |
|---|---|---|
| Gram tiles (`compute_gram`: libsvm, Nyström) | Masked join in column panels | `gram.py` |
| SMO kernel columns (`KernelColumnCache.compute_columns`) | l-mer index over the training set (`-t 1`-`-t 5`), else batched masked join | `solver.py` |
| Prediction `-t 0/1/2/4` | Exact l-mer table | `svm.py`, `lmer_table.py` |
| Prediction `-t 3`, other kernels; ISM; mutation impact | Cached l-mer index over the SVs, else masked join | `svm.py`, `ism.py`, `explain.py` |
| `to_deltasvm` | l-mer table, else scoring all 4^l l-mers | `svm.py` |
| GkmExplain attribution | Brute-force packed kernel (per-position decomposition) | `explain.py` |

The brute-force packed kernels remain for calls with fewer than 16 sequences on a side and as the test oracle (`kernel.use_masked = False`).

## MLX (Apple GPU)

Steps 1–2 below are implemented (the l-mer table runs on MLX in float32; the Metal masked join and the Metal l-mer index count in int32), and the index build moved to Metal as well. GkmExplain attribution still falls back to CPU. The plan is kept for its constraints.

Constraints that shape the design:

- **No float64 in Metal.** Every kernel value must come from exact integer counts, combined in float64 on the CPU. Unified memory makes moving data between CPU and GPU free, so the CPU can also do cheap stages like index building and the final combine.
- **`mx.fast.metal_kernel`** takes a kernel body with named inputs and outputs. `atomic_outputs=True` makes outputs `device atomic<...>`, for histograms and scatters. Threadgroup arrays need a size fixed at compile time, which is a `template` parameter. Threadgroup memory is 32 KB, so a counter row holds at most 8,192 int32; use 4,096-column tiles.
- **Dispatch:** `get_array_module` returns `mlx.core` for MLX arrays, and `is_mlx()` selects the branch, as in `pairwise_from_indices`.

Steps, cheapest first:

1. **l-mer table prediction on MLX.** `_score_lmer_table` already uses plain indexing and `sum`. The table is float64, which Metal can't use, so keep it as NumPy and gather on the CPU through unified memory, or pass a float32 copy to `mx.take`. Prediction already returns float32 on MLX, so float32 table values are acceptable there. Add an MLX case to `tests/test_lmer_table.py` (skipped without MLX) comparing with the CPU table.
2. **Masked join, Metal join with CPU-built index.** Build the column index with the existing Numba `_build_column_index`. It's cheap, and unified memory means `mx.array(offsets)` costs little. Then run a Metal join kernel with the same structure as the CUDA one:
   - one threadgroup per query sequence, with `threadgroup atomic_int counts[TILE]`;
   - lane groups of 16 scanning each bucket;
   - output **int32 counts** `[B, S]` per plan level, not float.

   The CPU combines `Σ coef_level × counts_level` in float64. That keeps results exact and identical to the CPU and CUDA paths. With 4 levels for `-t 2`, a 1,000×1,000 tile needs 16 MB of int32 counts. (`-t 4/5` window weights break per-level counting; MLX joins them on the CPU.)
3. **Move the index build to Metal**, if profiling shows it matters. Histogram and scatter use atomic outputs with `atomic_fetch_add_explicit`, and the prefix sum uses `mx.cumsum`.
4. **Self-kernel diagonals.** Keep them on CPU: the packed pair loop over the symmetric half (`direct._self_diagonal_packed_numba`), by mismatch count (`weighted._self_diagonal_weighted_numba` adds window weights for `-t 4`).

Validation: `tests/test_mlx.py` patterns plus exactness against the CPU masked join (integer counts make bitwise equality achievable). Benchmark: a dev check script (since removed) run on an M-series Mac, against the CPU masked path on the same machine. The CPU path is already fast there (1,000-sequence Gram in 1.3 s on an M4), so ship the Metal join only if it is clearly faster.

Testing needs `mlx` installed (`uv pip install -e ".[mlx]"`); it isn't in the current dev environment.

## Optimizations to evaluate after the baseline

- **Fewer hits for near-identical pairs.** An identical pair is hit 232 times. Alternative: use only size-(l−dmax) masks, compute m for each hit with one XOR+popcount, and add the pair only at its canonical mask, e.g. the mask that drops the mismatched positions plus the first (dmax − m) others. That counts every contributing pair exactly once with weight h(m), and the a_t basis isn't needed. It trades the extra levels for a popcount per hit. Measure which wins.
- **Distinct l-mers first.** Collapse each tile's windows to (l-mer, seq, multiplicity) before joining. Repeats then cost one hit per distinct pair times a multiplicity product.
- **Spaced-seed families.** Fewer, smaller masks plus verification, the read-mapping trade-off between the 165-mask family and the 4-block pigeonhole. Only worth trying if sorting dominates.

## Validation

- **Exactness.** For `-t 0`, `-t 1`, `-t 2` and `-t 3`, with RC on and off, normalization on and off, odd N and several tile sizes, Gram and rectangular blocks must match the brute-force path to ≤ 1e-12 relative. `tests/test_gram.py::TestSymmetricDiagonalTiles` is the template.
- **LS-GKM agreement.** Existing oracle fixtures (`tests/fixtures/`) and `benchmarks/agreement_lsgkm.py` must be unchanged.
- **End to end.** Retrain the S1 models (`make -B s1`); per-model AUPRCs must still match the paper to 4 decimals.
- **GPU vs CPU.** Same as now (< 3e-3 from the float32 diagonal), or better if the GPU path accumulates in float64.

## Benchmarks and success criteria

- `python benchmarks/timing_train.py --device cuda --sizes 1000,2000,5000,10000,20000`: compare against the table above, on the same server and thread count.
- `make -B s3`: solver sweep up to the full dataset.

Targets:

1. CPU training faster than LS-GKM at 16 threads for every N ≥ 1,000 in Fig. 1B's sizes.
2. GPU training ≥ 10× faster than LS-GKM on the full GM12878 set.
3. No regression at small N: keep brute force where it's faster, chosen by a measured crossover.

## Open questions

- How skewed the bucket sizes are on real data, and how much heavy-group handling matters. Measure the bucket-size distribution per mask on GM12878 first.
- Tile size and accumulator layout for CPU and GPU.
- Whether per-level integer counting or direct float accumulation is faster, and whether the canonical-mask variant beats the a_t basis.
- Whether the same machinery gives a faster `to_deltasvm` (all 4^l l-mers against the SVs). It's a join of a complete key set against the SV windows, so it plausibly does.

## Files

- `src/gkmsvm/kernels/packed.py`: `PackedGkmKernel` (`pack`, `raw_packed`, `raw_diag_packed`), `PAD_WINDOW`.
- `src/gkmsvm/kernels/direct.py`: brute-force packed kernels (`_fused_pairwise_packed_numba`, `_fused_pairwise_packed_sym_numba`, CUDA and Metal equivalents), `_min_matches`.
- `src/gkmsvm/kernels/masked.py`: mask plans (`mask_plan`) and the joins on CPU, CUDA and Metal.
- `src/gkmsvm/kernels/lmer_index.py`: the cached l-mer index.
- `src/gkmsvm/lmer_table.py`: exact l-mer table build.
- `src/gkmsvm/kernels/esttrunc.py`: `_build_estlmer_table` (h(m) for `-t 1`/`-t 2`).
- `src/gkmsvm/gram.py`: tiling, symmetry, column panels, normalization.
- `benchmarks/README.md`: benchmark pipeline. Run timings on one host and record it (result headers do this).
