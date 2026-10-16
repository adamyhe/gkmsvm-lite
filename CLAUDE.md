# CLAUDE.md

Agent-facing reference for working on this codebase. Human-readable docs are in `docs/` and `README.md`.

## Build and test

```bash
uv pip install -e "." --group dev    # CPU only
uv pip install -e ".[gpu]" --group dev  # with CuPy GPU support
pytest tests/ -v
pytest tests/test_codec.py          # single file
pytest tests/ -k "test_rc"          # pattern match
```

## Source layout

```
src/gkmsvm/
├── __init__.py          # public API re-exports
├── svm.py               # GkmSVM model, scoring, SV cache, to_deltasvm
├── lmer_table.py        # exact 4^l l-mer weight table (-t 0/1/2/4 prediction)
├── cli.py               # CLI entry point (gkmsvm command)
├── train.py             # train_gkmsvm() (C-SVC) + train_gkmsvr() (epsilon-SVR)
├── solver.py            # KernelColumnCache, smo_solve(), _csmo C extension — column-cached SMO
├── gram.py              # compute_gram() — tiled Gram matrix with symmetry
├── serialization.py     # save/load npz and LS-GKM text formats
├── codec.py             # one-hot encode/decode, RC, validation
├── ism.py               # in-silico mutagenesis
├── explain.py           # GkmExplain attribution
├── deltasvm.py          # DeltaSVM linear scoring
├── fasta.py             # FASTA I/O, extract_loci (vendored tangermeme)
├── backend.py           # get_array_module (numpy/cupy/mlx dispatch)
├── _threading.py        # Numba threading layer pin
├── kernels/
│   ├── base.py          # GkmKernel ABC (base / finish hooks)
│   ├── packed.py        # PackedGkmKernel: pack(), raw_packed(), raw_diag_packed()
│   ├── direct.py        # -t 0 gkm_cnt + brute-force packed kernels (CPU/CUDA/Metal)
│   ├── esttrunc.py      # -t 1 gkm_estfull, -t 2 gkm_esttrunc
│   ├── masked.py        # masked-join kernel (CPU/CUDA/Metal)
│   ├── lmer_index.py    # cached l-mer -> SV index for scoring (CPU/CUDA/Metal)
│   ├── rbf.py           # RbfKernel wrapper; -t 3 gkmrbf
│   └── weighted.py      # -t 4 wgkm, -t 5 wgkmrbf
└── importers/
    ├── lsgkm.py         # load_lsgkm_model()
    ├── classic.py       # load_classic_model() — C gkmSVM (LIBSVM-style)
    ├── __init__.py      # shared header parsing (parse_header_value) and build_model()
    ├── r_gkmsvm.py      # load_r_gkmsvm_model() — R gkmSVM (.gkmmodel, _svalpha.out)
    └── deltasvm.py      # load_deltasvm_model()
```

## Benchmarks

`benchmarks/` holds the application-note pipeline: one script → one `results/*.tsv` (provenance header from `_common.write_tsv`) → one `make` target per figure panel; `figures.ipynb` only plots. Experiments that feed no figure go in `benchmarks/dev/`. Map of panels and files: `benchmarks/README.md`.

## Conventions

- Python >=3.10, NumPy, Numba >=0.57, scikit-learn, tqdm. CuPy >=12 optional (`[gpu]` extra). MLX >=0.10 optional (`[mlx]` extra)
- Array format: `[batch, 4, length]` one-hot DNA, channel order A=0/C=1/G=2/T=3
- Output: `[batch, 1]` floating-point margin scores
- Arrays are numpy.ndarray (CPU), cupy.ndarray (NVIDIA GPU), or mlx.core.array (Apple GPU)
- `model.cuda()` / `model.mlx()` / `model.cpu()` moves arrays between devices (`backend.DeviceModel`, shared by `GkmSVM` and `DeltaSVM`)
- Kernel normalization on by default, RC equivalence on by default
- Score = `Σ coef_i × K(x, sv_i) + bias` where `bias = -rho` (LS-GKM) or `+rho` (classic gkmSVM)
- `resolve_kernel_type()` maps aliases and integers to canonical internal names
- Kernel modes: `-t 0` gkm_cnt/direct, `-t 1` gkm_estfull/estimated_full, `-t 2` gkm_esttrunc/estimated (default), `-t 3` gkmrbf/rbf, `-t 4` wgkm/weighted, `-t 5` wgkmrbf/weighted_rbf
- ISM: `ism(model, x, batch_size=N)` → `[B, 4, L]` score deltas (window-delta optimization)
- GkmExplain: `gkmexplain(model, x, mode=..., batch_size=50)` → `[B, 4, L]` attribution scores, `-t 0/1/2` only (others raise; use `mutation_impact`/ISM). MLX falls back to CPU (float32 violates completeness axiom). Modes (string or legacy int):
  - `"importance"` (0) — hypothetical × one-hot. Sum = score - bias (completeness axiom). For visualization.
  - `"hypothetical"` (1) — all 4 bases. For TF-MoDISco.
  - `"perturbation"` (2) — discrete kernel-value Δ per mutation (lsgkm C mode 3, `perturbation_eff=1`). Ignores the RBF `exp()`.
- Mutation impact: `mutation_impact(model, ref, alt)` = lsgkm `gkmexplain -m 5`, the paper's dsQTL VEP score. `score_variants(method="gkmexplain")` calls it. Normalizes ΔK by the ref norm only; RBF distributes `exp(γ(K-1)) - exp(-γ)` proportionally to ΔK/K. Not equal to ISM. Oracle: `tests/fixtures/lsgkm_mode5_*`.
- RBF (`-t 3`, `-t 5`): `K = exp(γ(K_norm - 1))` on the normalized base kernel, matching lsgkm.
- `to_deltasvm(query_norm=True)` makes each weight = gkmpredict on that l-mer (how lsgkm deltaSVM weights are built); default drops the l-mer norm.
- `batch_size` on `model()`, `score_variants()`, `ism()`, `gkmexplain()` chunks over inputs. Kernel scoring computes [rows, S] blocks of at most `GkmSVM._MAX_SCORE_BYTES`; `model.sv_chunk_size` only chunks SVs in GkmExplain. When scoring goes through the masked join (no l-mer table or index), small `batch_size` values are raised so each call fills ~256 MB of [batch, S] (`GkmSVM._min_batch_size`): every masked-join call indexes all SV windows.
- `verbose=True` on `model()`, `score_variants()`, `ism()`, `gkmexplain()` enables tqdm progress bars over input batches
- No PyTorch dependency. Gradient-based methods are incompatible — use GkmExplain or ISM
- tangermeme interop is vendored (pyfaidx for FASTA extraction)

## Key implementation details

- Forward pass, ISM, and GkmExplain use the packed uint32 path (XOR + popcount). GkmExplain has one kernel per device (`_explain_cpu`, CUDA `fused_explain`) with a `perturbation` flag; it fuses normalization + coefficient multiplication + SV-dimension reduction, accumulating directly into `[B, 4, L]` result (no per-SV intermediate). Forward + RC SV windows concatenated for single kernel launch. CPU: `prange(B)` with L1-resident per-thread result arrays. GPU: float64 `atomicAdd` (CC >= 6.0). Apple GPU: custom Metal shaders via `mx.fast.metal_kernel` (forward/ISM only; GkmExplain falls back to CPU). ISM builds the packed windows covering each position (padded with `PAD_WINDOW`, which cancels in new − old) and works for every kernel type, by path: with the l-mer table, table weights over those windows (`ism._ism_table`); on CPU with the l-mer index, one fused pass per (position, sequence) adds the changed windows' hits to the reference row and reduces over SVs (`LmerIndex.mutants`); otherwise per-SV values in blocks of at most `_MAX_SCORE_BYTES`. Mutant norms come from `raw_diag_mutants` (for `-t 0/1/2/3` on CPU, only pairs touching a changed window). Inputs the packed path can't represent (weighted with N) score all 4L mutants through `model()`.
- Kernel interface: `kernel.base` is the kernel giving raw values (the base of an RBF kernel, else itself) and `kernel.finish(k_norm)` applies the RBF. Base kernels are `PackedGkmKernel`s: `pack(x)` gives [B, W] uint32 windows on x's device (MLX packs on CPU), `raw_packed(bx, by, by_rc)` the raw kernel with the RC block in the same call (masked join, or brute force below 16 rows), `raw_diag_packed(bx, bx_rc)` the self-kernel. `packs(x)` is False for weighted kernels on inputs with N (one-hot fallback). Gram, SMO columns, scoring, ISM and mutation impact all use this, so `-t 3`/`-t 5` get packed caches, Gram panels and batched SMO columns.
- The model caches packed SV windows, their base self-kernels and the l-mer index per device (`GkmSVM._sv_cache`, cleared on device moves); `model.clear_caches()` also drops the l-mer table. `use_lmer_index`/`use_lmer_table` are checked on every call.
- CPU kernel scoring normalizes, applies the RBF and reduces over SVs in one numba pass. Through the l-mer index it is fused with the query (`LmerIndex.reduce`: each row's hits accumulate in a cache-resident length-S buffer, never a [B, S] block); otherwise `svm._normalized_dot` reduces the raw block. `to_deltasvm` on RBF models scores only canonical l-mers (index <= RC index) and copies weights to their reverse complements, as lsgkm's nrkmers.py pipeline does.
- Min-matches skip: for esttrunc l=11 k=7 d=3, `min_matches=8`. 99.88% of window pairs skipped.
- CPU self-kernels for `-t 0/1/2/3` (`_self_diagonal_packed_numba`, used for SV and query norms) visit each pair of the symmetric forward and RC blocks once, four rows per pass over the columns, with branch-free table lookups; rows with N (packed as A on both strands, so the RC block isn't symmetric) visit every RC pair.
- CPU inner loop: Numba `@njit(parallel=True, fastmath=True)`. NVIDIA GPU inner loop: CuPy RawKernel with shared-memory caching. Apple GPU inner loop: Metal kernel with per-thread accumulation and `popcount()`.
- MLX compatibility: `get_strides()` for arrays without `.strides`, `xp.ascontiguousarray()` for `.copy()`, `to_cpu(X)[indices]` for fancy indexing. DeltaSVM auto-chunks intermediates >256 MB.
- Scoring against the SVs uses the cached l-mer index (`kernels/lmer_index.py`) for kernels with at most `MAX_NEIGHBORS` l-mers within dmax mismatches (e.g. `-t 1`/`-t 2`/`-t 3` with d <= 3, l <= 12; `-t 4`/`-t 5` with l - k <= 3, since a weighted neighbor's weight is e_k over its agreeing positions, `lmer_index.index_neighbors`): CSR over 4^l l-mers listing the SVs containing each, built once; a query window visits its dmax-neighborhood, so a call costs nothing per SV. CPU/CUDA accumulate float64 (CUDA int32 per-level counting measured slower: the cost is the scattered atomics); Metal counts hits per distinct weight in int32 and combines on CPU (above `_METAL_MAX_LEVELS` distinct weights, as for weighted kernels, MLX queries on CPU). The index covers the packed SVs; `_sv_raw` joins the N rows' columns, and the fused CPU reductions run only without N rows. `model.use_lmer_index = False` forces the masked join.
- `raw_packed` uses the masked-join kernel (`kernels/masked.py`) when both sides have >= 16 sequences: h(m) as a signed sum of C(l-m, j), each term an exact-match join on masked l-mer keys. `kernel.use_masked = False` forces brute force. `compute_gram` then works in column panels so each column index is built once. MLX builds the index and joins in Metal, writing int32 counts that the CPU combines in float64. `-t 4`/`-t 5` use it too (per-mask coefficients; inputs with N fall back to the per-position path).
- `to_deltasvm` reads the l-mer table when it applies, else scores all 4^l l-mers through `_score_unbatched` (RBF always keeps the query normalization; non-RBF without `query_norm` multiplies weights back by sqrt(K(u,u))). It never mutates the model.
- Prediction for `-t 0`/`-t 1`/`-t 2`/`-t 4` (l <= 12) uses an exact 4^l l-mer weight table (`lmer_table.py`, built from marginals of the SV l-mer weights; with RC, a mask and its mirror share one marginal, read at rc(u)): f(x) = sum_i w(x_i) / sqrt(K(x,x)) + b. `model.use_lmer_table`: `"auto"` (>= 500 SVs), True, False. RBF uses the kernel path.
- `compute_gram` self-Gram: skips tiles below the diagonal; on CPU, diagonal tiles compute one triangle (`_fused_pairwise_packed_sym_numba`, also valid for the RC block). RBF tiles use the base kernel's packed path, then `kernel.finish`.
- Training: `solver="auto"` uses the precomputed Gram + sklearn when the N×N float64 Gram fits in 75% of available RAM and under `max_gram_gb` (the Gram always lives in host RAM; tiles are computed on the training device); else column-cached SMO. `train.py` dispatches every solver through `_fit`, which builds the model. SMO uses WSS3 working set selection (C extension `_csmo` with Python kernel column callback), shrinking, and kernel columns cached in C — memory is O(cache_size × N) not O(N²). No Python fallback: without the extension, `solver="smo"` raises. `solver="nystrom"` uses low-rank kernel approximation for fast approximate training (fewer kernel evaluations; still a dense N×N approximate Gram, so no memory saving). `max_gram_gb` caps Gram allocation on shared compute.
- `KernelColumnCache` packs all training sequences once (through `kernel.base`). When the l-mer index applies (`-t 1/2/3`), it indexes all training sequences once and a column is one query row (~3-6 ms on CPU for 300 bp, N=5-10k), so SMO prefetches only into empty cache slots (the first fill), then computes single requested columns. Otherwise columns are all rows against the requested columns and their RCs, so the masked join indexes only the batch, and misses prefetch up to `_SMO_PREFETCH` columns into free cache slots. Without the index, a cache smaller than N degrades to single brute-force columns once full.
- SVR (`train_gkmsvr`) uses sklearn epsilon-SVR with precomputed Gram. SMO SVR is not yet implemented.
- `scikit-learn` provides the LIBSVM C solver for both SVC and SVR via `SVC(kernel='precomputed')` / `SVR(kernel='precomputed')`. No direct `libsvm-official` dependency.

## Gotchas

- LS-GKM defaults are `-t 2 -l 11 -k 7 -d 3`. Always pass parameters explicitly when generating oracle fixtures.
- The `d` parameter zeros weight table entries for m > d. Without this, scores diverge ~1.5% from `gkmpredict`.
- `-t 0` and `-t 2` use completely different weight table math. Do not mix.
- `score(ALT) - score(REF)` differs from deltaSVM's k-mer-weight linear approximation.
- Original gkmSVM (`.gkmmodel`) uses opposite sign convention for bias vs LS-GKM. Use `load_classic_model()` not `load_lsgkm_model()`.
- `-t 4`/`-t 5` (wgkm/wgkmrbf) require M and H parameters. They use the masked join with per-mask weights; N agrees with nothing, which packing (N as A) can't express. Inputs with N positions use the per-position match path (chunked over columns to `_MATCH_BYTES`). SVs with N (`GkmSVM._sv_n_rows()`; ENCFF579AOX has 4) are split off: the other SVs stay packed, the N rows' columns come from `raw_packed_n` (a CPU pair loop over agreement bits with per-window N masks), and the `-t 4` l-mer table adds their windows only for masks that avoid their Ns.
- LS-GKM C is GPL v3 — cannot wrap or link against it (gkmsvm-lite is MIT).
- GPU float32 diagonal causes <3e-3 score difference vs CPU float64.
- Beer lab `deltasvm.pl` drops the first weight-file line as a bias header; with gkmpredict output on `nrkmers.py` l-mers that gives AAAAAAAAAA/TTTTTTTTTT raw score 0 (bias included, so -bias in a `to_deltasvm` table). Published deltaSVM scores from that script (e.g. the GkmExplain paper) include this; `dsqtl_auprc.deltasvm_pl()` reproduces it.
- The gkmexplain repo's summed deltaSVM AUPRCs (README 0.18943 / 0.18795) cover negsets 1, 3, 5 only (`combine_folds.sh` cuts the wrong columns). Five-model sums from its output files are 0.19021 / 0.18997 (`PAPER_SUM_AUPRC`). Reproduction write-up: `docs/validation.md`.
