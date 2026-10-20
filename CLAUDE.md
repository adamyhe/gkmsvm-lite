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
- Output: `[batch, 1]` margin scores. Precision doesn't follow the input: scores, ISM, attributions, Gram matrices, coefficients, cached norms and deltaSVM weights are float64 on CPU and CUDA, float32 on MLX (`backend.float_dtype`; `DeviceModel._FLOAT_ARRAYS` keep that across device moves, so moving to MLX rounds them). One-hot arrays keep the caller's dtype (float32 by default), which only sets their memory
- Arrays are numpy.ndarray (CPU), cupy.ndarray (NVIDIA GPU), or mlx.core.array (Apple GPU)
- `model.cuda()` / `model.mlx()` / `model.cpu()` moves arrays between devices (`backend.DeviceModel`, shared by `GkmSVM` and `DeltaSVM`)
- Kernel normalization on by default, RC equivalence on by default
- Score = `Σ coef_i × K(x, sv_i) + bias` where `bias = -rho` (LS-GKM) or `+rho` (classic gkmSVM)
- `resolve_kernel_type()` maps aliases and integers to canonical internal names
- Kernel modes: `-t 0` gkm_cnt/direct, `-t 1` gkm_estfull/estimated_full, `-t 2` gkm_esttrunc/estimated (default), `-t 3` gkmrbf/rbf, `-t 4` wgkm/weighted, `-t 5` wgkmrbf/weighted_rbf
- ISM: `ism(model, x, batch_size=N)` → `[B, 4, L]` score deltas (window-delta optimization)
- GkmExplain: `gkmexplain(model, x, mode=..., batch_size=50)` → `[B, 4, L]` attribution scores, every kernel type (`-t 4/5`: each window pair's contributions times w_i(x) w_j(sv); V and the index entries carry the SV windows' weights, every path scales query window i by w_i). For `-t 3`/`-t 5`, as lsgkm does, each SV's contributions scale by (exp(γ(K−1)) − exp(−γ))/K of its normalized base kernel value K (0 when K <= 0), so importance sums to score − bias − exp(−γ)·Σcoef. MLX runs it on CPU and returns NumPy float64 (Metal can't accumulate in float64, which the completeness axiom needs; a CPU copy of the model is cached in `_sv_cache`), as `mutation_impact` returns NumPy on every device. Every path accumulates in float64 and returns the input's dtype. Modes (string or legacy int):
  - `"importance"` (0) — hypothetical × one-hot, N counted as A (it packs as A; as in lsgkm). Sum = score - bias (completeness axiom). For visualization.
  - `"hypothetical"` (1) — all 4 bases. For TF-MoDISco.
  - `"perturbation"` (2) — discrete kernel-value Δ per mutation (lsgkm C mode 4: `perturbation_eff=1` over every pair with up to d+1 mismatches, i.e. every pair a substitution changes; mode 3 leaves out the d+1 pairs). Each SV's ΔK is divided by the reference's norm (not the mutant's), and RBF models spread exp(γ(K−1)) − exp(−γ) proportionally as in the other modes, so it approximates score(mutant) − score(ref) rather than equaling it, as lsgkm's does. Oracle: `tests/fixtures/*.explain4.txt.gz`.
- Oracles for every mode and kernel type (`-t 0`-`-t 5`): `tests/fixtures/*.explain{0,1,4,5}.txt.gz`, kundajelab lsgkm fork's gkmexplain (`benchmarks/dev/lsgkm_wgkm_fixtures.py --explain-bin`; the fork rejects a `norc` line, so the script drops it).
- Mutation impact: `mutation_impact(model, ref, alt)` = lsgkm `gkmexplain -m 5`, the paper's dsQTL VEP score. `score_variants(method="gkmexplain")` calls it. Normalizes ΔK by the ref norm only; RBF distributes `exp(γ(K-1)) - exp(-γ)` proportionally to ΔK/K. Not equal to ISM. Oracle: `tests/fixtures/lsgkm_mode5_*`.
- RBF (`-t 3`, `-t 5`): `K = exp(γ(K_norm - 1))` on the normalized base kernel, matching lsgkm.
- `to_deltasvm(query_norm=True)` makes each weight = gkmpredict on that l-mer (how lsgkm deltaSVM weights are built); default drops the l-mer norm.
- `batch_size` on `model()`, `score_variants()`, `ism()`, `gkmexplain()` chunks over inputs. Kernel scoring computes [rows, S] blocks of at most `GkmSVM._MAX_SCORE_BYTES`; `model.sv_chunk_size` only chunks SVs in GkmExplain. When scoring goes through the masked join (no l-mer table or index), small `batch_size` values are raised so each call fills ~256 MB of [batch, S] (`GkmSVM._min_batch_size`): every masked-join call indexes all SV windows.
- `verbose=True` on `model()`, `score_variants()`, `ism()`, `gkmexplain()` enables tqdm progress bars over input batches
- No PyTorch dependency. Gradient-based methods are incompatible — use GkmExplain or ISM
- tangermeme interop is vendored (pyfaidx for FASTA extraction)

## Key implementation details

- GkmExplain explain table (`-t 0/1/2`, `explain._build_explain_table`): a pair's weight at position k depends only on the mismatch count and the 2-bit delta d = (u ^ v)_k there, so every window's attribution A[u, d, k] = sum_v V[v] G_kd(u ^ v) is an XOR convolution over the 4^l l-mers, computed with Walsh-Hadamard transforms (H(G_kd) is H(G_0d) with digits 0 and k swapped; the 4l inverse transforms run together down A's [4^l, 4l] rows). [4^11, 4, 11] float64 = 1.4 GiB, built in ~1 s on an M4; then 0.035 ms per 300 bp sequence on CPU, agreeing with the other paths to ~1e-14. One table per mode (hypothetical/importance share one) in `model._explain_tables` (host memory, kept across device moves, dropped by `clear_caches`). `model.use_explain_table`: `"auto"` builds one when a call's rows times the per-sequence cost (`_EXPLAIN_SEQ_S`, CPU vs CUDA) exceed the estimated build time and it fits `_EXPLAIN_TABLE_BYTES` (2 GiB), or reuses one already built; True/False force it. Lookups run on CPU; results return to x's device. Not for RBF (query-dependent SV weights).
- GkmExplain attribution is linear in the SVs, so with l <= 12 and either <= `MAX_NEIGHBORS` l-mers within the cutoff or >= `_EXPLAIN_SV_WINDOWS_PER_NEIGHBOR` (6) SV windows per neighbor (`-t 0` l=11 k=7 without d, 143,980 neighbors: ~1.5k+ 300 bp SVs) it goes through `explain._explain_lmer`: V[v] = sum_s coef_s / sqrt(K(sv, sv)) times v's occurrences in sv_s and its RC (4^l float64, cached), and each query window visits its neighborhood, adding V[v] times the alpha/kappa (or perturbation) weight to each position. Nothing per SV (3.5 ms per 300 bp sequence for ENCFF579AOX on an M4). For `-t 3` the SV weights depend on the query, so each neighbor sums them over the l-mer index's CSR entries instead (`_explain_index_cpu`; 29 ms per sequence for the dsQTL `-t 3` model). CUDA runs it in `explain_lmer` (`_explain_lmer_gpu`): 16 lanes per query window, lane k owning position k, look up 16 neighbors at once and shuffle them; each lane keeps four register sums indexed by the mask's 2-bit delta (a dynamically indexed [4, l] array spilled to local memory and was 5-7x slower). 5-14x the CPU path on a Titan Xp host for `-t 2`/`-t 3` (more with more SVs). MLX runs it on CPU. `model.use_lmer_index = False` forces the per-SV kernels.
- Forward pass, ISM, and GkmExplain use the packed uint32 path (XOR + popcount). Otherwise GkmExplain has one kernel per device (`_explain_cpu`, CUDA `fused_explain`) with a `perturbation` flag; it fuses normalization + coefficient multiplication + SV-dimension reduction, accumulating directly into `[B, 4, L]` result (no per-SV intermediate). Forward + RC SV windows concatenated for single kernel launch. CPU: `prange(B)` with L1-resident per-thread result arrays. GPU: float64 `atomicAdd` (CC >= 6.0). Apple GPU: custom Metal shaders via `mx.fast.metal_kernel` (forward/ISM only; GkmExplain falls back to CPU). ISM builds the packed windows covering each position (padded with `PAD_WINDOW`, which cancels in new − old) and works for every kernel type, by path: with the l-mer table, table weights over those windows (`ism._ism_table`); on CPU with the l-mer index, one fused pass per (position, sequence) adds the changed windows' hits to the reference row and reduces over SVs (`LmerIndex.mutants`); otherwise per-SV values in blocks of at most `_MAX_SCORE_BYTES`. Mutant norms come from `raw_diag_mutants` (for every kernel, on CPU, only pairs touching a changed window). With window weights (`-t 4/5`) the padded window sets aren't whole sequences, so `ism._slot_weights` gives each slot its window's weight ([L, l], 0 for padding): the table and `LmerIndex.mutants` scale by it, and `_ism_kernel` scores each window as a one-window row (weight `window_weights(1)`, divided out) and sums with the slot weights. `-t 4/5` mutant norms: `weighted._mutant_diag_weighted_numba`.
- Kernel interface: `kernel.base` is the kernel giving raw values (the base of an RBF kernel, else itself) and `kernel.finish(k_norm)` applies the RBF. Base kernels are `PackedGkmKernel`s: `pack(x)` gives [B, W] uint32 windows on x's device (MLX packs on CPU; N packs as A), `pack_both(x)` adds the RC windows derived from the packed forward ones, so N is T on the reverse strand as in lsgkm (packing the one-hot RC would give A on both strands and break the RC symmetry the l-mer table relies on), `raw_packed(bx, by, by_rc)` the raw kernel with the RC block in the same call (masked join, or brute force below 16 rows), `raw_diag_packed(bx, bx_rc)` the self-kernel, `window_weights(W)` the [W] uint8 window weights (None except `-t 4/5`; the packed arrays passed around are whole sequences, so weights follow window positions). Gram, SMO columns, scoring, ISM and mutation impact all use this, so `-t 3`/`-t 5` get packed caches, Gram panels and batched SMO columns.
- The model caches packed SV windows, their base self-kernels and the l-mer index per device (`GkmSVM._sv_cache`, cleared on device moves); `model.clear_caches()` also drops the l-mer table. `use_lmer_index`/`use_lmer_table` are checked on every call.
- CPU kernel scoring normalizes, applies the RBF and reduces over SVs in one numba pass. Through the l-mer index it is fused with the query (`LmerIndex.reduce`: each row's hits accumulate in a cache-resident length-S buffer, never a [B, S] block); otherwise `svm._normalized_dot` reduces the raw block. `to_deltasvm` on RBF models scores only canonical l-mers (index <= RC index) and copies weights to their reverse complements, as lsgkm's nrkmers.py pipeline does.
- Min-matches skip: for esttrunc l=11 k=7 d=3, `min_matches=8`. 99.88% of window pairs skipped.
- CPU self-kernels for `-t 0`–`-t 5` (`_self_diagonal_packed_numba`, used for SV and query norms; `-t 4/5` use `weighted._self_diagonal_weighted_numba`) visit each pair of the symmetric forward and RC blocks once, four rows per pass over the columns, with branch-free table lookups; rows whose RC block isn't the reverse complement of the forward one visit every RC pair.
- CPU inner loop: Numba `@njit(parallel=True, fastmath=True)`. NVIDIA GPU inner loop: CuPy RawKernel with shared-memory caching. Apple GPU inner loop: Metal kernel with per-thread accumulation and `popcount()`.
- MLX compatibility: `get_strides()` for arrays without `.strides`, `xp.ascontiguousarray()` for `.copy()`, `to_cpu(X)[indices]` for fancy indexing. DeltaSVM auto-chunks intermediates >256 MB.
- Scoring against the SVs uses the cached l-mer index (`kernels/lmer_index.py`) for kernels with at most `MAX_NEIGHBORS` (8,192) l-mers within dmax mismatches, or `DENSE_MAX_NEIGHBORS` (32,768) when the indexed windows average >= 4 per l-mer, i.e. large SV sets (e.g. `-t 1`-`-t 5` with d <= 3, l <= 12; with window weights each CSR entry also stores its window's uint8 weight, `LmerIndex.vals`). SMO columns (one query row each) keep the 8,192 cap, where large neighborhoods were 4-8x slower than the join. Masks are sorted so neighbor lookups stay local: CSR over 4^l l-mers listing the SVs containing each, built once; a query window visits its dmax-neighborhood, so a call costs nothing per SV. CPU/CUDA accumulate float64 (CUDA int32 per-level counting measured slower: the cost is the scattered atomics); Metal counts hits per distinct weight in int32 and combines on CPU (above `_METAL_MAX_LEVELS` distinct weights, or with window weights, MLX queries on CPU). `model.use_lmer_index = False` forces the masked join.
- `raw_packed` uses the masked-join kernel (`kernels/masked.py`) when both sides have >= 16 sequences: h(m) as a signed sum of C(l-m, j), each term an exact-match join on masked l-mer keys. `kernel.use_masked = False` forces brute force. `compute_gram` then works in column panels so each column index is built once. MLX builds the index and joins in Metal, writing int32 counts that the CPU combines in float64. `-t 4`/`-t 5` use it too: each column window's weight sits next to its id (CPU float64 accumulation; the CUDA join counts in shared uint64, exact; MLX joins them on CPU).
- `to_deltasvm` reads the l-mer table when it applies, else scores all 4^l l-mers through `_score_unbatched` (RBF always keeps the query normalization; non-RBF without `query_norm` multiplies weights back by sqrt(K(u,u))). It never mutates the model.
- Prediction for `-t 0`/`-t 1`/`-t 2`/`-t 4` (l <= 12) uses an exact 4^l l-mer weight table (`lmer_table.py`, built from marginals of the SV l-mer weights; with RC, a mask and its mirror share one marginal, read at rc(u)): f(x) = sum_i wt_i w(x_i) / sqrt(K(x,x)) + b (wt_i: window weights, 1 except `-t 4`). `model.use_lmer_table`: `"auto"` (>= 500 SVs), True, False. RBF uses the kernel path.
- `compute_gram` self-Gram: skips tiles below the diagonal; on CPU, diagonal tiles compute one triangle (`_fused_pairwise_packed_sym_numba`, also valid for the RC block). RBF tiles use the base kernel's packed path, then `kernel.finish`.
- Training: `solver="auto"` uses the precomputed Gram + sklearn when the N×N float64 Gram fits in 75% of available RAM and under `max_gram_gb` (the Gram always lives in host RAM; tiles are computed on the training device); else column-cached SMO. `train.py` dispatches every solver through `_fit`, which builds the model. SMO uses WSS3 working set selection (C extension `_csmo` with Python kernel column callback), shrinking, and kernel columns cached in C — memory is O(cache_size × N) not O(N²). No Python fallback: without the extension, `solver="smo"` raises. `solver="nystrom"` uses low-rank kernel approximation for fast approximate training (fewer kernel evaluations; still a dense N×N approximate Gram, so no memory saving). `max_gram_gb` caps Gram allocation on shared compute.
- `KernelColumnCache` packs all training sequences once (through `kernel.base`). When the l-mer index applies (`-t 1`-`-t 5` with a small d-neighborhood), it indexes all training sequences once and a column is one query row (~3-6 ms on CPU for 300 bp, N=5-10k), so SMO prefetches only into empty cache slots (the first fill), then computes single requested columns. Otherwise columns are all rows against the requested columns and their RCs, so the masked join indexes only the batch, and misses prefetch up to `_SMO_PREFETCH` columns into free cache slots. Without the index, a cache smaller than N degrades to single brute-force columns once full.
- SVR (`train_gkmsvr`) uses sklearn epsilon-SVR with precomputed Gram. SMO SVR is not yet implemented.
- Sequence lengths (`docs/design/variable-length.md`): the main pipeline is fixed windows. CLI `--genome` reads BED/narrowPeak summit windows (`fasta.read_loci`/`extract_loci`), `--length` center-crops mixed FASTAs (`codec.center_crop`; default the model's SV length, 300 for training; shorter records are an error). Variable length is compatibility: `pack(x, lengths)` writes `PAD_WINDOW` past each row's windows (RC rows hold theirs first), and every kernel skips PAD explicitly (CPU: spans/counts from `direct._spans`/`_counts`; CUDA/Metal compare each window with `0xFFFFFFFFu`; one side suffices for brute force, since PAD against a real window falls below min_matches). `-t 4/5` weights are per row (`row_weights`: `[1, W]` when nothing is padded; the CUDA weighted join, index query and `fused_explain` take them with a row stride, 0 when shared). `GkmSVM.sv_lengths` (importers, npz, `save_lsgkm` trims); `model(x, lengths=)` or a list (`svm.as_batch`); shorter than l gives NaN with a warning. Every device runs padded input (`test_variable_length.py::test_*_on_device`). Training (`train_gkmsvm`/`train_gkmsvr` on lists of different lengths; `compute_gram(lengths=, y_lengths=)`, `KernelColumnCache(lengths)`, Nyström) keeps `sv_lengths`; sequences shorter than l are an error (gkmtrain silently builds a broken model). ISM, GkmExplain (`lengths=` or a list), `mutation_impact`/`score_variants` (`ref_lengths`/`alt_lengths`, indels allowed) and `DeltaSVM` (scored per length) take mixed-length queries: `[B, 4, L_max]` out, 0 past each length, NaN rows shorter than l. ISM: `_packed_mutations` keeps PAD, slot weights are `[L, 1 or B, l]`, the mutant self-kernels use each row's span. Oracles: `tests/fixtures/lsgkm_ragged_*`: gkmpredict on mixed-length queries; gkmtrain's models on `lsgkm_ragged_{pos,neg}.fa` (same SVs, coefficients to the solvers' 1e-3 tolerance, and gkmtrain's solution meets our KKT conditions to 6e-4); gkmexplain modes 0/1/4/5 on `lsgkm_ragged_query.fa` (odd lengths 31-151) for every fixture model (`*_query.explain*`).
- `scikit-learn` provides the LIBSVM C solver for both SVC and SVR via `SVC(kernel='precomputed')` / `SVR(kernel='precomputed')`. No direct `libsvm-official` dependency.

## Gotchas

- N: lsgkm encodes any non-ACGT base as A before taking the reverse complement, so N is A on the forward strand and T on the reverse one (`pack_both`), for every kernel type. Imported models' coefficients were fit with that kernel, so N is not treated as its own symbol. String inputs (CLI FASTAs, `train_gkmsvm`/`train_gkmsvr` DNA strings) accept N (`one_hot_encode(allow_n=True)`); other non-ACGT letters raise unless `allow_other=True` (`--allow-other-bases`, `allow_other_bases=`), which reads them as N, matching gkmpredict (LS-GKM reads every non-ACGT letter as A). Importers always read them as N, with a warning: the model was fit that way.
- LS-GKM defaults are `-t 2 -l 11 -k 7 -d 3`. Always pass parameters explicitly when generating oracle fixtures.
- The `d` parameter zeros weight table entries for m > d, for every kernel type including `-t 0` (LS-GKM's search visits only pairs within d mismatches; at l=11 k=7 d=3 that drops `-t 0`'s m = 4 term). Without this, scores diverge ~1.5% from `gkmpredict`. `DirectGkmKernel(d=None)` (a `-t 0` model with no `d` in its params, e.g. one saved before d applied to `-t 0`) counts every pair; training stores `d` for every kernel type. Oracle for `-t 0`/`-t 1` scores and attributions: `tests/fixtures/lsgkm_wgkm_t{0,1}_*`.
- `-t 0` and `-t 2` use completely different weight table math. Do not mix.
- `score(ALT) - score(REF)` differs from deltaSVM's k-mer-weight linear approximation.
- Original gkmSVM (`.gkmmodel`) uses opposite sign convention for bias vs LS-GKM. Use `load_classic_model()` not `load_lsgkm_model()`.
- `-t 4`/`-t 5` (wgkm/wgkmrbf) are the `-t 2` table (with `d`) times integer window weights from the window's distance to the sequence center: `min(M, floor(M 2^(-|W/2 - i|/H) + 1))` (`weighted._window_weights`, LS-GKM's `gkmkernel_new_object`; M 1-255 and H default to LS-GKM's 50). Not per-position weights inside the l-mer (an earlier version did that and disagreed with gkmpredict). An RC window carries its forward window's weight, so with an even window count the kernel isn't exactly RC-invariant. Oracle: `tests/fixtures/lsgkm_wgkm_*` (`benchmarks/dev/lsgkm_wgkm_fixtures.py`).
- Packed kernels take l <= 15 (`packed.MAX_L`): a uint32 holds 16 bases, and at l = 16 the all-T l-mer equals `PAD_WINDOW`. LS-GKM allows l <= 12.
- LS-GKM C is GPL v3 — cannot wrap or link against it (gkmsvm-lite is MIT).
- MLX computes in float32 (Metal has no float64), so its scores differ from CPU at the float32 level; MLX Gram tiles are normalized on the host in float64.
- Beer lab `deltasvm.pl` drops the first weight-file line as a bias header; with gkmpredict output on `nrkmers.py` l-mers that gives AAAAAAAAAA/TTTTTTTTTT raw score 0 (bias included, so -bias in a `to_deltasvm` table). Published deltaSVM scores from that script (e.g. the GkmExplain paper) include this; `dsqtl_auprc.deltasvm_pl()` reproduces it.
- The gkmexplain repo's summed deltaSVM AUPRCs (README 0.18943 / 0.18795) cover negsets 1, 3, 5 only (`combine_folds.sh` cuts the wrong columns). Five-model sums from its output files are 0.19021 / 0.18997 (`PAPER_SUM_AUPRC`). Reproduction write-up: `docs/validation.md`.
