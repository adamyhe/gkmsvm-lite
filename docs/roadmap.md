# Roadmap

## Completed

- [x] One-hot codec with RC, validation, batch padding
- [x] DirectGkmKernel (`-t 0` / `gkm_cnt` / `direct`) with combinatorial verification
- [x] EstTruncGkmKernel (`-t 2` / `gkm_esttrunc` / `estimated`) ported from LS-GKM C source
- [x] EstFullGkmKernel (`-t 1` / `gkm_estfull` / `estimated_full`) via `EstTruncGkmKernel(truncate=False)`
- [x] RbfGkmKernel (`-t 3` / `gkmrbf` / `rbf`)
- [x] CenterWeightedGkmKernel (`-t 4` / `wgkm` / `weighted`)
- [x] CenterWeightedRbfGkmKernel (`-t 5` / `wgkmrbf` / `weighted_rbf`)
- [x] `-t 6` / `gkm_cnt_rbf` / `direct_rbf` (RBF on `-t 0`, the kundajelab fork's): its gkmtrain, gkmpredict and gkmexplain (modes 0/1/4/5) as oracles
- [x] GkmSVM scoring with memory-capped [rows, S] blocks
- [x] LS-GKM model importer (plain text and gzip)
- [x] Classic (C++) and R gkmSVM importer — `gkmsvm_train`'s `_svalpha.out`/`_svseq.fa` (no bias; the kernel as arguments; non-ACGT dropped as the tools read FASTA), checked against each tool's `gkmsvm_kernel` (`tests/test_importers/test_gkmsvm_tools.py`)
- [x] DeltaSVM model + importer (linear gapped k-mer scoring)
- [x] FASTA I/O and pyfaidx locus extraction
- [x] Oracle validation against Dongwon-Lee/lsgkm `gkmpredict`
- [x] NumPy/Numba CPU reference implementation with cross-validation tests
- [x] Numba threading layer pin (workqueue, avoids OpenMP conflicts)
- [x] NumPy/CuPy backend — dropped PyTorch; NumPy+Numba CPU, CuPy GPU
- [x] Fused Numba/CuPy pairwise kernels (match-count + table-lookup + sum)
- [x] Int8 base-index path (4x fewer ops, 16x less memory vs float32 one-hot)
- [x] CUDA coalesced memory access (SV layout transposition)
- [x] ISM with window-delta optimization
- [x] ISM int8 index path
- [x] GkmExplain attribution (mode 0 + hypothetical mode 1)
- [x] GkmExplain default 2000-SV chunking (prevents GPU OOM)
- [x] Pre-cached SV diagonal and int8 index windows
- [x] Descriptive kernel aliases (`direct`, `estimated`, `rbf`, `weighted`, etc.)
- [x] `resolve_kernel_type()` for alias/integer resolution
- [x] Packed uint32 comparison (XOR + popcount, ~3x CPU / ~9x GPU over int8 loop)
- [x] Min-matches skip (mismatch table sparsity, 99.88% skip for esttrunc default)
- [x] Shared-memory query caching in CUDA kernel
- [x] Packed SV window caching (CPU + GPU, eliminates redundant packing)
- [x] tqdm progress bars for inference, ISM, GkmExplain
- [x] dsQTL benchmark: GkmExplain paper (Fig. 8) reproduced per model with the paper's models — ISM, GkmExplain mutation impact, deltaSVM ([validation](validation.md))
- [x] GkmExplain mutation impact score (lsgkm `gkmexplain -m 5`)
- [x] Training via precomputed Gram matrix (scikit-learn's LIBSVM)
- [x] Tiled Gram matrix computation with symmetry exploitation
- [x] Model serialization: npz (native) and LS-GKM text (interop) formats
- [x] `GkmSVM.save()` with auto-format detection
- [x] `load_model()` unified loader (npz + LS-GKM auto-detect)
- [x] Column-cached SMO solver (`solver="smo"`) — LRU-cached kernel columns, no N×N Gram matrix
- [x] `train_gkmsvm()` solver selection: `auto` / `libsvm` / `smo` / `nystrom`
- [x] Memory-aware solver auto-selection — Gram size against 75% of available RAM
- [x] Epsilon-SVR regression via `train_gkmsvr()` — continuous-valued prediction with configurable epsilon tube
- [x] MLX backend for Apple Silicon GPU inference (`model.mlx()`)
- [x] MLX shim for NumPy API compatibility (strides, copy, fancy indexing)
- [x] Custom Metal kernels via `mx.fast.metal_kernel` (fused XOR + popcount with per-thread early exit)
- [x] `mx.as_strided` sliding windows (replaces Python-loop concatenation)
- [x] MLX training support (libsvm + SMO solvers)
- [x] MLX ISM support
- [x] DeltaSVM auto-chunking for memory-bounded intermediates
- [x] Nanog replication on MLX (oracle match, RC invariance, ISM, training)
- [x] Nyström approximate solver for large-scale training (`solver="nystrom"`)
- [x] C WSS3 SMO solver with GPU kernel column callback
- [x] CLI — `gkmsvm` command-line tool with subcommands for predict, train, explain, ISM, deltaSVM, variant scoring, and model import
- [x] GkmExplain fused reduction — coefficient multiplication + SV-dimension reduction in inner kernel loop, eliminates `[B, 4, L, S_c]` intermediate
- [x] GkmExplain SV window caching — SV chunks outer / sequence batches inner loop order
- [x] GkmExplain forward + RC fusion — single kernel launch via concatenated SV windows
- [x] GkmExplain mode 0 = mode 1 × OHE — unified to single mode 1 kernel
- [x] GkmExplain CPU `prange(B)` — L1-resident per-thread result arrays
- [x] Masked-join kernel for Gram tiles and SMO columns, all six kernel types, CPU/CUDA/Metal ([design](design/mask-join-kernel.md))
- [x] Exact l-mer weight table for `-t 0/1/2/4` prediction and `to_deltasvm`
- [x] Packed kernel interface (`kernel.base`, `kernel.finish`, `raw_packed`) shared by Gram, SMO, scoring, ISM and mutation impact
- [x] Cached l-mer index for scoring, ISM and mutation impact against SVs, and for SMO columns over the training set (`-t 1`-`-t 5`)
- [x] ISM and mutation impact for every kernel type
- [x] `-t 4`/`-t 5` as LS-GKM's window-weighted kernel (integer weights by distance from the sequence center), every path on CPU/CUDA/MLX; gkmpredict oracle
- [x] Window-delta ISM for `-t 4`/`-t 5` (slot weights for the windows covering each position)
- [x] GkmExplain for every kernel type (`-t 4`/`-t 5` included): explain table, l-mer path (CPU and CUDA), per-SV kernels
- [x] lsgkm oracles for every kernel type (`-t 0` to `-t 5`): gkmpredict scores; gkmexplain modes 0/1/4 on every CPU explain path and mode 5 (mutation impact) on every scoring path, with N and odd/even window counts (kundajelab fork, `benchmarks/dev/lsgkm_wgkm_fixtures.py`)
- [x] N as in LS-GKM (A on the forward strand, T on the reverse); N accepted in every string input; `--allow-other-bases` / `allow_other` reads other letters as N (as gkmpredict does)
- [x] `-t 0` truncated at `d` mismatches, as LS-GKM
- [x] `-w` positive-class weight (`pos_weight`; LIBSVM and SMO match gkmtrain `-w 2` and `-w 0.5`)
- [x] SVR oracle: `train_gkmsvr` gives kundajelab/lsgkm-svr's epsilon-SVR models (`-t 2/3/4`: same SVs, coefficients to 1e-3, predictions to 5e-4), and its model files reproduce its gkmpredict
- [x] `-t 4/5` on CUDA and Metal: weighted brute force and self-kernels in both; on MLX the masked join and l-mer index count weighted hits in Metal, exact in 64 bits (uint32 word + carry). MLX 1500 x 1500 300 bp Gram 1.66 s -> 0.65 s (`-t 2` 0.58 s)
- [x] ISM mutant self-kernels on CUDA and Metal (every kernel type): only the pairs touching a changed window, as on CPU, replacing a W^2 pass per mutant (MLX) and a CPU round trip (`-t 4/5`). MLX ISM 32 x 300 bp against 2,000 SVs: `-t 2` 323 -> 41 ms, `-t 4` 157 -> 42 ms
- [x] ISM through the l-mer index on CUDA and Metal (RBF, and non-RBF without the l-mer table): one thread block per (position, sequence) scatters the changed windows' hits and reduces over the SVs, no [mutants, S] block. MLX `-t 3` 2,000 SVs 32 x 300 bp 3.4 s -> 1.0 s (CPU 2.2 s); dense indexes (> 6 entries per l-mer) run the CPU pass on MLX
- [x] GkmExplain's per-SV kernel in Metal: per-window float32 partials folded in float64 on CPU, a ballot over the lanes holding a pair; 2.3-5.7x the CPU on an M4, no CPU fallback left for GkmExplain on MLX
- [x] Mutation impact through the l-mer table (`-t 0/1/2/4`): alt minus ref table sums over the ref norm, as prediction. dsQTL `-t 2` model, 500 variants: 7.0 s -> 0.01 s on CPU, 13.8 s -> 0.03 s on MLX
- [x] MLX queries a dense l-mer index (> 2.5 entries per l-mer) on CPU: dsQTL `-t 3` model, 500 x 300 bp: prediction 11.7 s -> 5.3 s (CPU 4.9 s), mutation impact 23.0 s -> 10.1 s (CPU 9.8 s)
- [x] GkmExplain on MLX: the l-mer path in Metal, per-window float32 partials folded in float64 on CPU (`docs/design/mlx-gkmexplain.md`); within 6e-7 of the CPU's largest attribution, 1.2-5x the CPU on an M4
- [x] SMO SVR: `train_gkmsvr(solver="smo")` (and `"auto"` when the Gram doesn't fit), the C SMO over LIBSVM's 2N epsilon-SVR dual variables with N cached columns; lsgkm-svr's models to 2e-7, LIBSVM's on mixed lengths, KKT conditions to the tolerance, on CPU, CUDA and MLX
- [x] Training oracle: `train_gkmsvm` (LIBSVM and SMO) gives gkmtrain's support vectors for `-t 0` to `-t 5` on fixed- and mixed-length sets with N, coefficients to the solvers' 1e-3 tolerance and gkmpredict's scores to 5e-4; gkmtrain's solutions meet our KKT conditions (`tests/test_train_oracle.py`)
- [x] Fixed-length pipeline (`--genome` summit windows, `--length` center-crop, 300 bp for training) and sequences of different lengths for compatibility: padded SVs and queries on CPU, CUDA and Metal; imported mixed-length models, npz/LS-GKM round trips; training, ISM, GkmExplain, mutation impact, variant scores and DeltaSVM on mixed lengths (gkmpredict, gkmtrain and gkmexplain oracles)

## Next

Gaps found by checking against LS-GKM (2026-10-19), most important first:

- [ ] **Full dsQTL re-run on the CUDA host** — `benchmarks/agreement_lsgkm.py` with all five negsets and both kernels (`make` without `AGREE_ARGS`) for the figure; the default check (`-t 3`, negset 1) was re-run on CPU on 2026-10-19 after the N, padding and importer changes and matches the earlier run: mutation impact to 5.0e-6, ISM to 1.0e-5, deltaSVM (as `deltasvm.pl` scores it) to 3.4e-5 of the paper's outputs. CUDA explain paths are checked against CPU only, not against lsgkm directly.

Performance gaps:

- None open. RBF (`-t 3`/`-t 5`) has no explain table by construction: the table sums the SVs into one 4^l vector V before any query, and an RBF SV's weight depends on the query through exp(γ(K(x, sv) − 1)), so V would be per query and cost as much as the index path's query.

Features:

- [ ] **Binary Platt scaling** — calibrated probability output for binary SVC models. Fit sigmoid (A, B) on held-out decision values; store parameters in GkmSVM and serialize to npz; add `predict_proba()` method. Post-hoc calibration avoids the 5-fold CV overhead of sklearn's `probability=True`. See mLS-GKM (Howard & Harmston, 2026) for reference.
- [ ] **Multiclass SVC** — one-vs-one classification with K(K-1)/2 binary sub-models. Requires architectural changes to GkmSVM (multiple SV sets, per-pair coefficients/bias). Wu-Lin-Weng coupling converts pairwise probabilities into K-class vector (requires Platt scaling). See mLS-GKM (Howard & Harmston, 2026) for reference.
