# Roadmap

## Completed

- [x] One-hot codec with RC, validation, batch padding
- [x] DirectGkmKernel (`-t 0` / `gkm_cnt` / `direct`) with combinatorial verification
- [x] EstTruncGkmKernel (`-t 2` / `gkm_esttrunc` / `estimated`) ported from LS-GKM C source
- [x] EstFullGkmKernel (`-t 1` / `gkm_estfull` / `estimated_full`) via `EstTruncGkmKernel(truncate=False)`
- [x] RbfGkmKernel (`-t 3` / `gkmrbf` / `rbf`)
- [x] CenterWeightedGkmKernel (`-t 4` / `wgkm` / `weighted`)
- [x] CenterWeightedRbfGkmKernel (`-t 5` / `wgkmrbf` / `weighted_rbf`)
- [x] GkmSVM scoring with memory-capped [rows, S] blocks
- [x] LS-GKM model importer (plain text and gzip)
- [x] Classic gkmSVM importer — C gkmSVM LIBSVM-style format (opposite bias sign convention)
- [x] Classic (C++) and R gkmSVM importer — `gkmsvm_train`'s `_svalpha.out`/`_svseq.fa` (no bias; the kernel as arguments; non-ACGT dropped as the tools read FASTA), checked against each tool's `gkmsvm_kernel` (`tests/test_importers/test_gkmsvm_tools.py`); also LIBSVM-style and `.gkmmodel` files, which no released tool writes
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
- [x] Training oracle: `train_gkmsvm` (LIBSVM and SMO) gives gkmtrain's support vectors for `-t 0` to `-t 5` on fixed- and mixed-length sets with N, coefficients to the solvers' 1e-3 tolerance and gkmpredict's scores to 5e-4; gkmtrain's solutions meet our KKT conditions (`tests/test_train_oracle.py`)
- [x] Fixed-length pipeline (`--genome` summit windows, `--length` center-crop, 300 bp for training) and sequences of different lengths for compatibility: padded SVs and queries on CPU, CUDA and Metal; imported mixed-length models, npz/LS-GKM round trips; training, ISM, GkmExplain, mutation impact, variant scores and DeltaSVM on mixed lengths (gkmpredict, gkmtrain and gkmexplain oracles)

## Next

Gaps found by checking against LS-GKM (2026-10-19), most important first:

- [ ] **Re-run the dsQTL check** — `tests/test_gpu.py` passes on a CUDA host (2026-10-19, after the `-t 0` `d`, N and padding changes); `benchmarks/agreement_lsgkm.py` still to re-run (last run 2026-10-16, before the N packing change: mutation impact matched the paper's `gkmexplain -m 5` outputs to ~5e-6). CUDA explain paths are checked against CPU only, not against lsgkm directly.
- [ ] **`-t 6` gkm_cnt_rbf** — in the kundajelab fork only; not supported.

Performance gaps:

- [ ] `-t 4`/`-t 5` self-kernels and brute force (< 16 rows) run on CPU even on CUDA; MLX joins and queries weighted windows on CPU.
- [ ] GkmExplain on MLX runs on CPU and returns NumPy (Metal has no float64). Plan: per-window partials on the GPU folded in float64 on CPU, float-float within a window if float32 isn't enough; measure float32 error first (`docs/design/mlx-gkmexplain.md`). Not needed for the application note.
- [ ] No explain table for RBF (`-t 3`/`-t 5`): their SV weights depend on the query, so they take the index path.

Features:

- [ ] **SMO SVR** — column-cached SMO solver for epsilon-SVR (2N dual variables, epsilon-tube working set selection). Currently SVR uses precomputed Gram only; SMO SVR needed for large-scale regression where the Gram matrix exceeds available memory. Reference implementation: kundajelab/lsgkm-svr
- [ ] **Binary Platt scaling** — calibrated probability output for binary SVC models. Fit sigmoid (A, B) on held-out decision values; store parameters in GkmSVM and serialize to npz; add `predict_proba()` method. Post-hoc calibration avoids the 5-fold CV overhead of sklearn's `probability=True`. See mLS-GKM (Howard & Harmston, 2026) for reference.
- [ ] **Multiclass SVC** — one-vs-one classification with K(K-1)/2 binary sub-models. Requires architectural changes to GkmSVM (multiple SV sets, per-pair coefficients/bias). Wu-Lin-Weng coupling converts pairwise probabilities into K-class vector (requires Platt scaling). See mLS-GKM (Howard & Harmston, 2026) for reference.
