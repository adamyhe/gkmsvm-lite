# Roadmap

## Completed

- [x] One-hot codec with RC, validation, batch padding
- [x] DirectGkmKernel (`-t 0` / `gkm_cnt` / `direct`) with combinatorial verification
- [x] EstTruncGkmKernel (`-t 2` / `gkm_esttrunc` / `estimated`) ported from LS-GKM C source
- [x] EstFullGkmKernel (`-t 1` / `gkm_estfull` / `estimated_full`) via `EstTruncGkmKernel(truncate=False)`
- [x] RbfGkmKernel (`-t 3` / `gkmrbf` / `rbf`)
- [x] CenterWeightedGkmKernel (`-t 4` / `wgkm` / `weighted`)
- [x] CenterWeightedRbfGkmKernel (`-t 5` / `wgkmrbf` / `weighted_rbf`)
- [x] GkmSVM with chunked SV evaluation
- [x] LS-GKM model importer (plain text and gzip)
- [x] Classic gkmSVM importer — C gkmSVM LIBSVM-style format (opposite bias sign convention)
- [x] R gkmSVM importer — `.gkmmodel` unified and `_svalpha.out`/`_svseq.fa` two-file formats
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
- [x] dsQTL benchmark replication (AP=0.19, r=0.73, GPU 337 seq/s, CPU 28 seq/s at 19bp)
- [x] Training via precomputed Gram matrix (libsvm-official C solver)
- [x] Tiled Gram matrix computation with symmetry exploitation
- [x] Model serialization: npz (native) and LS-GKM text (interop) formats
- [x] `GkmSVM.save()` with auto-format detection
- [x] `load_model()` unified loader (npz + LS-GKM auto-detect)
- [x] Column-cached SMO solver (`solver="smo"`) — LRU-cached kernel columns, no N×N Gram matrix, scales to ATAC/ChIP-scale (80K+ sequences)
- [x] `train_gkmsvm()` solver selection: `auto` / `smo` / `libsvm`
- [x] Memory-aware solver auto-selection — estimates Gram matrix size against available RAM/VRAM (75% budget)
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
- [x] 365 tests passing

## Next

- [ ] **SMO SVR** — column-cached SMO solver for epsilon-SVR (2N dual variables, epsilon-tube working set selection). Currently SVR uses precomputed Gram only; SMO SVR needed for large-scale regression where the Gram matrix exceeds available memory. Reference implementation: kundajelab/lsgkm-svr
- [ ] **Binary Platt scaling** — calibrated probability output for binary SVC models. Fit sigmoid (A, B) on held-out decision values; store parameters in GkmSVM and serialize to npz; add `predict_proba()` method. Post-hoc calibration avoids the 5-fold CV overhead of sklearn's `probability=True`. See mLS-GKM (Howard & Harmston, 2026) for reference.
- [ ] **Multiclass SVC** — one-vs-one classification with K(K-1)/2 binary sub-models. Requires architectural changes to GkmSVM (multiple SV sets, per-pair coefficients/bias). Wu-Lin-Weng coupling converts pairwise probabilities into K-class vector (requires Platt scaling). See mLS-GKM (Howard & Harmston, 2026) for reference.
