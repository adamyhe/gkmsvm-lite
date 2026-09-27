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
- [x] Memory-aware solver auto-selection — estimates Gram matrix size against available RAM/VRAM (50% budget)
- [x] Replaced scikit-learn with libsvm-official (114 KB vs ~100 MB)
- [x] Epsilon-SVR regression via `train_gkmsvr()` — continuous-valued prediction with configurable epsilon tube
- [x] 273 tests passing
- [x] MLX backend for Apple Silicon GPU inference (`model.mlx()`)
- [x] MLX shim for NumPy API compatibility (strides, copy, fancy indexing)
- [x] Custom Metal kernels via `mx.fast.metal_kernel` (fused XOR + popcount with per-thread early exit)
- [x] `mx.as_strided` sliding windows (replaces Python-loop concatenation)
- [x] MLX training support (libsvm + SMO solvers)
- [x] MLX ISM support
- [x] DeltaSVM auto-chunking for memory-bounded intermediates
- [x] Nanog replication on MLX (oracle match, RC invariance, ISM, training)
- [x] 316 tests passing (29 MLX + 287 others)

## Next

- [ ] **CLI** — command-line interface for training, scoring, and variant effect prediction
