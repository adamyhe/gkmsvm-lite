# CLAUDE.md

Agent-facing reference for working on this codebase. Human-readable docs are in `docs/` and `README.md`.

## Build and test

```bash
uv pip install -e ".[dev]"          # CPU only
uv pip install -e ".[dev,gpu]"      # with CuPy GPU support
pytest tests/ -v
pytest tests/test_codec.py          # single file
pytest tests/ -k "test_rc"          # pattern match
```

## Source layout

```
src/gkmsvm/
├── __init__.py          # public API re-exports
├── svm.py               # GkmSVM model, scoring, chunked inference
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
│   ├── base.py          # GkmKernel ABC
│   ├── direct.py        # -t 0 gkm_cnt + packed uint32 fused kernels
│   ├── esttrunc.py      # -t 1 gkm_estfull, -t 2 gkm_esttrunc
│   ├── rbf.py           # -t 3 gkmrbf
│   └── weighted.py      # -t 4 wgkm, -t 5 wgkmrbf
└── importers/
    ├── lsgkm.py         # load_lsgkm_model()
    ├── classic.py       # load_classic_model() — C gkmSVM (LIBSVM-style)
    ├── r_gkmsvm.py      # load_r_gkmsvm_model() — R gkmSVM (.gkmmodel, _svalpha.out)
    └── deltasvm.py      # load_deltasvm_model()
```

## Conventions

- Python >=3.10, NumPy, Numba >=0.57, scikit-learn, tqdm. CuPy >=12 optional (`[gpu]` extra). MLX >=0.10 optional (`[mlx]` extra)
- Array format: `[batch, 4, length]` one-hot DNA, channel order A=0/C=1/G=2/T=3
- Output: `[batch, 1]` floating-point margin scores
- Arrays are numpy.ndarray (CPU), cupy.ndarray (NVIDIA GPU), or mlx.core.array (Apple GPU)
- `model.cuda()` / `model.mlx()` / `model.cpu()` moves arrays between devices
- Kernel normalization on by default, RC equivalence on by default
- Score = `Σ coef_i × K(x, sv_i) + bias` where `bias = -rho` (LS-GKM) or `+rho` (classic gkmSVM)
- `resolve_kernel_type()` maps aliases and integers to canonical internal names
- Kernel modes: `-t 0` gkm_cnt/direct, `-t 1` gkm_estfull/estimated_full, `-t 2` gkm_esttrunc/estimated (default), `-t 3` gkmrbf/rbf, `-t 4` wgkm/weighted, `-t 5` wgkmrbf/weighted_rbf
- ISM: `ism(model, x)` → `[B, 4, L]` score deltas (window-delta optimization)
- GkmExplain: `gkmexplain(model, x, mode=0|1)` → `[B, 4, L]` attribution scores. Mode 0 = mode 1 × OHE (single unified kernel). MLX unsupported (float32 violates completion axiom)
- `verbose=True` on `model()`, `score_variants()`, `ism()`, `gkmexplain()` enables tqdm progress bars
- No PyTorch dependency. Gradient-based methods are incompatible — use GkmExplain or ISM
- tangermeme interop is vendored (pyfaidx for FASTA extraction)

## Key implementation details

- Forward pass, ISM, and GkmExplain use the packed uint32 path (XOR + popcount). GkmExplain's inner kernel fuses normalization + coefficient multiplication + SV-dimension reduction, accumulating directly into `[B, 4, L]` result (no per-SV intermediate). Forward + RC SV windows concatenated for single kernel launch. CPU: `prange(B)` with L1-resident per-thread result arrays. GPU: float64 `atomicAdd` (CC >= 6.0). Apple GPU: custom Metal shaders via `mx.fast.metal_kernel` (forward/ISM only; GkmExplain falls back to CPU). Weighted kernels use the float one-hot path.
- Packed SV windows are cached on the model for CPU, NVIDIA GPU, and MLX. First call packs; subsequent calls reuse.
- Min-matches skip: for esttrunc l=11 k=7 d=3, `min_matches=8`. 99.88% of window pairs skipped.
- CPU inner loop: Numba `@njit(parallel=True, fastmath=True)`. NVIDIA GPU inner loop: CuPy RawKernel with shared-memory caching. Apple GPU inner loop: Metal kernel with per-thread accumulation and `popcount()`.
- MLX compatibility: `get_strides()` for arrays without `.strides`, `xp.ascontiguousarray()` for `.copy()`, `to_cpu(X)[indices]` for fancy indexing. DeltaSVM auto-chunks intermediates >256 MB.
- SV diagonal is cached after first computation.
- Training: `solver="auto"` estimates Gram matrix size — uses precomputed Gram + sklearn when it fits in device memory (< 75%); on GPU, falls back to GPU-computed Gram + CPU-side sklearn when it fits in system RAM (< 75%); else column-cached SMO. SMO uses WSS3 working set selection (C extension `_csmo` with Python kernel column callback), shrinking, and LRU-cached kernel columns — memory is O(cache_size × N) not O(N²). `solver="nystrom"` uses low-rank kernel approximation for fast approximate training. `max_gram_gb` caps Gram allocation on shared compute.
- `KernelColumnCache` pre-packs all training windows once, computes single columns via `pairwise_from_indices(bx[1,W,l], by_all)` on cache miss.
- SVR (`train_gkmsvr`) uses sklearn epsilon-SVR with precomputed Gram. SMO SVR is not yet implemented.
- `scikit-learn` provides the LIBSVM C solver for both SVC and SVR via `SVC(kernel='precomputed')` / `SVR(kernel='precomputed')`. No direct `libsvm-official` dependency.

## Gotchas

- LS-GKM defaults are `-t 2 -l 11 -k 7 -d 3`. Always pass parameters explicitly when generating oracle fixtures.
- The `d` parameter zeros weight table entries for m > d. Without this, scores diverge ~1.5% from `gkmpredict`.
- `-t 0` and `-t 2` use completely different weight table math. Do not mix.
- `score(ALT) - score(REF)` differs from deltaSVM's k-mer-weight linear approximation.
- Original gkmSVM (`.gkmmodel`) uses opposite sign convention for bias vs LS-GKM. Use `load_classic_model()` not `load_lsgkm_model()`.
- `-t 4`/`-t 5` (wgkm/wgkmrbf) require M and H parameters; these use per-position DP, not the matmul+table path.
- LS-GKM C is GPL v3 — cannot wrap or link against it (gkmsvm-lite is MIT).
- GPU float32 diagonal causes <3e-3 score difference vs CPU float64.
