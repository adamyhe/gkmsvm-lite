# CLAUDE.md

Agent-facing reference for working on this codebase. Human-readable docs are in `docs/` and `README.md`.

## Build and test

```bash
uv pip install -e ".[dev]"          # CPU only
uv pip install -e ".[dev,gpu]"      # with CuPy GPU support
uv pip install -e ".[dev,bench]"    # with benchmark dependencies
pytest tests/ -v
pytest tests/test_codec.py          # single file
pytest tests/ -k "test_rc"          # pattern match
```

## Source layout

```
src/gkmsvm/
├── __init__.py          # public API re-exports
├── svm.py               # GkmSVM model, scoring, chunked inference
├── train.py             # train_gkmsvm() (C-SVC) + train_gkmsvr() (epsilon-SVR)
├── solver.py            # KernelColumnCache, smo_solve() — column-cached SMO
├── gram.py              # compute_gram() — tiled Gram matrix with symmetry
├── serialization.py     # save/load npz and LS-GKM text formats
├── codec.py             # one-hot encode/decode, RC, validation
├── ism.py               # in-silico mutagenesis
├── explain.py           # GkmExplain attribution
├── deltasvm.py          # DeltaSVM linear scoring
├── fasta.py             # FASTA I/O, extract_loci (vendored tangermeme)
├── backend.py           # get_array_module (numpy/cupy dispatch)
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
    └── deltasvm.py      # load_deltasvm_weights()
```

## Conventions

- Python >=3.10, NumPy, Numba >=0.57, tqdm. CuPy >=12 optional (`[gpu]` extra)
- Array format: `[batch, 4, length]` one-hot DNA, channel order A=0/C=1/G=2/T=3
- Output: `[batch, 1]` floating-point margin scores
- Arrays are numpy.ndarray (CPU) or cupy.ndarray (GPU)
- `model.cuda()` / `model.cpu()` moves arrays between devices
- Kernel normalization on by default, RC equivalence on by default
- Score = `Σ coef_i × K(x, sv_i) + bias` where `bias = -rho` (LS-GKM) or `+rho` (classic gkmSVM)
- `resolve_kernel_type()` maps aliases and integers to canonical internal names
- Kernel modes: `-t 0` gkm_cnt/direct, `-t 1` gkm_estfull/estimated_full, `-t 2` gkm_esttrunc/estimated (default), `-t 3` gkmrbf/rbf, `-t 4` wgkm/weighted, `-t 5` wgkmrbf/weighted_rbf
- ISM: `ism(model, x)` → `[B, 4, L]` score deltas (window-delta optimization)
- GkmExplain: `gkmexplain(model, x, mode=0|1)` → `[B, 4, L]` attribution scores
- `verbose=True` on `model()`, `score_variants()`, `ism()`, `gkmexplain()` enables tqdm progress bars
- No PyTorch dependency. Gradient-based methods are incompatible — use GkmExplain or ISM
- tangermeme interop is vendored (pyfaidx for FASTA extraction)

## Key implementation details

- Forward pass and ISM use the packed uint32 path (XOR + popcount). GkmExplain uses packed pre-filter + bit extraction from packed uint32 (no float intermediates). CPU: fused Numba kernel. GPU: fused CuPy RawKernel with coalesced access and shared memory. Weighted kernels use the float one-hot path.
- Packed SV windows are cached on the model for both CPU and GPU. First call packs; subsequent calls reuse.
- Min-matches skip: for esttrunc l=11 k=7 d=3, `min_matches=8`. 99.88% of window pairs skipped.
- CPU inner loop: Numba `@njit(parallel=True, fastmath=True)`. GPU inner loop: CuPy RawKernel with shared-memory caching.
- SV diagonal is cached after first computation.
- Training: `solver="auto"` estimates Gram matrix size against available RAM/VRAM — uses precomputed Gram + libsvm-official when it fits (< 50% available memory), column-cached SMO otherwise. SMO uses LRU-cached kernel columns — memory is O(cache_size × N) not O(N²).
- `KernelColumnCache` pre-packs all training windows once, computes single columns via `pairwise_from_indices(bx[1,W,l], by_all)` on cache miss.
- SVR (`train_gkmsvr`) uses libsvm epsilon-SVR (`-s 3`) with precomputed Gram. SMO SVR is not yet implemented.
- `libsvm-official` (114 KB, BSD) provides the C solver for both SVC and SVR. No scikit-learn dependency.

## Gotchas

- LS-GKM defaults are `-t 2 -l 11 -k 7 -d 3`. Always pass parameters explicitly when generating oracle fixtures.
- The `d` parameter zeros weight table entries for m > d. Without this, scores diverge ~1.5% from `gkmpredict`.
- `-t 0` and `-t 2` use completely different weight table math. Do not mix.
- `score(ALT) - score(REF)` differs from deltaSVM's k-mer-weight linear approximation.
- Original gkmSVM (`.gkmmodel`) uses opposite sign convention for bias vs LS-GKM. Use `load_classic_model()` not `load_lsgkm_model()`.
- `-t 4`/`-t 5` (wgkm/wgkmrbf) require M and H parameters; these use per-position DP, not the matmul+table path.
- LS-GKM C is GPL v3 — cannot wrap or link against it (gkmsvm-lite is MIT).
- GPU float32 diagonal causes <3e-3 score difference vs CPU float64.
