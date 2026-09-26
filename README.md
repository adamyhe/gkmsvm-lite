# gkmsvm-lite

A NumPy/CuPy implementation of gapped k-mer SVMs (gkm-SVMs) for DNA sequence classification. Loads existing LS-GKM models and scores sequences, validated against `gkmpredict` to floating-point precision.

## Install

```bash
pip install -e "."               # CPU (NumPy + Numba)
pip install -e ".[gpu]"          # GPU (+ CuPy)
pip install -e ".[dev]"          # + pytest
```

## Quick start

```python
from gkmsvm import load_lsgkm_model, ism, gkmexplain

# Load a trained LS-GKM model
model = load_lsgkm_model("model.txt.gz")

# Score sequences: [B, 4, L] one-hot → [B, 1] margin scores
scores = model(x)

# Move to GPU for ~6x speedup
model.cuda()
scores = model(x_gpu)

# Attribution
ism_scores = ism(model, x)                 # [B, 4, L] score deltas
attr = gkmexplain(model, x, mode=0)        # [B, 4, L] importance scores
```

## Variant scoring

Score REF/ALT pairs for variant effect prediction:

```python
from gkmsvm import load_lsgkm_model, one_hot_encode

model = load_lsgkm_model("model.txt.gz")

ref = one_hot_encode("ACGTACGTACG...")
alt = one_hot_encode("ACGTACGAACG...")  # single base change

ref_score, alt_score = model(ref), model(alt)
delta = alt_score - ref_score
```

For batch variant scoring with progress bars:

```python
scores = model.score_variants(ref_seqs, alt_seqs, verbose=True)
```

## Kernel modes

| LS-GKM flag | Internal name  | Alias            | Description                            |
|-------------|----------------|------------------|----------------------------------------|
| `-t 0`      | `gkm_cnt`      | `direct`         | Exact gapped k-mer count               |
| `-t 1`      | `gkm_estfull`  | `estimated_full` | Estimated l-mer kernel, full filter     |
| `-t 2`      | `gkm_esttrunc` | `estimated`      | Estimated l-mer kernel, truncated (default) |
| `-t 3`      | `gkmrbf`       | `rbf`            | RBF on estimated kernel                |
| `-t 4`      | `wgkm`         | `weighted`       | Center-weighted gapped k-mer           |
| `-t 5`      | `wgkmrbf`      | `weighted_rbf`   | Center-weighted RBF                    |

`GkmSVM` accepts any of: internal name, alias, or integer (`-t N` value).

```python
from gkmsvm import resolve_kernel_type
resolve_kernel_type("estimated")  # → "gkm_esttrunc"
resolve_kernel_type(2)            # → "gkm_esttrunc"
```

## Attribution

Gradient-based methods (DeepLIFT, SHAP, captum) are incompatible with gkm-SVMs — the kernel involves discrete k-mer counting with no meaningful gradient. Use:

- **GkmExplain** — analytical decomposition of the SVM decision function into per-base importance scores. 20-30x faster than ISM. Supports importance (mode 0) and hypothetical (mode 1) scores.
- **ISM** — exhaustive single-base mutagenesis with a window-delta optimization that recomputes only the ~l affected windows per mutation.

See [docs/attribution.md](docs/attribution.md) for details.

## Performance

Benchmarked on ENCODE ENCFF579AOX (72K SVs, esttrunc l=11 k=7 d=3):

| Sequence length | GPU (RTX 3080) | CPU (Numba) | Speedup |
|---|---|---|---|
| 19bp (dsQTL variants) | 342.9 seq/s | 52.8 seq/s | 6.5x |
| 100bp | 33.0 seq/s | 5.9 seq/s | 5.6x |
| 300bp (ENCODE peaks) | 10.4 seq/s | 1.9 seq/s | 5.5x |

Key optimizations:
- **Packed uint32 comparison**: l-mer windows packed to uint32, match counting via XOR + popcount instead of per-base loops
- **Min-matches skip**: 99.88% of window pairs contribute zero weight and are skipped
- **Shared-memory caching**: GPU loads query windows into shared memory once per thread block
- **Cached SV packing**: packed support vector windows are computed once and reused across calls

See [docs/performance.md](docs/performance.md) for full benchmark tables and tuning guidance.

## Memory management

For large models (>10K SVs), use chunked SV inference to bound memory:

```python
model = load_lsgkm_model("model.txt.gz", sv_chunk_size=5000)
```

GPU batch size has minimal impact on throughput. Use batch_size=16-64 to save VRAM.

## Architecture

- **Backend**: NumPy + Numba `@njit(parallel=True)` on CPU, CuPy RawKernel on GPU
- **Array format**: `[batch, 4, length]` one-hot DNA (A=0, C=1, G=2, T=3)
- **No PyTorch dependency** — gkm-SVMs are not differentiable; autograd provides no value
- **Score formula**: `score(x) = Σ coef_i × K(x, sv_i) + bias`

See [docs/design.md](docs/design.md) for design decisions.

## Documentation

| Document | Description |
|---|---|
| [docs/design.md](docs/design.md) | Design decisions and implementation rationale |
| [docs/performance.md](docs/performance.md) | Benchmarks, optimization details, tuning guidance |
| [docs/attribution.md](docs/attribution.md) | Attribution methods and why gradients don't work |
| [docs/roadmap.md](docs/roadmap.md) | Completed features and what's next |

## License

MIT. LS-GKM C (Dongwon-Lee/lsgkm) is GPL v3 — gkmsvm-lite is a clean-room reimplementation.
