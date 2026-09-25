# gkmsvm-lite

A NumPy/CuPy implementation of gapped k-mer SVMs (gkm-SVMs) for DNA sequence classification. Loads existing LS-GKM models and scores sequences, validated against `gkmpredict` to floating-point precision.

## Install

```bash
pip install -e ".[dev]"          # CPU (NumPy + Numba)
pip install -e ".[dev,gpu]"      # GPU (CuPy)
```

## Quick start

```python
from gkmsvm import load_lsgkm_model, ism, gkmexplain

model = load_lsgkm_model("model.txt.gz")
scores = model(x)                          # [B, 4, L] → [B, 1]

model.cuda()                               # move to GPU
scores = model(x_gpu)

ism_scores = ism(model, x)                 # [B, 4, L] score deltas
attr = gkmexplain(model, x, mode=0)        # [B, 4, L] importance scores
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

Gradient-based methods (DeepLIFT, SHAP, captum) are incompatible with gkm-SVMs. Use:

- **GkmExplain** — analytical decomposition, 20-30x faster than ISM
- **ISM** — exhaustive single-base mutagenesis with window-delta optimization

See `docs/attribution.md` for details.

## Architecture

- **Backend**: NumPy + Numba `@njit(parallel=True)` on CPU, CuPy RawKernel on GPU
- **Array format**: `[batch, 4, length]` one-hot DNA (A=0, C=1, G=2, T=3)
- **No PyTorch dependency** — gkm-SVMs are not differentiable
- **Score formula**: `score(x) = sum(coef_i * K(x, sv_i)) + bias`

See `docs/design.md` for design decisions and `docs/performance.md` for benchmarks.

## License

MIT. LS-GKM C (Dongwon-Lee/lsgkm) is GPL v3 — gkmsvm-lite is a clean-room reimplementation.
