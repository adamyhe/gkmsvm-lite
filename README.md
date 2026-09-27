# gkmsvm-lite

[![PyPI](https://img.shields.io/pypi/v/gkmsvm-lite)](https://pypi.org/project/gkmsvm-lite/)
[![CI](https://github.com/adamyhe/gkmsvm-lite/actions/workflows/ci.yml/badge.svg)](https://github.com/adamyhe/gkmsvm-lite/actions/workflows/ci.yml)
[![Downloads](https://static.pepy.tech/badge/gkmsvm-lite)](https://pepy.tech/projects/gkmsvm-lite)

A pure-Python implementation of gapped k-mer SVMs (gkm-SVMs) for DNA sequence analysis. Supports classification (C-SVC) and regression (epsilon-SVR), all six kernel types, GPU acceleration via CuPy (NVIDIA) and MLX (Apple Silicon), and compatibility with models from LS-GKM, classic gkmSVM, and the R gkmSVM package.

## Install

```bash
pip install gkmsvm-lite              # CPU (NumPy + Numba)
pip install gkmsvm-lite[gpu]         # NVIDIA GPU (+ CuPy)
pip install gkmsvm-lite[mlx]         # Apple Silicon GPU (+ MLX)
```

For development installation from source, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Quick start

```python
from gkmsvm import load_lsgkm_model, one_hot_encode, ism, gkmexplain

# Load an existing model
model = load_lsgkm_model("model.txt.gz")

# Score a sequence
x = one_hot_encode("ACGTACGTACGTACGT")[None, ...]  # [1, 4, 16]
score = model(x)  # [1, 1]

# Variant effect prediction
ref = one_hot_encode("ACGTACGTACGTACGT")[None, ...]
alt = one_hot_encode("ACGTACGAACGTACGT")[None, ...]
delta = model(alt).item() - model(ref).item()

# Attribution
attr = gkmexplain(model, x, mode=0)  # [1, 4, 16] importance scores
deltas = ism(model, x)                # [1, 4, 16] mutation deltas

# GPU acceleration
model.cuda()  # NVIDIA (~6x speedup)
model.mlx()   # Apple Silicon (~2-6x speedup)
```

## Train a model

```python
from gkmsvm import train_gkmsvm, train_gkmsvr

# Classification
model = train_gkmsvm(pos_seqs, neg_seqs, l=11, k=7, d=3, C=1.0)

# Regression (e.g., chromatin accessibility)
model = train_gkmsvr(sequences, labels, l=11, k=7, d=3, epsilon=0.1)

model.save("my_model.npz")
```

## Load models from any gkm-SVM package

```python
from gkmsvm import (
    load_lsgkm_model,      # LS-GKM / kundajelab/lsgkm
    load_classic_model,     # C gkmSVM (LIBSVM-style format)
    load_r_gkmsvm_model,   # R gkmSVM (Ghandi et al. 2014)
    load_deltasvm_weights,  # DeltaSVM k-mer weights
    load_model,             # auto-detect npz or LS-GKM text
)
```

See [docs/models.md](docs/models.md) for format details, training options, and examples.

## Kernel types

All six LS-GKM kernel types are supported:

| `-t` | Name | Description |
|------|------|-------------|
| 0 | gkm | Direct gapped k-mer count |
| 1 | gkm_estfull | Estimated full l-mer |
| 2 | gkm_esttrunc | Estimated truncated l-mer (LS-GKM default) |
| 3 | gkmrbf | RBF-transformed estimated |
| 4 | wgkm | Center-weighted gapped k-mer |
| 5 | wgkmrbf | Center-weighted RBF |

## Documentation

| Guide | Contents |
|---|---|
| [Loading and training models](docs/models.md) | Importing from all formats, training SVC/SVR, solver selection, kernel types |
| [Scoring and variant effects](docs/scoring.md) | Inference, batch scoring, VEP, GPU acceleration, memory management |
| [Interpretation](docs/interpretation.md) | GkmExplain, ISM, when to use each, why gradients don't work |
| [Performance](docs/performance.md) | Benchmarks, optimization details, tuning guidance |
| [Design decisions](docs/design.md) | Architecture, kernel math, format specifications |
| [Roadmap](docs/roadmap.md) | Completed features and what's next |

## Citation

If you use gkmsvm-lite, please cite the relevant methods papers:

> Ghandi M, Lee D, Mohammad-Noori M, Beer MA. Enhanced regulatory sequence prediction using gapped k-mer features. *PLoS Comput Biol* 10(7):e1003711 (2014).

> Lee D. LS-GKM: a new gkm-SVM for large-scale datasets. *Bioinformatics* 32(14):2196-2198 (2016).

If you use GkmExplain:

> Shrikumar A, Prakash E, Kundaje A. GkmExplain: fast and accurate interpretation of nonlinear gapped k-mer SVMs. *Bioinformatics* 35(14):i173-i182 (2019).

If you use DeltaSVM:

> Lee D, Gorkin DU, Baker M, Strober BJ, Asoni AL, McCallion AS, Beer MA. A method to predict the impact of regulatory variants from DNA sequence. *Nat Genet* 47(8):955-961 (2015).

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

## License

MIT. LS-GKM C (Dongwon-Lee/lsgkm) is GPL v3 — gkmsvm-lite is a clean-room reimplementation.
