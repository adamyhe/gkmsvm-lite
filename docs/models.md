# Loading and training models

## Loading existing models

gkmsvm-lite can load models from all major gkm-SVM implementations. Every loader returns a `GkmSVM` instance with the same scoring interface.

### LS-GKM (Dongwon-Lee/lsgkm, kundajelab/lsgkm)

The most common format. LS-GKM models are plain text (or gzipped) files with key-value headers followed by an `SV` marker and support vector lines.

```python
from gkmsvm import load_lsgkm_model

# Plain text or gzip — auto-detected
model = load_lsgkm_model("model.txt")
model = load_lsgkm_model("model.txt.gz")

# Control memory usage for large models
model = load_lsgkm_model("model.txt.gz", sv_chunk_size=5000)
```

Models trained with kundajelab/lsgkm (which adds GkmExplain) and kundajelab/lsgkm-svr (which adds SVR) use the same format and load with this function.

Bias convention: `bias = -rho` (LIBSVM convention).

### Classic gkmSVM (C implementation)

The C gkmSVM implementation uses LIBSVM-style headers with an `SV` marker. Supports both embedded SVs (single file) and two-file format (model + FASTA).

```python
from gkmsvm import load_classic_model

# Single file with embedded SVs
model = load_classic_model("model.gkmmodel")

# Two-file format: model + separate FASTA
model = load_classic_model("model.txt", svseq_path="model.svseq.fa")
```

Bias convention: `bias = +rho` (opposite from LS-GKM). This is handled automatically — you don't need to adjust scores.

### R gkmSVM (R/kernlab package)

The original R package (Ghandi et al. 2014) uses `#`-prefixed headers and FASTA-style support vectors.

```python
from gkmsvm import load_r_gkmsvm_model

# Unified .gkmmodel format
model = load_r_gkmsvm_model("model.gkmmodel")

# Legacy two-file format (auto-detects _svseq.fa companion)
model = load_r_gkmsvm_model("mymodel_svalpha.out")

# Or specify the FASTA file explicitly
model = load_r_gkmsvm_model("alphas.out", svseq_path="sequences.fa")
```

Bias convention: `bias = +rho` (same as C gkmSVM).

### DeltaSVM weights

DeltaSVM models are k-mer weight files, not full SVMs. They provide fast linear approximations for variant effect scoring.

```python
from gkmsvm import load_deltasvm_weights
from gkmsvm import DeltaSVM

# Load from k-mer weight file
model = load_deltasvm_weights("weights.txt")

# Score sequences
scores = model(x)  # [B, 1]

# Score variants
delta = model.score_variants(ref_seqs, alt_seqs)
```

DeltaSVM scoring is much faster than full SVM scoring since it uses a linear k-mer lookup instead of kernel evaluation against all support vectors. However, it is an approximation — `score(ALT) - score(REF)` from a full SVM differs from deltaSVM's linear estimate.

### Native format (npz)

Models trained or saved with gkmsvm-lite use a NumPy `.npz` format. The unified `load_model()` function auto-detects npz vs LS-GKM text format.

```python
from gkmsvm import load_model

# Auto-detects format
model = load_model("model.npz")        # native npz
model = load_model("model.txt.gz")     # LS-GKM text
```

### Saving models

Any `GkmSVM` model can be saved in either format:

```python
# Native npz (recommended — preserves full precision)
model.save("model.npz")

# LS-GKM text (interoperable with gkmpredict)
model.save("model.txt")
model.save("model.txt.gz")  # gzipped

# Explicit format override
model.save("model.bin", format="npz")
```

## Training new models

### Classification (C-SVC)

Train a binary classifier on positive and negative DNA sequences:

```python
from gkmsvm import train_gkmsvm

model = train_gkmsvm(
    pos_seqs=["ACGTACGT...", ...],   # positive sequences (strings or arrays)
    neg_seqs=["TGCATGCA...", ...],   # negative sequences
    kernel_type="estimated",          # -t 2 (default), or "direct", "rbf", etc.
    l=11, k=7, d=3,                  # kernel parameters
    C=1.0,                            # regularization
)

# Score new sequences
scores = model(x)  # positive = predicted positive class
```

Input sequences can be DNA strings or pre-encoded `[N, 4, L]` one-hot arrays.

### Regression (epsilon-SVR)

Train a regression model on sequences with continuous labels (e.g., chromatin accessibility, gene expression):

```python
from gkmsvm import train_gkmsvr

model = train_gkmsvr(
    sequences=["ACGTACGT...", ...],   # DNA sequences
    labels=[0.5, 1.2, -0.3, ...],    # continuous target values
    kernel_type="estimated",
    l=11, k=7, d=3,
    C=1.0,
    epsilon=0.1,                      # tube width — errors within ±epsilon are ignored
)

# Predict continuous values
predictions = model(x)  # [B, 1]
```

### Solver selection

Training requires computing kernel values between all pairs of training sequences. Two solvers are available:

- **Precomputed Gram + LIBSVM** (`solver="libsvm"`): Computes the full N x N kernel matrix, then passes it to LIBSVM's C solver. Fast, but requires O(N^2) memory. For N=10K sequences, the Gram matrix is ~800 MB.

- **Column-cached SMO** (`solver="smo"`): Computes kernel columns on demand with an LRU cache. Memory is O(cache_size x N) instead of O(N^2). Slower per iteration but scales to 80K+ sequences where the Gram matrix would exceed available RAM.

- **Auto** (`solver="auto"`, default): Estimates whether the Gram matrix fits in 50% of available RAM (CPU) or VRAM (GPU). Uses precomputed Gram when it fits, SMO otherwise.

```python
# Force a specific solver
model = train_gkmsvm(pos, neg, solver="libsvm")   # fast, needs N^2 memory
model = train_gkmsvm(pos, neg, solver="smo", cache_size=512)  # large-scale

# SVR always uses the libsvm solver (SMO SVR not yet implemented)
model = train_gkmsvr(seqs, labels)
```

Training works on all backends (CPU, NVIDIA GPU, Apple Silicon MLX). On MLX, the Metal kernels accelerate Gram matrix and kernel column computation. After training, use `model.mlx()` or `model.cuda()` to move the model for inference.

### Kernel types

All six kernel types are supported for both training and inference:

| Flag | Alias | Use case |
|---|---|---|
| `-t 0` | `direct` | Exact k-mer counting. Best for short sequences or small k. |
| `-t 2` | `estimated` | Default. Truncated estimated kernel, fastest for large l. |
| `-t 1` | `estimated_full` | Like estimated but without truncation. |
| `-t 3` | `rbf` | RBF transform on estimated kernel. Better for some tasks. |
| `-t 4` | `weighted` | Center-weighted positions. Requires `M` and `H` parameters. |
| `-t 5` | `weighted_rbf` | Center-weighted + RBF. Requires `M`, `H`, and `gamma`. |

```python
# Weighted kernel example
model = train_gkmsvm(
    pos, neg,
    kernel_type="weighted",
    l=11, k=7,
    M=5,      # center-weight window size
    H=1.0,    # decay half-life
)
```

### Training workflow example

A typical workflow for training on ATAC-seq peaks:

```python
from gkmsvm import (
    train_gkmsvm, read_fasta, one_hot_encode, ism, gkmexplain
)

# Load sequences
pos_records = read_fasta("positive_peaks.fa")
neg_records = read_fasta("negative_peaks.fa")
pos_seqs = [seq for _, seq in pos_records]
neg_seqs = [seq for _, seq in neg_records]

# Train
model = train_gkmsvm(pos_seqs, neg_seqs, l=11, k=7, d=3, C=1.0)

# Save
model.save("my_model.npz")

# Score new sequences
test_records = read_fasta("test.fa")
test_seqs = [one_hot_encode(seq) for _, seq in test_records]
import numpy as np
x = np.stack(test_seqs)
scores = model(x)

# Interpret
attr = gkmexplain(model, x, mode=0)
```
