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

# Straight onto a GPU
model = load_lsgkm_model("model.txt.gz", device="cuda")
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

The R package (Ghandi et al. 2016) uses `#`-prefixed headers and FASTA-style support vectors. The kernel is `gkm_esttrunc` unless a `#kernel_type` header (name or `-t` integer) says otherwise.

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
from gkmsvm import load_deltasvm_model
from gkmsvm import DeltaSVM

# Load from k-mer weight file
model = load_deltasvm_model("weights.txt")

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
model.save("model.bin", fmt="npz")
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

Training needs kernel values between pairs of training sequences.

- **Precomputed Gram + sklearn** (`solver="libsvm"`): computes the full N x N kernel matrix in host RAM (tiles on the training device) and passes it to scikit-learn's LIBSVM-backed `SVC(kernel="precomputed")`. Fast, but O(N^2) memory: ~800 MB at N=10K.
- **Column-cached SMO** (`solver="smo"`): WSS3 working set selection with shrinking and an LRU cache of kernel columns in a C extension; memory is O(cache_size x N). For `-t 1`-`-t 5` a column is one query of an l-mer index over the training set; other kernels compute columns in batches through the masked join. With a cache smaller than N, evicted columns are recomputed, so it is slower than libsvm when the Gram fits.
- **Nyström approximation** (`solver="nystrom"`): kernel values against `n_components` landmarks only (O(N m) instead of O(N^2)), then an approximate N x N Gram for sklearn. Saves time, not memory; the solution is approximate.
- **Auto** (`solver="auto"`, default): libsvm when the Gram fits in 75% of available RAM (and under `max_gram_gb`), else SMO with a warning.

```python
model = train_gkmsvm(pos, neg, solver="libsvm")                      # needs N^2 memory
model = train_gkmsvm(pos, neg, solver="smo", cache_size=2048)        # O(cache_size x N)
model = train_gkmsvm(pos, neg, solver="nystrom", n_components=500)   # approximate

# SVR uses the precomputed Gram (or Nyström); SMO SVR is not implemented,
# so SVR training needs O(N^2) memory.
model = train_gkmsvr(seqs, labels)
```

Training works on all backends (CPU, NVIDIA GPU, Apple Silicon MLX). The returned model is on CPU; use `model.cuda()` or `model.mlx()` to move it for inference. Timings are in [performance](performance.md#measurements).

### Kernel types

All six kernel types are supported for both training and inference:

| Flag | Alias | Use case |
|---|---|---|
| `-t 0` | `direct` | Exact k-mer counting. Best for short sequences or small k. |
| `-t 2` | `estimated` | Default. Truncated estimated kernel, fastest for large l. |
| `-t 1` | `estimated_full` | Like estimated but without truncation. |
| `-t 3` | `rbf` | RBF transform on estimated kernel. Better for some tasks. |
| `-t 4` | `weighted` | `-t 2` with l-mers near the sequence center weighted up to `M` times more (`M`, `H`; LS-GKM defaults 50, 50). |
| `-t 5` | `weighted_rbf` | Center-weighted + RBF (`M`, `H`, `gamma`). |

```python
# Weighted kernel example
model = train_gkmsvm(
    pos, neg,
    kernel_type="weighted",
    l=11, k=7, d=3,
    M=50,     # weight of the center window (1-255)
    H=50.0,   # half-life of the weights, in bp from the center
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
attr = gkmexplain(model, x, mode="importance")
```
