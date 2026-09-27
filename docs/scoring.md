# Scoring and variant effect prediction

## Basic scoring

All models (loaded or trained) use the same interface. Input is a `[batch, 4, length]` one-hot encoded DNA array; output is `[batch, 1]` margin scores.

```python
from gkmsvm import load_model, one_hot_encode
import numpy as np

model = load_model("model.npz")

# Encode a single sequence
x = one_hot_encode("ACGTACGTACGTACGTACGT")  # [4, 20]

# Score (add batch dimension)
score = model(x[None, ...])  # [1, 1]
print(f"Score: {score.item():.4f}")
```

### Batch scoring

Score multiple sequences at once:

```python
from gkmsvm import encode_batch

# From strings (automatically pads to max length)
sequences = ["ACGTACGT...", "TGCATGCA...", ...]
x = encode_batch(sequences)  # [N, 4, max_len]

scores = model(x)  # [N, 1]
```

### Scoring from FASTA files

```python
from gkmsvm import read_fasta, encode_batch

records = read_fasta("sequences.fa")
sequences = [seq for _, seq in records]
x = encode_batch(sequences)

scores = model(x, verbose=True)  # progress bar for large batches
```

## Variant effect prediction

### Single variant

```python
ref = one_hot_encode("ACGTACGTACGTACGT")
alt = one_hot_encode("ACGTACGAACGTACGT")  # T→A at position 8

ref_score = model(ref[None, ...]).item()
alt_score = model(alt[None, ...]).item()
delta = alt_score - ref_score
```

### Batch variant scoring

For scoring many REF/ALT pairs efficiently:

```python
ref_seqs = np.stack([one_hot_encode(s) for s in ref_strings])
alt_seqs = np.stack([one_hot_encode(s) for s in alt_strings])

# Returns [N, 1] delta scores (alt - ref)
deltas = model.score_variants(ref_seqs, alt_seqs, verbose=True)
```

### DeltaSVM (linear approximation)

DeltaSVM provides a faster but approximate variant effect score using pre-computed k-mer weights instead of full kernel evaluation:

```python
from gkmsvm import load_deltasvm_weights

dsvm = load_deltasvm_weights("weights.txt")
scores = dsvm(x)
deltas = dsvm.score_variants(ref_seqs, alt_seqs)
```

DeltaSVM is orders of magnitude faster than full SVM scoring but is a linear approximation. For accurate scores, use the full model's `score_variants()`.

## GPU acceleration

### NVIDIA (CuPy)

Move models to GPU for ~6x speedup:

```python
model.cuda()           # move to GPU
scores = model(x_gpu)  # input must also be on GPU

import cupy as cp
x_gpu = cp.asarray(x)
scores = model(x_gpu)

model.cpu()            # move back to CPU
```

GPU throughput is roughly flat across batch sizes (compute-bound). Use batch_size=16-64 to save VRAM without sacrificing speed.

### Apple Silicon (MLX)

Move models to MLX for 2-6x speedup on Apple Silicon:

```python
from gkmsvm.backend import to_mlx

model.mlx()            # move to Apple GPU
x_mlx = to_mlx(x)
scores = model(x_mlx)

model.cpu()            # move back to CPU
```

MLX uses custom Metal shaders with the same packed uint32 XOR+popcount approach as the CUDA path. Speedup scales with batch size — use batch_size=4-16 for best throughput. Apple Silicon's unified memory is shared between CPU and GPU; avoid running memory-heavy background processes during large jobs.

## Memory management

For large models (>10K support vectors), use chunked SV inference to bound peak memory:

```python
# At load time
model = load_model("large_model.npz", sv_chunk_size=5000)

# Or on an existing model
model.sv_chunk_size = 5000
```

Recommended chunk sizes for single-sequence scoring at 300bp:

| GPU VRAM | `sv_chunk_size` | Peak memory |
|---|---|---|
| 10 GB NVIDIA | 5000 | ~3.4 GB |
| 16+ GB NVIDIA | 10000 | ~6.8 GB |
| 16 GB Apple Silicon | 5000 | ~3.4 GB (unified memory shared with system) |
| CPU | 5000 or None | N/A |

For batch scoring, reduce proportionally (`5000 / batch_size`).

**Apple Silicon note:** MLX uses unified memory shared between CPU and GPU. Keep batch sizes moderate (4-16) and be aware that other applications (browsers, other processes) compete for the same memory pool. DeltaSVM auto-chunks large batches to prevent memory thrashing.

## One-hot encoding

DNA sequences are represented as `[4, length]` one-hot arrays with channel order A=0, C=1, G=2, T=3.

```python
from gkmsvm import one_hot_encode, one_hot_decode, reverse_complement

# Encode
x = one_hot_encode("ACGT")  # [4, 4] float64

# Decode back to string
seq = one_hot_decode(x)  # "ACGT"

# Reverse complement
rc = reverse_complement(x)  # [4, 4]
one_hot_decode(rc)  # "ACGT" (palindrome in this case)

# Batch encode with padding
from gkmsvm import encode_batch
x = encode_batch(["ACGT", "ACGTACGT"])  # [2, 4, 8], shorter seqs zero-padded
```

## Reverse-complement invariance

By default, kernels are RC-equivalent: `K(x, y) = K(x, y) + K(x, RC(y))`. This means scores are the same for a sequence and its reverse complement:

```python
x = one_hot_encode("ACGTACGTACGT")[None, ...]
rc_x = reverse_complement(x)

score = model(x).item()
rc_score = model(rc_x).item()
assert abs(score - rc_score) < 1e-10
```

RC equivalence is on by default and matches LS-GKM's `norc=0` setting. Models loaded from LS-GKM files respect the `norc` header if present.
