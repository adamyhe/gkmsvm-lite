# Interpretation and attribution

## Why gradient methods don't work

gkm-SVMs compute kernel values through discrete k-mer counting and mismatch table lookups. There is no meaningful gradient through these operations. **Gradient-based methods (DeepLIFT, SHAP, captum, ledidi) are not compatible with gkm-SVMs.** Do not attempt to differentiate through the kernel — use the methods below instead.

## GkmExplain

GkmExplain analytically decomposes the SVM decision function into per-base importance scores. It is 20-30x faster than ISM on CPU and provides theoretically grounded attributions with connections to Integrated Gradients.

```python
from gkmsvm import gkmexplain

# Mode 0: importance scores
# Non-zero only at reference bases (what the model "sees")
attr = gkmexplain(model, x, mode=0)  # [B, 4, L]

# Mode 1: hypothetical importance scores
# All 4 bases get values at each position (what the model "would see")
hyp = gkmexplain(model, x, mode=1)   # [B, 4, L]
```

### Mode 0 vs Mode 1

**Mode 0 (importance)** attributes the kernel value to the positions that actually match between the query and each support vector. Only the reference base at each position gets a non-zero score. Use this for understanding what drives the current prediction.

**Mode 1 (hypothetical)** computes what the importance would be if each base were present at each position. All four channels can be non-zero. Use this for:
- Motif discovery with TF-MoDISco
- Understanding what mutations would do (similar to ISM but faster)
- Generating sequence logos

### Practical usage

```python
import numpy as np
from gkmsvm import load_model, one_hot_encode, gkmexplain

model = load_model("model.npz")
x = one_hot_encode("ACGTACGTACGTACGTACGT")[None, ...]

# Get importance scores
attr = gkmexplain(model, x, mode=0)  # [1, 4, 20]

# Find most important positions
importance = attr[0].sum(axis=0)  # [20] — sum over bases
top_positions = np.argsort(-np.abs(importance))[:5]
print(f"Top 5 positions: {top_positions}")

# Get hypothetical scores for motif discovery
hyp = gkmexplain(model, x, mode=1)  # [1, 4, 20]
```

### Memory and chunking

GkmExplain processes support vectors in chunks to avoid GPU OOM on large models. The default chunk size is 2000 SVs. For very large models:

```python
attr = gkmexplain(model, x, mode=0, sv_chunk_size=1000)
```

GkmExplain uses the float one-hot kernel path (not packed uint32), so it is slower than forward scoring but still 20-30x faster than ISM.

## In-silico mutagenesis (ISM)

ISM exhaustively scores all possible single-base mutations and reports the score change relative to the reference. It is evaluation-based (no gradients needed) and gives exact delta scores.

```python
from gkmsvm import ism

# Score all single-base mutations
deltas = ism(model, x)  # [B, 4, L] — delta scores

# Reference base positions are always 0.0
# Other bases show the score change if that mutation were made
```

### Window-delta optimization

ISM in gkmsvm-lite uses a window-delta optimization: when a single base changes, only ~l of the W = L - l + 1 kernel windows are affected. Only these windows are recomputed rather than the full kernel, making ISM ~(W/l)x faster than naive re-scoring.

### When to use ISM vs GkmExplain

| | GkmExplain | ISM |
|---|---|---|
| Speed | 20-30x faster | Slower (one kernel eval per mutation) |
| Output | Analytical decomposition | Exact score deltas |
| Interpretation | What contributes to the kernel | What happens if you mutate |
| Mode 1 | Hypothetical importance at all bases | N/A (ISM is inherently "hypothetical") |
| Use case | Motif discovery, fast screening | Ground-truth mutation effects |

GkmExplain mode 1 delta scores closely match ISM delta scores but are not identical — GkmExplain decomposes the kernel differently than literal re-evaluation. For most applications, GkmExplain is preferred for speed. Use ISM when you need exact mutation impact scores.

## Paired REF/ALT scoring

For specific variants (not exhaustive mutagenesis), score REF and ALT sequences directly:

```python
from gkmsvm import one_hot_encode

ref = one_hot_encode("ACGTACGTACGTACGT")
alt = one_hot_encode("ACGTACGAACGTACGT")  # T→A at position 8

delta = model(alt[None, ...]).item() - model(ref[None, ...]).item()
```

This gives the exact score change for a specific variant. For batch scoring, use `model.score_variants()`.

Note: `score(ALT) - score(REF)` from the full SVM is **not** identical to deltaSVM's k-mer weight approximation. DeltaSVM is a linear approximation that is faster but less accurate.

## Sequence design

For sequence design tasks (finding sequences that maximize or minimize the SVM score), use black-box optimization methods like simulated annealing. These work because they only require score evaluation, not gradients. This replaces the role that gradient-based tools like ledidi play for neural network models.

## Progress bars

All methods support `verbose=True` for tqdm progress bars on large inputs:

```python
scores = model(x, verbose=True)
deltas = ism(model, x, verbose=True)
attr = gkmexplain(model, x, mode=0, verbose=True)
```
