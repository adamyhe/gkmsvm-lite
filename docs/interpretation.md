# Interpretation and attribution

## Why gradient methods don't work

gkm-SVMs compute kernel values through discrete k-mer counting and mismatch table lookups. There is no meaningful gradient through these operations. **Gradient-based attribution methods are not compatible with gkm-SVMs.** This includes DeepLIFT, DeepSHAP, Integrated Gradients, Saliency, GradCAM, and gradient-based methods in captum. Ledidi (gradient-based sequence design) is also incompatible.

**Perturbation-based methods do work** — they only need to evaluate the model, not differentiate through it. This includes KernelSHAP, Occlusion, Shapley Value Sampling, and Feature Ablation (all available in captum). However, these require many kernel evaluations per sequence and are much slower than GkmExplain or ISM. GkmExplain has theoretical connections to Integrated Gradients and is the recommended attribution method for gkm-SVMs: it handles epistatic motif logic (OR/redundancy) that saturates ISM.

## GkmExplain

GkmExplain analytically decomposes the SVM decision function into per-base importance scores. It provides theoretically grounded attributions with connections to Integrated Gradients. Attribution modes need a count-table base kernel (`-t 0/1/2`, or `-t 3` on top of one). For `-t 3`, as in lsgkm, each SV's contributions are scaled by (exp(γ(K−1)) − exp(−γ))/K of its kernel value K with the sequence, so importance scores sum to score − bias − exp(−γ)·Σcoef. For weighted models (`-t 4/5`) use `mutation_impact()` or ISM.

```python
from gkmsvm import gkmexplain

# Importance scores — non-zero only at reference bases
attr = gkmexplain(model, x, mode="importance")    # [B, 4, L]

# Hypothetical importance — all 4 bases at each position
hyp = gkmexplain(model, x, mode="hypothetical")   # [B, 4, L]

# Perturbation effect — discrete kernel Δ per single-base mutation
pert = gkmexplain(model, x, mode="perturbation")  # [B, 4, L]
```

### Modes

**Importance** (`mode="importance"`, legacy `mode=0`) attributes the kernel value to the positions that actually match between the query and each support vector. Only the reference base at each position gets a non-zero score. Satisfies the completeness axiom: attributions sum to `score - bias`. Use for understanding what drives the current prediction.

**Hypothetical** (`mode="hypothetical"`, legacy `mode=1`) computes what the importance would be if each base were present at each position. All four channels can be non-zero. Corresponds to lsgkm C mode 1 (`perturbation_eff=0`). Use for:
- Motif discovery with TF-MoDISco (hypothetical contribution input)
- Generating sequence logos

Importance is exactly `hypothetical * one_hot_input`. Internally, only the hypothetical computation runs — importance is derived by element-wise multiplication with the input one-hot.

**Perturbation** (`mode="perturbation"`, legacy `mode=2`) computes the discrete kernel-value change from each possible single-base mutation. At each position, the reference base channel is zero; alternate base channels contain the predicted score change from that mutation. Corresponds to lsgkm C mode 3 (`perturbation_eff=1`). Uses `gamma[m] = mismatch_table[m+1] - mismatch_table[m]` for match-to-mismatch and `kappa[m] = mismatch_table[m-1] - mismatch_table[m]` for mismatch-to-match. Exact for linear kernels; does not account for the RBF `exp()` in `-t 3`/`-t 5`. Use for per-position mutation maps.

**Mutation impact score** (`mutation_impact(model, ref, alt)`, or `model.score_variants(ref, alt, method="gkmexplain")`) is the variant effect score used in the GkmExplain paper's dsQTL evaluation (lsgkm `gkmexplain -m 5`). Per support vector, the base-kernel change is normalized by the *reference* sequence's norm; for RBF kernels the kernel value above its floor, `exp(γ(K-1)) - exp(-γ)`, is distributed in proportion to `ΔK / K`. It is a linearization, so it differs slightly from the kernel delta `score(alt) - score(ref)`. It works for every kernel type. Verified against lsgkm `gkmexplain -m 5` output. With the paper's models it reproduces the paper's dsQTL AUPRCs exactly ([validation](validation.md#dsqtl-variant-effect-prediction)).

### Practical usage

```python
import numpy as np
from gkmsvm import load_model, one_hot_encode, gkmexplain

model = load_model("model.npz")
x = one_hot_encode("ACGTACGTACGTACGTACGT")[None, ...]

# Get importance scores
attr = gkmexplain(model, x, mode="importance")  # [1, 4, 20]

# Find most important positions
importance = attr[0].sum(axis=0)  # [20] — sum over bases
top_positions = np.argsort(-np.abs(importance))[:5]
print(f"Top 5 positions: {top_positions}")

# Get hypothetical scores for motif discovery
hyp = gkmexplain(model, x, mode="hypothetical")  # [1, 4, 20]

# Variant effect prediction
pert = gkmexplain(model, x, mode="perturbation")  # [1, 4, 20]
```

### Batching and memory

GkmExplain processes input sequences in batches (default 50) and support vectors in internal chunks (default 2000 SVs, tunable via `model.sv_chunk_size`). For large inputs, `batch_size` controls memory and enables progress bars:

```python
attr = gkmexplain(model, x, mode="importance", batch_size=20, verbose=True)
```

GkmExplain compares packed uint32 windows (XOR + popcount), skipping window pairs beyond the mismatch cutoff (~99.9% for l=11 k=7 d=3), and decomposes only the contributing pairs per position. The kernel multiplies by the normalized coefficient and reduces over SVs in its inner loop, accumulating into the `[B, 4, L]` result, so there is no per-SV intermediate. On CPU a Numba kernel parallelizes over the batch; on NVIDIA GPUs a CuPy kernel runs one thread per (sequence, SV) pair and accumulates with float64 `atomicAdd` (compute capability >= 6.0). On Apple Silicon it falls back to CPU (Metal has no float64, which the completeness axiom needs).

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

ISM works for every kernel type and on all backends (CPU, NVIDIA GPU, and Apple Silicon MLX). The affected windows are packed (padded with a window value that cancels in the difference) and scored through the same structures as `model()`: models with the exact l-mer table (`-t 0/1/2/4`) add table weights, so ISM costs a few lookups per mutation (about 4 ms per 300 bp sequence for the 72,145-SV ENCFF579AOX model on an M4); others score the changed windows against the SVs through the cached l-mer index or the masked join, about 0.5 s per 300 bp sequence for a 32,715-SV `-t 3` model. Weighted models (`-t 4/5`) score all 4L mutants through `model()` instead.

### When to use what

| | GkmExplain | ISM | KernelSHAP |
|---|---|---|---|
| Speed | All window pairs within the cutoff | Affected windows only, via the l-mer index where it applies | Orders of magnitude slower |
| Output | Analytical decomposition | Exact score deltas | Shapley values |
| Interpretation | What contributes to the kernel | What happens if you mutate | Feature importance with interactions |
| Epistatic logic | Handles OR/redundant motifs correctly | Saturates (misses redundant motifs) | Handles interactions |
| Hypothetical | What each base would contribute | N/A (inherently hypothetical) | N/A |
| Perturbation | Discrete kernel Δ per mutation | N/A | N/A |
| Use case | Recommended default | Exact mutation effects | When Shapley guarantees are needed |

GkmExplain is more informative than ISM. ISM suffers from **saturation effects**: when multiple motifs can independently drive the score (OR logic), mutating one motif has no effect if the other is intact, so ISM reports both as unimportant. GkmExplain analytically decomposes the kernel and correctly attributes importance to all contributing motifs regardless of redundancy. Hypothetical importance scores enable motif discovery with TF-MoDISco and are not available from ISM.

For variant effect prediction, use `model.score_variants(method="gkmexplain")` (the paper's mutation impact score) or `method="kernel"` for the exact score change. KernelSHAP provides formal Shapley value guarantees but is impractical for routine use due to the cost of repeated kernel evaluations.

## Paired REF/ALT scoring

For specific variants (not exhaustive mutagenesis), score REF and ALT sequences directly:

```python
from gkmsvm import one_hot_encode

ref = one_hot_encode("ACGTACGTACGTACGT")
alt = one_hot_encode("ACGTACGAACGTACGT")  # T→A at position 8

delta = model.score_variants(ref[None], alt[None]).item()  # score(alt) - score(ref)
```

This gives the exact score change for a specific variant; pass [N, 4, L] arrays to score many at once.

Note: `score(ALT) - score(REF)` from the full SVM is **not** identical to deltaSVM's k-mer weight approximation. DeltaSVM is a linear approximation that is faster but less accurate.

## Sequence design

For sequence design tasks (finding sequences that maximize or minimize the SVM score), use black-box optimization methods like simulated annealing. These work because they only require score evaluation, not gradients. This replaces the role that gradient-based tools like ledidi play for neural network models.

## Progress bars

All methods support `verbose=True` for tqdm progress bars on large inputs:

```python
scores = model(x, verbose=True)
deltas = ism(model, x, verbose=True)
attr = gkmexplain(model, x, mode="importance", verbose=True)
```
