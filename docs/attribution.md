# Attribution and interpretability

## Why gradient methods don't work

**Gradient-based methods (DeepLIFT, SHAP, captum, ledidi) are NOT compatible with gkm-SVMs.** The kernel involves discrete k-mer counting and mismatch lookups — no meaningful gradient exists through these operations. Do not attempt to differentiate through the kernel.

Ledidi is gradient-based (uses backprop for sequence design) and therefore incompatible.

## Supported methods

### GkmExplain (primary)
Analytically decomposes the SVM decision function into per-base importance scores without gradients. 20-30x faster than ISM on CPU. Ported from kundajelab/lsgkm C implementation (Shrikumar et al. 2019).

- Mode 0: importance scores — attributes kernel value to matching positions
- Mode 1: hypothetical importance scores — all 4 bases get values at each position

```python
from gkmsvm import gkmexplain
scores = gkmexplain(model, x, mode=0)         # [B, 4, L]
hyp_scores = gkmexplain(model, x, mode=1)     # [B, 4, L]
```

### ISM (in-silico mutagenesis)
Exhaustively score all single-base mutations. Evaluation-based, no gradients needed. Slower but exact. Can leverage tangermeme.ersatz.substitute for sequence manipulation.

### Paired REF/ALT scoring
`score(ALT) - score(REF)` for variant effect prediction. Evaluation-based. For VCF-scale scoring, use `tangermeme.variant_effect.substitution_effect` which accepts any `nn.Module`.

Note: `score(ALT) - score(REF)` is NOT identical to deltaSVM's k-mer-weight approximation — deltaSVM is a linear approximation.

### Simulated annealing / black-box optimization
For sequence design tasks. Works because it only requires score evaluation, not gradients. Replaces the role ledidi plays for neural models.

## tangermeme utilities

- `tangermeme.ersatz`: substitute, shuffle, dinucleotide_shuffle, randomize — all accept `[B, 4, L]` tensors
- `tangermeme.variant_effect`: substitution_effect, deletion_effect, insertion_effect — work with any nn.Module
- `tangermeme.match.extract_matching_loci`: GC-matched negative loci generation for training
