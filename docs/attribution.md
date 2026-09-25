# Attribution and interpretability

## Why gradient methods don't work

**Gradient-based methods (DeepLIFT, SHAP, captum, ledidi) are NOT compatible with gkm-SVMs.** The kernel involves discrete k-mer counting and mismatch lookups — no meaningful gradient exists through these operations. Do not attempt to differentiate through the kernel.

Ledidi is gradient-based (uses backprop for sequence design) and therefore incompatible.

## Supported methods

### GkmExplain (primary)
Analytically decomposes the SVM decision function into per-base importance scores without gradients. Much faster than ISM. No Python implementation exists — must be ported from C (kundajelab/lsgkm). Reference: Shrikumar et al. 2019.

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
