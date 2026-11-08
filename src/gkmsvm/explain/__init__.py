"""GkmExplain: analytical attribution for gapped k-mer SVMs.

Decomposes the SVM decision function into per-position, per-base
importance scores without gradients. Ported from the C implementation
in kundajelab/lsgkm (Shrikumar et al. 2019).

The attribution is linear in the SVs (before any RBF), so it goes by the
first of these that applies: the explain table (every l-mer's attribution,
precomputed with Walsh-Hadamard transforms; non-RBF), the l-mer path (each
query window visits its neighbors' summed SV weights V, or for RBF sums the
per-query SV weights over the l-mer index), or the per-SV kernels, which
visit every (query window, SV window) pair on packed uint32 windows and
skip those below min_matches.

Modules: core (gkmexplain, path dispatch), table (explain table), lmer (the
l-mer path), cpu / cuda / metal (per-device kernels), weights (pair weights by
mismatch count), impact (mutation_impact), common (shared helpers).
"""

from gkmsvm.explain.core import gkmexplain
from gkmsvm.explain.impact import mutation_impact

__all__ = ["gkmexplain", "mutation_impact"]
