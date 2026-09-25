from __future__ import annotations

import torch


def gkmexplain(
    model,
    sequences: torch.Tensor,
) -> torch.Tensor:
    """Compute per-base importance scores using the GkmExplain algorithm.

    Analytically decomposes the SVM decision function into per-base
    contributions without requiring gradients. Port of the C implementation
    from kundajelab/lsgkm (Shrikumar et al. 2019).

    Gradient-based attribution methods (DeepLIFT, SHAP, captum, ledidi)
    are NOT compatible with gkm-SVMs because the kernel involves discrete
    k-mer counting with no meaningful gradient.

    Args:
        model: A GkmSVM model instance.
        sequences: [B, 4, L] one-hot encoded DNA sequences.

    Returns:
        [B, 4, L] per-base importance scores.
    """
    raise NotImplementedError(
        "GkmExplain not yet implemented. "
        "Reference: kundajelab/lsgkm libsvm_gkm.c kmertree_dfs_withexplanation(). "
        "Streaming variant: harmstonlab/mLS-GKM (O(L+SV) memory vs O(L*SV))."
    )


def ism(
    model,
    sequences: torch.Tensor,
) -> torch.Tensor:
    """In-silico mutagenesis: score all single-base mutations.

    For each position, substitutes each of the 3 alternative bases and
    computes the score difference. Evaluation-based, no gradients needed.
    Exploits window locality: only ~l windows per mutation are affected.

    Args:
        model: A GkmSVM model instance.
        sequences: [B, 4, L] one-hot encoded DNA sequences.

    Returns:
        [B, 4, L] scores where entry [b, c, p] is the score when
        position p is mutated to base c.
    """
    raise NotImplementedError
