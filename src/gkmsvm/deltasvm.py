"""DeltaSVM: linear scoring model over gapped k-mer weights.

A deltaSVM model assigns a precomputed weight to each k-mer of length k.
Scoring sums the weights of all gapped k-mers (C(l,k) per window, W
windows per sequence). This is a LINEAR approximation of the full kernel
SVM — score(ALT) - score(REF) from deltaSVM is NOT identical to the same
difference computed via full kernel evaluation.
"""

from __future__ import annotations

from itertools import combinations

import torch
from torch import nn

from gkmsvm.codec import reverse_complement

_BASE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def _kmer_to_index(kmer: str) -> int:
    idx = 0
    for base in kmer:
        idx = idx * 4 + _BASE_MAP[base]
    return idx


def _index_to_kmer(idx: int, k: int) -> str:
    bases = "ACGT"
    chars = []
    for _ in range(k):
        chars.append(bases[idx % 4])
        idx //= 4
    return "".join(reversed(chars))


class DeltaSVM(nn.Module):
    """Linear gapped k-mer scoring model.

    Stores a weight table of size 4^k and scores sequences by summing
    weights of all gapped k-mers found in sliding windows of length l.

    Score = Σ_{windows} Σ_{C(l,k) combos} weight[kmer_index]
    """

    def __init__(
        self,
        weights: torch.Tensor,
        l: int,
        k: int,
        *,
        include_rc: bool = True,
        bias: float = 0.0,
    ):
        super().__init__()
        expected = 4**k
        if weights.shape != (expected,):
            raise ValueError(f"weights must be [{expected}], got {list(weights.shape)}")
        if k > l:
            raise ValueError(f"k ({k}) must be <= l ({l})")

        self.register_buffer("weights", weights)
        self.register_buffer("_bias", torch.tensor(bias, dtype=weights.dtype))
        self.l = l
        self.k = k
        self.include_rc = include_rc

        combos = list(combinations(range(l), k))
        self.register_buffer(
            "_combos", torch.tensor(combos, dtype=torch.long)
        )
        powers = 4 ** torch.arange(k - 1, -1, -1, dtype=torch.long)
        self.register_buffer("_powers", powers)

    @property
    def bias(self) -> float:
        return self._bias.item()

    @property
    def num_kmers(self) -> int:
        return int((self.weights != 0).sum().item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score sequences by summing gapped k-mer weights.

        Args:
            x: [B, 4, L] one-hot encoded DNA sequences.

        Returns:
            [B, 1] scores.
        """
        score = self._score_strand(x)
        if self.include_rc:
            score = score + self._score_strand(reverse_complement(x))
        return score + self._bias

    def _score_strand(self, x: torch.Tensor) -> torch.Tensor:
        """Score one strand."""
        B, C, L = x.shape
        W = L - self.l + 1
        if W < 1:
            return torch.zeros(B, 1, dtype=x.dtype, device=x.device)

        windows = x.unfold(2, self.l, 1)
        base_idx = windows.argmax(dim=1)

        selected = base_idx[:, :, self._combos]
        kmer_idx = (selected * self._powers).sum(dim=-1)
        scores = self.weights[kmer_idx].sum(dim=(1, 2))
        return scores.unsqueeze(1)

    def score_variants(self, ref: torch.Tensor, alt: torch.Tensor) -> torch.Tensor:
        """Compute variant effect scores as score(alt) - score(ref).

        Args:
            ref: [B, 4, L] reference sequences.
            alt: [B, 4, L] alternate sequences.

        Returns:
            [B, 1] score differences.
        """
        return self.forward(alt) - self.forward(ref)
