"""DeltaSVM: linear scoring model over gapped k-mer weights.

A deltaSVM model assigns a precomputed weight to each k-mer of length k.
Scoring sums the weights of all gapped k-mers (C(l,k) per window, W
windows per sequence). This is a LINEAR approximation of the full kernel
SVM — score(ALT) - score(REF) from deltaSVM is NOT identical to the same
difference computed via full kernel evaluation.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from gkmsvm.backend import get_array_module
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


class DeltaSVM:
    """Linear gapped k-mer scoring model.

    Stores a weight table of size 4^k and scores sequences by summing
    weights of all gapped k-mers found in sliding windows of length l.

    Score = Σ_{windows} Σ_{C(l,k) combos} weight[kmer_index]
    """

    def __init__(
        self,
        weights: np.ndarray,
        l: int,
        k: int,
        *,
        include_rc: bool = True,
        bias: float = 0.0,
    ):
        expected = 4**k
        if weights.shape != (expected,):
            raise ValueError(f"weights must be [{expected}], got {list(weights.shape)}")
        if k > l:
            raise ValueError(f"k ({k}) must be <= l ({l})")

        self.weights = weights
        self.bias = bias
        self.l = l
        self.k = k
        self.include_rc = include_rc

        self._combos = np.array(list(combinations(range(l), k)), dtype=np.int64)
        self._powers = 4 ** np.arange(k - 1, -1, -1, dtype=np.int64)

    @property
    def num_kmers(self) -> int:
        return int((self.weights != 0).sum())

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Score sequences by summing gapped k-mer weights.

        Args:
            x: [B, 4, L] one-hot encoded DNA sequences.

        Returns:
            [B, 1] scores.
        """
        score = self._score_strand(x)
        if self.include_rc:
            score = score + self._score_strand(reverse_complement(x))
        return score + self.bias

    def _score_strand(self, x: np.ndarray) -> np.ndarray:
        """Score one strand."""
        xp = get_array_module(x)
        B, C, L = x.shape
        W = L - self.l + 1
        if W < 1:
            return xp.zeros((B, 1), dtype=x.dtype)

        # Sliding windows
        sx = x.strides
        windows = xp.lib.stride_tricks.as_strided(
            x, shape=(B, C, W, self.l), strides=(sx[0], sx[1], sx[2], sx[2])
        )
        base_idx = xp.argmax(windows, axis=1)  # [B, W, l]

        combos = xp.asarray(self._combos)
        powers = xp.asarray(self._powers)
        weights = xp.asarray(self.weights)

        selected = base_idx[:, :, combos]  # [B, W, n_combos, k]
        kmer_idx = (selected * powers).sum(axis=-1)  # [B, W, n_combos]
        scores = weights[kmer_idx].sum(axis=(1, 2))
        return scores[:, None]

    def score_variants(self, ref: np.ndarray, alt: np.ndarray) -> np.ndarray:
        """Compute variant effect scores as score(alt) - score(ref)."""
        return self(alt) - self(ref)
