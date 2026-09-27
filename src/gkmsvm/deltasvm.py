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
from numba import njit, prange

from gkmsvm.backend import get_array_module, get_strides, is_gpu, is_mlx, to_cpu, to_gpu, to_mlx
from gkmsvm.codec import reverse_complement


@njit(parallel=True, fastmath=True, cache=True)
def _score_strand_numba(x, combos, powers, weights, l):
    """Fused argmax + gapped k-mer weight lookup. No intermediate arrays."""
    B = x.shape[0]
    L = x.shape[2]
    W = L - l + 1
    n_combos = combos.shape[0]
    k = combos.shape[1]
    scores = np.zeros(B, dtype=np.float64)
    if W < 1:
        return scores
    for b in prange(B):
        s = 0.0
        for w in range(W):
            for c in range(n_combos):
                idx = 0
                for j in range(k):
                    pos = w + combos[c, j]
                    base = 0
                    for ch in range(1, 4):
                        if x[b, ch, pos] > x[b, base, pos]:
                            base = ch
                    idx += base * powers[j]
                s += weights[idx]
        scores[b] = s
    return scores

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
        device: str = "cpu",
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
        self._on_gpu = False
        self._on_mlx = False

        self._combos = np.array(list(combinations(range(l), k)), dtype=np.int64)
        self._powers = 4 ** np.arange(k - 1, -1, -1, dtype=np.int64)

        if device != "cpu":
            self._to_device(device)

    @property
    def num_kmers(self) -> int:
        return int((self.weights != 0).sum())

    def _to_device(self, device: str) -> None:
        from gkmsvm.backend import HAS_CUPY, HAS_MLX
        if device in ("cuda", "gpu"):
            self.cuda()
        elif device == "mlx":
            self.mlx()
        elif device == "auto":
            if HAS_CUPY:
                self.cuda()
            elif HAS_MLX:
                self.mlx()
        elif device != "cpu":
            raise ValueError(
                f"Unknown device {device!r}. Use 'auto', 'cpu', 'cuda', or 'mlx'."
            )

    def cuda(self) -> DeltaSVM:
        """Move model arrays to GPU (CuPy)."""
        self.weights = to_gpu(self.weights)
        self._on_gpu = True
        self._on_mlx = False
        return self

    def mlx(self) -> DeltaSVM:
        """Move model arrays to Apple GPU (MLX)."""
        self.weights = to_mlx(to_cpu(self.weights))
        self._on_gpu = False
        self._on_mlx = True
        return self

    def cpu(self) -> DeltaSVM:
        """Move model arrays to CPU (NumPy)."""
        self.weights = to_cpu(self.weights)
        self._on_gpu = False
        self._on_mlx = False
        return self

    def _match_device(self, x: np.ndarray) -> np.ndarray:
        """Convert input array to match the model's device."""
        if self._on_gpu and not is_gpu(x):
            return to_gpu(np.asarray(x) if is_mlx(x) else x)
        if self._on_mlx and not is_mlx(x):
            return to_mlx(to_cpu(x) if is_gpu(x) else x)
        if not self._on_gpu and not self._on_mlx and (is_gpu(x) or is_mlx(x)):
            return to_cpu(x)
        return x

    def __call__(
        self, x: np.ndarray, *, verbose: bool = False,
    ) -> np.ndarray:
        """Score sequences by summing gapped k-mer weights.

        Args:
            x: [B, 4, L] one-hot encoded DNA sequences.
            verbose: Show tqdm progress bar over batch chunks.

        Returns:
            [B, 1] scores.
        """
        x = self._match_device(x)
        score = self._score_strand(x, verbose=verbose)
        if self.include_rc:
            score = score + self._score_strand(reverse_complement(x))
        return score + self.bias

    _INTERMEDIATE_BUDGET = 256 * 1024 * 1024  # 256 MB

    def _score_strand(
        self, x: np.ndarray, *, verbose: bool = False,
    ) -> np.ndarray:
        """Score one strand."""
        xp = get_array_module(x)
        B, C, L = x.shape
        W = L - self.l + 1
        if W < 1:
            return xp.zeros((B, 1), dtype=x.dtype)

        if not self._on_gpu and not self._on_mlx:
            return self._score_strand_cpu(x, verbose=verbose)
        return self._score_strand_array(x, verbose=verbose)

    def _score_strand_cpu(
        self, x: np.ndarray, *, verbose: bool = False,
    ) -> np.ndarray:
        """Numba-accelerated CPU path. No intermediate arrays."""
        B = x.shape[0]
        x64 = np.ascontiguousarray(x, dtype=np.float64)
        weights64 = np.asarray(self.weights, dtype=np.float64)
        if verbose and B > 100:
            from tqdm import tqdm
            chunk = max(1, B // 20)
            chunks = list(range(0, B, chunk))
            parts = []
            for i in tqdm(chunks, desc="DeltaSVM"):
                parts.append(_score_strand_numba(
                    x64[i:i + chunk], self._combos, self._powers, weights64, self.l
                ))
            scores = np.concatenate(parts)
        else:
            scores = _score_strand_numba(
                x64, self._combos, self._powers, weights64, self.l
            )
        return scores[:, None].astype(x.dtype)

    def _score_strand_array(
        self, x: np.ndarray, *, verbose: bool = False,
    ) -> np.ndarray:
        """GPU/MLX path using array operations."""
        xp = get_array_module(x)
        B, C, L = x.shape
        W = L - self.l + 1

        n_combos = len(self._combos)
        est_bytes = B * W * n_combos * self.k * 4
        if est_bytes > self._INTERMEDIATE_BUDGET:
            chunk = max(1, self._INTERMEDIATE_BUDGET // (W * n_combos * self.k * 4))
            chunks = range(0, B, chunk)
            if verbose:
                from tqdm import tqdm
                chunks = tqdm(chunks, desc="DeltaSVM", total=(B + chunk - 1) // chunk)
            parts = [self._score_strand_array(x[i:i + chunk]) for i in chunks]
            return xp.concatenate(parts)

        sx = get_strides(x)
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

    def score_variants(
        self, ref: np.ndarray, alt: np.ndarray, *, verbose: bool = False,
    ) -> np.ndarray:
        """Compute variant effect scores as score(alt) - score(ref).

        Args:
            ref: [B, 4, L] reference sequences.
            alt: [B, 4, L] alternate sequences.
            verbose: Show tqdm progress bar.

        Returns:
            [B, 1] score differences.
        """
        ref = self._match_device(ref)
        alt = self._match_device(alt)
        return self(alt, verbose=verbose) - self(ref, verbose=verbose)
