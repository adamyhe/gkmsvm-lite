from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class GkmKernel(ABC):
    """Base class for gapped k-mer kernels.

    Subclasses implement specific kernel modes (-t 0, -t 2, etc.).
    """

    def __init__(
        self, l: int, k: int, *, normalize: bool = True, include_rc: bool = True
    ):
        if k > l:
            raise ValueError(f"k ({k}) must be <= l ({l})")
        if l < 1:
            raise ValueError(f"l must be >= 1, got {l}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.l = l
        self.k = k
        self.normalize = normalize
        self.include_rc = include_rc

    @abstractmethod
    def _raw_pairwise(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute unnormalized kernel values between x and y.

        Args:
            x: [B, 4, Lx] one-hot encoded sequences.
            y: [S, 4, Ly] one-hot encoded sequences.

        Returns:
            [B, S] tensor of raw kernel values.
        """

    @abstractmethod
    def _raw_diagonal(
        self, x: torch.Tensor, *, chunk_size: int | None = None
    ) -> torch.Tensor:
        """Compute unnormalized self-kernel values.

        Args:
            x: [B, 4, L] one-hot encoded sequences.
            chunk_size: Process in chunks to limit memory.

        Returns:
            [B] tensor of K(x_i, x_i) values.
        """

    _DIAG_CHUNK_THRESHOLD = 10000

    def pairwise(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute kernel values between all pairs of sequences in x and y.

        Args:
            x: [B, 4, Lx] one-hot encoded sequences.
            y: [S, 4, Ly] one-hot encoded sequences.

        Returns:
            [B, S] tensor of kernel values.
        """
        raw = self._raw_pairwise(x, y)
        if not self.normalize:
            return raw
        diag_x = self._raw_diagonal(x)
        cs = 1000 if y.shape[0] > self._DIAG_CHUNK_THRESHOLD else None
        diag_y = self._raw_diagonal(y, chunk_size=cs)
        norm = torch.sqrt(diag_x.unsqueeze(1) * diag_y.unsqueeze(0))
        norm = torch.clamp(norm, min=1e-10)
        return raw / norm

    def diagonal(self, x: torch.Tensor) -> torch.Tensor:
        """Compute self-kernel values K(x_i, x_i).

        Args:
            x: [B, 4, L] one-hot encoded sequences.

        Returns:
            [B] tensor. Always 1.0 when normalized.
        """
        if self.normalize:
            return torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
        return self._raw_diagonal(x)
