from __future__ import annotations

from math import comb

import numpy as np

from gkmsvm.backend import get_array_module
from gkmsvm.codec import reverse_complement
from gkmsvm.kernels.base import GkmKernel


class DirectGkmKernel(GkmKernel):
    """Direct gapped k-mer kernel (LS-GKM -t 0 / gkm_cnt).

    For window length l and k informative positions, two windows differing
    in m positions share C(l-m, k) gapped k-mer features. The kernel sums
    these shared features over all pairs of windows from the two sequences.

    Kernel computation uses matmul on flattened one-hot windows, avoiding
    the O(B*S*W*W*4*l) intermediate of the broadcast approach.
    """

    def __init__(
        self, l: int, k: int, *, normalize: bool = True, include_rc: bool = True
    ):
        super().__init__(l, k, normalize=normalize, include_rc=include_rc)
        self._mismatch_table = np.array(
            [comb(l - m, k) if l - m >= k else 0 for m in range(l + 1)],
            dtype=np.float64,
        )

    def flat_windows(self, x: np.ndarray) -> np.ndarray:
        """Extract length-l windows and flatten channels for matmul.

        Args:
            x: [B, 4, L] one-hot array.

        Returns:
            [B, W, 4*l] array where W = L - l + 1.
        """
        xp = get_array_module(x)
        B, C, L = x.shape
        if L < self.l:
            raise ValueError(
                f"Sequence length {L} is shorter than window length {self.l}"
            )
        W = L - self.l + 1
        # Sliding window: extract [B, 4, W, l] then reshape to [B, W, 4*l]
        strides = x.strides
        shape = (B, C, W, self.l)
        new_strides = (strides[0], strides[1], strides[2], strides[2])
        wx = xp.lib.stride_tricks.as_strided(x, shape=shape, strides=new_strides)
        wx = xp.ascontiguousarray(wx.transpose(0, 2, 1, 3))  # [B, W, 4, l]
        return wx.reshape(B, W, C * self.l)

    def _apply_table(self, matches: np.ndarray) -> np.ndarray:
        """Look up weight table from match counts, sum over window dims.

        Args:
            matches: [..., Wx, Wy] float array of per-window-pair match counts.

        Returns:
            [...] array with window dimensions summed out.
        """
        xp = get_array_module(matches)
        table = xp.asarray(self._mismatch_table)
        mismatches = xp.clip(xp.rint(self.l - matches).astype(np.int64), 0, self.l)
        return table[mismatches].sum(axis=(-2, -1))

    def pairwise_from_windows(
        self, wx: np.ndarray, wy: np.ndarray
    ) -> np.ndarray:
        """Raw kernel from pre-extracted flat windows (no RC, no normalization).

        Args:
            wx: [B, Wx, F] flat windows from query sequences.
            wy: [S, Wy, F] flat windows from support sequences.

        Returns:
            [B, S] raw kernel values.
        """
        xp = get_array_module(wx)
        matches = xp.einsum("bif,sjf->bsij", wx, wy)
        return self._apply_table(matches)

    def diagonal_from_windows(self, wx: np.ndarray) -> np.ndarray:
        """Raw self-kernel from pre-extracted flat windows (no RC, no normalization).

        Args:
            wx: [B, W, F] flat windows.

        Returns:
            [B] raw self-kernel values.
        """
        xp = get_array_module(wx)
        # bmm: [B, W, F] @ [B, F, W] -> [B, W, W]
        self_matches = xp.matmul(wx, wx.transpose(0, 2, 1))
        return self._apply_table(self_matches)

    def _raw_pairwise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        wx = self.flat_windows(x)
        wy = self.flat_windows(y)
        result = self.pairwise_from_windows(wx, wy)

        if self.include_rc:
            wy_rc = self.flat_windows(reverse_complement(y))
            result = result + self.pairwise_from_windows(wx, wy_rc)

        return result.astype(x.dtype)

    def _raw_diagonal(
        self, x: np.ndarray, *, chunk_size: int | None = None
    ) -> np.ndarray:
        xp = get_array_module(x)
        B = x.shape[0]
        if chunk_size is None or chunk_size >= B:
            wx = self.flat_windows(x)
            result = self.diagonal_from_windows(wx)
            if self.include_rc:
                wx_rc = self.flat_windows(reverse_complement(x))
                rc_matches = xp.matmul(wx, wx_rc.transpose(0, 2, 1))
                result = result + self._apply_table(rc_matches)
            return result.astype(x.dtype)

        result = xp.empty(B, dtype=x.dtype)
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            result[start:end] = self._raw_diagonal(x[start:end])
        return result
