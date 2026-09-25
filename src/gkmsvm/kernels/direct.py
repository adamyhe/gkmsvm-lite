from __future__ import annotations

from math import comb

import torch

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
        self._mismatch_table = torch.tensor(
            [comb(l - m, k) if l - m >= k else 0 for m in range(l + 1)],
            dtype=torch.float64,
        )

    def flat_windows(self, x: torch.Tensor) -> torch.Tensor:
        """Extract length-l windows and flatten channels for matmul.

        Args:
            x: [B, 4, L] one-hot tensor.

        Returns:
            [B, W, 4*l] tensor where W = L - l + 1.
        """
        B, C, L = x.shape
        if L < self.l:
            raise ValueError(
                f"Sequence length {L} is shorter than window length {self.l}"
            )
        wx = x.unfold(2, self.l, 1)  # [B, 4, W, l]
        wx = wx.permute(0, 2, 1, 3)  # [B, W, 4, l]
        return wx.reshape(B, -1, C * self.l)  # [B, W, 4*l]

    def _apply_table(self, matches: torch.Tensor) -> torch.Tensor:
        """Look up weight table from match counts, sum over window dims.

        On GPU, uses a histogram loop that avoids materializing the full
        [B, S, Wx, Wy] gather result — 17x less peak memory, preventing OOM
        on large models. On CPU, uses direct gather which is faster.

        Args:
            matches: [..., Wx, Wy] float tensor of per-window-pair match counts.

        Returns:
            [...] tensor with window dimensions summed out.
        """
        dtype = (
            torch.float32
            if matches.device.type == "mps"
            else self._mismatch_table.dtype
        )
        table = self._mismatch_table.to(device=matches.device, dtype=dtype)
        if matches.device.type in ("cuda", "mps"):
            return self._apply_table_histogram(matches, table)
        return self._apply_table_eager(matches, table)

    def _apply_table_eager(
        self, matches: torch.Tensor, table: torch.Tensor
    ) -> torch.Tensor:
        mismatches = (self.l - matches).round().long().clamp(0, self.l)
        return table[mismatches].sum(dim=(-2, -1))

    def _apply_table_histogram(
        self, matches: torch.Tensor, table: torch.Tensor
    ) -> torch.Tensor:
        rounded = matches.round()
        result = torch.zeros(
            matches.shape[:-2], dtype=table.dtype, device=matches.device
        )
        for m in range(self.l + 1):
            w = table[m].item()
            if w == 0.0:
                continue
            count = (rounded == (self.l - m)).sum(dim=(-2, -1))
            result = result + w * count.to(table.dtype)
        return result

    def pairwise_from_windows(self, wx: torch.Tensor, wy: torch.Tensor) -> torch.Tensor:
        """Raw kernel from pre-extracted flat windows (no RC, no normalization).

        Useful for training where windows can be cached across iterations.

        Args:
            wx: [B, Wx, F] flat windows from query sequences.
            wy: [S, Wy, F] flat windows from support sequences.

        Returns:
            [B, S] raw kernel values.
        """
        matches = torch.einsum("bif,sjf->bsij", wx, wy)
        return self._apply_table(matches)

    def diagonal_from_windows(self, wx: torch.Tensor) -> torch.Tensor:
        """Raw self-kernel from pre-extracted flat windows (no RC, no normalization).

        Args:
            wx: [B, W, F] flat windows.

        Returns:
            [B] raw self-kernel values.
        """
        self_matches = torch.bmm(wx, wx.transpose(1, 2))
        return self._apply_table(self_matches)

    def _raw_pairwise(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        wx = self.flat_windows(x)
        wy = self.flat_windows(y)
        result = self.pairwise_from_windows(wx, wy)

        if self.include_rc:
            wy_rc = self.flat_windows(reverse_complement(y))
            result = result + self.pairwise_from_windows(wx, wy_rc)

        return result.to(x.dtype)

    def _raw_diagonal(self, x: torch.Tensor) -> torch.Tensor:
        wx = self.flat_windows(x)
        result = self.diagonal_from_windows(wx)

        if self.include_rc:
            wx_rc = self.flat_windows(reverse_complement(x))
            rc_matches = torch.bmm(wx, wx_rc.transpose(1, 2))
            result = result + self._apply_table(rc_matches)

        return result.to(x.dtype)
