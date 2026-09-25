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
    """

    def __init__(
        self, l: int, k: int, *, normalize: bool = True, include_rc: bool = True
    ):
        super().__init__(l, k, normalize=normalize, include_rc=include_rc)
        self._mismatch_table = torch.tensor(
            [comb(l - m, k) if l - m >= k else 0 for m in range(l + 1)],
            dtype=torch.float64,
        )

    def _windows(self, x: torch.Tensor) -> torch.Tensor:
        """Extract all length-l windows from one-hot sequences.

        Args:
            x: [B, 4, L] one-hot tensor.

        Returns:
            [B, W, 4, l] tensor where W = L - l + 1.
        """
        B, C, L = x.shape
        if L < self.l:
            raise ValueError(
                f"Sequence length {L} is shorter than window length {self.l}"
            )
        return x.unfold(2, self.l, 1).permute(0, 2, 1, 3)

    def _count_mismatches(
        self, wx: torch.Tensor, wy: torch.Tensor
    ) -> torch.Tensor:
        """Count per-position mismatches between all pairs of windows.

        Args:
            wx: [B, Wx, 4, l] windows from sequence x.
            wy: [S, Wy, 4, l] windows from sequence y.

        Returns:
            [B, S, Wx, Wy] integer tensor of mismatch counts.
        """
        # matches[b, s, wx, wy, pos] = 1 where bases match
        # wx: [B, Wx, 4, l] -> [B, 1, Wx, 1, 4, l]
        # wy: [S, Wy, 4, l] -> [1, S, 1, Wy, 4, l]
        matches = (
            wx[:, None, :, None, :, :] * wy[None, :, None, :, :, :]
        ).sum(dim=-2)  # [B, S, Wx, Wy, l]
        return self.l - matches.sum(dim=-1)  # [B, S, Wx, Wy]

    def _kernel_from_mismatches(self, mismatches: torch.Tensor) -> torch.Tensor:
        """Look up shared features from mismatch counts and sum.

        Args:
            mismatches: [B, S, Wx, Wy] integer mismatch counts.

        Returns:
            [B, S] kernel values.
        """
        dtype = torch.float32 if mismatches.device.type == "mps" else self._mismatch_table.dtype
        table = self._mismatch_table.to(device=mismatches.device, dtype=dtype)
        mismatches = mismatches.long().clamp(0, self.l)
        shared = table[mismatches]
        return shared.sum(dim=(-2, -1))

    def _raw_pairwise(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        wx = self._windows(x)
        wy = self._windows(y)
        mismatches = self._count_mismatches(wx, wy)
        result = self._kernel_from_mismatches(mismatches)

        if self.include_rc:
            y_rc = reverse_complement(y)
            wy_rc = self._windows(y_rc)
            mismatches_rc = self._count_mismatches(wx, wy_rc)
            result = result + self._kernel_from_mismatches(mismatches_rc)

        return result.to(x.dtype)

    def _raw_diagonal(self, x: torch.Tensor) -> torch.Tensor:
        wx = self._windows(x)
        B, W, C, l = wx.shape
        # Self-kernel: every window pair (i, i) has 0 mismatches
        zero_mismatch_contribution = self._mismatch_table[0]
        # Cross-window pairs within the same sequence
        mismatches = self._count_mismatches(wx, wx)  # [B, B, W, W]
        diag_mismatches = torch.diagonal(mismatches, dim1=0, dim2=1)  # [W, W, B]
        diag_mismatches = diag_mismatches.permute(2, 0, 1)  # [B, W, W]
        dtype = torch.float32 if x.device.type == "mps" else self._mismatch_table.dtype
        table = self._mismatch_table.to(device=x.device, dtype=dtype)
        shared = table[diag_mismatches.long().clamp(0, self.l)]
        result = shared.sum(dim=(-2, -1))

        if self.include_rc:
            x_rc = reverse_complement(x)
            wx_rc = self._windows(x_rc)
            mismatches_rc = self._count_mismatches(wx, wx_rc)
            diag_rc = torch.diagonal(mismatches_rc, dim1=0, dim2=1)
            diag_rc = diag_rc.permute(2, 0, 1)
            shared_rc = table[diag_rc.long().clamp(0, self.l)]
            result = result + shared_rc.sum(dim=(-2, -1))

        return result.to(x.dtype)
