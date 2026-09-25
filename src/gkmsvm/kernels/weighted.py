from __future__ import annotations

import torch

from gkmsvm.codec import reverse_complement
from gkmsvm.kernels.base import GkmKernel


def _center_weights(l: int, M: int, H: float) -> torch.Tensor:
    """Position weights for center-weighted kernels.

    Positions within M/2 of center get weight 1.0. Beyond that, weight
    decays as 2^(-distance/H), floored at 0.001.
    """
    weights = torch.zeros(l, dtype=torch.float64)
    center = (l - 1) / 2.0
    half_M = M / 2.0
    for i in range(l):
        dist = abs(i - center) - half_M
        if dist <= 0:
            weights[i] = 1.0
        else:
            weights[i] = max(0.001, 2.0 ** (-dist / H))
    return weights


def _elementary_symmetric_k(weights: torch.Tensor, k: int) -> float:
    """k-th elementary symmetric polynomial of weights via DP."""
    n = len(weights)
    dp = torch.zeros(k + 1, dtype=torch.float64)
    dp[0] = 1.0
    for i in range(n):
        for j in range(min(i + 1, k), 0, -1):
            dp[j] = dp[j] + weights[i] * dp[j - 1]
    return dp[k].item()


def _weighted_kernel_from_matches(
    matches: torch.Tensor,
    pos_weights: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Compute weighted gapped k-mer kernel from per-position matches.

    For each element in the batch dimensions, computes the k-th elementary
    symmetric polynomial of (w_p * match_p) for p in 0..l-1, then sums
    over all window pairs.

    Args:
        matches: [..., W1, W2, l] per-position match indicators.
        pos_weights: [l] position weights.
        k: number of informative positions.

    Returns:
        [...] tensor with window and position dimensions reduced.
    """
    weighted = matches * pos_weights
    l = matches.shape[-1]
    batch_shape = matches.shape[:-1]

    dp = matches.new_zeros(*batch_shape, k + 1)
    dp[..., 0] = 1.0
    for p in range(l):
        v = weighted[..., p]
        for j in range(min(p + 1, k), 0, -1):
            dp[..., j] = dp[..., j] + v * dp[..., j - 1]

    return dp[..., k].sum(dim=(-2, -1))


class CenterWeightedGkmKernel(GkmKernel):
    """Center-weighted gapped k-mer kernel (-t 4 / wgkm).

    Positions near the center of the l-mer window contribute more to
    the kernel than positions at the edges. Each gapped k-mer with
    informative positions {p1,...,pk} has weight prod(w[pi]).

    Uses per-position match computation with an elementary symmetric
    polynomial DP to compute exact position-weighted kernel values.

    Parameters M and H control the weight profile:
    - M: number of fully-weighted center positions
    - H: half-life for exponential decay beyond the center region
    """

    def __init__(
        self,
        l: int,
        k: int,
        *,
        M: int,
        H: float,
        normalize: bool = True,
        include_rc: bool = True,
    ):
        super().__init__(l, k, normalize=normalize, include_rc=include_rc)
        self.M = M
        self.H = H
        self._pos_weights = _center_weights(l, M, H)

    def _per_position_matches(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-position match indicators between all window pairs.

        Args:
            x: [B, 4, Lx] one-hot sequences.
            y: [S, 4, Ly] one-hot sequences.

        Returns:
            [B, S, Wx, Wy, l] binary match indicators.
        """
        wx = x.unfold(2, self.l, 1)  # [B, 4, Wx, l]
        wy = y.unfold(2, self.l, 1)  # [S, 4, Wy, l]
        wx = wx.permute(0, 2, 3, 1)  # [B, Wx, l, 4]
        wy = wy.permute(0, 2, 3, 1)  # [S, Wy, l, 4]
        matches = torch.einsum("bplc,sqlc->bspql", wx, wy)
        return matches

    def _per_position_self_matches(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-position match indicators for self-kernel.

        Args:
            x: [B, 4, L] one-hot sequences.

        Returns:
            [B, W, W, l] binary match indicators.
        """
        wx = x.unfold(2, self.l, 1).permute(0, 2, 3, 1)  # [B, W, l, 4]
        matches = torch.einsum("bplc,bqlc->bpql", wx, wx)
        return matches

    def _raw_pairwise(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pw = self._pos_weights.to(device=x.device, dtype=x.dtype)
        matches = self._per_position_matches(x, y)
        result = _weighted_kernel_from_matches(matches, pw, self.k)

        if self.include_rc:
            y_rc = reverse_complement(y)
            matches_rc = self._per_position_matches(x, y_rc)
            result = result + _weighted_kernel_from_matches(matches_rc, pw, self.k)

        return result

    def _raw_diagonal(self, x: torch.Tensor) -> torch.Tensor:
        pw = self._pos_weights.to(device=x.device, dtype=x.dtype)
        matches = self._per_position_self_matches(x)
        result = _weighted_kernel_from_matches(matches, pw, self.k)

        if self.include_rc:
            x_rc = reverse_complement(x)
            wx = x.unfold(2, self.l, 1).permute(0, 2, 3, 1)
            wx_rc = x_rc.unfold(2, self.l, 1).permute(0, 2, 3, 1)
            rc_matches = torch.einsum("bplc,bqlc->bpql", wx, wx_rc)
            result = result + _weighted_kernel_from_matches(rc_matches, pw, self.k)

        return result


class CenterWeightedRbfGkmKernel(GkmKernel):
    """Center-weighted RBF gapped k-mer kernel (-t 5 / wgkmrbf).

    Combines center-weighted position importance with RBF distance
    transformation. Uses the center-weighted kernel as the base for
    computing squared distances, then applies exp(-gamma * dist²).
    """

    def __init__(
        self,
        l: int,
        k: int,
        *,
        M: int,
        H: float,
        gamma: float = 1.0,
        normalize: bool = True,
        include_rc: bool = True,
    ):
        super().__init__(l, k, normalize=normalize, include_rc=include_rc)
        self.M = M
        self.H = H
        self.gamma = gamma
        self._base = CenterWeightedGkmKernel(
            l, k, M=M, H=H, normalize=False, include_rc=include_rc
        )

    def _raw_pairwise(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        K_xy = self._base._raw_pairwise(x, y)
        K_xx = self._base._raw_diagonal(x)
        K_yy = self._base._raw_diagonal(y)
        dist_sq = K_xx.unsqueeze(1) + K_yy.unsqueeze(0) - 2 * K_xy
        return torch.exp(-self.gamma * torch.clamp(dist_sq, min=0))

    def _raw_diagonal(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
