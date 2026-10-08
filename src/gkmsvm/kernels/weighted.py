from __future__ import annotations

import numpy as np

from gkmsvm.backend import get_array_module, get_strides
from gkmsvm.codec import reverse_complement
from gkmsvm.kernels.base import GkmKernel


def _center_weights(l: int, M: int, H: float) -> np.ndarray:
    """Per-position weights for center-weighted kernels.

    Args:
        l: Window length.
        M: Width of the flat center region.
        H: Half-life for exponential decay from center.

    Returns:
        [l] array of position weights (1.0 at center, decaying to edges).
    """
    weights = np.zeros(l, dtype=np.float64)
    center = (l - 1) / 2.0
    half_M = M / 2.0
    for i in range(l):
        dist = abs(i - center) - half_M
        if dist <= 0:
            weights[i] = 1.0
        else:
            weights[i] = max(0.001, 2.0 ** (-dist / H))
    return weights


def _elementary_symmetric_k(weights: np.ndarray, k: int) -> float:
    """k-th elementary symmetric polynomial of weights via DP."""
    n = len(weights)
    dp = np.zeros(k + 1, dtype=np.float64)
    dp[0] = 1.0
    for i in range(n):
        for j in range(min(i + 1, k), 0, -1):
            dp[j] = dp[j] + weights[i] * dp[j - 1]
    return float(dp[k])


def _weighted_kernel_from_matches(
    matches: np.ndarray,
    pos_weights: np.ndarray,
    k: int,
) -> np.ndarray:
    """Weighted gapped k-mer kernel via elementary symmetric polynomial DP.

    Args:
        matches: [..., W1, W2, l] per-position match indicators.
        pos_weights: [l] position weights.
        k: Number of informative positions.

    Returns:
        [...] kernel values with window and position dims reduced.
    """
    xp = get_array_module(matches)
    weighted = matches * pos_weights
    l = matches.shape[-1]
    batch_shape = matches.shape[:-1]

    dp = xp.zeros((*batch_shape, k + 1), dtype=matches.dtype)
    dp[..., 0] = 1.0
    for p in range(l):
        v = weighted[..., p]
        for j in range(min(p + 1, k), 0, -1):
            dp[..., j] = dp[..., j] + v * dp[..., j - 1]

    return dp[..., k].sum(axis=(-2, -1))


class CenterWeightedGkmKernel(GkmKernel):
    """Center-weighted gapped k-mer kernel (-t 4 / wgkm).

    Positions near the center of the l-mer window contribute more to
    the kernel than positions at the edges.
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

    def _per_position_matches(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Per-position match indicators between all window pairs.

        Returns:
            [B, S, Wx, Wy, l] binary match indicators.
        """
        xp = get_array_module(x)
        B, C, Lx = x.shape
        S, _, Ly = y.shape
        Wx = Lx - self.l + 1
        Wy = Ly - self.l + 1

        # Sliding windows: [B, 4, Wx, l] -> [B, Wx, l, 4]
        sx = get_strides(x)
        wx = xp.lib.stride_tricks.as_strided(
            x, shape=(B, C, Wx, self.l), strides=(sx[0], sx[1], sx[2], sx[2])
        )
        wx = xp.ascontiguousarray(wx.transpose(0, 2, 3, 1))

        sy = get_strides(y)
        wy = xp.lib.stride_tricks.as_strided(
            y, shape=(S, C, Wy, self.l), strides=(sy[0], sy[1], sy[2], sy[2])
        )
        wy = xp.ascontiguousarray(wy.transpose(0, 2, 3, 1))

        return xp.einsum("bplc,sqlc->bspql", wx, wy)

    def _per_position_self_matches(self, x: np.ndarray) -> np.ndarray:
        """Compute per-position match indicators for self-kernel.

        Returns:
            [B, W, W, l] binary match indicators.
        """
        xp = get_array_module(x)
        B, C, L = x.shape
        W = L - self.l + 1

        sx = get_strides(x)
        wx = xp.lib.stride_tricks.as_strided(
            x, shape=(B, C, W, self.l), strides=(sx[0], sx[1], sx[2], sx[2])
        )
        wx = xp.ascontiguousarray(wx.transpose(0, 2, 3, 1))  # [B, W, l, 4]
        return xp.einsum("bplc,bqlc->bpql", wx, wx)

    def _raw_pairwise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xp = get_array_module(x)
        pw = xp.asarray(self._pos_weights).astype(x.dtype)
        matches = self._per_position_matches(x, y)
        result = _weighted_kernel_from_matches(matches, pw, self.k)

        if self.include_rc:
            y_rc = reverse_complement(y)
            matches_rc = self._per_position_matches(x, y_rc)
            result = result + _weighted_kernel_from_matches(matches_rc, pw, self.k)

        return result

    def _raw_diagonal(
        self, x: np.ndarray, *, chunk_size: int | None = None
    ) -> np.ndarray:
        xp = get_array_module(x)
        B = x.shape[0]
        if chunk_size is not None and chunk_size < B:
            chunks = []
            for start in range(0, B, chunk_size):
                end = min(start + chunk_size, B)
                chunks.append(self._raw_diagonal(x[start:end]))
            return xp.concatenate(chunks)

        pw = xp.asarray(self._pos_weights).astype(x.dtype)
        matches = self._per_position_self_matches(x)
        result = _weighted_kernel_from_matches(matches, pw, self.k)

        if self.include_rc:
            x_rc = reverse_complement(x)

            sx = get_strides(x)
            B, C, L = x.shape
            W = L - self.l + 1
            wx = xp.lib.stride_tricks.as_strided(
                x, shape=(B, C, W, self.l), strides=(sx[0], sx[1], sx[2], sx[2])
            )
            wx = xp.ascontiguousarray(wx.transpose(0, 2, 3, 1))

            sxr = get_strides(x_rc)
            wx_rc = xp.lib.stride_tricks.as_strided(
                x_rc, shape=(B, C, W, self.l), strides=(sxr[0], sxr[1], sxr[2], sxr[2])
            )
            wx_rc = xp.ascontiguousarray(wx_rc.transpose(0, 2, 3, 1))

            rc_matches = xp.einsum("bplc,bqlc->bpql", wx, wx_rc)
            result = result + _weighted_kernel_from_matches(rc_matches, pw, self.k)

        return result


class CenterWeightedRbfGkmKernel(GkmKernel):
    """Center-weighted RBF gapped k-mer kernel (-t 5 / wgkmrbf)."""

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
            l, k, M=M, H=H, normalize=True, include_rc=include_rc
        )

    def _raw_pairwise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xp = get_array_module(x)
        return xp.exp(self.gamma * (self._base.pairwise(x, y) - 1))

    def _raw_diagonal(
        self, x: np.ndarray, *, chunk_size: int | None = None
    ) -> np.ndarray:
        xp = get_array_module(x)
        return xp.ones(x.shape[0], dtype=x.dtype)
