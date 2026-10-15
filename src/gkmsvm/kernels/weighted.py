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

    # Per-call cap on the per-position match tensor
    _MATCH_BYTES = 256 * 2**20

    # Use the masked-join kernel (kernels/masked.py) when the inputs have no N
    # positions; set False to force the per-position path
    use_masked = True

    def _join_plan(self):
        """Mask plan: every size-k mask, coefficient prod of its weights."""
        if getattr(self, "_plan_cache", None) is None:
            from gkmsvm.kernels.masked import weighted_mask_plan

            self._plan_cache = weighted_mask_plan(self.l, self.k, self._pos_weights)
        return self._plan_cache

    def _masked_plan(self):
        """_join_plan() for the pairwise kernel, unless use_masked is off."""
        return self._join_plan() if self.use_masked else None

    def base_index_windows(self, x: np.ndarray) -> np.ndarray:
        """[B, W, l] int8 base indices of sliding windows (as DirectGkmKernel)."""
        from gkmsvm.kernels.direct import DirectGkmKernel

        return DirectGkmKernel.base_index_windows(self, x)

    def _packed_exact(self, x) -> bool:
        """The packed path is exact only without N (all-zero) positions.

        Packing maps N to A, while this kernel's one-hot path lets N match
        nothing.
        """
        return not bool((x.max(axis=1) == 0).any())

    def _packed(self, x):
        from gkmsvm.backend import is_gpu, is_mlx, to_cpu
        from gkmsvm.kernels.direct import _pack_windows_cpu, _pack_windows_uint32

        xp = get_array_module(x)
        bw = self.base_index_windows(x)
        if is_gpu(bw) and not is_mlx(bw):
            return _pack_windows_uint32(bw, xp)
        return _pack_windows_cpu(np.ascontiguousarray(to_cpu(bw)))

    def _masked_pairwise(self, x, y):
        """Masked-join K_raw(x, y), or None when it doesn't apply."""
        from gkmsvm.backend import is_gpu, is_mlx
        from gkmsvm.kernels.direct import _MASKED_MIN_ROWS

        plan = self._masked_plan()
        if (plan is None or min(x.shape[0], y.shape[0]) < _MASKED_MIN_ROWS
                or not (self._packed_exact(x) and self._packed_exact(y))):
            return None
        bx, by = self._packed(x), self._packed(y)
        by_rc = self._packed(reverse_complement(y)) if self.include_rc else None
        if is_mlx(x):
            import mlx.core as mx

            from gkmsvm.kernels.masked import masked_pairwise_mlx

            return mx.array(masked_pairwise_mlx(bx, by, plan, by_rc).astype(np.float32))
        if is_gpu(bx):
            from gkmsvm.kernels.masked import masked_pairwise_gpu

            return masked_pairwise_gpu(bx, by, plan, by_rc)
        from gkmsvm.kernels.masked import masked_pairwise

        return masked_pairwise(bx, by, plan, by_rc)

    def _masked_diagonal(self, x):
        """Masked-join K_raw(x, x) (computed on CPU), or None when it doesn't apply."""
        from gkmsvm.backend import to_cpu
        from gkmsvm.kernels.masked import masked_self_kernel

        plan = self._masked_plan()
        if plan is None or not self._packed_exact(x):
            return None
        xc = to_cpu(x)
        brc = self._packed(reverse_complement(xc)) if self.include_rc else None
        return masked_self_kernel(self._packed(xc), plan, brc)

    def _raw_pairwise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xp = get_array_module(x)
        masked = self._masked_pairwise(x, y)
        if masked is not None:
            return masked.astype(x.dtype)
        pw = xp.asarray(self._pos_weights).astype(x.dtype)
        y_rc = reverse_complement(y) if self.include_rc else None
        # The [B, S, Wx, Wy, l] match tensor is chunked over S to fit
        # _MATCH_BYTES (one SMO column against 10k sequences would be ~74 GB)
        Wx, Wy = x.shape[2] - self.l + 1, y.shape[2] - self.l + 1
        per_col = x.shape[0] * Wx * Wy * self.l * x.dtype.itemsize
        step = max(1, self._MATCH_BYTES // max(per_col, 1))
        parts = []
        for s in range(0, y.shape[0], step):
            part = _weighted_kernel_from_matches(
                self._per_position_matches(x, y[s:s + step]), pw, self.k)
            if y_rc is not None:
                part = part + _weighted_kernel_from_matches(
                    self._per_position_matches(x, y_rc[s:s + step]), pw, self.k)
            parts.append(part)
        return parts[0] if len(parts) == 1 else xp.concatenate(parts, axis=1)

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

        masked = self._masked_diagonal(x)
        if masked is not None:
            return xp.asarray(masked).astype(x.dtype)
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
