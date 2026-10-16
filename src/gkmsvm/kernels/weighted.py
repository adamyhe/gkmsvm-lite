from __future__ import annotations

import numpy as np
from numba import njit, prange

from gkmsvm.backend import get_array_module, get_strides, to_cpu
from gkmsvm.codec import reverse_complement
from gkmsvm.kernels.packed import PackedGkmKernel
from gkmsvm.kernels.rbf import RbfKernel


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


@njit(parallel=True, cache=True, fastmath=True)
def _brute_nmask_numba(bx, by, by_n, g, l):
    """[B, S]: sum over window pairs of g[agreement bits], N positions agreeing with nothing.

    bx [B, Wx] packed query windows; by, by_n [S, Wy] packed column windows and
    their N masks (0b11 at N positions); g [2^l] the pair weight of each set
    of agreeing positions.
    """
    B, Wx = bx.shape
    S, Wy = by.shape
    even = np.uint32((1 << (2 * l)) - 1) & np.uint32(0x55555555)
    out = np.zeros((B, S))
    for item in prange(B * S):
        b, s = item // S, item % S
        acc = 0.0
        for i in range(Wx):
            u = bx[b, i]
            for j in range(Wy):
                x = u ^ by[s, j]
                agree = ~((x | (x >> np.uint32(1))) | by_n[s, j]) & even
                idx = 0
                for p in range(l):
                    idx |= ((agree >> np.uint32(2 * p)) & np.uint32(1)) << p
                acc += g[idx]
        out[b, s] = acc
    return out


def _agreement_weights(pos_weights: np.ndarray, k: int) -> np.ndarray:
    """[2^l]: e_k of the position weights over each set of agreeing positions."""
    l = len(pos_weights)
    bits = (np.arange(2**l)[:, None] >> np.arange(l)) & 1
    dp = np.zeros((2**l, k + 1))
    dp[:, 0] = 1.0
    for p in range(l):
        v = bits[:, p] * pos_weights[p]
        for j in range(min(p + 1, k), 0, -1):
            dp[:, j] += v * dp[:, j - 1]
    return dp[:, k]


def _on_device(out: np.ndarray, like):
    """CPU result out on like's device (float32 on MLX)."""
    from gkmsvm.kernels.packed import device_of

    dev = device_of(like)
    if dev == "mlx":
        import mlx.core as mx

        return mx.array(out.astype(np.float32))
    return out if dev == "cpu" else get_array_module(like).asarray(out)


class CenterWeightedGkmKernel(PackedGkmKernel):
    """Center-weighted gapped k-mer kernel (-t 4 / wgkm).

    Positions near the center of the l-mer window contribute more to
    the kernel than positions at the edges. The packed path is the masked
    join with per-mask weights; inputs with N positions use per-position
    match indicators.
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

    def _onehot_windows(self, x: np.ndarray) -> np.ndarray:
        """[B, W, l, 4] one-hot windows."""
        xp = get_array_module(x)
        B, C, L = x.shape
        W = L - self.l + 1
        sx = get_strides(x)
        wx = xp.lib.stride_tricks.as_strided(
            x, shape=(B, C, W, self.l), strides=(sx[0], sx[1], sx[2], sx[2])
        )
        return xp.ascontiguousarray(wx.transpose(0, 2, 3, 1))

    def _per_position_matches(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """[B, S, Wx, Wy, l] per-position match indicators between all window pairs."""
        xp = get_array_module(x)
        return xp.einsum("bplc,sqlc->bspql", self._onehot_windows(x), self._onehot_windows(y))

    # Per-call cap on the per-position match tensor
    _MATCH_BYTES = 256 * 2**20

    def _join_plan(self):
        """Mask plan: every size-k mask, coefficient prod of its weights."""
        if getattr(self, "_plan_cache", None) is None:
            from gkmsvm.kernels.masked import weighted_mask_plan

            self._plan_cache = weighted_mask_plan(self.l, self.k, self._pos_weights)
        return self._plan_cache

    def packs(self, x) -> bool:
        """The packed path needs the masked join and no N (all-zero) positions.

        Packing maps N to A, while this kernel's one-hot path lets N match
        nothing.
        """
        return self.use_masked and not bool((x.max(axis=1) == 0).any())

    def pack_nmask(self, x):
        """[B, W] uint32 N masks of x's packed windows (CPU): 0b11 at each N position."""
        n = (to_cpu(x).max(axis=1) == 0).astype(np.uint32)  # [B, L]
        W = n.shape[1] - self.l + 1
        out = np.zeros((n.shape[0], W), dtype=np.uint32)
        for k in range(self.l):
            out |= (n[:, k:k + W] * np.uint32(3)) << np.uint32(2 * k)
        return out

    def raw_packed_n(self, bx, by, by_n, by_rc=None, by_rc_n=None):
        """K_raw [B, S] of packed query rows bx (no N) against columns with N.

        by, by_n: the columns' packed windows (N packed as A) and N masks
        (pack_nmask); by_rc, by_rc_n the same for their reverse complements.
        A brute-force CPU pair loop, for the few SVs that contain N; the result
        is on bx's device.
        """
        if getattr(self, "_agree_g", None) is None:
            self._agree_g = _agreement_weights(self._pos_weights, self.k)
        q = np.ascontiguousarray(to_cpu(bx))
        out = _brute_nmask_numba(q, np.ascontiguousarray(by), np.ascontiguousarray(by_n),
                                 self._agree_g, self.l)
        if by_rc is not None:
            out += _brute_nmask_numba(q, np.ascontiguousarray(by_rc),
                                      np.ascontiguousarray(by_rc_n), self._agree_g, self.l)
        return _on_device(out, bx)

    def _pattern_weights(self) -> np.ndarray:
        """[2^l] pair weight by mismatch pattern: e_k over the agreeing positions."""
        if getattr(self, "_pattern_g", None) is None:
            if getattr(self, "_agree_g", None) is None:
                self._agree_g = _agreement_weights(self._pos_weights, self.k)
            self._pattern_g = self._agree_g[~np.arange(2**self.l) & (2**self.l - 1)]
        return self._pattern_g

    def raw_diag_packed(self, bx, bx_rc=None):
        """K_raw(x, x): each pair of the symmetric forward and RC blocks once, by
        mismatch pattern (direct._self_diagonal_packed_numba). On CPU; the
        result is on bx's device."""
        from gkmsvm.kernels.direct import _self_diagonal_packed_numba

        fwd = np.ascontiguousarray(to_cpu(bx))
        by = fwd if bx_rc is None else np.concatenate([fwd, to_cpu(bx_rc)], axis=1)
        return _on_device(_self_diagonal_packed_numba(
            fwd, np.ascontiguousarray(by), self._pattern_weights(), self.l, True), bx)

    def raw_diag_mutants(self, bx, bx_rc, new, new_rc, diag):
        """See PackedGkmKernel.raw_diag_mutants: only pairs touching a changed
        window, by mismatch pattern. On CPU; the result is on bx's device."""
        from gkmsvm.kernels.direct import _mutant_diag_numba

        B = bx.shape[0]
        rc = (np.zeros((B, 0), np.uint32) if bx_rc is None
              else np.ascontiguousarray(to_cpu(bx_rc)))
        new_rc = np.zeros((4, 1, 1, 1), np.uint32) if new_rc is None else new_rc
        return _on_device(_mutant_diag_numba(
            np.ascontiguousarray(to_cpu(bx)), rc, np.ascontiguousarray(new),
            np.ascontiguousarray(new_rc), np.asarray(to_cpu(diag), dtype=np.float64),
            self._pattern_weights(), self.l, True), bx)

    def _raw_pairwise_onehot(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xp = get_array_module(x)
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

    def _raw_diagonal_onehot(self, x: np.ndarray) -> np.ndarray:
        xp = get_array_module(x)
        pw = xp.asarray(self._pos_weights).astype(x.dtype)
        wx = self._onehot_windows(x)
        result = _weighted_kernel_from_matches(
            xp.einsum("bplc,bqlc->bpql", wx, wx), pw, self.k)
        if self.include_rc:
            wx_rc = self._onehot_windows(reverse_complement(x))
            result = result + _weighted_kernel_from_matches(
                xp.einsum("bplc,bqlc->bpql", wx, wx_rc), pw, self.k)
        return result


class CenterWeightedRbfGkmKernel(RbfKernel):
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
        super().__init__(
            CenterWeightedGkmKernel(l, k, M=M, H=H, normalize=True, include_rc=include_rc),
            gamma,
        )
        self.M = M
        self.H = H
