from __future__ import annotations

import numpy as np

from gkmsvm.backend import float_dtype, get_array_module
from gkmsvm.kernels.base import GkmKernel
from gkmsvm.kernels.esttrunc import EstTruncGkmKernel


class RbfKernel(GkmKernel):
    """RBF on a normalized base kernel: K = exp(gamma * (K_norm - 1)).

    K_norm = K_raw(x, y) / sqrt(K_raw(x, x) * K_raw(y, y)) of the base kernel
    (lsgkm's -t 3 / -t 5 convention). Self-similarity is always 1, so the
    normalize flag of this kernel is a no-op.
    """

    def __init__(self, base: GkmKernel, gamma: float):
        super().__init__(base.l, base.k, normalize=True, include_rc=base.include_rc)
        self._base = base
        self.gamma = gamma

    @property
    def base(self) -> GkmKernel:
        return self._base

    def finish(self, k_norm):
        xp = get_array_module(k_norm)
        return xp.exp(self.gamma * (k_norm - 1))

    def _raw_pairwise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.finish(self._base.pairwise(x, y))

    def _raw_diagonal(
        self, x: np.ndarray, *, chunk_size: int | None = None
    ) -> np.ndarray:
        xp = get_array_module(x)
        return xp.ones(x.shape[0], dtype=float_dtype(x))


class RbfGkmKernel(RbfKernel):
    """RBF kernel on top of the estimated truncated gkm kernel (-t 3 / gkmrbf)."""

    def __init__(
        self,
        l: int,
        k: int,
        *,
        d: int = 3,
        gamma: float = 1.0,
        normalize: bool = True,
        include_rc: bool = True,
    ):
        super().__init__(
            EstTruncGkmKernel(l, k, d=d, normalize=True, include_rc=include_rc), gamma)
        self.d = d
