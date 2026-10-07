from __future__ import annotations

import numpy as np

from gkmsvm.backend import get_array_module
from gkmsvm.kernels.base import GkmKernel
from gkmsvm.kernels.esttrunc import EstTruncGkmKernel


class RbfGkmKernel(GkmKernel):
    """RBF kernel on top of the estimated truncated gkm kernel (-t 3 / gkmrbf).

    K_rbf(x, y) = exp(gamma * (K_norm(x, y) - 1))

    where K_norm = K_raw(x,y) / sqrt(K_raw(x,x) * K_raw(y,y)) is the
    normalized base kernel (matching lsgkm's -t 3 convention). Self-similarity
    is always 1, so the normalize flag on the outer kernel is a no-op.
    """

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
        super().__init__(l, k, normalize=normalize, include_rc=include_rc)
        self.gamma = gamma
        self.d = d
        self._base = EstTruncGkmKernel(
            l, k, d=d, normalize=True, include_rc=include_rc
        )

    def _raw_pairwise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xp = get_array_module(x)
        K_norm = self._base.pairwise(x, y)
        return xp.exp(self.gamma * (K_norm - 1))

    def _raw_diagonal(
        self, x: np.ndarray, *, chunk_size: int | None = None
    ) -> np.ndarray:
        xp = get_array_module(x)
        return xp.ones(x.shape[0], dtype=x.dtype)
