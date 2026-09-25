from __future__ import annotations

import numpy as np

from gkmsvm.backend import get_array_module
from gkmsvm.kernels.base import GkmKernel
from gkmsvm.kernels.esttrunc import EstTruncGkmKernel


class RbfGkmKernel(GkmKernel):
    """RBF kernel on top of the estimated truncated gkm kernel (-t 3 / gkmrbf).

    K_rbf(x, y) = exp(-gamma * (K_raw(x,x) + K_raw(y,y) - 2*K_raw(x,y)))

    where K_raw is the unnormalized gkm_esttrunc kernel. Self-similarity is
    always 1 (distance to self is zero), so the normalize flag is a no-op.
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
            l, k, d=d, normalize=False, include_rc=include_rc
        )

    def _raw_pairwise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xp = get_array_module(x)
        K_xy = self._base._raw_pairwise(x, y)
        K_xx = self._base._raw_diagonal(x)
        cs = 1000 if y.shape[0] > self._DIAG_CHUNK_THRESHOLD else None
        K_yy = self._base._raw_diagonal(y, chunk_size=cs)
        dist_sq = K_xx[:, None] + K_yy[None, :] - 2 * K_xy
        return xp.exp(-self.gamma * xp.clip(dist_sq, 0, None))

    def _raw_diagonal(
        self, x: np.ndarray, *, chunk_size: int | None = None
    ) -> np.ndarray:
        xp = get_array_module(x)
        return xp.ones(x.shape[0], dtype=x.dtype)
