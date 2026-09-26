"""Column-cached SMO solver for gkm-SVM training."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from gkmsvm.backend import get_array_module
from gkmsvm.codec import reverse_complement
from gkmsvm.kernels.base import GkmKernel


class KernelColumnCache:
    """LRU cache for normalized kernel columns K(i, :).

    Pre-computes packed uint32 window representations for all training
    sequences once. Individual kernel columns are computed on demand
    and cached with LRU eviction.

    Memory per column: N x 8 bytes. For N=80K and 256 columns = ~160 MB.
    """

    def __init__(self, kernel: GkmKernel, X: np.ndarray, max_columns: int = 256):
        xp = get_array_module(X)
        self._kernel = kernel
        self._xp = xp
        self._N = X.shape[0]
        self._use_fast = hasattr(kernel, "pairwise_from_indices")

        if self._use_fast:
            self._idx_windows = kernel.base_index_windows(X)
            self._rc_idx_windows = (
                kernel.base_index_windows(reverse_complement(X))
                if kernel.include_rc
                else None
            )
            self._packed_t = None
            self._rc_packed_t = None
            if hasattr(kernel, "_min_matches"):
                from gkmsvm.kernels.direct import (
                    _pack_windows_cpu,
                    _pack_windows_uint32,
                )

                if xp is not np:
                    packed = _pack_windows_uint32(self._idx_windows, xp)
                    self._packed_t = xp.ascontiguousarray(packed.T)
                    if self._rc_idx_windows is not None:
                        rc_packed = _pack_windows_uint32(
                            self._rc_idx_windows, xp
                        )
                        self._rc_packed_t = xp.ascontiguousarray(rc_packed.T)
                else:
                    self._packed_t = _pack_windows_cpu(
                        np.ascontiguousarray(self._idx_windows)
                    )
                    if self._rc_idx_windows is not None:
                        self._rc_packed_t = _pack_windows_cpu(
                            np.ascontiguousarray(self._rc_idx_windows)
                        )
            self._X = None
        else:
            self._X = X
            self._idx_windows = None
            self._rc_idx_windows = None
            self._packed_t = None
            self._rc_packed_t = None

        self._diag = kernel._raw_diagonal(X).astype(np.float64)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._max = max_columns
        self.hits = 0
        self.misses = 0

    def get_column(self, i: int) -> np.ndarray:
        """Return normalized kernel column K(i, :). Compute on cache miss."""
        if i in self._cache:
            self._cache.move_to_end(i)
            self.hits += 1
            return self._cache[i]

        col = self._compute_column(i)
        self._cache[i] = col
        self.misses += 1
        if len(self._cache) > self._max:
            self._cache.popitem(last=False)
        return col

    def _compute_column(self, i: int) -> np.ndarray:
        """Compute normalized K(i, :) using the fastest available path."""
        xp = self._xp
        kernel = self._kernel

        if self._use_fast:
            bx_i = self._idx_windows[i : i + 1]
            raw = kernel.pairwise_from_indices(
                bx_i, self._idx_windows, by_packed_t=self._packed_t
            )
            if kernel.include_rc and self._rc_idx_windows is not None:
                raw = raw + kernel.pairwise_from_indices(
                    bx_i,
                    self._rc_idx_windows,
                    by_packed_t=self._rc_packed_t,
                )
        else:
            raw = kernel._raw_pairwise(self._X[i : i + 1], self._X)

        raw = raw.astype(np.float64)

        if kernel.normalize:
            norm = xp.sqrt(self._diag[i] * self._diag)
            raw = raw / xp.clip(norm, 1e-10, None)

        return raw[0]


def smo_solve(
    kernel: GkmKernel,
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    tol: float = 1e-3,
    max_iter: int = 10_000_000,
    cache_size: int = 256,
    verbose: bool = False,
) -> tuple[np.ndarray, float]:
    """Column-cached SMO solver for C-SVM.

    Uses maximal violating pair working set selection (WSS1) with
    LRU-cached kernel column evaluation. Kernel columns use the
    packed uint32 path when available.

    Args:
        kernel: GkmKernel instance.
        X: [N, 4, L] one-hot encoded training sequences.
        y: [N] labels, +1 or -1.
        C: Regularization parameter.
        tol: KKT violation tolerance for convergence.
        max_iter: Maximum SMO iterations.
        cache_size: Number of kernel columns to cache.
        verbose: Print progress every 1000 iterations.

    Returns:
        coefficients: [N] signed dual coefficients (alpha_i * y_i).
        bias: Decision boundary offset (score = K*coef + bias).
    """
    xp = get_array_module(X)
    N = X.shape[0]
    y = xp.asarray(y, dtype=np.float64)

    cache = KernelColumnCache(kernel, X, max_columns=cache_size)

    alpha = xp.zeros(N, dtype=np.float64)
    G = -xp.ones(N, dtype=np.float64)

    Q_diag = (
        xp.ones(N, dtype=np.float64)
        if kernel.normalize
        else cache._diag.copy()
    )

    gap = float("inf")

    for iteration in range(max_iter):
        I_up = ((y > 0) & (alpha < C)) | ((y < 0) & (alpha > 0))
        I_low = ((y > 0) & (alpha > 0)) | ((y < 0) & (alpha < C))

        neg_yG = -y * G

        if not xp.any(I_up) or not xp.any(I_low):
            break

        up_vals = xp.where(I_up, neg_yG, -np.inf)
        low_vals = xp.where(I_low, neg_yG, np.inf)

        i = int(xp.argmax(up_vals))
        j = int(xp.argmin(low_vals))

        m_val = float(neg_yG[i])
        M_val = float(neg_yG[j])
        gap = m_val - M_val

        if verbose and iteration % 1000 == 0:
            n_sv = int(xp.sum(alpha > 1e-10))
            total = cache.hits + cache.misses
            hr = cache.hits / max(1, total) * 100
            print(
                f"  iter {iteration:>8d}  gap={gap:.4e}  "
                f"SVs={n_sv}  cache hit={hr:.0f}%"
            )

        if gap < tol:
            break

        K_col_i = cache.get_column(i)
        K_col_j = cache.get_column(j)

        K_ij = float(K_col_i[j])
        a = float(Q_diag[i]) + float(Q_diag[j]) - 2.0 * K_ij
        if a <= 0:
            a = 1e-12

        s = float(y[i] * y[j])
        b = -s * float(G[i]) + float(G[j])
        d_j = -b / a

        alpha_i = float(alpha[i])
        alpha_j = float(alpha[j])

        if s > 0:
            lo = max(-alpha_j, alpha_i - C)
            hi = min(C - alpha_j, alpha_i)
        else:
            lo = max(-alpha_j, -alpha_i)
            hi = min(C - alpha_j, C - alpha_i)

        d_j = max(lo, min(hi, d_j))
        d_i = -s * d_j

        alpha[i] += d_i
        alpha[j] += d_j

        G += d_i * y[i] * y * K_col_i + d_j * y[j] * y * K_col_j

    if verbose:
        n_sv = int(xp.sum(alpha > 1e-10))
        total = cache.hits + cache.misses
        hr = cache.hits / max(1, total) * 100
        print(
            f"  SMO done: {iteration + 1} iters, {n_sv} SVs, "
            f"gap={gap:.2e}, cache hit={hr:.1f}%"
        )

    free = (alpha > 1e-10) & (alpha < C - 1e-10)
    if xp.any(free):
        rho = float(xp.mean(y[free] * G[free]))
    else:
        rho = -(m_val + M_val) / 2.0

    return (alpha * y).astype(np.float64), float(-rho)
