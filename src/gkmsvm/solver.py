"""Column-cached SMO solver for gkm-SVM training."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from gkmsvm.backend import get_array_module, is_mlx, to_cpu
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

                if is_mlx(self._idx_windows):
                    self._packed_t = xp.asarray(_pack_windows_cpu(
                        np.ascontiguousarray(to_cpu(self._idx_windows))
                    ))
                    if self._rc_idx_windows is not None:
                        self._rc_packed_t = xp.asarray(_pack_windows_cpu(
                            np.ascontiguousarray(to_cpu(self._rc_idx_windows))
                        ))
                elif xp is not np:
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

        self._mlx = is_mlx(X)
        diag = kernel._raw_diagonal(X)
        if self._mlx:
            diag = to_cpu(diag)
        self._diag = diag.astype(np.float64)
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

    def _compute_column(self, i: int):
        """Compute normalized K(i, :) using the fastest available path.

        Returns numpy for MLX inputs, stays on device for CuPy.
        """
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

        if self._mlx:
            raw = to_cpu(raw)
            xp = np
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
    cache_size: int = 2048,
    verbose: bool = False,
) -> tuple[np.ndarray, float]:
    """Column-cached SMO solver for C-SVM.

    Uses a C implementation of LIBSVM-style serial WSS3 SMO (Fan et al.
    2005) with shrinking.  Kernel columns are computed by
    KernelColumnCache using the fastest available backend (Numba CPU,
    CuPy GPU, or MLX).  Falls back to a pure-Python serial solver when
    the C extension is not available or the kernel is not normalized.

    Args:
        kernel: GkmKernel instance.
        X: [N, 4, L] one-hot encoded training sequences.
        y: [N] labels, +1 or -1.
        C: Regularization parameter.
        tol: KKT violation tolerance for convergence.
        max_iter: Maximum SMO iterations.
        cache_size: Number of kernel columns to cache.
        verbose: Print progress.

    Returns:
        coefficients: [N] signed dual coefficients (alpha_i * y_i).
        bias: Decision boundary offset (score = K*coef + bias).
    """
    if _get_csmo() is not None and kernel.normalize:
        return _smo_csolver(
            kernel, X, y, C, tol, max_iter, cache_size, verbose,
        )

    mlx_input = is_mlx(X)
    xp = np if mlx_input else get_array_module(X)
    return _smo_serial(
        kernel, X, y, C, tol, max_iter, cache_size, verbose, xp, mlx_input,
    )


# ---------------------------------------------------------------------------
# Serial SMO (CPU / MLX / fallback when C extension unavailable)
# ---------------------------------------------------------------------------


def _smo_serial(kernel, X, y, C, tol, max_iter, cache_size, verbose, xp,
                mlx_input):
    """Serial WSS2 SMO with shrinking."""
    N = X.shape[0]
    y = xp.asarray(to_cpu(y) if mlx_input else y, dtype=np.float64)

    cache = KernelColumnCache(kernel, X, max_columns=cache_size)

    alpha = xp.zeros(N, dtype=np.float64)
    G = -xp.ones(N, dtype=np.float64)

    Q_diag = (
        xp.ones(N, dtype=np.float64)
        if kernel.normalize
        else cache._diag.copy()
    )

    import time as _time

    active = xp.ones(N, dtype=bool)
    n_active = N
    shrink_interval = max(N, 1000)
    unshrink_needed = False

    gap = float("inf")
    m_val = float("inf")
    M_val = float("-inf")
    t_start = _time.monotonic()

    for iteration in range(max_iter):
        I_up = active & (((y > 0) & (alpha < C)) | ((y < 0) & (alpha > 0)))
        I_low = active & (((y > 0) & (alpha > 0)) | ((y < 0) & (alpha < C)))

        neg_yG = -y * G

        if not xp.any(I_up) or not xp.any(I_low):
            if unshrink_needed:
                active[:] = True
                n_active = N
                unshrink_needed = False
                continue
            break

        up_vals = xp.where(I_up, neg_yG, -np.inf)
        low_vals = xp.where(I_low, neg_yG, np.inf)

        i = int(xp.argmax(up_vals))
        m_val = float(neg_yG[i])
        M_val = float(xp.min(low_vals))
        gap = m_val - M_val

        if verbose and iteration % 1000 == 0:
            n_sv = int(xp.sum(alpha > 1e-10))
            total = cache.hits + cache.misses
            hr = cache.hits / max(1, total) * 100
            elapsed = _time.monotonic() - t_start
            print(
                f"  iter {iteration:>8d}  gap={gap:.4e}  "
                f"SVs={n_sv}  active={n_active}/{N}  "
                f"cache hit={hr:.0f}%  {elapsed:.1f}s"
            )

        if gap < tol:
            if unshrink_needed:
                active[:] = True
                n_active = N
                unshrink_needed = False
                continue
            break

        # Shrinking
        if iteration > 0 and iteration % shrink_interval == 0:
            shrunk = _shrink(alpha, neg_yG, active, y, C, m_val, M_val, xp)
            if shrunk > 0:
                n_active = int(xp.sum(active))
                unshrink_needed = True
                if verbose:
                    print(f"  shrink: removed {shrunk}, active={n_active}/{N}")
                continue

        # WSS2: select j using second-order information
        K_col_i = cache.get_column(i)

        candidates = I_low & (neg_yG < m_val)
        b_sq = (m_val - neg_yG) ** 2
        a_wss = Q_diag[i] + Q_diag - 2.0 * K_col_i
        a_wss = xp.maximum(a_wss, 1e-12)
        gain = xp.where(candidates, b_sq / a_wss, -np.inf)
        j = int(xp.argmax(gain))

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
        elapsed = _time.monotonic() - t_start
        print(
            f"  SMO done: {iteration + 1} iters, {n_sv} SVs, "
            f"gap={gap:.2e}, cache hit={hr:.1f}%, {elapsed:.1f}s"
        )

    free = (alpha > 1e-10) & (alpha < C - 1e-10)
    if xp.any(free):
        rho = float(xp.mean(y[free] * G[free]))
    else:
        rho = -(m_val + M_val) / 2.0

    return (alpha * y).astype(np.float64), float(-rho)


def _shrink(alpha, neg_yG, active, y, C, m_val, M_val, xp):
    """Remove bounded variables unlikely to change from the active set."""
    at_zero = alpha < 1e-10
    at_C = alpha > C - 1e-10

    shrink_zero_pos = at_zero & (y > 0) & (neg_yG < M_val) & active
    shrink_zero_neg = at_zero & (y < 0) & (neg_yG > m_val) & active
    shrink_C_pos = at_C & (y > 0) & (neg_yG > m_val) & active
    shrink_C_neg = at_C & (y < 0) & (neg_yG < M_val) & active

    to_shrink = shrink_zero_pos | shrink_zero_neg | shrink_C_pos | shrink_C_neg
    n_shrunk = int(xp.sum(to_shrink))
    if n_shrunk > 0:
        active[to_shrink] = False
    return n_shrunk


# ---------------------------------------------------------------------------
# C solver with kernel column callback (LIBSVM-style architecture)
# ---------------------------------------------------------------------------

def _load_csmo():
    """Load the compiled C SMO solver. Returns None if unavailable."""
    import ctypes

    try:
        import gkmsvm._csmo as _csmo_mod
        return ctypes.CDLL(_csmo_mod.__file__)
    except (ImportError, OSError):
        return None


_csmo_lib = None
_csmo_checked = False


def _get_csmo():
    global _csmo_lib, _csmo_checked
    if not _csmo_checked:
        _csmo_lib = _load_csmo()
        _csmo_checked = True
    return _csmo_lib


import ctypes as _ct

_COLUMN_CB = _ct.CFUNCTYPE(
    None, _ct.c_int, _ct.c_int,
    _ct.POINTER(_ct.c_double), _ct.c_void_p,
)


def _smo_csolver(kernel, X, y, C, tol, max_iter, cache_size, verbose):
    """WSS3 SMO using the C solver with kernel column callback.

    The C solver handles the optimization loop (WSS3, shrinking, gradient
    updates) while kernel columns are computed by our KernelColumnCache
    (Numba CPU or CuPy GPU).
    """
    lib = _get_csmo()
    if lib is None:
        raise RuntimeError("C SMO solver not available")

    N = X.shape[0]
    xp = get_array_module(X)
    mlx_input = is_mlx(X)

    cache = KernelColumnCache(kernel, X, max_columns=cache_size)

    y_np = to_cpu(y).astype(np.float64) if mlx_input else np.asarray(
        to_cpu(y) if xp is not np else y, dtype=np.float64
    )

    def _column_callback(idx, n, out_ptr, _userdata):
        col = cache.get_column(idx)
        if xp is not np or mlx_input:
            col = to_cpu(col)
        col = np.ascontiguousarray(col, dtype=np.float64)
        _ct.memmove(out_ptr, col.ctypes.data, n * 8)

    cb = _COLUMN_CB(_column_callback)

    alpha_out = np.zeros(N, dtype=np.float64)
    rho_out = np.zeros(1, dtype=np.float64)

    lib.csmo_solve.restype = _ct.c_int
    lib.csmo_solve.argtypes = [
        _ct.c_int,
        _ct.POINTER(_ct.c_double),
        _ct.c_double,
        _ct.c_double,
        _ct.c_int,
        _ct.c_int,
        _COLUMN_CB,
        _ct.c_void_p,
        _ct.POINTER(_ct.c_double),
        _ct.POINTER(_ct.c_double),
        _ct.c_int,
    ]

    n_iter = lib.csmo_solve(
        N,
        y_np.ctypes.data_as(_ct.POINTER(_ct.c_double)),
        C,
        tol,
        max_iter,
        min(cache_size, N),
        cb,
        None,
        alpha_out.ctypes.data_as(_ct.POINTER(_ct.c_double)),
        rho_out.ctypes.data_as(_ct.POINTER(_ct.c_double)),
        int(verbose),
    )

    if verbose:
        total = cache.hits + cache.misses
        hr = cache.hits / max(1, total) * 100
        print(
            f"  Python-side cache: {cache.hits} hits, {cache.misses} misses "
            f"({hr:.0f}% hit rate)"
        )

    return alpha_out, float(rho_out[0])
