"""Column-cached SMO solver for gkm-SVM training."""

from __future__ import annotations

import numpy as np

from gkmsvm.backend import get_array_module, is_mlx, to_cpu
from gkmsvm.codec import reverse_complement
from gkmsvm.kernels.base import GkmKernel


class KernelColumnCache:
    """Normalized kernel columns K(i, :) for the C SMO solver.

    Pre-computes packed uint32 window representations for all training
    sequences once and computes columns on request. The solver keeps the
    columns in its own cache.
    """

    def __init__(self, kernel: GkmKernel, X: np.ndarray):
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

    def compute_columns(self, indices) -> dict:
        """Normalized columns {i: K(i, :)} for `indices`, in one kernel call (uncached).

        Uses symmetry, K(i, :) = K(:, i): all training sequences against the
        batch (and its reverse complements), so a masked-join kernel indexes
        only the small batch side.
        """
        indices = [int(i) for i in indices]
        if not indices:
            return {}
        if not self._use_fast or len(indices) == 1:
            return {i: self._compute_column(i) for i in indices}
        xp = self._xp
        kernel = self._kernel
        sel = xp.asarray(np.asarray(indices, dtype=np.int64))
        if self._mlx:
            import mlx.core as mx

            sel = mx.array(np.asarray(indices, dtype=np.int32))
        by = self._idx_windows[sel]
        bx_packed = self._packed_t if (xp is np and self._packed_t is not None) else None
        raw = kernel.pairwise_from_indices(self._idx_windows, by, bx_packed=bx_packed)
        if kernel.include_rc and self._rc_idx_windows is not None:
            raw = raw + kernel.pairwise_from_indices(
                self._idx_windows, self._rc_idx_windows[sel], bx_packed=bx_packed,
            )
        if self._mlx:
            raw = to_cpu(raw)
            xp = np
        raw = raw.astype(np.float64)  # [N, B]
        if kernel.normalize:
            d = xp.asarray(self._diag) if xp is not np else self._diag
            norm = xp.sqrt(d[:, None] * d[xp.asarray(np.asarray(indices, dtype=np.int64))][None, :])
            raw = raw / xp.clip(norm, 1e-10, None)
        return {i: xp.ascontiguousarray(raw[:, b]) for b, i in enumerate(indices)}

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
    cache_size: int | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, float]:
    """Column-cached SMO solver for C-SVM.

    Uses a C implementation of LIBSVM-style serial WSS3 SMO (Fan et al.
    2005) with shrinking.  Kernel columns are computed by
    KernelColumnCache using the fastest available backend (Numba CPU,
    CuPy GPU, or MLX).  Needs the ``_csmo`` C extension and a normalized
    kernel (the solver assumes K(i, i) = 1).

    Args:
        kernel: GkmKernel instance.
        X: [N, 4, L] one-hot encoded training sequences.
        y: [N] labels, +1 or -1.
        C: Regularization parameter.
        tol: KKT violation tolerance for convergence.
        max_iter: Maximum SMO iterations.
        cache_size: Number of kernel columns to cache; None sizes it from
            available RAM (all N columns when they fit in a quarter of it).
        verbose: Print progress.

    Returns:
        coefficients: [N] signed dual coefficients (alpha_i * y_i).
        bias: Decision boundary offset (score = K*coef + bias).
    """
    if not kernel.normalize:
        raise ValueError("smo_solve needs a normalized kernel")
    if _get_csmo() is None:
        raise RuntimeError(
            "The C SMO extension (gkmsvm._csmo) is not built; reinstall with a C "
            "compiler, or train with solver='libsvm'"
        )
    if cache_size is None:
        cache_size = auto_cache_columns(X.shape[0])
    return _smo_csolver(kernel, X, y, C, tol, max_iter, cache_size, verbose)


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
    _ct.POINTER(_ct.c_double), _ct.c_void_p, _ct.c_void_p,
)


# Most columns computed per kernel-cache miss in the C solver: the requested
# one plus the likeliest next working-set members, which go into free cache
# slots (or slots of unrequested prefetches), never over columns the solver
# asked for. Batches let the masked-join kernel amortize its index. 0 or 1
# disables prefetching.
_SMO_PREFETCH = 256
# Fraction of available RAM for the C solver's column cache when cache_size
# is None
_SMO_CACHE_RAM_FRACTION = 0.25


def auto_cache_columns(N: int) -> int:
    """Column-cache size from available RAM: all N columns if they fit."""
    from gkmsvm.train import _available_memory

    cols = int(_available_memory(np) * _SMO_CACHE_RAM_FRACTION // (8 * max(N, 1)))
    return max(min(256, N), min(N, cols))


def _prefetch_candidates(state, y, C, k, fresh):
    """Up to k indices WSS3 is likely to select next, among `fresh` ones.

    Working-set pairs come from the ends of -y*G: the largest values in
    I_up and the smallest in I_low. Take half from each. `fresh` excludes
    columns already cached or staged (the likeliest ones usually are).
    """
    N = y.shape[0]
    alpha, G = state[:N], state[N:2 * N]
    score = -y * G
    up = fresh & (((y > 0) & (alpha < C)) | ((y < 0) & (alpha > 0)))
    low = fresh & (((y > 0) & (alpha > 0)) | ((y < 0) & (alpha < C)))
    half = max(1, k // 2)
    out = []
    for mask, sign in ((up, -1.0), (low, 1.0)):
        cand = np.flatnonzero(mask)
        if cand.size == 0:
            continue
        vals = sign * score[cand]
        take = min(half, cand.size)
        top = cand[np.argpartition(vals, take - 1)[:take]]
        out.extend(int(t) for t in top)
    return list(dict.fromkeys(out))[:k]


def _smo_csolver(kernel, X, y, C, tol, max_iter, cache_size, verbose):
    """WSS3 SMO using the C solver with kernel column callback.

    The C solver handles the optimization loop (WSS3, shrinking, gradient
    updates) while kernel columns are computed by our KernelColumnCache
    (Numba CPU or CuPy GPU).
    """
    lib = _get_csmo()
    N = X.shape[0]
    xp = get_array_module(X)
    mlx_input = is_mlx(X)

    # Columns live in the C solver's cache; this object only computes them
    cache = KernelColumnCache(kernel, X)

    y_np = to_cpu(y).astype(np.float64) if mlx_input else np.asarray(
        to_cpu(y) if xp is not np else y, dtype=np.float64
    )

    # Written by the C solver: alpha in state[:N], gradient in state[N:2N],
    # 1.0 in state[2N:] for columns in its cache
    state = np.zeros(3 * N, dtype=np.float64)
    in_cache = state[2 * N:]
    max_batch = max(1, _SMO_PREFETCH) if cache._use_fast else 1
    lib.csmo_cache_room.restype = _ct.c_int
    lib.csmo_cache_room.argtypes = [_ct.c_void_p]
    lib.csmo_cache_insert.restype = _ct.c_int
    lib.csmo_cache_insert.argtypes = [_ct.c_void_p, _ct.c_int, _ct.c_void_p]
    # ctypes swallows exceptions raised in the callback and the solver would
    # carry on with an unwritten column, so record them and raise afterwards
    errors: list[BaseException] = []
    stats = {"batches": 0, "prefetched": 0}

    def _host(col):
        if xp is not np or mlx_input:
            col = to_cpu(col)
        return np.ascontiguousarray(col, dtype=np.float64)

    def _columns(idx, c_cache):
        """{idx: column, plus prefetched columns that fit in the cache}."""
        extra = []
        if max_batch > 1:
            room = lib.csmo_cache_room(c_cache)  # excludes idx's slot
            if room > 0:
                fresh = in_cache == 0
                fresh[idx] = False
                extra = _prefetch_candidates(state, y_np, C, min(max_batch - 1, room), fresh)
        try:
            return cache.compute_columns([idx, *extra])
        except Exception as exc:  # noqa: BLE001 - fall back to one column
            if not extra:
                raise
            import warnings

            warnings.warn(f"SMO column prefetch failed ({exc!r}); computing columns singly",
                          RuntimeWarning, stacklevel=2)
            return {idx: cache._compute_column(idx)}

    def _column_callback(idx, n, out_ptr, _userdata, c_cache):
        try:
            cols = _columns(idx, c_cache)
            col = _host(cols.pop(idx))
            if cols:
                stats["batches"] += 1
            for j, cj in cols.items():
                cj = _host(cj)
                stats["prefetched"] += lib.csmo_cache_insert(c_cache, j, cj.ctypes.data)
        except BaseException as exc:  # noqa: BLE001 - re-raised after the solver returns
            errors.append(exc)
            col = np.full(n, np.nan)
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
        _ct.c_void_p,
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
        state.ctypes.data_as(_ct.c_void_p),
    )
    if errors:
        raise RuntimeError("SMO kernel column computation failed") from errors[0]
    if verbose:
        print(f"  Column cache {cache_size} columns; {stats['prefetched']} columns "
              f"prefetched in {stats['batches']} batches")

    return alpha_out, float(rho_out[0])
