"""Column-cached SMO solver for gkm-SVM training."""

from __future__ import annotations

import numpy as np

from gkmsvm.backend import float_dtype, get_array_module, is_mlx, to_cpu
from gkmsvm.kernels.base import GkmKernel
from gkmsvm.kernels.packed import PackedGkmKernel


class KernelColumnCache:
    """Normalized kernel columns K(i, :) for the C SMO solver.

    Packs all training sequences once (through the base kernel for -t 3/-t 5)
    and computes columns on request. The solver keeps the columns in its own
    cache. When the l-mer index applies (kernels/lmer_index.py), it indexes
    all training sequences once, so one column costs one query row; else
    columns go through raw_packed, which needs batches to amortize the
    masked join.
    """

    def __init__(self, kernel: GkmKernel, X: np.ndarray):
        self._kernel = kernel
        self._xp = get_array_module(X)
        self._N = X.shape[0]
        self._mlx = is_mlx(X)
        base = kernel.base
        self._use_fast = isinstance(base, PackedGkmKernel)
        if self._use_fast:
            self._packed, self._packed_rc = base.pack_both(X)
            self._X = None
            diag = base.raw_diag_packed(self._packed, self._packed_rc).astype(
                float_dtype(X)
            )
            from gkmsvm.kernels import lmer_index

            self._index = (
                lmer_index.LmerIndex(base, self._packed, self._packed_rc)
                if lmer_index.applies(base)
                else None
            )
        else:
            self._X = X
            self._index = None
            diag = base._raw_diagonal(X)
        self._diag = to_cpu(diag).astype(np.float64)
        # The diagonal where the columns are normalized (CuPy: on the GPU)
        self._diag_xp = (
            self._diag if self._xp is np or self._mlx else self._xp.asarray(self._diag)
        )

    def compute_columns(self, indices) -> dict:
        """Normalized columns {i: K(i, :)} for `indices`, in one kernel call (uncached).

        Uses symmetry, K(i, :) = K(:, i): all training sequences against the
        batch and its reverse complements, so a masked-join kernel indexes
        only the small batch side.
        """
        indices = [int(i) for i in indices]
        if not indices:
            return {}
        xp = self._xp
        kernel = self._kernel
        base = kernel.base
        idx = np.asarray(indices, dtype=np.int64)
        if not self._use_fast:
            cols = kernel.pairwise(self._X, self._X[xp.asarray(idx)])
            cols = to_cpu(cols) if self._mlx else cols
            return {
                i: xp.ascontiguousarray(cols[:, b])
                if not self._mlx
                else np.ascontiguousarray(cols[:, b])
                for b, i in enumerate(indices)
            }
        if self._mlx:
            import mlx.core as mx

            sel = mx.array(idx.astype(np.int32))
        else:
            sel = xp.asarray(idx)
        if self._index is not None:
            raw = self._index.pairwise(self._packed[sel]).T  # K symmetric
        else:
            rc = None if self._packed_rc is None else self._packed_rc[sel]
            raw = base.raw_packed(self._packed, self._packed[sel], rc)  # [N, B]
        if self._mlx:
            raw = to_cpu(raw)
            xp = np
        raw = raw.astype(np.float64)
        if base.normalize:
            d = self._diag if xp is np else self._diag_xp
            norm = xp.sqrt(d[:, None] * d[xp.asarray(idx)][None, :])
            raw = raw / xp.clip(norm, 1e-10, None)
        raw = kernel.finish(raw)
        return {i: xp.ascontiguousarray(raw[:, b]) for b, i in enumerate(indices)}


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
    None, _ct.c_int, _ct.c_int, _ct.POINTER(_ct.c_double), _ct.c_void_p
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
    alpha, G = state[:N], state[N : 2 * N]
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

    y_np = (
        to_cpu(y).astype(np.float64)
        if mlx_input
        else np.asarray(to_cpu(y) if xp is not np else y, dtype=np.float64)
    )

    # Written by the C solver: alpha in state[:N], gradient in state[N:2N],
    # 1.0 in state[2N:] for columns in its cache
    state = np.zeros(3 * N, dtype=np.float64)
    in_cache = state[2 * N :]
    max_batch = max(1, _SMO_PREFETCH) if cache._use_fast else 1
    # With the l-mer index a column costs one query row, so prefetch only
    # fills empty slots: replacing unrequested prefetches would just churn
    capacity = min(cache_size, N)
    free_only = cache._index is not None
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
            if free_only:
                room = min(room, capacity - 1 - int(in_cache.sum()))
            if room > 0:
                fresh = in_cache == 0
                fresh[idx] = False
                extra = _prefetch_candidates(
                    state, y_np, C, min(max_batch - 1, room), fresh
                )
        try:
            return cache.compute_columns([idx, *extra])
        except Exception as exc:  # noqa: BLE001 - fall back to one column
            if not extra:
                raise
            import warnings

            warnings.warn(
                f"SMO column prefetch failed ({exc!r}); computing columns singly",
                RuntimeWarning,
                stacklevel=2,
            )
            return cache.compute_columns([idx])

    def _column_callback(idx, n, out_ptr, c_cache):
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
        alpha_out.ctypes.data_as(_ct.POINTER(_ct.c_double)),
        rho_out.ctypes.data_as(_ct.POINTER(_ct.c_double)),
        int(verbose),
        state.ctypes.data_as(_ct.c_void_p),
    )
    if errors:
        raise RuntimeError("SMO kernel column computation failed") from errors[0]
    if verbose:
        print(
            f"  Column cache {cache_size} columns; {stats['prefetched']} columns "
            f"prefetched in {stats['batches']} batches"
        )

    return alpha_out, float(rho_out[0])
