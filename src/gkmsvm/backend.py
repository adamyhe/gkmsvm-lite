"""Array backend: numpy (CPU), cupy (NVIDIA GPU), or mlx (Apple GPU).

All gkmsvm computation uses numpy-style arrays. When cupy or mlx is
installed and arrays live on GPU, the matching backend's API is used
transparently.

Install extras for GPU support::

    pip install gkmsvm-lite[gpu]   # NVIDIA (CuPy)
    pip install gkmsvm-lite[mlx]   # Apple Silicon (MLX)
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    cp = None  # type: ignore[assignment]
    HAS_CUPY = False

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    mx = None  # type: ignore[assignment]
    HAS_MLX = False


def get_array_module(x: Any) -> Any:
    """Return numpy, cupy, or the mlx shim depending on where *x* lives."""
    if HAS_CUPY and isinstance(x, cp.ndarray):
        return cp
    if HAS_MLX and isinstance(x, mx.array):
        return _get_mlx_shim()
    return np


def to_gpu(x: np.ndarray) -> Any:
    """Move a numpy array to GPU (CuPy)."""
    if not HAS_CUPY:
        raise RuntimeError(
            "CuPy is not installed. Install with: pip install gkmsvm-lite[gpu]"
        )
    return cp.asarray(x)


def to_mlx(x: np.ndarray) -> Any:
    """Move a numpy array to MLX."""
    if not HAS_MLX:
        raise RuntimeError(
            "MLX is not installed. Install with: pip install gkmsvm-lite[mlx]"
        )
    return mx.array(x)


def to_cpu(x: Any) -> np.ndarray:
    """Move an array to CPU (numpy)."""
    if HAS_CUPY and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    if HAS_MLX and isinstance(x, mx.array):
        return np.array(x)
    return np.asarray(x)


def is_gpu(x: Any) -> bool:
    """Check if an array lives on a GPU backend."""
    if HAS_CUPY and isinstance(x, cp.ndarray):
        return True
    if HAS_MLX and isinstance(x, mx.array):
        return True
    return False


def is_mlx(x: Any) -> bool:
    """Check if an array is an MLX array."""
    return HAS_MLX and isinstance(x, mx.array)


def get_strides(x: Any) -> tuple:
    """Return array strides, or zeros for MLX (which has no strides)."""
    if hasattr(x, "strides"):
        return x.strides
    return (0,) * x.ndim


# ---------------------------------------------------------------------------
# MLX shim — adapts mlx.core to look like numpy/cupy where they differ
# ---------------------------------------------------------------------------

_mlx_shim_instance = None


def _get_mlx_shim():
    global _mlx_shim_instance
    if _mlx_shim_instance is None:
        _mlx_shim_instance = _MlxShim()
    return _mlx_shim_instance


class _MlxShim:
    """Wraps mlx.core to provide numpy-compatible operations.

    Forwards most attributes to mlx.core. Overrides operations where
    the API differs: rint, ascontiguousarray, as_strided, einsum
    argument order, etc.
    """

    def __getattr__(self, name: str) -> Any:
        return getattr(mx, name)

    # --- numpy-compatible aliases ---

    def rint(self, x: Any) -> Any:
        return mx.round(x)

    def ascontiguousarray(self, x: Any, dtype: Any = None) -> Any:
        if dtype is not None:
            return mx.array(x, dtype=dtype)
        return mx.array(x)

    def asarray(self, x: Any, dtype: Any = None) -> Any:
        if isinstance(x, np.ndarray):
            arr = mx.array(x)
        elif isinstance(x, mx.array):
            arr = x
        else:
            arr = mx.array(np.asarray(x))
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    def asnumpy(self, x: Any) -> np.ndarray:
        return np.array(x)

    @property
    def newaxis(self) -> None:
        return None

    # --- stride tricks ---

    class _LibStrideTricks:
        class stride_tricks:
            @staticmethod
            def as_strided(x: Any, shape: tuple, strides: tuple) -> Any:
                return _mlx_sliding_windows(x, shape, strides)

    @property
    def lib(self) -> Any:
        return self._LibStrideTricks

    # --- dtype aliases ---

    @property
    def float64(self) -> Any:
        return mx.float32

    @property
    def float32(self) -> Any:
        return mx.float32

    @property
    def int64(self) -> Any:
        return mx.int32

    @property
    def int8(self) -> Any:
        return mx.int8

    @property
    def uint32(self) -> Any:
        return mx.uint32

    @property
    def bool_(self) -> Any:
        return mx.bool_

    # --- module identity ---

    @property
    def __name__(self) -> str:
        return "mlx_shim"


def _mlx_sliding_windows(x: Any, shape: tuple, strides: tuple) -> Any:
    """Implement as_strided for MLX using mx.as_strided.

    Only handles the sliding-window patterns used in gkmsvm:
    - [B, C, W, l] from [B, C, L] (flat_windows)
    - [B, W, l] from [B, L] (base_index_windows)
    """
    ndim_in = x.ndim
    ndim_out = len(shape)

    if ndim_in == 3 and ndim_out == 4:
        B, C, L = x.shape
        _, _, W, l = shape
        return mx.as_strided(x, shape=(B, C, W, l), strides=(C * L, L, 1, 1))

    if ndim_in == 2 and ndim_out == 3:
        B, L = x.shape
        _, W, l = shape
        return mx.as_strided(x, shape=(B, W, l), strides=(L, 1, 1))

    raise NotImplementedError(
        f"MLX sliding windows: {ndim_in}D -> {ndim_out}D not implemented"
    )
