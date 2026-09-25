"""Array backend: numpy (CPU) or cupy (GPU).

All gkmsvm computation uses numpy-style arrays. When cupy is installed
and arrays live on GPU, cupy's API is used transparently. Install the
[gpu] extra for GPU support: ``pip install gkmsvm-lite[gpu]``.
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


def get_array_module(x: Any) -> Any:
    """Return numpy or cupy depending on where *x* lives."""
    if HAS_CUPY and isinstance(x, cp.ndarray):
        return cp
    return np


def to_gpu(x: np.ndarray) -> Any:
    """Move a numpy array to GPU (cupy)."""
    if not HAS_CUPY:
        raise RuntimeError(
            "CuPy is not installed. Install with: pip install gkmsvm-lite[gpu]"
        )
    return cp.asarray(x)


def to_cpu(x: Any) -> np.ndarray:
    """Move an array to CPU (numpy)."""
    if HAS_CUPY and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def is_gpu(x: Any) -> bool:
    """Check if an array lives on GPU."""
    return HAS_CUPY and isinstance(x, cp.ndarray)
