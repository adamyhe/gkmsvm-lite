"""Pin Numba's threading layer before anything can pick a crashing one.

Numba's default OpenMP threading layer and PyTorch's OpenMP runtime load
distinct copies of LLVM's libomp, and whichever initializes second crashes
the process with SIGSEGV. Since torch is a required dependency of gkmsvm,
we always pin to ``workqueue`` which avoids OpenMP entirely.

Adapted from scprism._threading (see scprism PR #8).
"""

from __future__ import annotations

import warnings

_OVERRIDABLE = ("default", "omp")


def pin_threading_layer() -> str:
    """Choose Numba's threading layer now, while the choice is still available.

    Returns the layer in effect, or ``"unavailable"`` if Numba is not installed.
    Idempotent and cheap (~1.7 ms first call, ~2 us after).
    """
    try:
        import numba
    except ImportError:
        return "unavailable"

    try:
        current: str = str(numba.threading_layer())
    except ValueError:
        pass
    else:
        if current == "omp":
            warnings.warn(
                "Numba's threading layer is already initialized as 'omp'; "
                "Numba prange calls may segfault alongside PyTorch. "
                "Set NUMBA_THREADING_LAYER=workqueue, or import gkmsvm before "
                "any code that initializes Numba.",
                RuntimeWarning,
                stacklevel=2,
            )
        return current

    if numba.config.THREADING_LAYER in _OVERRIDABLE:
        numba.config.THREADING_LAYER = "workqueue"

    try:
        numba.get_num_threads()
        return str(numba.threading_layer())
    except Exception:
        return "uninitialized"
