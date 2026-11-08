"""Packed-window helpers shared by the kernels: mismatch counts by XOR +
popcount, the RC window, row strides of window weights, and the span and
count of each row's real (non-PAD) windows."""

from __future__ import annotations

import numpy as np
from numba import njit

from gkmsvm.kernels.packed import PAD_WINDOW

_PAD = PAD_WINDOW

# ---------------------------------------------------------------------------
# Numba-accelerated kernels
# ---------------------------------------------------------------------------


@njit(inline="always")
def _popcount(x):
    """Hamming weight (number of set bits) for uint32."""
    x = np.uint32(x - ((x >> 1) & np.uint32(0x55555555)))
    x = np.uint32((x & np.uint32(0x33333333)) + ((x >> 2) & np.uint32(0x33333333)))
    x = np.uint32((x + (x >> 4)) & np.uint32(0x0F0F0F0F))
    return int(np.uint32(x * np.uint32(0x01010101)) >> 24)


@njit(inline="always")
def _packed_mismatches(a, b):
    """Count mismatching 2-bit positions between two packed uint32 values."""
    x = a ^ b
    return _popcount(((x >> 1) | x) & np.uint32(0x55555555))


def row_stride(w, W: int) -> int:
    """Row stride of [rows or 1, W] window weights: 0 when every row shares one."""
    return W if w.shape[0] > 1 else 0


@njit(cache=True)
def _spans(packed):
    """[R, 2] (start, end) of each row's non-PAD windows (one contiguous run:
    first in forward rows, last in RC rows)."""
    R, W = packed.shape
    out = np.zeros((R, 2), dtype=np.int64)
    for r in range(R):
        start = 0
        while start < W and packed[r, start] == _PAD:
            start += 1
        end = start
        while end < W and packed[r, end] != _PAD:
            end += 1
        out[r, 0] = start
        out[r, 1] = end
    return out


@njit(cache=True)
def _counts(packed):
    """Number of windows before the first PAD in each row (forward rows)."""
    R, W = packed.shape
    out = np.empty(R, dtype=np.int64)
    for r in range(R):
        n = 0
        while n < W and packed[r, n] != _PAD:
            n += 1
        out[r] = n
    return out


@njit(inline="always")
def _rc_window(w, l):
    """Packed reverse complement of one packed l-mer."""
    out = np.uint32(0)
    for k in range(l):
        out |= (np.uint32(3) - ((w >> np.uint32(2 * k)) & np.uint32(3))) << np.uint32(
            2 * (l - 1 - k)
        )
    return out
