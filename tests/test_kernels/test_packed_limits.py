"""Packed kernels take l <= 15: at l = 16 the all-T l-mer is PAD_WINDOW."""

import pytest

from gkmsvm.kernels.esttrunc import EstTruncGkmKernel
from gkmsvm.kernels.packed import MAX_L, PAD_WINDOW


def test_pad_is_no_l_mer():
    assert int(PAD_WINDOW) >= 4**MAX_L


def test_l_above_max_raises():
    EstTruncGkmKernel(MAX_L, 7, d=3)
    with pytest.raises(ValueError, match="l must be <= 15"):
        EstTruncGkmKernel(16, 8, d=3)
