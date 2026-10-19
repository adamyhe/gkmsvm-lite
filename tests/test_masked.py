"""Masked-join CPU kernel must equal the brute-force packed kernel."""

import numpy as np
import pytest

from gkmsvm.codec import reverse_complement
from gkmsvm.kernels.direct import (
    DirectGkmKernel,
    _fused_pairwise_packed_numba,
)
from gkmsvm.kernels.esttrunc import EstTruncGkmKernel
from gkmsvm.kernels.masked import mask_plan, masked_pairwise
from gkmsvm.kernels.rbf import RbfGkmKernel


def _onehot(n, length, rng):
    idx = rng.integers(0, 4, size=(n, length))
    x = np.zeros((n, 4, length))
    np.put_along_axis(x, idx[:, None, :], 1.0, axis=1)
    return x


def _packed(kernel, X):
    return kernel.pack(X)


KERNELS = {
    "t0_l8k5": lambda: DirectGkmKernel(8, 5),
    "t1_l9k6": lambda: EstTruncGkmKernel(9, 6, d=3, truncate=False),
    "t2_l9k6d3": lambda: EstTruncGkmKernel(9, 6, d=3),
    "t2_l10k6d2": lambda: EstTruncGkmKernel(10, 6, d=2),
    "t2_l11k7d3": lambda: EstTruncGkmKernel(11, 7, d=3),
}


@pytest.mark.parametrize("name", KERNELS)
def test_masked_equals_brute(name):
    kernel = KERNELS[name]()
    rng = np.random.default_rng(1)
    # Repeated subsequences make many exact and near matches
    X = _onehot(19, 80, rng)
    X[3, :, 10:50] = X[7, :, 20:60]
    Y = _onehot(23, 60, rng)
    bx, by = _packed(kernel, X), _packed(kernel, Y)
    by_rc = _packed(kernel, reverse_complement(Y))
    t, l, mm = kernel._mismatch_table, kernel.l, kernel._min_matches
    plan = mask_plan(tuple(t), l, mm)
    assert plan is not None

    ref = _fused_pairwise_packed_numba(bx, by, t, l, mm)
    np.testing.assert_allclose(masked_pairwise(bx, by, plan), ref, rtol=1e-12, atol=1e-12)

    ref_rc = ref + _fused_pairwise_packed_numba(bx, by_rc, t, l, mm)
    np.testing.assert_allclose(masked_pairwise(bx, by, plan, by_rc), ref_rc,
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", ["t2", "t3"])
def test_pairwise_dispatch_matches_brute(kind):
    """kernel.pairwise uses the masked path above _MASKED_MIN_ROWS; compare to forced brute."""
    def make():
        if kind == "t3":
            return RbfGkmKernel(9, 6, d=3, gamma=2.0)
        return EstTruncGkmKernel(9, 6, d=3)

    rng = np.random.default_rng(2)
    X, Y = _onehot(40, 70, rng), _onehot(33, 70, rng)
    fast, brute = make(), make()
    for k in (brute, getattr(brute, "_base", None)):
        if k is not None:
            k.use_masked = False
    np.testing.assert_allclose(fast.pairwise(X, Y), brute.pairwise(X, Y), rtol=1e-12)


def test_plan_none_without_cutoff():
    # min_matches = 0 means every pair contributes: no masked plan
    assert mask_plan((1.0, 1.0, 1.0), 2, 0) is None


class TestWeighted:
    """-t 4 / -t 5 masked path (window weights) vs the weighted brute-force loop."""

    @staticmethod
    def _pair(rc, rbf=False):
        from gkmsvm.kernels.weighted import CenterWeightedGkmKernel, CenterWeightedRbfGkmKernel

        if rbf:
            a = CenterWeightedRbfGkmKernel(8, 5, M=3, H=2.0, gamma=2.0, include_rc=rc)
            b = CenterWeightedRbfGkmKernel(8, 5, M=3, H=2.0, gamma=2.0, include_rc=rc)
            b._base.use_masked = False
        else:
            a = CenterWeightedGkmKernel(8, 5, M=3, H=2.0, include_rc=rc)
            b = CenterWeightedGkmKernel(8, 5, M=3, H=2.0, include_rc=rc)
            b.use_masked = False
        return a, b

    @pytest.mark.parametrize("rc", [True, False])
    @pytest.mark.parametrize("rbf", [False, True])
    def test_pairwise_and_diagonal(self, rc, rbf):
        rng = np.random.default_rng(4)
        X, Y = _onehot(20, 50, rng), _onehot(18, 40, rng)
        X[2, :, 5:35] = Y[4, :, 5:35]
        fast, ref = self._pair(rc, rbf)
        np.testing.assert_allclose(fast.pairwise(X, Y), ref.pairwise(X, Y), rtol=1e-12)
        if not rbf:
            np.testing.assert_allclose(fast._raw_diagonal(X), ref._raw_diagonal(X), rtol=1e-12)

    def test_self_gram_panels(self):
        """Self-Gram column panels (diag_offset) weight windows too."""
        from gkmsvm.kernels.weighted import CenterWeightedGkmKernel

        rng = np.random.default_rng(5)
        X = _onehot(40, 41, rng)  # even window count
        kernel = CenterWeightedGkmKernel(8, 5, M=20, H=3.0)
        bx, brc = kernel.pack_both(X)
        full = kernel.raw_packed(bx, bx, brc)
        upper = kernel.raw_packed(bx, bx[10:], brc[10:], diag_offset=10)
        keep = np.arange(40)[:, None] <= np.arange(10, 40)[None, :]
        np.testing.assert_allclose(upper[keep], full[:, 10:][keep], rtol=1e-12)
