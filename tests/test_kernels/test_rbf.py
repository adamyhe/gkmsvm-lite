import random

import numpy as np
import pytest

from gkmsvm.codec import one_hot_encode, reverse_complement
from gkmsvm.kernels.rbf import RbfGkmKernel
from gkmsvm.kernels.esttrunc import EstTruncGkmKernel


def _make_seqs(n, length, seed=42):
    rng = random.Random(seed)
    return ["".join(rng.choice("ACGT") for _ in range(length)) for _ in range(n)]


class TestRbfGkmKernelBasic:
    def setup_method(self):
        self.kernel = RbfGkmKernel(l=5, k=3, d=3, gamma=1.0, include_rc=False)

    def test_self_similarity_is_one(self):
        x = one_hot_encode("ACGTACGTAC")[np.newaxis]
        k = self.kernel.pairwise(x, x)
        assert float(k[0, 0]) == pytest.approx(1.0)

    def test_output_range(self):
        seqs = _make_seqs(4, 15, seed=1)
        x = np.stack([one_hot_encode(s) for s in seqs[:2]])
        y = np.stack([one_hot_encode(s) for s in seqs[2:]])
        k = self.kernel.pairwise(x, y)
        assert k.shape == (2, 2)
        assert (k >= 0).all()
        assert (k <= 1).all()

    def test_symmetry(self):
        x = one_hot_encode("ACGTACGTAC")[np.newaxis]
        y = one_hot_encode("TGCATGCATG")[np.newaxis]
        kxy = self.kernel.pairwise(x, y)
        kyx = self.kernel.pairwise(y, x)
        np.testing.assert_allclose(kxy, kyx, atol=1e-6)

    def test_identical_sequences_max_similarity(self):
        seqs = _make_seqs(3, 15, seed=2)
        x = np.stack([one_hot_encode(s) for s in seqs])
        k = self.kernel.pairwise(x, x)
        for i in range(3):
            assert float(k[i, i]) == pytest.approx(1.0)
            for j in range(3):
                if i != j:
                    assert float(k[i, j]) <= 1.0


class TestRbfGkmKernelGamma:
    def test_higher_gamma_sharper_falloff(self):
        x = one_hot_encode("ACGTACGTAC")[np.newaxis]
        y = one_hot_encode("ACGTACGTAG")[np.newaxis]

        k_low = RbfGkmKernel(l=5, k=3, gamma=0.1, include_rc=False)
        k_high = RbfGkmKernel(l=5, k=3, gamma=10.0, include_rc=False)

        sim_low = float(k_low.pairwise(x, y)[0, 0])
        sim_high = float(k_high.pairwise(x, y)[0, 0])
        assert sim_low > sim_high

    def test_gamma_zero_all_ones(self):
        kernel = RbfGkmKernel(l=5, k=3, gamma=0.0, include_rc=False)
        seqs = _make_seqs(3, 15, seed=3)
        x = np.stack([one_hot_encode(s) for s in seqs])
        k = kernel.pairwise(x, x)
        np.testing.assert_allclose(k, np.ones_like(k), atol=1e-6)


class TestRbfGkmKernelRC:
    def test_rc_invariance(self):
        kernel = RbfGkmKernel(l=5, k=3, gamma=1.0, include_rc=True)
        x = one_hot_encode("ACGTACGTAC")[np.newaxis]
        y = one_hot_encode("TGCATGCATG")[np.newaxis]
        x_rc = reverse_complement(x)

        k_xy = kernel.pairwise(x, y)
        k_rcx_y = kernel.pairwise(x_rc, y)
        np.testing.assert_allclose(k_xy, k_rcx_y, atol=1e-5)


class TestRbfGkmKernelConsistency:
    def test_matches_manual_rbf(self):
        """RBF kernel should equal exp(gamma * (K_norm - 1)) matching lsgkm."""
        gamma = 2.0
        base = EstTruncGkmKernel(l=5, k=3, d=3, normalize=True, include_rc=False)
        rbf = RbfGkmKernel(l=5, k=3, d=3, gamma=gamma, include_rc=False)

        seqs = _make_seqs(3, 15, seed=4)
        x = np.stack([one_hot_encode(s) for s in seqs[:2]])
        y = np.stack([one_hot_encode(s) for s in seqs[2:]])

        K_norm = base.pairwise(x, y)
        expected = np.exp(gamma * (K_norm - 1))

        got = rbf.pairwise(x, y)
        np.testing.assert_allclose(
            got.astype(np.float64), expected.astype(np.float64), atol=1e-6
        )
