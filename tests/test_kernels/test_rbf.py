import random

import pytest
import torch

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
        x = one_hot_encode("ACGTACGTAC").unsqueeze(0)
        k = self.kernel.pairwise(x, x)
        assert k.item() == pytest.approx(1.0)

    def test_output_range(self):
        seqs = _make_seqs(4, 15, seed=1)
        x = torch.stack([one_hot_encode(s) for s in seqs[:2]])
        y = torch.stack([one_hot_encode(s) for s in seqs[2:]])
        k = self.kernel.pairwise(x, y)
        assert k.shape == (2, 2)
        assert (k >= 0).all()
        assert (k <= 1).all()

    def test_symmetry(self):
        x = one_hot_encode("ACGTACGTAC").unsqueeze(0)
        y = one_hot_encode("TGCATGCATG").unsqueeze(0)
        kxy = self.kernel.pairwise(x, y)
        kyx = self.kernel.pairwise(y, x)
        assert torch.allclose(kxy, kyx, atol=1e-6)

    def test_identical_sequences_max_similarity(self):
        seqs = _make_seqs(3, 15, seed=2)
        x = torch.stack([one_hot_encode(s) for s in seqs])
        k = self.kernel.pairwise(x, x)
        for i in range(3):
            assert k[i, i].item() == pytest.approx(1.0)
            for j in range(3):
                if i != j:
                    assert k[i, j].item() <= 1.0


class TestRbfGkmKernelGamma:
    def test_higher_gamma_sharper_falloff(self):
        x = one_hot_encode("ACGTACGTAC").unsqueeze(0)
        y = one_hot_encode("ACGTACGTAG").unsqueeze(0)

        k_low = RbfGkmKernel(l=5, k=3, gamma=0.1, include_rc=False)
        k_high = RbfGkmKernel(l=5, k=3, gamma=10.0, include_rc=False)

        sim_low = k_low.pairwise(x, y).item()
        sim_high = k_high.pairwise(x, y).item()
        assert sim_low > sim_high

    def test_gamma_zero_all_ones(self):
        kernel = RbfGkmKernel(l=5, k=3, gamma=0.0, include_rc=False)
        seqs = _make_seqs(3, 15, seed=3)
        x = torch.stack([one_hot_encode(s) for s in seqs])
        k = kernel.pairwise(x, x)
        assert torch.allclose(k, torch.ones_like(k), atol=1e-6)


class TestRbfGkmKernelRC:
    def test_rc_invariance(self):
        kernel = RbfGkmKernel(l=5, k=3, gamma=1.0, include_rc=True)
        x = one_hot_encode("ACGTACGTAC").unsqueeze(0)
        y = one_hot_encode("TGCATGCATG").unsqueeze(0)
        x_rc = reverse_complement(x)

        k_xy = kernel.pairwise(x, y)
        k_rcx_y = kernel.pairwise(x_rc, y)
        assert torch.allclose(k_xy, k_rcx_y, atol=1e-5)


class TestRbfGkmKernelConsistency:
    def test_matches_manual_rbf(self):
        """RBF kernel should equal exp(-gamma * dist²) where
        dist² = K(x,x) + K(y,y) - 2*K(x,y) using unnormalized base kernel."""
        gamma = 2.0
        base = EstTruncGkmKernel(l=5, k=3, d=3, normalize=False, include_rc=False)
        rbf = RbfGkmKernel(l=5, k=3, d=3, gamma=gamma, include_rc=False)

        seqs = _make_seqs(3, 15, seed=4)
        x = torch.stack([one_hot_encode(s) for s in seqs[:2]])
        y = torch.stack([one_hot_encode(s) for s in seqs[2:]])

        K_xy = base._raw_pairwise(x, y)
        K_xx = base._raw_diagonal(x)
        K_yy = base._raw_diagonal(y)
        dist_sq = K_xx.unsqueeze(1) + K_yy.unsqueeze(0) - 2 * K_xy
        expected = torch.exp(-gamma * torch.clamp(dist_sq, min=0))

        got = rbf.pairwise(x, y)
        assert torch.allclose(got.double(), expected.double(), atol=1e-6)


class TestRbfGkmKernelDevice:
    def test_to_mps(self):
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")
        kernel = RbfGkmKernel(l=5, k=3, gamma=1.0, include_rc=True)
        x = one_hot_encode("ACGTACGTAC").unsqueeze(0)
        y = one_hot_encode("TGCATGCATG").unsqueeze(0)

        k_cpu = kernel.pairwise(x, y)
        k_mps = kernel.pairwise(x.to("mps"), y.to("mps"))
        assert torch.allclose(k_cpu, k_mps.cpu(), atol=1e-4)
