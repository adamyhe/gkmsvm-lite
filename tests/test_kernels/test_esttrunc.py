import pytest
import torch
from math import comb

from gkmsvm.codec import one_hot_encode, reverse_complement
from gkmsvm.kernels.esttrunc import EstTruncGkmKernel, _build_estlmer_table


class TestWeightTable:
    """Verify the weight table computation against known properties."""

    def test_table_length(self):
        table = _build_estlmer_table(l=11, k=7, truncate=True)
        assert table.shape == (12,)

    def test_zero_mismatch_is_largest(self):
        table = _build_estlmer_table(l=11, k=7, truncate=True)
        assert table[0] > 0
        assert table[0] >= table[1]

    def test_weights_nonnegative(self):
        table = _build_estlmer_table(l=11, k=7, truncate=True)
        assert (table >= 0).all()

    def test_high_mismatches_truncated_to_zero(self):
        table = _build_estlmer_table(l=11, k=7, truncate=True)
        assert table[-1] == 0.0

    def test_truncated_vs_full(self):
        table_tr = _build_estlmer_table(l=11, k=7, truncate=True)
        table_full = _build_estlmer_table(l=11, k=7, truncate=False)
        # Both should be valid weight tables
        assert table_tr[0] > 0
        assert table_full[0] > 0
        # Truncated zeros out high-mismatch terms that full keeps
        tr_zeros = (table_tr == 0).sum()
        full_zeros = (table_full == 0).sum()
        assert tr_zeros >= full_zeros

    def test_differs_from_direct(self):
        """EstTrunc weight table differs from the direct (gkm_cnt) table."""
        direct_table = torch.tensor(
            [comb(11 - m, 7) if 11 - m >= 7 else 0 for m in range(12)],
            dtype=torch.float64,
        )
        esttrunc_table = _build_estlmer_table(l=11, k=7, truncate=True)
        # They should not be equal
        assert not torch.allclose(direct_table, esttrunc_table)

    def test_small_l_k(self):
        """Smoke test with small parameters."""
        table = _build_estlmer_table(l=3, k=2, truncate=True)
        assert table.shape == (4,)
        assert table[0] > 0


class TestEstTruncKernelBasic:
    """Test with l=3, k=2 for basic correctness."""

    def setup_method(self):
        self.kernel = EstTruncGkmKernel(
            l=3, k=2, d=1, normalize=False, include_rc=False
        )

    def test_self_kernel_positive(self):
        seq = one_hot_encode("ACGT").unsqueeze(0)
        diag = self.kernel._raw_diagonal(seq)
        assert diag.item() > 0

    def test_identical_sequences(self):
        x = one_hot_encode("ACGT").unsqueeze(0)
        k = self.kernel._raw_pairwise(x, x)
        diag = self.kernel._raw_diagonal(x)
        assert k.item() == pytest.approx(diag.item())

    def test_symmetry(self):
        x = one_hot_encode("ACGT").unsqueeze(0)
        y = one_hot_encode("ACGA").unsqueeze(0)
        kxy = self.kernel._raw_pairwise(x, y)
        kyx = self.kernel._raw_pairwise(y, x)
        assert torch.allclose(kxy, kyx)

    def test_completely_different_less_than_identical(self):
        x = one_hot_encode("AAAA").unsqueeze(0)
        y = one_hot_encode("CCCC").unsqueeze(0)
        k_diff = self.kernel._raw_pairwise(x, y)
        k_self = self.kernel._raw_pairwise(x, x)
        assert k_diff.item() <= k_self.item()


class TestEstTruncKernelNormalized:
    def setup_method(self):
        self.kernel = EstTruncGkmKernel(
            l=3, k=2, d=1, normalize=True, include_rc=False
        )

    def test_self_kernel_is_one(self):
        for seq in ["ACGT", "AAAA", "GGGGGG"]:
            x = one_hot_encode(seq).unsqueeze(0)
            diag = self.kernel.diagonal(x)
            assert diag.item() == pytest.approx(1.0)

    def test_identical_score_one(self):
        x = one_hot_encode("ACGTACGT").unsqueeze(0)
        k = self.kernel.pairwise(x, x)
        assert k.item() == pytest.approx(1.0)

    def test_range(self):
        x = one_hot_encode("ACGT").unsqueeze(0)
        y = one_hot_encode("ACGA").unsqueeze(0)
        k = self.kernel.pairwise(x, y)
        assert 0.0 <= k.item() <= 1.0


class TestEstTruncKernelRC:
    def setup_method(self):
        self.kernel = EstTruncGkmKernel(
            l=3, k=2, d=1, normalize=True, include_rc=True
        )

    def test_rc_invariance(self):
        x = one_hot_encode("ACGTAC").unsqueeze(0)
        y = one_hot_encode("TGCATG").unsqueeze(0)
        x_rc = reverse_complement(x)
        k_xy = self.kernel.pairwise(x, y)
        k_rcx_y = self.kernel.pairwise(x_rc, y)
        assert torch.allclose(k_xy, k_rcx_y, atol=1e-5)


class TestEstTruncKernelDefaultParams:
    """Test with L=11, k=7, d=3 — the LS-GKM / ENCODE default."""

    def setup_method(self):
        self.kernel = EstTruncGkmKernel(
            l=11, k=7, d=3, normalize=True, include_rc=True
        )

    def test_runs_with_encode_length_sequences(self):
        seq = "A" * 50
        x = one_hot_encode(seq).unsqueeze(0)
        k = self.kernel.pairwise(x, x)
        assert k.item() == pytest.approx(1.0)

    def test_batch_pairwise(self):
        x = torch.stack([
            one_hot_encode("ACGTACGTACGTACGT"),
            one_hot_encode("TGCATGCATGCATGCA"),
        ])
        y = torch.stack([
            one_hot_encode("ACGTACGTACGTACGT"),
        ])
        k = self.kernel.pairwise(x, y)
        assert k.shape == (2, 1)
        assert torch.isfinite(k).all()
        assert k[0, 0].item() == pytest.approx(1.0)  # identical
        assert k[1, 0].item() < 1.0  # different

    def test_different_lengths(self):
        x = one_hot_encode("A" * 20).unsqueeze(0)
        y = one_hot_encode("A" * 30).unsqueeze(0)
        k = self.kernel.pairwise(x, y)
        assert torch.isfinite(k).all()
        assert 0.0 < k.item() <= 1.0
