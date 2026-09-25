import pytest
import torch
from math import comb

from gkmsvm.codec import one_hot_encode, reverse_complement
from gkmsvm.kernels.direct import DirectGkmKernel


class TestDirectGkmKernelBasic:
    """Test with l=3, k=2 for hand-verifiable results."""

    def setup_method(self):
        self.kernel = DirectGkmKernel(l=3, k=2, normalize=False, include_rc=False)

    def test_mismatch_table(self):
        # C(3-m, 2) for m=0..3: C(3,2)=3, C(2,2)=1, C(1,2)=0, C(0,2)=0
        expected = [3, 1, 0, 0]
        assert self.kernel._mismatch_table.tolist() == expected

    def test_self_kernel_identical_windows(self):
        # "AAA": 1 window of length 3. Self-kernel = C(3,2) = 3
        seq = one_hot_encode("AAA").unsqueeze(0)
        diag = self.kernel._raw_diagonal(seq)
        assert diag.item() == pytest.approx(3.0)

    def test_self_kernel_longer_sequence(self):
        # "AAAA": 2 windows [AAA, AAA]. All identical.
        # Sum over 2x2 window pairs, each contributing C(3,2)=3 -> 4*3 = 12
        seq = one_hot_encode("AAAA").unsqueeze(0)
        diag = self.kernel._raw_diagonal(seq)
        assert diag.item() == pytest.approx(12.0)

    def test_completely_different(self):
        # "AAA" vs "CCC": 1 window each, 3 mismatches -> C(0,2)=0
        x = one_hot_encode("AAA").unsqueeze(0)
        y = one_hot_encode("CCC").unsqueeze(0)
        k = self.kernel._raw_pairwise(x, y)
        assert k.item() == pytest.approx(0.0)

    def test_one_mismatch(self):
        # "AAA" vs "AAC": 1 window each, 1 mismatch -> C(2,2)=1
        x = one_hot_encode("AAA").unsqueeze(0)
        y = one_hot_encode("AAC").unsqueeze(0)
        k = self.kernel._raw_pairwise(x, y)
        assert k.item() == pytest.approx(1.0)

    def test_symmetry(self):
        x = one_hot_encode("ACGT").unsqueeze(0)
        y = one_hot_encode("ACGA").unsqueeze(0)
        kxy = self.kernel._raw_pairwise(x, y)
        kyx = self.kernel._raw_pairwise(y, x)
        assert torch.allclose(kxy, kyx)

    def test_batch_pairwise(self):
        x = torch.stack([one_hot_encode("AAA"), one_hot_encode("CCC")])
        y = torch.stack([one_hot_encode("AAA"), one_hot_encode("AAC")])
        k = self.kernel._raw_pairwise(x, y)
        assert k.shape == (2, 2)
        assert k[0, 0].item() == pytest.approx(3.0)  # AAA vs AAA
        assert k[0, 1].item() == pytest.approx(1.0)  # AAA vs AAC
        assert k[1, 0].item() == pytest.approx(0.0)  # CCC vs AAA
        assert k[1, 1].item() == pytest.approx(0.0)  # CCC vs AAC


class TestDirectGkmKernelNormalized:
    def setup_method(self):
        self.kernel = DirectGkmKernel(l=3, k=2, normalize=True, include_rc=False)

    def test_self_kernel_is_one(self):
        for seq in ["AAA", "ACGT", "GGGGGG"]:
            x = one_hot_encode(seq).unsqueeze(0)
            diag = self.kernel.diagonal(x)
            assert diag.item() == pytest.approx(1.0)

    def test_normalized_range(self):
        x = one_hot_encode("ACGT").unsqueeze(0)
        y = one_hot_encode("ACGA").unsqueeze(0)
        k = self.kernel.pairwise(x, y)
        assert 0.0 <= k.item() <= 1.0

    def test_identical_sequences_score_one(self):
        x = one_hot_encode("ACGTACGT").unsqueeze(0)
        k = self.kernel.pairwise(x, x)
        assert k.item() == pytest.approx(1.0)


class TestDirectGkmKernelRC:
    def setup_method(self):
        self.kernel = DirectGkmKernel(l=3, k=2, normalize=True, include_rc=True)
        self.kernel_no_rc = DirectGkmKernel(l=3, k=2, normalize=True, include_rc=False)

    def test_rc_invariance(self):
        x = one_hot_encode("ACGTAC").unsqueeze(0)
        y = one_hot_encode("TGCATG").unsqueeze(0)
        x_rc = reverse_complement(x)

        k_xy = self.kernel.pairwise(x, y)
        k_rcx_y = self.kernel.pairwise(x_rc, y)
        assert torch.allclose(k_xy, k_rcx_y, atol=1e-5)

    def test_rc_increases_or_equals_kernel(self):
        x = one_hot_encode("ACGTAC").unsqueeze(0)
        y = one_hot_encode("TGCATG").unsqueeze(0)

        k_rc = self.kernel._raw_pairwise(x, y)
        k_no_rc = self.kernel_no_rc._raw_pairwise(x, y)
        assert k_rc.item() >= k_no_rc.item()

    def test_palindrome_rc(self):
        # ACGT is its own reverse complement
        x = one_hot_encode("ACGT").unsqueeze(0)
        k = self.kernel.pairwise(x, x)
        assert k.item() == pytest.approx(1.0)


class TestDirectGkmKernelEdgeCases:
    def test_sequence_too_short(self):
        kernel = DirectGkmKernel(l=5, k=3)
        x = one_hot_encode("ACGT").unsqueeze(0)
        with pytest.raises(ValueError, match="shorter than window length"):
            kernel.pairwise(x, x)

    def test_minimum_length(self):
        kernel = DirectGkmKernel(l=3, k=2, normalize=False, include_rc=False)
        x = one_hot_encode("ACG").unsqueeze(0)
        k = kernel.pairwise(x, x)
        assert k.item() == pytest.approx(comb(3, 2))

    def test_different_lengths(self):
        kernel = DirectGkmKernel(l=3, k=2, normalize=True, include_rc=False)
        x = one_hot_encode("ACGT").unsqueeze(0)     # 2 windows
        y = one_hot_encode("ACGTAC").unsqueeze(0)   # 4 windows
        k = kernel.pairwise(x, y)
        assert k.shape == (1, 1)
        assert 0.0 <= k.item() <= 1.0

    def test_larger_parameters(self):
        kernel = DirectGkmKernel(l=5, k=3, normalize=True, include_rc=False)
        x = one_hot_encode("ACGTACGTAC").unsqueeze(0)
        y = one_hot_encode("TGCATGCATG").unsqueeze(0)
        k = kernel.pairwise(x, y)
        assert k.shape == (1, 1)
        # Just check it runs and gives valid output
        assert torch.isfinite(k).all()
        assert 0.0 <= k.item() <= 1.0
