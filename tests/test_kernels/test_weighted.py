import random
from itertools import combinations

import numpy as np
import pytest

from gkmsvm.codec import one_hot_encode, reverse_complement
from gkmsvm.kernels.direct import DirectGkmKernel
from gkmsvm.kernels.weighted import (
    CenterWeightedGkmKernel,
    CenterWeightedRbfGkmKernel,
    _center_weights,
    _elementary_symmetric_k,
)


def _make_seqs(n, length, seed=42):
    rng = random.Random(seed)
    return ["".join(rng.choice("ACGT") for _ in range(length)) for _ in range(n)]


def _naive_weighted_kernel(seq_x, seq_y, l, k, pos_weights, include_rc=False):
    """Brute-force weighted kernel for verification."""
    combos = list(combinations(range(l), k))

    def _score_pair(sx, sy):
        Wx = len(sx) - l + 1
        Wy = len(sy) - l + 1
        total = 0.0
        for i in range(Wx):
            for j in range(Wy):
                for combo in combos:
                    weight = 1.0
                    match = True
                    for p in combo:
                        weight *= pos_weights[p]
                        if sx[i + p] != sy[j + p]:
                            match = False
                            break
                    if match:
                        total += weight
        return total

    result = _score_pair(seq_x, seq_y)
    if include_rc:
        rc_map = {"A": "T", "T": "A", "C": "G", "G": "C"}
        rc_y = "".join(rc_map[b] for b in reversed(seq_y))
        result += _score_pair(seq_x, rc_y)
    return result


class TestCenterWeights:
    def test_all_center(self):
        w = _center_weights(11, M=11, H=1.0)
        np.testing.assert_allclose(w, np.ones(11, dtype=np.float64))

    def test_symmetry(self):
        w = _center_weights(11, M=5, H=2.0)
        for i in range(11):
            assert float(w[i]) == pytest.approx(float(w[10 - i]))

    def test_center_positions_full_weight(self):
        w = _center_weights(11, M=5, H=2.0)
        for i in [3, 4, 5, 6, 7]:
            assert float(w[i]) == pytest.approx(1.0)

    def test_edge_decay(self):
        w = _center_weights(11, M=5, H=2.0)
        assert float(w[0]) < float(w[3])
        assert float(w[1]) < float(w[3])
        assert float(w[0]) < float(w[1])

    def test_half_life(self):
        H = 2.0
        w = _center_weights(11, M=3, H=H)
        center = 5
        half_M = 1.5
        w_at_edge = float(w[4])
        assert w_at_edge == pytest.approx(1.0)
        w_at_H = float(w[2])
        dist = abs(2 - center) - half_M
        expected = 2.0 ** (-dist / H)
        assert w_at_H == pytest.approx(expected)


class TestElementarySymmetric:
    def test_uniform_weights(self):
        from math import comb
        for n in [5, 7, 11]:
            w = np.ones(n, dtype=np.float64)
            for k in range(1, n + 1):
                assert _elementary_symmetric_k(w, k) == pytest.approx(comb(n, k))

    def test_e1_is_sum(self):
        w = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        assert _elementary_symmetric_k(w, 1) == pytest.approx(10.0)

    def test_e2_is_sum_of_products(self):
        w = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        expected = 1 * 2 + 1 * 3 + 2 * 3
        assert _elementary_symmetric_k(w, 2) == pytest.approx(expected)

    def test_e_n_is_product(self):
        w = np.array([2.0, 3.0, 5.0], dtype=np.float64)
        assert _elementary_symmetric_k(w, 3) == pytest.approx(30.0)


class TestCenterWeightedGkmKernel:
    def test_matches_naive(self):
        """Weighted kernel should match brute-force computation."""
        l, k = 5, 3
        M, H = 3, 1.0
        pos_weights = _center_weights(l, M, H)
        kernel = CenterWeightedGkmKernel(
            l=l, k=k, M=M, H=H, normalize=False, include_rc=False
        )
        seqs = _make_seqs(3, 12, seed=1)
        for seq_x in seqs:
            for seq_y in seqs:
                x = one_hot_encode(seq_x)[np.newaxis]
                y = one_hot_encode(seq_y)[np.newaxis]
                got = float(kernel._raw_pairwise(x, y)[0, 0])
                expected = _naive_weighted_kernel(seq_x, seq_y, l, k, pos_weights)
                assert got == pytest.approx(expected, abs=1e-4), (
                    f"{seq_x} vs {seq_y}: {got} != {expected}"
                )

    def test_matches_naive_with_rc(self):
        l, k = 5, 3
        M, H = 3, 1.0
        pos_weights = _center_weights(l, M, H)
        kernel = CenterWeightedGkmKernel(
            l=l, k=k, M=M, H=H, normalize=False, include_rc=True
        )
        seqs = _make_seqs(2, 12, seed=2)
        for seq_x in seqs:
            for seq_y in seqs:
                x = one_hot_encode(seq_x)[np.newaxis]
                y = one_hot_encode(seq_y)[np.newaxis]
                got = float(kernel._raw_pairwise(x, y)[0, 0])
                expected = _naive_weighted_kernel(
                    seq_x, seq_y, l, k, pos_weights, include_rc=True
                )
                assert got == pytest.approx(expected, abs=1e-4)

    def test_self_similarity_positive(self):
        kernel = CenterWeightedGkmKernel(
            l=5, k=3, M=3, H=1.0, normalize=False, include_rc=False
        )
        x = one_hot_encode("ACGTACGTAC")[np.newaxis]
        diag = kernel._raw_diagonal(x)
        assert float(diag[0]) > 0

    def test_symmetry(self):
        kernel = CenterWeightedGkmKernel(
            l=5, k=3, M=3, H=1.0, normalize=False, include_rc=False
        )
        x = one_hot_encode("ACGTACGTAC")[np.newaxis]
        y = one_hot_encode("TGCATGCATG")[np.newaxis]
        kxy = kernel._raw_pairwise(x, y)
        kyx = kernel._raw_pairwise(y, x)
        np.testing.assert_allclose(kxy, kyx, atol=1e-6)

    def test_normalized_self_is_one(self):
        kernel = CenterWeightedGkmKernel(
            l=5, k=3, M=3, H=1.0, normalize=True, include_rc=False
        )
        x = one_hot_encode("ACGTACGTAC")[np.newaxis]
        k = kernel.pairwise(x, x)
        assert float(k[0, 0]) == pytest.approx(1.0)

    def test_normalized_range(self):
        kernel = CenterWeightedGkmKernel(
            l=5, k=3, M=3, H=1.0, normalize=True, include_rc=False
        )
        seqs = _make_seqs(3, 15, seed=10)
        x = np.stack([one_hot_encode(s) for s in seqs])
        k = kernel.pairwise(x, x)
        assert (k >= -1e-6).all()
        assert (k <= 1.0 + 1e-6).all()

    def test_differs_from_direct(self):
        """Normalized center-weighted kernel should differ from unweighted."""
        direct = DirectGkmKernel(l=5, k=3, normalize=True, include_rc=False)
        weighted = CenterWeightedGkmKernel(
            l=5, k=3, M=3, H=1.0, normalize=True, include_rc=False
        )
        seqs = _make_seqs(4, 15, seed=10)
        x = np.stack([one_hot_encode(s) for s in seqs[:2]])
        y = np.stack([one_hot_encode(s) for s in seqs[2:]])

        k_direct = direct.pairwise(x, y)
        k_weighted = weighted.pairwise(x, y)
        assert not np.allclose(k_direct, k_weighted, atol=1e-4)

    def test_full_M_matches_direct(self):
        """When M >= l (all weights 1.0), should match direct kernel."""
        direct = DirectGkmKernel(l=5, k=3, normalize=False, include_rc=False)
        weighted = CenterWeightedGkmKernel(
            l=5, k=3, M=5, H=1.0, normalize=False, include_rc=False
        )
        seqs = _make_seqs(3, 15, seed=20)
        x = np.stack([one_hot_encode(s) for s in seqs[:2]])
        y = np.stack([one_hot_encode(s) for s in seqs[2:]])

        k_direct = direct._raw_pairwise(x, y)
        k_weighted = weighted._raw_pairwise(x, y)
        np.testing.assert_allclose(k_direct, k_weighted, atol=1e-4)

    def test_batch_pairwise(self):
        kernel = CenterWeightedGkmKernel(
            l=5, k=3, M=3, H=1.0, normalize=True, include_rc=True
        )
        seqs = _make_seqs(4, 15, seed=30)
        x = np.stack([one_hot_encode(s) for s in seqs[:2]])
        y = np.stack([one_hot_encode(s) for s in seqs[2:]])
        k = kernel.pairwise(x, y)
        assert k.shape == (2, 2)
        assert np.isfinite(k).all()


class TestCenterWeightedGkmKernelRC:
    def test_rc_invariance(self):
        kernel = CenterWeightedGkmKernel(
            l=5, k=3, M=3, H=1.0, normalize=True, include_rc=True
        )
        x = one_hot_encode("ACGTACGTAC")[np.newaxis]
        y = one_hot_encode("TGCATGCATG")[np.newaxis]
        x_rc = reverse_complement(x)

        k_xy = kernel.pairwise(x, y)
        k_rcx_y = kernel.pairwise(x_rc, y)
        np.testing.assert_allclose(k_xy, k_rcx_y, atol=1e-5)


class TestCenterWeightedRbfGkmKernel:
    def test_self_similarity_is_one(self):
        kernel = CenterWeightedRbfGkmKernel(
            l=5, k=3, M=3, H=1.0, gamma=1.0, include_rc=False
        )
        x = one_hot_encode("ACGTACGTAC")[np.newaxis]
        k = kernel.pairwise(x, x)
        assert float(k[0, 0]) == pytest.approx(1.0)

    def test_output_range(self):
        kernel = CenterWeightedRbfGkmKernel(
            l=5, k=3, M=3, H=1.0, gamma=1.0, include_rc=False
        )
        seqs = _make_seqs(3, 15, seed=40)
        x = np.stack([one_hot_encode(s) for s in seqs])
        k = kernel.pairwise(x, x)
        assert (k >= 0).all()
        assert (k <= 1.0 + 1e-6).all()

    def test_gamma_effect(self):
        x = one_hot_encode("ACGTACGTAC")[np.newaxis]
        y = one_hot_encode("ACGTACGTAG")[np.newaxis]

        k_low = CenterWeightedRbfGkmKernel(
            l=5, k=3, M=3, H=1.0, gamma=0.1, include_rc=False
        )
        k_high = CenterWeightedRbfGkmKernel(
            l=5, k=3, M=3, H=1.0, gamma=10.0, include_rc=False
        )
        assert float(k_low.pairwise(x, y)[0, 0]) > float(k_high.pairwise(x, y)[0, 0])

    def test_rc_invariance(self):
        kernel = CenterWeightedRbfGkmKernel(
            l=5, k=3, M=3, H=1.0, gamma=1.0, include_rc=True
        )
        x = one_hot_encode("ACGTACGTAC")[np.newaxis]
        y = one_hot_encode("TGCATGCATG")[np.newaxis]
        x_rc = reverse_complement(x)

        k_xy = kernel.pairwise(x, y)
        k_rcx_y = kernel.pairwise(x_rc, y)
        np.testing.assert_allclose(k_xy, k_rcx_y, atol=1e-5)
