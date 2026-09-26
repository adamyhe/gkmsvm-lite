"""Tests for Gram matrix computation."""

import numpy as np
import pytest

from gkmsvm.codec import one_hot_encode
from gkmsvm.gram import compute_gram
from gkmsvm.kernels.direct import DirectGkmKernel
from gkmsvm.kernels.esttrunc import EstTruncGkmKernel


def _random_seqs(n, length, rng):
    seqs = []
    for _ in range(n):
        idx = rng.integers(0, 4, size=length)
        x = np.zeros((4, length), dtype=np.float64)
        x[idx, np.arange(length)] = 1.0
        seqs.append(x)
    return np.stack(seqs)


class TestComputeGram:
    def test_self_gram_matches_pairwise(self):
        rng = np.random.default_rng(42)
        X = _random_seqs(10, 20, rng)
        kernel = DirectGkmKernel(l=7, k=5, normalize=True, include_rc=True)

        gram = compute_gram(kernel, X)
        expected = kernel.pairwise(X, X)

        np.testing.assert_allclose(gram, expected, atol=1e-10)

    def test_symmetry(self):
        rng = np.random.default_rng(42)
        X = _random_seqs(15, 20, rng)
        kernel = EstTruncGkmKernel(l=7, k=5, d=3, normalize=True)

        gram = compute_gram(kernel, X)

        np.testing.assert_allclose(gram, gram.T, atol=1e-10)

    def test_diagonal_is_one_when_normalized(self):
        rng = np.random.default_rng(42)
        X = _random_seqs(8, 20, rng)
        kernel = DirectGkmKernel(l=7, k=5, normalize=True, include_rc=True)

        gram = compute_gram(kernel, X)

        np.testing.assert_allclose(np.diag(gram), 1.0, atol=1e-10)

    def test_chunked_matches_full(self):
        rng = np.random.default_rng(42)
        X = _random_seqs(12, 20, rng)
        kernel = DirectGkmKernel(l=7, k=5, normalize=True, include_rc=True)

        gram_full = compute_gram(kernel, X, chunk_size=100)
        gram_chunked = compute_gram(kernel, X, chunk_size=4)

        np.testing.assert_allclose(gram_chunked, gram_full, atol=1e-10)

    def test_asymmetric_xy(self):
        rng = np.random.default_rng(42)
        X = _random_seqs(6, 20, rng)
        Y = _random_seqs(8, 20, rng)
        kernel = DirectGkmKernel(l=7, k=5, normalize=True, include_rc=True)

        gram = compute_gram(kernel, X, Y)
        expected = kernel.pairwise(X, Y)

        assert gram.shape == (6, 8)
        np.testing.assert_allclose(gram, expected, atol=1e-10)

    def test_asymmetric_chunked(self):
        rng = np.random.default_rng(42)
        X = _random_seqs(6, 20, rng)
        Y = _random_seqs(8, 20, rng)
        kernel = DirectGkmKernel(l=7, k=5, normalize=True, include_rc=True)

        gram_full = compute_gram(kernel, X, Y, chunk_size=100)
        gram_chunked = compute_gram(kernel, X, Y, chunk_size=3)

        np.testing.assert_allclose(gram_chunked, gram_full, atol=1e-10)

    def test_esttrunc_gram(self):
        rng = np.random.default_rng(42)
        X = _random_seqs(10, 20, rng)
        kernel = EstTruncGkmKernel(l=7, k=5, d=3, normalize=True, include_rc=True)

        gram = compute_gram(kernel, X, chunk_size=4)
        expected = kernel.pairwise(X, X)

        np.testing.assert_allclose(gram, expected, atol=1e-10)
