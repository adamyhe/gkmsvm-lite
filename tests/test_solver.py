"""Tests for the column-cached SMO solver."""

import numpy as np
import pytest

from gkmsvm.kernels.direct import DirectGkmKernel
from gkmsvm.kernels.esttrunc import EstTruncGkmKernel
from gkmsvm.solver import KernelColumnCache, smo_solve


def _random_seqs(n, length, rng):
    seqs = []
    for _ in range(n):
        idx = rng.integers(0, 4, size=length)
        x = np.zeros((4, length), dtype=np.float64)
        x[idx, np.arange(length)] = 1.0
        seqs.append(x)
    return np.stack(seqs)


def _biased_seqs(n, length, rng, bias_base=0):
    """Generate sequences biased toward a specific base."""
    seqs = []
    for _ in range(n):
        probs = [0.1, 0.1, 0.1, 0.1]
        probs[bias_base] = 0.7
        idx = rng.choice(4, size=length, p=probs)
        x = np.zeros((4, length), dtype=np.float64)
        x[idx, np.arange(length)] = 1.0
        seqs.append(x)
    return np.stack(seqs)


class TestKernelColumnCache:
    def test_column_matches_pairwise(self):
        rng = np.random.default_rng(42)
        X = _random_seqs(20, 20, rng)
        kernel = DirectGkmKernel(l=7, k=5, normalize=True, include_rc=True)

        cache = KernelColumnCache(kernel, X, max_columns=10)
        expected = kernel.pairwise(X, X)

        for i in range(X.shape[0]):
            col = cache.get_column(i)
            np.testing.assert_allclose(col, expected[i], atol=1e-10)

    def test_column_matches_esttrunc(self):
        rng = np.random.default_rng(42)
        X = _random_seqs(15, 20, rng)
        kernel = EstTruncGkmKernel(
            l=7, k=5, d=3, normalize=True, include_rc=True
        )

        cache = KernelColumnCache(kernel, X, max_columns=10)
        expected = kernel.pairwise(X, X)

        for i in range(X.shape[0]):
            col = cache.get_column(i)
            np.testing.assert_allclose(col, expected[i], atol=1e-10)

    def test_cache_hit_returns_same(self):
        rng = np.random.default_rng(42)
        X = _random_seqs(10, 20, rng)
        kernel = DirectGkmKernel(l=7, k=5, normalize=True, include_rc=True)

        cache = KernelColumnCache(kernel, X, max_columns=5)
        col1 = cache.get_column(3)
        col2 = cache.get_column(3)

        assert col1 is col2
        assert cache.hits == 1
        assert cache.misses == 1

    def test_lru_eviction(self):
        rng = np.random.default_rng(42)
        X = _random_seqs(10, 20, rng)
        kernel = DirectGkmKernel(l=7, k=5, normalize=True, include_rc=True)

        cache = KernelColumnCache(kernel, X, max_columns=3)

        cache.get_column(0)
        cache.get_column(1)
        cache.get_column(2)
        assert len(cache._cache) == 3

        cache.get_column(3)
        assert len(cache._cache) == 3
        assert 0 not in cache._cache
        assert 3 in cache._cache

    def test_lru_touch_prevents_eviction(self):
        rng = np.random.default_rng(42)
        X = _random_seqs(10, 20, rng)
        kernel = DirectGkmKernel(l=7, k=5, normalize=True, include_rc=True)

        cache = KernelColumnCache(kernel, X, max_columns=3)

        cache.get_column(0)
        cache.get_column(1)
        cache.get_column(2)
        cache.get_column(0)  # touch 0, making 1 the oldest
        cache.get_column(3)  # evicts 1, not 0

        assert 0 in cache._cache
        assert 1 not in cache._cache

    def test_column_symmetry(self):
        rng = np.random.default_rng(42)
        X = _random_seqs(15, 20, rng)
        kernel = DirectGkmKernel(l=7, k=5, normalize=True, include_rc=True)

        cache = KernelColumnCache(kernel, X, max_columns=15)

        for i in range(5):
            for j in range(i + 1, 5):
                col_i = cache.get_column(i)
                col_j = cache.get_column(j)
                assert col_i[j] == pytest.approx(col_j[i], abs=1e-10)


class TestSmoSolve:
    def test_convergence_separable(self):
        rng = np.random.default_rng(42)
        pos = _biased_seqs(30, 20, rng, bias_base=0)
        neg = _biased_seqs(30, 20, rng, bias_base=3)
        X = np.concatenate([pos, neg])
        y = np.array([1.0] * 30 + [-1.0] * 30)

        kernel = DirectGkmKernel(l=7, k=5, normalize=True, include_rc=True)
        coef, bias = smo_solve(kernel, X, y, C=10.0, tol=1e-3)

        n_sv = np.sum(np.abs(coef) > 1e-10)
        assert n_sv > 0

        score_pos = np.mean(
            [float(np.dot(coef, kernel.pairwise(pos[i:i+1], X)[0]) + bias)
             for i in range(5)]
        )
        score_neg = np.mean(
            [float(np.dot(coef, kernel.pairwise(neg[i:i+1], X)[0]) + bias)
             for i in range(5)]
        )
        assert score_pos > 0
        assert score_neg < 0

    def test_smo_vs_libsvm_agreement(self):
        from libsvm.svmutil import svm_train

        from gkmsvm.gram import compute_gram

        rng = np.random.default_rng(123)
        pos = _biased_seqs(25, 20, rng, bias_base=0)
        neg = _biased_seqs(25, 20, rng, bias_base=3)
        X = np.concatenate([pos, neg])
        y = np.array([1.0] * 25 + [-1.0] * 25)
        N = X.shape[0]

        kernel = DirectGkmKernel(l=7, k=5, normalize=True, include_rc=True)

        coef_smo, bias_smo = smo_solve(
            kernel, X, y, C=1.0, tol=1e-4, cache_size=50
        )

        gram = compute_gram(kernel, X)
        ids = np.arange(1, N + 1, dtype=np.float64).reshape(-1, 1)
        x_train = np.hstack([ids, gram])
        model = svm_train(y.tolist(), x_train, "-s 0 -t 4 -c 1.0 -q")

        n_sv = model.l
        coef_lib = np.zeros(N)
        for i in range(n_sv):
            coef_lib[model.sv_indices[i] - 1] = model.sv_coef[0][i]
        bias_lib = float(-model.rho[0])

        test_seqs = _random_seqs(10, 20, rng)
        scores_smo = np.array([
            float(np.dot(coef_smo, kernel.pairwise(t[None], X)[0]) + bias_smo)
            for t in test_seqs
        ])
        scores_lib = np.array([
            float(np.dot(coef_lib, kernel.pairwise(t[None], X)[0]) + bias_lib)
            for t in test_seqs
        ])

        np.testing.assert_allclose(scores_smo, scores_lib, atol=0.05)

    def test_smo_esttrunc(self):
        rng = np.random.default_rng(42)
        pos = _biased_seqs(20, 20, rng, bias_base=0)
        neg = _biased_seqs(20, 20, rng, bias_base=3)
        X = np.concatenate([pos, neg])
        y = np.array([1.0] * 20 + [-1.0] * 20)

        kernel = EstTruncGkmKernel(
            l=7, k=5, d=3, normalize=True, include_rc=True
        )
        coef, bias = smo_solve(kernel, X, y, C=1.0, tol=1e-3)

        n_sv = np.sum(np.abs(coef) > 1e-10)
        assert n_sv > 0

    def test_train_api_smo(self):
        from gkmsvm import train_gkmsvm

        rng = np.random.default_rng(42)
        pos = ["AAAAAAAAAAAAAAACCCCC"] * 10
        neg = ["TTTTTTTTTTTTTTTGGGGG"] * 10

        model = train_gkmsvm(
            pos, neg, kernel_type="direct", l=7, k=5, C=1.0,
            solver="smo", cache_size=20,
        )

        assert model.num_support_vectors > 0

        from gkmsvm.codec import one_hot_encode
        x_pos = one_hot_encode("AAAAAAAAAAAAAAACCCCC")[None, ...]
        x_neg = one_hot_encode("TTTTTTTTTTTTTTTGGGGG")[None, ...]

        assert model(x_pos).item() > model(x_neg).item()

    def test_smo_respects_C(self):
        rng = np.random.default_rng(42)
        X = _random_seqs(30, 20, rng)
        y = np.array([1.0] * 15 + [-1.0] * 15)

        kernel = DirectGkmKernel(l=7, k=5, normalize=True, include_rc=True)
        coef, _ = smo_solve(kernel, X, y, C=0.5, tol=1e-3)

        alpha = np.abs(coef)
        assert np.all(alpha <= 0.5 + 1e-8)

    def test_no_gram_materialized(self):
        """SMO should work without building the full N×N matrix."""
        rng = np.random.default_rng(42)
        N = 200
        X = _random_seqs(N, 20, rng)
        y = np.array([1.0] * (N // 2) + [-1.0] * (N // 2))

        kernel = DirectGkmKernel(l=7, k=5, normalize=True, include_rc=True)
        coef, bias = smo_solve(
            kernel, X, y, C=1.0, tol=1e-3, cache_size=32,
        )

        n_sv = np.sum(np.abs(coef) > 1e-10)
        assert n_sv > 0

    def test_solver_auto_selection(self):
        """solver='auto' uses libsvm for small N."""
        from gkmsvm import train_gkmsvm

        rng = np.random.default_rng(42)
        pos = ["ACGT" * 5] * 10
        neg = ["TGCA" * 5] * 10

        model = train_gkmsvm(
            pos, neg, kernel_type="direct", l=7, k=5, C=1.0,
            solver="auto",
        )
        assert model.num_support_vectors > 0

    def test_invalid_solver_raises(self):
        from gkmsvm import train_gkmsvm

        with pytest.raises(ValueError, match="Unknown solver"):
            train_gkmsvm(
                ["ACGT" * 5] * 5, ["TGCA" * 5] * 5,
                kernel_type="direct", l=7, k=5, solver="bad",
            )
