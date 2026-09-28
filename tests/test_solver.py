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

    def test_smo_vs_sklearn_agreement(self):
        from sklearn.svm import SVC

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
        clf = SVC(C=1.0, kernel="precomputed").fit(gram, y)

        coef_lib = np.zeros(N)
        for i, idx in enumerate(clf.support_):
            coef_lib[idx] = clf.dual_coef_[0, i]
        bias_lib = float(clf.intercept_[0])

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


class TestTrainGkmsvr:
    def test_svr_basic(self):
        """SVR trains and produces predictions correlated with labels."""
        from gkmsvm import train_gkmsvr

        rng = np.random.default_rng(42)
        n = 40
        seqs = []
        labels = []
        for i in range(n):
            a_frac = i / (n - 1)
            probs = [a_frac * 0.6 + 0.1, 0.1, 0.1, (1 - a_frac) * 0.6 + 0.1]
            idx = rng.choice(4, size=20, p=probs)
            x = np.zeros((4, 20), dtype=np.float64)
            x[idx, np.arange(20)] = 1.0
            seqs.append(x)
            labels.append(float(i) / n)

        X = np.stack(seqs)
        model = train_gkmsvr(
            X, labels, kernel_type="direct", l=7, k=5, C=10.0, epsilon=0.05,
        )

        assert model.num_support_vectors > 0
        preds = np.array([model(X[i:i+1]).item() for i in range(n)])
        corr = np.corrcoef(preds, labels)[0, 1]
        assert corr > 0.5

    def test_svr_string_input(self):
        """SVR accepts string sequences."""
        from gkmsvm import train_gkmsvr

        seqs = [
            "AAAAAAAAAAAAAAACCCCC",
            "AAAAAAAAAAAAAACCCCCG",
            "AAAAAAAAAAAAACCCCGGG",
            "TTTTTTTTTTTTTTTGGGGG",
            "TTTTTTTTTTTTTGGGGGGC",
            "TTTTTTTTTTTTGGGGGCCC",
        ]
        labels = [1.0, 0.8, 0.6, -1.0, -0.8, -0.6]

        model = train_gkmsvr(
            seqs, labels, kernel_type="direct", l=7, k=5, C=1.0,
        )
        assert model.num_support_vectors > 0

    def test_svr_label_mismatch_raises(self):
        """Mismatched sequence/label counts raise ValueError."""
        from gkmsvm import train_gkmsvr

        seqs = ["ACGT" * 5] * 10
        labels = [1.0] * 5

        with pytest.raises(ValueError, match="must match"):
            train_gkmsvr(seqs, labels, kernel_type="direct", l=7, k=5)

    def test_svr_epsilon_effect(self):
        """Larger epsilon produces fewer support vectors (wider tube)."""
        from gkmsvm import train_gkmsvr

        rng = np.random.default_rng(99)
        seqs = _biased_seqs(30, 20, rng, bias_base=0)
        labels = np.linspace(0, 1, 30).tolist()

        model_tight = train_gkmsvr(
            seqs, labels, kernel_type="direct", l=7, k=5,
            C=10.0, epsilon=0.01,
        )
        model_wide = train_gkmsvr(
            seqs, labels, kernel_type="direct", l=7, k=5,
            C=10.0, epsilon=0.3,
        )
        assert model_tight.num_support_vectors >= model_wide.num_support_vectors

    def test_svr_save_load_roundtrip(self):
        """SVR model can be saved and loaded with identical predictions."""
        import tempfile

        from gkmsvm import train_gkmsvr
        from gkmsvm.serialization import load_model

        seqs = [
            "AAAAAAAAAAAAAAACCCCC",
            "AAAAAAAAAAAAAACCCCCG",
            "TTTTTTTTTTTTTTTGGGGG",
            "TTTTTTTTTTTTTGGGGGGC",
        ]
        labels = [1.0, 0.8, -1.0, -0.8]

        model = train_gkmsvr(
            seqs, labels, kernel_type="direct", l=7, k=5, C=1.0,
        )

        from gkmsvm.codec import one_hot_encode
        test_x = one_hot_encode("AAAAAAAAAAAAAAACCCCC")[None, ...]
        score_before = model(test_x).item()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            model.save(f.name)
            loaded = load_model(f.name)

        score_after = loaded(test_x).item()
        assert score_before == pytest.approx(score_after, abs=1e-10)

    def test_svr_esttrunc(self):
        """SVR works with esttrunc kernel."""
        from gkmsvm import train_gkmsvr

        rng = np.random.default_rng(42)
        seqs = _random_seqs(20, 20, rng)
        labels = np.linspace(-1, 1, 20).tolist()

        model = train_gkmsvr(
            seqs, labels, kernel_type="estimated", l=7, k=5, d=3,
            C=1.0, epsilon=0.1,
        )
        assert model.num_support_vectors > 0
