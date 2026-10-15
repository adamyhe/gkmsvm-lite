"""Tests for in silico mutagenesis with window-delta optimization."""
import random

import numpy as np
import pytest

from gkmsvm.codec import one_hot_encode
from gkmsvm.ism import ism
from gkmsvm.svm import GkmSVM


def _make_seqs(n, length, seed=42):
    rng = random.Random(seed)
    return ["".join(rng.choice("ACGT") for _ in range(length)) for _ in range(n)]


def _naive_ism(model, x):
    """Brute-force ISM: score every mutant individually."""
    B, C, L = x.shape
    ref_score = model(x).squeeze(-1)
    result = np.zeros((B, C, L), dtype=x.dtype)
    for p in range(L):
        for base in range(4):
            x_mut = x.copy()
            x_mut[:, :, p] = 0
            x_mut[:, base, p] = 1
            score_mut = model(x_mut).squeeze(-1)
            result[:, base, p] = score_mut - ref_score
    return result


class TestISMCorrectness:
    """Compare window-delta ISM against naive brute-force."""

    def test_small_no_rc_no_norm(self):
        svs = np.stack([one_hot_encode(s) for s in _make_seqs(3, 10, seed=1)])
        coefs = np.array([0.5, -0.3, 0.2], dtype=np.float32)
        model = GkmSVM(svs, coefs, -0.1, "gkm_cnt",
                        {"L": 3, "k": 2, "include_rc": False})
        model.kernel.normalize = False
        x = np.stack([one_hot_encode(s) for s in _make_seqs(2, 10, seed=2)])

        got = ism(model, x)
        expected = _naive_ism(model, x)
        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_small_with_norm(self):
        svs = np.stack([one_hot_encode(s) for s in _make_seqs(3, 10, seed=3)])
        coefs = np.array([0.5, -0.3, 0.2], dtype=np.float32)
        model = GkmSVM(svs, coefs, -0.1, "gkm_cnt",
                        {"L": 3, "k": 2, "include_rc": False})
        x = np.stack([one_hot_encode(s) for s in _make_seqs(2, 10, seed=4)])

        got = ism(model, x)
        expected = _naive_ism(model, x)
        np.testing.assert_allclose(got, expected, atol=1e-5)

    def test_small_with_rc(self):
        svs = np.stack([one_hot_encode(s) for s in _make_seqs(3, 10, seed=5)])
        coefs = np.array([0.5, -0.3, 0.2], dtype=np.float32)
        model = GkmSVM(svs, coefs, -0.1, "gkm_cnt",
                        {"L": 3, "k": 2, "include_rc": True})
        x = np.stack([one_hot_encode(s) for s in _make_seqs(2, 10, seed=6)])

        got = ism(model, x)
        expected = _naive_ism(model, x)
        np.testing.assert_allclose(got, expected, atol=1e-5)

    def test_esttrunc_with_rc_norm(self):
        svs = np.stack([one_hot_encode(s) for s in _make_seqs(5, 20, seed=7)])
        rng = np.random.default_rng(7)
        coefs = rng.standard_normal(5).astype(np.float32)
        model = GkmSVM(svs, coefs, -0.15, "gkm_esttrunc",
                        {"L": 11, "k": 7, "d": 3, "include_rc": True})
        x = one_hot_encode(_make_seqs(1, 20, seed=8)[0])[np.newaxis]

        got = ism(model, x)
        expected = _naive_ism(model, x)
        np.testing.assert_allclose(got, expected, atol=1e-5)

    def test_esttrunc_no_rc_no_norm(self):
        svs = np.stack([one_hot_encode(s) for s in _make_seqs(5, 20, seed=9)])
        rng = np.random.default_rng(9)
        coefs = rng.standard_normal(5).astype(np.float32)
        model = GkmSVM(svs, coefs, -0.15, "gkm_esttrunc",
                        {"L": 11, "k": 7, "d": 3, "include_rc": False})
        model.kernel.normalize = False
        x = one_hot_encode(_make_seqs(1, 20, seed=10)[0])[np.newaxis]

        got = ism(model, x)
        expected = _naive_ism(model, x)
        np.testing.assert_allclose(got, expected, atol=1e-6)


class TestISMProperties:
    """Test expected properties of ISM output."""

    @pytest.fixture
    def model_and_query(self):
        svs = np.stack([one_hot_encode(s) for s in _make_seqs(5, 20, seed=11)])
        rng = np.random.default_rng(11)
        coefs = rng.standard_normal(5).astype(np.float32)
        model = GkmSVM(svs, coefs, -0.15, "gkm_esttrunc",
                        {"L": 11, "k": 7, "d": 3, "include_rc": True})
        x = one_hot_encode(_make_seqs(1, 20, seed=12)[0])[np.newaxis]
        return model, x

    def test_output_shape(self, model_and_query):
        model, x = model_and_query
        result = ism(model, x)
        assert result.shape == x.shape

    def test_ref_base_is_zero(self, model_and_query):
        model, x = model_and_query
        result = ism(model, x)
        ref_bases = x.argmax(axis=1)
        for b in range(x.shape[0]):
            for p in range(x.shape[2]):
                ref = int(ref_bases[b, p])
                assert abs(float(result[b, ref, p])) < 1e-6

    def test_batch_consistent(self):
        svs = np.stack([one_hot_encode(s) for s in _make_seqs(3, 15, seed=13)])
        rng = np.random.default_rng(13)
        coefs = rng.standard_normal(3).astype(np.float32)
        model = GkmSVM(svs, coefs, -0.1, "gkm_cnt",
                        {"L": 3, "k": 2, "include_rc": True})

        seqs = _make_seqs(3, 15, seed=14)
        x_batch = np.stack([one_hot_encode(s) for s in seqs])
        result_batch = ism(model, x_batch)

        for i in range(3):
            result_single = ism(model, x_batch[i:i+1])
            np.testing.assert_allclose(result_batch[i:i+1], result_single, atol=1e-6)


class TestISMChunked:
    def test_chunked_matches_full(self):
        svs = np.stack([one_hot_encode(s) for s in _make_seqs(10, 20, seed=15)])
        rng = np.random.default_rng(15)
        coefs = rng.standard_normal(10).astype(np.float32)
        model = GkmSVM(svs, coefs, -0.15, "gkm_esttrunc",
                        {"L": 11, "k": 7, "d": 3, "include_rc": True})
        x = one_hot_encode(_make_seqs(1, 20, seed=16)[0])[np.newaxis]

        full = ism(model, x)
        model.sv_chunk_size = 3
        chunked = ism(model, x)
        np.testing.assert_allclose(full, chunked, atol=1e-5)


@pytest.mark.parametrize("kernel_type, params", [
    ("gkmrbf", {"L": 6, "k": 4, "d": 2, "gamma": 2.0}),
    ("wgkm", {"L": 6, "k": 4, "M": 3, "H": 2.0}),
    ("wgkmrbf", {"L": 6, "k": 4, "M": 3, "H": 2.0, "gamma": 2.0}),
])
def test_ism_scores_mutants_without_index_path(kernel_type, params):
    """-t 3/4/5 ISM equals score(mutant) - score(ref)."""
    from gkmsvm.svm import GkmSVM

    rng = np.random.default_rng(3)
    X = np.eye(4)[rng.integers(0, 4, (12, 20))].transpose(0, 2, 1)
    model = GkmSVM(X, rng.normal(size=12), 0.1, kernel_type, params)
    d = ism(model, X[:2])
    assert d.shape == (2, 4, 20)
    for b, p in [(0, 0), (1, 9), (1, 19)]:
        ref_base = X[b, :, p].argmax()
        assert d[b, ref_base, p] == 0
        v = (ref_base + 1) % 4
        mut = X[b].copy()
        mut[:, p] = 0
        mut[v, p] = 1
        expected = model(mut[None]) - model(X[b:b + 1])
        np.testing.assert_allclose(d[b, v, p], expected[0, 0], rtol=1e-9, atol=1e-12)
