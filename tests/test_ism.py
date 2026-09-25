"""Tests for in silico mutagenesis with window-delta optimization."""
import random

import pytest
import torch

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
    result = torch.zeros(B, C, L, dtype=x.dtype, device=x.device)
    for p in range(L):
        for base in range(4):
            x_mut = x.clone()
            x_mut[:, :, p] = 0
            x_mut[:, base, p] = 1
            score_mut = model(x_mut).squeeze(-1)
            result[:, base, p] = score_mut - ref_score
    return result


class TestISMCorrectness:
    """Compare window-delta ISM against naive brute-force."""

    def test_small_no_rc_no_norm(self):
        svs = torch.stack([one_hot_encode(s) for s in _make_seqs(3, 10, seed=1)])
        coefs = torch.tensor([0.5, -0.3, 0.2])
        model = GkmSVM(svs, coefs, -0.1, "gkm_cnt",
                        {"L": 3, "k": 2, "include_rc": False})
        model.kernel.normalize = False
        x = torch.stack([one_hot_encode(s) for s in _make_seqs(2, 10, seed=2)])

        got = ism(model, x)
        expected = _naive_ism(model, x)
        assert torch.allclose(got, expected, atol=1e-6), \
            f"max diff: {(got - expected).abs().max()}"

    def test_small_with_norm(self):
        svs = torch.stack([one_hot_encode(s) for s in _make_seqs(3, 10, seed=3)])
        coefs = torch.tensor([0.5, -0.3, 0.2])
        model = GkmSVM(svs, coefs, -0.1, "gkm_cnt",
                        {"L": 3, "k": 2, "include_rc": False})
        x = torch.stack([one_hot_encode(s) for s in _make_seqs(2, 10, seed=4)])

        got = ism(model, x)
        expected = _naive_ism(model, x)
        assert torch.allclose(got, expected, atol=1e-5), \
            f"max diff: {(got - expected).abs().max()}"

    def test_small_with_rc(self):
        svs = torch.stack([one_hot_encode(s) for s in _make_seqs(3, 10, seed=5)])
        coefs = torch.tensor([0.5, -0.3, 0.2])
        model = GkmSVM(svs, coefs, -0.1, "gkm_cnt",
                        {"L": 3, "k": 2, "include_rc": True})
        x = torch.stack([one_hot_encode(s) for s in _make_seqs(2, 10, seed=6)])

        got = ism(model, x)
        expected = _naive_ism(model, x)
        assert torch.allclose(got, expected, atol=1e-5), \
            f"max diff: {(got - expected).abs().max()}"

    def test_esttrunc_with_rc_norm(self):
        svs = torch.stack([one_hot_encode(s) for s in _make_seqs(5, 20, seed=7)])
        coefs = torch.randn(5)
        model = GkmSVM(svs, coefs, -0.15, "gkm_esttrunc",
                        {"L": 11, "k": 7, "d": 3, "include_rc": True})
        x = one_hot_encode(_make_seqs(1, 20, seed=8)[0]).unsqueeze(0)

        got = ism(model, x)
        expected = _naive_ism(model, x)
        assert torch.allclose(got, expected, atol=1e-5), \
            f"max diff: {(got - expected).abs().max()}"

    def test_esttrunc_no_rc_no_norm(self):
        svs = torch.stack([one_hot_encode(s) for s in _make_seqs(5, 20, seed=9)])
        coefs = torch.randn(5)
        model = GkmSVM(svs, coefs, -0.15, "gkm_esttrunc",
                        {"L": 11, "k": 7, "d": 3, "include_rc": False})
        model.kernel.normalize = False
        x = one_hot_encode(_make_seqs(1, 20, seed=10)[0]).unsqueeze(0)

        got = ism(model, x)
        expected = _naive_ism(model, x)
        assert torch.allclose(got, expected, atol=1e-6), \
            f"max diff: {(got - expected).abs().max()}"


class TestISMProperties:
    """Test expected properties of ISM output."""

    @pytest.fixture
    def model_and_query(self):
        svs = torch.stack([one_hot_encode(s) for s in _make_seqs(5, 20, seed=11)])
        coefs = torch.randn(5)
        model = GkmSVM(svs, coefs, -0.15, "gkm_esttrunc",
                        {"L": 11, "k": 7, "d": 3, "include_rc": True})
        x = one_hot_encode(_make_seqs(1, 20, seed=12)[0]).unsqueeze(0)
        return model, x

    def test_output_shape(self, model_and_query):
        model, x = model_and_query
        result = ism(model, x)
        assert result.shape == x.shape

    def test_ref_base_is_zero(self, model_and_query):
        model, x = model_and_query
        result = ism(model, x)
        ref_bases = x.argmax(dim=1)
        for b in range(x.shape[0]):
            for p in range(x.shape[2]):
                ref = ref_bases[b, p].item()
                assert abs(result[b, ref, p].item()) < 1e-6

    def test_batch_consistent(self):
        svs = torch.stack([one_hot_encode(s) for s in _make_seqs(3, 15, seed=13)])
        coefs = torch.randn(3)
        model = GkmSVM(svs, coefs, -0.1, "gkm_cnt",
                        {"L": 3, "k": 2, "include_rc": True})

        seqs = _make_seqs(3, 15, seed=14)
        x_batch = torch.stack([one_hot_encode(s) for s in seqs])
        result_batch = ism(model, x_batch)

        for i in range(3):
            result_single = ism(model, x_batch[i:i+1])
            assert torch.allclose(result_batch[i:i+1], result_single, atol=1e-6), \
                f"batch[{i}] max diff: {(result_batch[i:i+1] - result_single).abs().max()}"


class TestISMChunked:
    def test_chunked_matches_full(self):
        svs = torch.stack([one_hot_encode(s) for s in _make_seqs(10, 20, seed=15)])
        coefs = torch.randn(10)
        model = GkmSVM(svs, coefs, -0.15, "gkm_esttrunc",
                        {"L": 11, "k": 7, "d": 3, "include_rc": True})
        x = one_hot_encode(_make_seqs(1, 20, seed=16)[0]).unsqueeze(0)

        full = ism(model, x)
        chunked = ism(model, x, sv_chunk_size=3)
        assert torch.allclose(full, chunked, atol=1e-5), \
            f"max diff: {(full - chunked).abs().max()}"
