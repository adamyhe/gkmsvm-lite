"""Tests for GkmExplain attribution."""
import random

import pytest
import torch

from gkmsvm.codec import one_hot_encode
from gkmsvm.explain import gkmexplain
from gkmsvm.ism import ism
from gkmsvm.svm import GkmSVM


def _make_seqs(n, length, seed=42):
    rng = random.Random(seed)
    return ["".join(rng.choice("ACGT") for _ in range(length)) for _ in range(n)]


def _make_model(n_sv, seqlen, kernel_type, kernel_params, seed=1):
    svs = torch.stack([one_hot_encode(s) for s in _make_seqs(n_sv, seqlen, seed)])
    coefs = torch.randn(n_sv, generator=torch.Generator().manual_seed(seed))
    return GkmSVM(svs, coefs, -0.15, kernel_type, kernel_params)


class TestGkmExplainDecomposition:
    """Verify that explanation scores decompose the SVM decision value."""

    def test_sum_equals_score_no_rc_no_norm(self):
        model = _make_model(3, 10, "gkm_cnt", {"L": 3, "k": 2, "include_rc": False})
        model.kernel.normalize = False
        x = torch.stack([one_hot_encode(s) for s in _make_seqs(2, 10, seed=2)])

        exp = gkmexplain(model, x, mode=0)
        score = model(x).squeeze(-1)
        exp_sum = (exp * x).sum(dim=(1, 2))
        expected = score - model.bias
        assert torch.allclose(exp_sum, expected.to(exp_sum.dtype), atol=1e-8), \
            f"max diff: {(exp_sum - expected).abs().max()}"

    def test_sum_equals_score_with_norm(self):
        model = _make_model(3, 10, "gkm_cnt", {"L": 3, "k": 2, "include_rc": False})
        x = torch.stack([one_hot_encode(s) for s in _make_seqs(2, 10, seed=3)])

        exp = gkmexplain(model, x, mode=0)
        score = model(x).squeeze(-1)
        exp_sum = (exp * x).sum(dim=(1, 2))
        expected = score - model.bias
        assert torch.allclose(exp_sum, expected.to(exp_sum.dtype), atol=1e-6), \
            f"max diff: {(exp_sum - expected).abs().max()}"

    def test_sum_equals_score_with_rc(self):
        model = _make_model(3, 10, "gkm_cnt", {"L": 3, "k": 2, "include_rc": True})
        x = torch.stack([one_hot_encode(s) for s in _make_seqs(2, 10, seed=4)])

        exp = gkmexplain(model, x, mode=0)
        score = model(x).squeeze(-1)
        exp_sum = (exp * x).sum(dim=(1, 2))
        expected = score - model.bias
        assert torch.allclose(exp_sum, expected.to(exp_sum.dtype), atol=1e-6), \
            f"max diff: {(exp_sum - expected).abs().max()}"

    def test_sum_equals_score_esttrunc(self):
        model = _make_model(
            5, 20, "gkm_esttrunc",
            {"L": 11, "k": 7, "d": 3, "include_rc": True}, seed=5
        )
        x = one_hot_encode(_make_seqs(1, 20, seed=6)[0]).unsqueeze(0)

        exp = gkmexplain(model, x, mode=0)
        score = model(x).squeeze(-1)
        exp_sum = (exp * x).sum(dim=(1, 2))
        expected = score - model.bias
        assert torch.allclose(exp_sum, expected.to(exp_sum.dtype), atol=1e-5), \
            f"max diff: {(exp_sum - expected).abs().max()}"

    def test_sum_equals_score_esttrunc_no_rc_no_norm(self):
        model = _make_model(
            5, 20, "gkm_esttrunc",
            {"L": 11, "k": 7, "d": 3, "include_rc": False}, seed=7
        )
        model.kernel.normalize = False
        x = one_hot_encode(_make_seqs(1, 20, seed=8)[0]).unsqueeze(0)

        exp = gkmexplain(model, x, mode=0)
        score = model(x).squeeze(-1)
        exp_sum = (exp * x).sum(dim=(1, 2))
        expected = score - model.bias
        assert torch.allclose(exp_sum, expected.to(exp_sum.dtype), atol=1e-8), \
            f"max diff: {(exp_sum - expected).abs().max()}"


class TestGkmExplainMode0Properties:
    """Test structural properties of mode 0 output."""

    @pytest.fixture
    def model_and_query(self):
        model = _make_model(
            5, 20, "gkm_esttrunc",
            {"L": 11, "k": 7, "d": 3, "include_rc": True}, seed=9
        )
        x = one_hot_encode(_make_seqs(1, 20, seed=10)[0]).unsqueeze(0)
        return model, x

    def test_output_shape(self, model_and_query):
        model, x = model_and_query
        exp = gkmexplain(model, x, mode=0)
        assert exp.shape == x.shape

    def test_nonref_bases_are_zero(self, model_and_query):
        model, x = model_and_query
        exp = gkmexplain(model, x, mode=0)
        non_ref = (1 - x).bool()
        assert (exp[non_ref].abs() < 1e-10).all(), \
            f"max nonref value: {exp[non_ref].abs().max()}"

    def test_batch_consistent(self):
        model = _make_model(3, 15, "gkm_cnt", {"L": 3, "k": 2, "include_rc": True})
        seqs = _make_seqs(3, 15, seed=11)
        x_batch = torch.stack([one_hot_encode(s) for s in seqs])
        result_batch = gkmexplain(model, x_batch, mode=0)

        for i in range(3):
            result_single = gkmexplain(model, x_batch[i:i + 1], mode=0)
            assert torch.allclose(result_batch[i:i + 1], result_single, atol=1e-8), \
                f"batch[{i}] max diff: {(result_batch[i:i+1] - result_single).abs().max()}"


class TestGkmExplainMode1:
    """Test hypothetical importance scores (mode 1)."""

    def test_output_shape(self):
        model = _make_model(3, 15, "gkm_cnt", {"L": 3, "k": 2, "include_rc": True})
        x = one_hot_encode(_make_seqs(1, 15, seed=12)[0]).unsqueeze(0)
        exp = gkmexplain(model, x, mode=1)
        assert exp.shape == x.shape

    def test_mode1_matches_mode0_at_ref_bases(self):
        """At reference base positions, mode 1 should equal mode 0."""
        model = _make_model(
            5, 20, "gkm_esttrunc",
            {"L": 11, "k": 7, "d": 3, "include_rc": True}, seed=13
        )
        x = one_hot_encode(_make_seqs(1, 20, seed=14)[0]).unsqueeze(0)
        exp0 = gkmexplain(model, x, mode=0)
        exp1 = gkmexplain(model, x, mode=1)
        ref_mask = x.bool()
        assert torch.allclose(
            exp0[ref_mask], exp1[ref_mask], atol=1e-8
        ), f"max diff: {(exp0[ref_mask] - exp1[ref_mask]).abs().max()}"

    def test_mode1_nonref_nonzero(self):
        """Mode 1 should have nonzero values at non-reference bases."""
        model = _make_model(
            3, 15, "gkm_cnt", {"L": 3, "k": 2, "include_rc": True}, seed=15
        )
        x = one_hot_encode(_make_seqs(1, 15, seed=16)[0]).unsqueeze(0)
        exp1 = gkmexplain(model, x, mode=1)
        non_ref = (1 - x).bool()
        assert exp1[non_ref].abs().max() > 1e-10, \
            "mode 1 should have nonzero values at non-reference bases"

    def test_mode1_no_rc_no_norm(self):
        model = _make_model(3, 12, "gkm_cnt", {"L": 3, "k": 2, "include_rc": False})
        model.kernel.normalize = False
        x = torch.stack([one_hot_encode(s) for s in _make_seqs(2, 12, seed=17)])
        exp = gkmexplain(model, x, mode=1)
        assert exp.shape == x.shape


class TestGkmExplainChunked:
    """Test that SV chunking produces identical results."""

    def test_chunked_mode0(self):
        model = _make_model(
            10, 20, "gkm_esttrunc",
            {"L": 11, "k": 7, "d": 3, "include_rc": True}, seed=18
        )
        x = one_hot_encode(_make_seqs(1, 20, seed=19)[0]).unsqueeze(0)

        full = gkmexplain(model, x, mode=0)
        chunked = gkmexplain(model, x, mode=0, sv_chunk_size=3)
        assert torch.allclose(full, chunked, atol=1e-8), \
            f"max diff: {(full - chunked).abs().max()}"

    def test_chunked_mode1(self):
        model = _make_model(
            10, 20, "gkm_esttrunc",
            {"L": 11, "k": 7, "d": 3, "include_rc": True}, seed=20
        )
        x = one_hot_encode(_make_seqs(1, 20, seed=21)[0]).unsqueeze(0)

        full = gkmexplain(model, x, mode=1)
        chunked = gkmexplain(model, x, mode=1, sv_chunk_size=4)
        assert torch.allclose(full, chunked, atol=1e-8), \
            f"max diff: {(full - chunked).abs().max()}"


class TestGkmExplainVsISM:
    """Cross-validate GkmExplain against ISM where applicable."""

    def test_mode1_delta_matches_ism(self):
        """Hypothetical deltas (mode1[alt] - mode1[ref]) should correlate with ISM."""
        model = _make_model(
            5, 15, "gkm_cnt", {"L": 3, "k": 2, "include_rc": True}, seed=22
        )
        x = one_hot_encode(_make_seqs(1, 15, seed=23)[0]).unsqueeze(0)

        exp1 = gkmexplain(model, x, mode=1)
        ref_scores = (exp1 * x).sum(dim=1, keepdim=True)
        hyp_delta = exp1 - ref_scores

        ism_result = ism(model, x)

        corr = torch.corrcoef(
            torch.stack([hyp_delta.flatten(), ism_result.flatten().to(hyp_delta.dtype)])
        )[0, 1]
        assert corr > 0.5, f"correlation {corr:.3f} too low"


class TestGkmExplainEdgeCases:
    """Edge case tests."""

    def test_invalid_mode(self):
        model = _make_model(3, 10, "gkm_cnt", {"L": 3, "k": 2, "include_rc": True})
        x = one_hot_encode(_make_seqs(1, 10, seed=24)[0]).unsqueeze(0)
        with pytest.raises(ValueError, match="mode must be 0 or 1"):
            gkmexplain(model, x, mode=5)

    def test_single_sv(self):
        model = _make_model(1, 10, "gkm_cnt", {"L": 3, "k": 2, "include_rc": True})
        x = one_hot_encode(_make_seqs(1, 10, seed=25)[0]).unsqueeze(0)
        exp = gkmexplain(model, x, mode=0)
        score = model(x).squeeze(-1)
        exp_sum = (exp * x).sum(dim=(1, 2))
        expected = score - model.bias
        assert torch.allclose(exp_sum, expected.to(exp_sum.dtype), atol=1e-6)
