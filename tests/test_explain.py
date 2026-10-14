"""Tests for GkmExplain attribution."""
import random

import numpy as np
import pytest

from gkmsvm.codec import one_hot_encode
from gkmsvm.explain import gkmexplain
from gkmsvm.ism import ism
from gkmsvm.svm import GkmSVM


def _make_seqs(n, length, seed=42):
    rng = random.Random(seed)
    return ["".join(rng.choice("ACGT") for _ in range(length)) for _ in range(n)]


def _make_model(n_sv, seqlen, kernel_type, kernel_params, seed=1):
    svs = np.stack([one_hot_encode(s) for s in _make_seqs(n_sv, seqlen, seed)])
    rng = np.random.default_rng(seed)
    coefs = rng.standard_normal(n_sv).astype(np.float32)
    return GkmSVM(svs, coefs, -0.15, kernel_type, kernel_params)


class TestGkmExplainDecomposition:
    """Verify that explanation scores decompose the SVM decision value."""

    def test_sum_equals_score_no_rc_no_norm(self):
        model = _make_model(3, 10, "gkm_cnt", {"L": 3, "k": 2, "include_rc": False})
        model.kernel.normalize = False
        x = np.stack([one_hot_encode(s) for s in _make_seqs(2, 10, seed=2)])

        exp = gkmexplain(model, x, mode="importance")
        score = model(x).squeeze(-1)
        exp_sum = (exp * x).sum(axis=(1, 2))
        expected = score - model.bias
        np.testing.assert_allclose(exp_sum, expected.astype(exp_sum.dtype), atol=1e-8)

    def test_sum_equals_score_with_norm(self):
        model = _make_model(3, 10, "gkm_cnt", {"L": 3, "k": 2, "include_rc": False})
        x = np.stack([one_hot_encode(s) for s in _make_seqs(2, 10, seed=3)])

        exp = gkmexplain(model, x, mode="importance")
        score = model(x).squeeze(-1)
        exp_sum = (exp * x).sum(axis=(1, 2))
        expected = score - model.bias
        np.testing.assert_allclose(exp_sum, expected.astype(exp_sum.dtype), atol=1e-6)

    def test_sum_equals_score_with_rc(self):
        model = _make_model(3, 10, "gkm_cnt", {"L": 3, "k": 2, "include_rc": True})
        x = np.stack([one_hot_encode(s) for s in _make_seqs(2, 10, seed=4)])

        exp = gkmexplain(model, x, mode="importance")
        score = model(x).squeeze(-1)
        exp_sum = (exp * x).sum(axis=(1, 2))
        expected = score - model.bias
        np.testing.assert_allclose(exp_sum, expected.astype(exp_sum.dtype), atol=1e-6)

    def test_sum_equals_score_esttrunc(self):
        model = _make_model(
            5, 20, "gkm_esttrunc",
            {"L": 11, "k": 7, "d": 3, "include_rc": True}, seed=5
        )
        x = one_hot_encode(_make_seqs(1, 20, seed=6)[0])[np.newaxis]

        exp = gkmexplain(model, x, mode="importance")
        score = model(x).squeeze(-1)
        exp_sum = (exp * x).sum(axis=(1, 2))
        expected = score - model.bias
        np.testing.assert_allclose(exp_sum, expected.astype(exp_sum.dtype), atol=1e-5)

    def test_sum_equals_score_esttrunc_no_rc_no_norm(self):
        model = _make_model(
            5, 20, "gkm_esttrunc",
            {"L": 11, "k": 7, "d": 3, "include_rc": False}, seed=7
        )
        model.kernel.normalize = False
        x = one_hot_encode(_make_seqs(1, 20, seed=8)[0])[np.newaxis]

        exp = gkmexplain(model, x, mode="importance")
        score = model(x).squeeze(-1)
        exp_sum = (exp * x).sum(axis=(1, 2))
        expected = score - model.bias
        np.testing.assert_allclose(exp_sum, expected.astype(exp_sum.dtype), atol=1e-8)


class TestGkmExplainImportanceProperties:
    """Test structural properties of importance mode output."""

    @pytest.fixture
    def model_and_query(self):
        model = _make_model(
            5, 20, "gkm_esttrunc",
            {"L": 11, "k": 7, "d": 3, "include_rc": True}, seed=9
        )
        x = one_hot_encode(_make_seqs(1, 20, seed=10)[0])[np.newaxis]
        return model, x

    def test_output_shape(self, model_and_query):
        model, x = model_and_query
        exp = gkmexplain(model, x, mode="importance")
        assert exp.shape == x.shape

    def test_nonref_bases_are_zero(self, model_and_query):
        model, x = model_and_query
        exp = gkmexplain(model, x, mode="importance")
        non_ref = (1 - x).astype(bool)
        assert (np.abs(exp[non_ref]) < 1e-10).all(), \
            f"max nonref value: {np.abs(exp[non_ref]).max()}"

    def test_batch_consistent(self):
        model = _make_model(3, 15, "gkm_cnt", {"L": 3, "k": 2, "include_rc": True})
        seqs = _make_seqs(3, 15, seed=11)
        x_batch = np.stack([one_hot_encode(s) for s in seqs])
        result_batch = gkmexplain(model, x_batch, mode="importance")

        for i in range(3):
            result_single = gkmexplain(model, x_batch[i:i + 1], mode="importance")
            np.testing.assert_allclose(result_batch[i:i + 1], result_single, atol=1e-8)

    def test_legacy_int_mode(self, model_and_query):
        """Integer mode aliases still work."""
        model, x = model_and_query
        exp_str = gkmexplain(model, x, mode="importance")
        exp_int = gkmexplain(model, x, mode=0)
        np.testing.assert_allclose(exp_str, exp_int, atol=1e-10)


class TestGkmExplainHypothetical:
    """Test hypothetical importance scores."""

    def test_output_shape(self):
        model = _make_model(3, 15, "gkm_cnt", {"L": 3, "k": 2, "include_rc": True})
        x = one_hot_encode(_make_seqs(1, 15, seed=12)[0])[np.newaxis]
        exp = gkmexplain(model, x, mode="hypothetical")
        assert exp.shape == x.shape

    def test_hypothetical_matches_importance_at_ref_bases(self):
        """At reference base positions, hypothetical should equal importance."""
        model = _make_model(
            5, 20, "gkm_esttrunc",
            {"L": 11, "k": 7, "d": 3, "include_rc": True}, seed=13
        )
        x = one_hot_encode(_make_seqs(1, 20, seed=14)[0])[np.newaxis]
        exp0 = gkmexplain(model, x, mode="importance")
        exp1 = gkmexplain(model, x, mode="hypothetical")
        ref_mask = x.astype(bool)
        np.testing.assert_allclose(
            exp0[ref_mask], exp1[ref_mask], atol=1e-8
        )

    def test_hypothetical_nonref_nonzero(self):
        """Hypothetical should have nonzero values at non-reference bases."""
        model = _make_model(
            3, 15, "gkm_cnt", {"L": 3, "k": 2, "include_rc": True}, seed=15
        )
        x = one_hot_encode(_make_seqs(1, 15, seed=16)[0])[np.newaxis]
        exp1 = gkmexplain(model, x, mode="hypothetical")
        non_ref = (1 - x).astype(bool)
        assert np.abs(exp1[non_ref]).max() > 1e-10, \
            "hypothetical should have nonzero values at non-reference bases"

    def test_hypothetical_no_rc_no_norm(self):
        model = _make_model(3, 12, "gkm_cnt", {"L": 3, "k": 2, "include_rc": False})
        model.kernel.normalize = False
        x = np.stack([one_hot_encode(s) for s in _make_seqs(2, 12, seed=17)])
        exp = gkmexplain(model, x, mode="hypothetical")
        assert exp.shape == x.shape


class TestGkmExplainChunked:
    """Test that SV chunking produces identical results."""

    def test_chunked_importance(self):
        model = _make_model(
            10, 20, "gkm_esttrunc",
            {"L": 11, "k": 7, "d": 3, "include_rc": True}, seed=18
        )
        x = one_hot_encode(_make_seqs(1, 20, seed=19)[0])[np.newaxis]

        full = gkmexplain(model, x, mode="importance")
        model.sv_chunk_size = 3
        chunked = gkmexplain(model, x, mode="importance")
        np.testing.assert_allclose(full, chunked, atol=1e-8)

    def test_chunked_hypothetical(self):
        model = _make_model(
            10, 20, "gkm_esttrunc",
            {"L": 11, "k": 7, "d": 3, "include_rc": True}, seed=20
        )
        x = one_hot_encode(_make_seqs(1, 20, seed=21)[0])[np.newaxis]

        full = gkmexplain(model, x, mode="hypothetical")
        model.sv_chunk_size = 4
        chunked = gkmexplain(model, x, mode="hypothetical")
        np.testing.assert_allclose(full, chunked, atol=1e-8)

    def test_chunked_perturbation(self):
        model = _make_model(
            10, 20, "gkm_esttrunc",
            {"L": 11, "k": 7, "d": 3, "include_rc": True}, seed=20
        )
        x = one_hot_encode(_make_seqs(1, 20, seed=21)[0])[np.newaxis]

        full = gkmexplain(model, x, mode="perturbation")
        model.sv_chunk_size = 4
        chunked = gkmexplain(model, x, mode="perturbation")
        np.testing.assert_allclose(full, chunked, atol=1e-8)


class TestGkmExplainVsISM:
    """Cross-validate GkmExplain against ISM where applicable."""

    def test_hypothetical_delta_matches_ism(self):
        """Hypothetical deltas (hyp[alt] - hyp[ref]) should correlate with ISM."""
        model = _make_model(
            5, 15, "gkm_cnt", {"L": 3, "k": 2, "include_rc": True}, seed=22
        )
        x = one_hot_encode(_make_seqs(1, 15, seed=23)[0])[np.newaxis]

        exp1 = gkmexplain(model, x, mode="hypothetical")
        ref_scores = (exp1 * x).sum(axis=1, keepdims=True)
        hyp_delta = exp1 - ref_scores

        ism_result = ism(model, x)

        corr = np.corrcoef(
            hyp_delta.flatten().astype(np.float64),
            ism_result.flatten().astype(np.float64),
        )[0, 1]
        assert corr > 0.5, f"correlation {corr:.3f} too low"

    def test_perturbation_matches_ism(self):
        """Perturbation mode should closely match ISM (both estimate score change)."""
        model = _make_model(
            5, 20, "gkm_esttrunc",
            {"L": 11, "k": 7, "d": 3, "include_rc": True}, seed=22,
        )
        x = one_hot_encode(_make_seqs(1, 20, seed=23)[0])[np.newaxis]

        pert = gkmexplain(model, x, mode="perturbation")
        ism_result = ism(model, x)

        non_ref = (1 - x).astype(bool)
        corr = np.corrcoef(
            pert[non_ref].flatten().astype(np.float64),
            ism_result[non_ref].flatten().astype(np.float64),
        )[0, 1]
        assert corr > 0.9, f"perturbation vs ISM correlation {corr:.3f} too low"


class TestCompletenessAxiom:
    """Completeness axiom: sum of importance attributions equals score minus bias.

    For any input x: Σ_{b,p} gkmexplain(x, mode="importance")[b, p] = f(x) - bias
    where f(x) = Σ_s coef_s * K_norm(x, sv_s).

    This holds because importance mode decomposes K(x, s) into per-position
    contributions that sum to K(x, s), and normalization is folded into
    the effective coefficient as a constant.
    """

    @pytest.mark.parametrize("kernel_type,params,seqlen,n_sv,atol", [
        ("gkm_cnt", {"L": 3, "k": 2, "include_rc": False}, 10, 5, 1e-8),
        ("gkm_cnt", {"L": 3, "k": 2, "include_rc": True}, 10, 5, 1e-6),
        ("gkm_cnt", {"L": 5, "k": 3, "include_rc": True}, 15, 8, 1e-6),
        ("gkm_esttrunc", {"L": 11, "k": 7, "d": 3, "include_rc": True}, 20, 5, 1e-5),
        ("gkm_esttrunc", {"L": 11, "k": 7, "d": 3, "include_rc": False}, 20, 5, 1e-5),
        ("gkm_esttrunc", {"L": 7, "k": 4, "d": 2, "include_rc": True}, 15, 6, 1e-5),
    ])
    def test_completeness_normalized(self, kernel_type, params, seqlen, n_sv, atol):
        model = _make_model(n_sv, seqlen, kernel_type, params, seed=100)
        x = np.stack([one_hot_encode(s) for s in _make_seqs(3, seqlen, seed=200)])

        attr = gkmexplain(model, x, mode="importance")
        scores = model(x).squeeze(-1)
        attr_sum = attr.sum(axis=(1, 2))
        np.testing.assert_allclose(attr_sum, scores - model.bias, atol=atol)

    @pytest.mark.parametrize("kernel_type,params,seqlen,n_sv", [
        ("gkm_cnt", {"L": 3, "k": 2, "include_rc": False}, 10, 5),
        ("gkm_cnt", {"L": 3, "k": 2, "include_rc": True}, 12, 4),
        ("gkm_esttrunc", {"L": 11, "k": 7, "d": 3, "include_rc": False}, 20, 5),
    ])
    def test_completeness_unnormalized(self, kernel_type, params, seqlen, n_sv):
        model = _make_model(n_sv, seqlen, kernel_type, params, seed=101)
        model.kernel.normalize = False
        x = np.stack([one_hot_encode(s) for s in _make_seqs(2, seqlen, seed=201)])

        attr = gkmexplain(model, x, mode="importance")
        scores = model(x).squeeze(-1)
        attr_sum = attr.sum(axis=(1, 2))
        np.testing.assert_allclose(attr_sum, scores - model.bias, atol=1e-5)

    def test_completeness_dense_fallback(self):
        """d=l forces dense path (min_matches=0, no packed skip)."""
        model = _make_model(
            4, 12, "gkm_esttrunc",
            {"L": 5, "k": 3, "d": 5, "include_rc": True}, seed=102,
        )
        x = np.stack([one_hot_encode(s) for s in _make_seqs(2, 12, seed=202)])

        attr = gkmexplain(model, x, mode="importance")
        scores = model(x).squeeze(-1)
        attr_sum = attr.sum(axis=(1, 2))
        np.testing.assert_allclose(attr_sum, scores - model.bias, atol=1e-6)

    def test_completeness_chunked_matches_full(self):
        """Chunked SV processing preserves the completeness axiom."""
        model = _make_model(
            12, 20, "gkm_esttrunc",
            {"L": 11, "k": 7, "d": 3, "include_rc": True}, seed=103,
        )
        x = np.stack([one_hot_encode(s) for s in _make_seqs(2, 20, seed=203)])

        attr_full = gkmexplain(model, x, mode="importance")
        model.sv_chunk_size = 4
        attr_chunked = gkmexplain(model, x, mode="importance")

        scores = model(x).squeeze(-1)
        expected = scores - model.bias

        np.testing.assert_allclose(attr_full.sum(axis=(1, 2)), expected, atol=1e-5)
        np.testing.assert_allclose(attr_chunked.sum(axis=(1, 2)), expected, atol=1e-5)
        np.testing.assert_allclose(attr_full, attr_chunked, atol=1e-8)


class TestGkmExplainPerturbation:
    """Test perturbation effect mode (for variant scoring)."""

    def test_output_shape(self):
        model = _make_model(3, 15, "gkm_cnt", {"L": 3, "k": 2, "include_rc": True})
        x = one_hot_encode(_make_seqs(1, 15, seed=30)[0])[np.newaxis]
        pert = gkmexplain(model, x, mode="perturbation")
        assert pert.shape == x.shape

    def test_ref_bases_are_zero(self):
        """Perturbation effect at reference base should be zero (no mutation)."""
        model = _make_model(
            5, 20, "gkm_esttrunc",
            {"L": 11, "k": 7, "d": 3, "include_rc": True}, seed=31,
        )
        x = one_hot_encode(_make_seqs(1, 20, seed=32)[0])[np.newaxis]
        pert = gkmexplain(model, x, mode="perturbation")
        ref_mask = x.astype(bool)
        assert (np.abs(pert[ref_mask]) < 1e-10).all(), \
            f"max ref-base perturbation value: {np.abs(pert[ref_mask]).max()}"

    def test_perturbation_nonref_nonzero(self):
        model = _make_model(
            3, 15, "gkm_cnt", {"L": 3, "k": 2, "include_rc": True}, seed=33,
        )
        x = one_hot_encode(_make_seqs(1, 15, seed=34)[0])[np.newaxis]
        pert = gkmexplain(model, x, mode="perturbation")
        non_ref = (1 - x).astype(bool)
        assert np.abs(pert[non_ref]).max() > 1e-10

    def test_perturbation_score_variants_matches_kernel(self):
        """score_variants(method='gkmexplain') should correlate well with kernel VEP."""
        model = _make_model(
            5, 15, "gkm_cnt", {"L": 3, "k": 2, "include_rc": True}, seed=35,
        )
        seqs = _make_seqs(4, 15, seed=36)
        ref = np.stack([one_hot_encode(s) for s in seqs])
        alt = ref.copy()
        for i in range(4):
            pos = (i * 3 + 2) % 15
            ref_base = ref[i, :, pos].argmax()
            alt_base = (ref_base + 1) % 4
            alt[i, :, pos] = 0
            alt[i, alt_base, pos] = 1

        kernel_scores = model.score_variants(ref, alt, method="kernel").flatten()
        explain_scores = model.score_variants(ref, alt, method="gkmexplain").flatten()

        corr = np.corrcoef(kernel_scores, explain_scores)[0, 1]
        assert corr > 0.9, f"kernel vs gkmexplain VEP correlation {corr:.3f} too low"

    def test_perturbation_dense_fallback(self):
        """d=l forces dense path for perturbation mode."""
        model = _make_model(
            4, 12, "gkm_esttrunc",
            {"L": 5, "k": 3, "d": 5, "include_rc": True}, seed=37,
        )
        x = one_hot_encode(_make_seqs(1, 12, seed=38)[0])[np.newaxis]
        pert = gkmexplain(model, x, mode="perturbation")
        ref_mask = x.astype(bool)
        assert (np.abs(pert[ref_mask]) < 1e-10).all()


class TestGkmExplainEdgeCases:
    """Edge case tests."""

    def test_invalid_mode(self):
        model = _make_model(3, 10, "gkm_cnt", {"L": 3, "k": 2, "include_rc": True})
        x = one_hot_encode(_make_seqs(1, 10, seed=24)[0])[np.newaxis]
        with pytest.raises(ValueError, match="mode must be"):
            gkmexplain(model, x, mode="invalid")

    def test_invalid_int_mode(self):
        model = _make_model(3, 10, "gkm_cnt", {"L": 3, "k": 2, "include_rc": True})
        x = one_hot_encode(_make_seqs(1, 10, seed=24)[0])[np.newaxis]
        with pytest.raises(ValueError, match="mode must be"):
            gkmexplain(model, x, mode=5)

    def test_single_sv(self):
        model = _make_model(1, 10, "gkm_cnt", {"L": 3, "k": 2, "include_rc": True})
        x = one_hot_encode(_make_seqs(1, 10, seed=25)[0])[np.newaxis]
        exp = gkmexplain(model, x, mode="importance")
        score = model(x).squeeze(-1)
        exp_sum = (exp * x).sum(axis=(1, 2))
        expected = score - model.bias
        np.testing.assert_allclose(exp_sum, expected.astype(exp_sum.dtype), atol=1e-6)


class TestMutationImpactOracle:
    """Mutation impact vs lsgkm ``gkmexplain -m 5`` on -t 2 and -t 3 models."""

    FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"

    def _load(self, kind):
        from gkmsvm.fasta import read_fasta
        from gkmsvm.importers.lsgkm import load_lsgkm_model

        model = load_lsgkm_model(
            str(self.FIXTURES / f"lsgkm_mode5_{kind}.model.txt.gz"), dtype=np.float64
        )
        seqs = [s for _, s in read_fasta(str(self.FIXTURES / "lsgkm_mode5_test51.fa"))]
        rows = [l.split("\t") for l in open(self.FIXTURES / f"lsgkm_mode5_{kind}.txt")]
        oracle = np.array([[float(v) for v in r[2].split(",")] for r in rows])
        pred = np.array(
            [float(l.split("\t")[1]) for l in open(self.FIXTURES / f"lsgkm_mode5_{kind}.pred.txt")]
        )
        return model, seqs, oracle, pred

    @pytest.mark.parametrize("kind", ["rbf", "est"])
    def test_matches_lsgkm_mode5(self, kind):
        from gkmsvm.explain import mutation_impact

        model, seqs, oracle, _ = self._load(kind)
        c = len(seqs[0]) // 2
        ref, alt = [], []
        for s in seqs:
            for b in "ACGT":
                ref.append(one_hot_encode(s, dtype=np.float64))
                alt.append(one_hot_encode(s[:c] + b + s[c + 1:], dtype=np.float64))
        got = mutation_impact(model, np.stack(ref), np.stack(alt)).reshape(-1, 4)
        np.testing.assert_allclose(got, oracle, rtol=1e-4, atol=1e-6)

    @pytest.mark.parametrize("kind", ["rbf", "est"])
    def test_forward_matches_gkmpredict(self, kind):
        model, seqs, _, pred = self._load(kind)
        x = np.stack([one_hot_encode(s, dtype=np.float64) for s in seqs])
        np.testing.assert_allclose(model(x).ravel(), pred, rtol=1e-4, atol=1e-6)

    def test_score_variants_uses_mutation_impact(self):
        from gkmsvm.explain import mutation_impact

        model, seqs, _, _ = self._load("rbf")
        c = len(seqs[0]) // 2
        ref = np.stack([one_hot_encode(s, dtype=np.float64) for s in seqs])
        alt = np.stack([
            one_hot_encode(s[:c] + ("A" if s[c] != "A" else "C") + s[c + 1:], dtype=np.float64)
            for s in seqs
        ])
        np.testing.assert_allclose(
            model.score_variants(ref, alt, method="gkmexplain", batch_size=7),
            mutation_impact(model, ref, alt),
        )
