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
            5, 20, "gkm_esttrunc", {"L": 11, "k": 7, "d": 3, "include_rc": True}, seed=5
        )
        x = one_hot_encode(_make_seqs(1, 20, seed=6)[0])[np.newaxis]

        exp = gkmexplain(model, x, mode="importance")
        score = model(x).squeeze(-1)
        exp_sum = (exp * x).sum(axis=(1, 2))
        expected = score - model.bias
        np.testing.assert_allclose(exp_sum, expected.astype(exp_sum.dtype), atol=1e-5)

    def test_sum_equals_score_esttrunc_no_rc_no_norm(self):
        model = _make_model(
            5,
            20,
            "gkm_esttrunc",
            {"L": 11, "k": 7, "d": 3, "include_rc": False},
            seed=7,
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
            5, 20, "gkm_esttrunc", {"L": 11, "k": 7, "d": 3, "include_rc": True}, seed=9
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
        assert (np.abs(exp[non_ref]) < 1e-10).all(), (
            f"max nonref value: {np.abs(exp[non_ref]).max()}"
        )

    def test_batch_consistent(self):
        model = _make_model(3, 15, "gkm_cnt", {"L": 3, "k": 2, "include_rc": True})
        seqs = _make_seqs(3, 15, seed=11)
        x_batch = np.stack([one_hot_encode(s) for s in seqs])
        result_batch = gkmexplain(model, x_batch, mode="importance")

        for i in range(3):
            result_single = gkmexplain(model, x_batch[i : i + 1], mode="importance")
            np.testing.assert_allclose(
                result_batch[i : i + 1], result_single, atol=1e-8
            )

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
            5,
            20,
            "gkm_esttrunc",
            {"L": 11, "k": 7, "d": 3, "include_rc": True},
            seed=13,
        )
        x = one_hot_encode(_make_seqs(1, 20, seed=14)[0])[np.newaxis]
        exp0 = gkmexplain(model, x, mode="importance")
        exp1 = gkmexplain(model, x, mode="hypothetical")
        ref_mask = x.astype(bool)
        np.testing.assert_allclose(exp0[ref_mask], exp1[ref_mask], atol=1e-8)

    def test_hypothetical_nonref_nonzero(self):
        """Hypothetical should have nonzero values at non-reference bases."""
        model = _make_model(
            3, 15, "gkm_cnt", {"L": 3, "k": 2, "include_rc": True}, seed=15
        )
        x = one_hot_encode(_make_seqs(1, 15, seed=16)[0])[np.newaxis]
        exp1 = gkmexplain(model, x, mode="hypothetical")
        non_ref = (1 - x).astype(bool)
        assert np.abs(exp1[non_ref]).max() > 1e-10, (
            "hypothetical should have nonzero values at non-reference bases"
        )

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
            10,
            20,
            "gkm_esttrunc",
            {"L": 11, "k": 7, "d": 3, "include_rc": True},
            seed=18,
        )
        x = one_hot_encode(_make_seqs(1, 20, seed=19)[0])[np.newaxis]

        full = gkmexplain(model, x, mode="importance")
        model.sv_chunk_size = 3
        chunked = gkmexplain(model, x, mode="importance")
        np.testing.assert_allclose(full, chunked, atol=1e-8)

    def test_chunked_hypothetical(self):
        model = _make_model(
            10,
            20,
            "gkm_esttrunc",
            {"L": 11, "k": 7, "d": 3, "include_rc": True},
            seed=20,
        )
        x = one_hot_encode(_make_seqs(1, 20, seed=21)[0])[np.newaxis]

        full = gkmexplain(model, x, mode="hypothetical")
        model.sv_chunk_size = 4
        chunked = gkmexplain(model, x, mode="hypothetical")
        np.testing.assert_allclose(full, chunked, atol=1e-8)

    def test_chunked_perturbation(self):
        model = _make_model(
            10,
            20,
            "gkm_esttrunc",
            {"L": 11, "k": 7, "d": 3, "include_rc": True},
            seed=20,
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
            5,
            20,
            "gkm_esttrunc",
            {"L": 11, "k": 7, "d": 3, "include_rc": True},
            seed=22,
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

    @pytest.mark.parametrize(
        "kernel_type,params,seqlen,n_sv,atol",
        [
            ("gkm_cnt", {"L": 3, "k": 2, "include_rc": False}, 10, 5, 1e-6),
            ("gkm_cnt", {"L": 3, "k": 2, "include_rc": True}, 10, 5, 1e-6),
            ("gkm_cnt", {"L": 5, "k": 3, "include_rc": True}, 15, 8, 1e-6),
            (
                "gkm_esttrunc",
                {"L": 11, "k": 7, "d": 3, "include_rc": True},
                20,
                5,
                1e-5,
            ),
            (
                "gkm_esttrunc",
                {"L": 11, "k": 7, "d": 3, "include_rc": False},
                20,
                5,
                1e-5,
            ),
            ("gkm_esttrunc", {"L": 7, "k": 4, "d": 2, "include_rc": True}, 15, 6, 1e-5),
        ],
    )
    def test_completeness_normalized(self, kernel_type, params, seqlen, n_sv, atol):
        model = _make_model(n_sv, seqlen, kernel_type, params, seed=100)
        x = np.stack([one_hot_encode(s) for s in _make_seqs(3, seqlen, seed=200)])

        attr = gkmexplain(model, x, mode="importance")
        scores = model(x).squeeze(-1)
        attr_sum = attr.sum(axis=(1, 2))
        np.testing.assert_allclose(attr_sum, scores - model.bias, atol=atol)

    @pytest.mark.parametrize(
        "kernel_type,params,seqlen,n_sv",
        [
            ("gkm_cnt", {"L": 3, "k": 2, "include_rc": False}, 10, 5),
            ("gkm_cnt", {"L": 3, "k": 2, "include_rc": True}, 12, 4),
            ("gkm_esttrunc", {"L": 11, "k": 7, "d": 3, "include_rc": False}, 20, 5),
        ],
    )
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
            4,
            12,
            "gkm_esttrunc",
            {"L": 5, "k": 3, "d": 5, "include_rc": True},
            seed=102,
        )
        x = np.stack([one_hot_encode(s) for s in _make_seqs(2, 12, seed=202)])

        attr = gkmexplain(model, x, mode="importance")
        scores = model(x).squeeze(-1)
        attr_sum = attr.sum(axis=(1, 2))
        np.testing.assert_allclose(attr_sum, scores - model.bias, atol=1e-6)

    def test_completeness_chunked_matches_full(self):
        """Chunked SV processing preserves the completeness axiom."""
        model = _make_model(
            12,
            20,
            "gkm_esttrunc",
            {"L": 11, "k": 7, "d": 3, "include_rc": True},
            seed=103,
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
            5,
            20,
            "gkm_esttrunc",
            {"L": 11, "k": 7, "d": 3, "include_rc": True},
            seed=31,
        )
        x = one_hot_encode(_make_seqs(1, 20, seed=32)[0])[np.newaxis]
        pert = gkmexplain(model, x, mode="perturbation")
        ref_mask = x.astype(bool)
        assert (np.abs(pert[ref_mask]) < 1e-10).all(), (
            f"max ref-base perturbation value: {np.abs(pert[ref_mask]).max()}"
        )

    def test_perturbation_nonref_nonzero(self):
        model = _make_model(
            3,
            15,
            "gkm_cnt",
            {"L": 3, "k": 2, "include_rc": True},
            seed=33,
        )
        x = one_hot_encode(_make_seqs(1, 15, seed=34)[0])[np.newaxis]
        pert = gkmexplain(model, x, mode="perturbation")
        non_ref = (1 - x).astype(bool)
        assert np.abs(pert[non_ref]).max() > 1e-10

    def test_perturbation_score_variants_matches_kernel(self):
        """score_variants(method='gkmexplain') should correlate well with kernel VEP."""
        model = _make_model(
            5,
            15,
            "gkm_cnt",
            {"L": 3, "k": 2, "include_rc": True},
            seed=35,
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
            4,
            12,
            "gkm_esttrunc",
            {"L": 5, "k": 3, "d": 5, "include_rc": True},
            seed=37,
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
            [
                float(l.split("\t")[1])
                for l in open(self.FIXTURES / f"lsgkm_mode5_{kind}.pred.txt")
            ]
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
                alt.append(one_hot_encode(s[:c] + b + s[c + 1 :], dtype=np.float64))
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
        alt = np.stack(
            [
                one_hot_encode(
                    s[:c] + ("A" if s[c] != "A" else "C") + s[c + 1 :], dtype=np.float64
                )
                for s in seqs
            ]
        )
        np.testing.assert_allclose(
            model.score_variants(ref, alt, method="gkmexplain", batch_size=7),
            mutation_impact(model, ref, alt),
        )


@pytest.mark.parametrize("mode", ["hypothetical", "perturbation"])
@pytest.mark.parametrize(
    "kernel_type, params",
    [
        ("gkm_esttrunc", {"L": 8, "k": 5, "d": 2}),
        ("gkm_esttrunc", {"L": 8, "k": 5, "d": 2, "include_rc": False}),
        ("gkm_cnt", {"L": 6, "k": 4}),
    ],
)
def test_lmer_path_matches_pair_loop(kernel_type, params, mode):
    """The l-mer neighborhood path equals the per-SV pair loop."""
    from gkmsvm.explain import gkmexplain
    from gkmsvm.svm import GkmSVM

    rng = np.random.default_rng(5)
    sv = np.eye(4)[rng.integers(0, 4, (30, 25))].transpose(0, 2, 1)
    model = GkmSVM(sv, rng.normal(size=30), 0.2, kernel_type, params)
    x = np.eye(4)[rng.integers(0, 4, (3, 25))].transpose(0, 2, 1)
    fast = gkmexplain(model, x, mode=mode)
    model.use_lmer_index = False  # forces the pair loop
    np.testing.assert_allclose(
        fast, gkmexplain(model, x, mode=mode), rtol=1e-12, atol=1e-15
    )


def test_lmer_path_past_neighbor_cap(monkeypatch):
    """-t 0 l=11 k=7 (143,980 neighbors) takes the l-mer path once the SVs have
    enough windows per neighbor, and equals the pair loop."""
    from gkmsvm import explain
    from gkmsvm.svm import GkmSVM

    rng = np.random.default_rng(6)
    sv = np.eye(4)[rng.integers(0, 4, (30, 40))].transpose(0, 2, 1)
    model = GkmSVM(sv, rng.normal(size=30), 0.2, "gkm_cnt", {"L": 11, "k": 7})
    x = np.eye(4)[rng.integers(0, 4, (3, 40))].transpose(0, 2, 1)
    calls = []
    lmer_cpu = explain._explain_lmer_cpu
    monkeypatch.setattr(
        explain, "_explain_lmer_cpu", lambda *a: calls.append(1) or lmer_cpu(*a)
    )
    pair = explain.gkmexplain(model, x, mode="hypothetical")
    assert calls == []  # 1,800 SV windows: pair loop
    monkeypatch.setattr(explain, "_EXPLAIN_SV_WINDOWS_PER_NEIGHBOR", 0.01)
    lmer = explain.gkmexplain(model, x, mode="hypothetical")
    assert calls
    np.testing.assert_allclose(lmer, pair, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("pair_loop", [False, True])
def test_rbf_attribution(pair_loop):
    """-t 3: LS-GKM scales each SV's contributions by (exp(g(K-1)) - exp(-g)) / K, so
    importance sums to score - bias - exp(-g) sum(coef); both paths agree."""
    from gkmsvm.explain import gkmexplain
    from gkmsvm.svm import GkmSVM

    rng = np.random.default_rng(8)
    sv = np.eye(4)[rng.integers(0, 4, (30, 25))].transpose(0, 2, 1)
    coef = rng.normal(size=30)
    model = GkmSVM(sv, coef, 0.2, "gkmrbf", {"L": 8, "k": 5, "d": 2, "gamma": 1.5})
    x = np.eye(4)[rng.integers(0, 4, (3, 25))].transpose(0, 2, 1)
    fast = gkmexplain(model, x, mode="hypothetical")
    if pair_loop:
        model.use_lmer_index = False
        np.testing.assert_allclose(
            gkmexplain(model, x, mode="hypothetical"), fast, rtol=1e-12, atol=1e-15
        )
    imp = gkmexplain(model, x, mode="importance")
    expected = model(x).ravel() - 0.2 - np.exp(-1.5) * coef.sum()
    np.testing.assert_allclose(imp.sum(axis=(1, 2)), expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("mode", ["hypothetical", "perturbation", "importance"])
@pytest.mark.parametrize(
    "kernel_type, params",
    [
        ("gkm_esttrunc", {"L": 8, "k": 5, "d": 2}),
        ("gkm_esttrunc", {"L": 8, "k": 5, "d": 2, "include_rc": False}),
        ("gkm_estfull", {"L": 7, "k": 4, "d": 3}),
        ("gkm_cnt", {"L": 7, "k": 4}),
    ],
)
def test_explain_table_matches_lmer_path(kernel_type, params, mode):
    """Attributions read from the explain table equal the l-mer path's."""
    from gkmsvm.explain import gkmexplain
    from gkmsvm.svm import GkmSVM

    rng = np.random.default_rng(9)
    sv = np.eye(4)[rng.integers(0, 4, (40, 30))].transpose(0, 2, 1)
    model = GkmSVM(sv, rng.normal(size=40), 0.2, kernel_type, params)
    x = np.eye(4)[rng.integers(0, 4, (4, 33))].transpose(0, 2, 1)
    model.use_explain_table = False
    ref = gkmexplain(model, x, mode=mode)
    model.use_explain_table = True
    np.testing.assert_allclose(
        gkmexplain(model, x, mode=mode), ref, rtol=1e-12, atol=1e-14
    )
    assert model._explain_tables


def test_explain_table_auto_and_caching():
    """ "auto" builds a table only for batches large enough to pay for it, then
    reuses it; RBF models never get one; clear_caches drops it."""
    from gkmsvm.explain import gkmexplain
    from gkmsvm.svm import GkmSVM

    rng = np.random.default_rng(10)
    sv = np.eye(4)[rng.integers(0, 4, (20, 30))].transpose(0, 2, 1)
    model = GkmSVM(
        sv, rng.normal(size=20), 0.2, "gkm_esttrunc", {"L": 8, "k": 5, "d": 2}
    )
    x = np.eye(4)[rng.integers(0, 4, (40, 30))].transpose(0, 2, 1)
    gkmexplain(model, x[:2], mode="hypothetical")
    assert model._explain_tables == {}
    full = gkmexplain(model, x, mode="hypothetical")
    assert list(model._explain_tables) == [False]
    np.testing.assert_allclose(
        gkmexplain(model, x[:2], mode="hypothetical"), full[:2], rtol=1e-12, atol=1e-14
    )
    model.clear_caches()
    assert model._explain_tables == {}

    rbf = GkmSVM(
        sv, rng.normal(size=20), 0.2, "gkmrbf", {"L": 8, "k": 5, "d": 2, "gamma": 1.5}
    )
    rbf.use_explain_table = True
    gkmexplain(rbf, x, mode="hypothetical")
    assert rbf._explain_tables == {}


_FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"
# (model, test FASTA, gkmexplain output prefix): -t 0 to -t 5, with
# N stretches and odd/even window counts (benchmarks/dev/lsgkm_wgkm_fixtures.py)
_LSGKM_EXPLAIN_SETS = {
    "t2": ("lsgkm_mode5_est.model.txt.gz", "lsgkm_mode5_test51.fa", "lsgkm_mode5_est"),
    "t3": ("lsgkm_mode5_rbf.model.txt.gz", "lsgkm_mode5_test51.fa", "lsgkm_mode5_rbf"),
    **{
        f"{t}_{n}": (
            f"lsgkm_mode5_{kind}.model.txt.gz",
            f"lsgkm_wgkm_test{n}.fa",
            f"lsgkm_mode5_{kind}_test{n}",
        )
        for t, kind in (("t2", "est"), ("t3", "rbf"))
        for n in (100, 85)
    },
    **{
        f"{t}_{n}": (
            f"lsgkm_wgkm_{t}.model.txt.gz",
            f"lsgkm_wgkm_test{n}.fa",
            f"lsgkm_wgkm_{t}_test{n}",
        )
        for t in ("t0", "t1", "t4", "t5")
        for n in (100, 85)
    },
    # Models gkmtrain trained on sequences of different lengths (60-140 bp)
    **{
        f"ragged_{t}_{n}": (
            f"lsgkm_ragged_{t}.model.txt.gz",
            f"lsgkm_wgkm_test{n}.fa",
            f"lsgkm_ragged_{t}_test{n}",
        )
        for t in ("t2", "t4", "t5")
        for n in (100, 85)
    },
}


def _read_lsgkm_explain(path):
    """gkmexplain output: name, score, then A,C,G,T per position split by ';'."""
    import gzip

    with gzip.open(path, "rt") as f:
        rows = [line.rstrip("\n").split("\t") for line in f]
    return np.stack(
        [
            np.array([[float(v) for v in pos.split(",")] for pos in r[2].split(";")]).T
            for r in rows
        ]
    )


@pytest.mark.parametrize("name", _LSGKM_EXPLAIN_SETS)
@pytest.mark.parametrize(
    "lsgkm_mode,mode", [(0, "importance"), (1, "hypothetical"), (4, "perturbation")]
)
def test_matches_lsgkm_gkmexplain(name, lsgkm_mode, mode):
    """Attributions equal the kundajelab/lsgkm fork's gkmexplain. Perturbation is
    its mode 4, over every pair a substitution changes (mode 3 leaves out pairs
    with d + 1 mismatches); importance credits N to A, which N packs as."""
    from gkmsvm.fasta import read_fasta
    from gkmsvm.importers.lsgkm import load_lsgkm_model

    model_file, fasta, prefix = _LSGKM_EXPLAIN_SETS[name]
    oracle = _FIXTURES / f"{prefix}.explain{lsgkm_mode}.txt.gz"
    if not oracle.exists():
        pytest.skip("gkmexplain fixtures not generated")
    model = load_lsgkm_model(_FIXTURES / model_file)
    x = np.stack(
        [
            one_hot_encode(s, allow_n=True, dtype=np.float64)
            for _, s in read_fasta(str(_FIXTURES / fasta))
        ]
    )
    expected = _read_lsgkm_explain(oracle)
    for table, index in [(True, True), (False, True), (False, False)]:
        model.use_explain_table, model.use_lmer_index = table, index
        model.clear_caches()
        # gkmexplain prints 6 significant digits
        np.testing.assert_allclose(
            gkmexplain(model, x, mode=mode),
            expected,
            rtol=2e-5,
            atol=5e-6 * np.abs(expected).max(),
            err_msg=f"{table=} {index=}",
        )


@pytest.mark.parametrize(
    "name",
    [
        "t0_85",
        "t1_85",
        "t2_85",
        "t3_85",
        "t4_85",
        "t5_85",
        "ragged_t2_85",
        "ragged_t4_85",
        "ragged_t5_85",
    ],
)
@pytest.mark.parametrize("path", ["index", "join", "brute"])
def test_matches_lsgkm_mode5_fixture_models(name, path):
    """Mutation impact equals gkmexplain -m 5 through the l-mer index, the
    masked join and the brute-force kernel."""
    from gkmsvm.explain import mutation_impact
    from gkmsvm.fasta import read_fasta
    from gkmsvm.importers.lsgkm import load_lsgkm_model

    model_file, fasta, prefix = _LSGKM_EXPLAIN_SETS[name]
    oracle = _FIXTURES / f"{prefix}.explain5.txt.gz"
    if not oracle.exists():
        pytest.skip("gkmexplain fixtures not generated")
    model = load_lsgkm_model(_FIXTURES / model_file)
    model.use_lmer_index = path == "index"
    model.kernel.base.use_masked = path != "brute"
    seqs = [s for _, s in read_fasta(str(_FIXTURES / fasta))]
    c = len(seqs[0]) // 2
    ref = [
        one_hot_encode(s, allow_n=True, dtype=np.float64) for s in seqs for _ in "ACGT"
    ]
    alt = [
        one_hot_encode(s[:c] + b + s[c + 1 :], allow_n=True, dtype=np.float64)
        for s in seqs
        for b in "ACGT"
    ]
    got = mutation_impact(model, np.stack(ref), np.stack(alt), batch_size=16).reshape(
        -1, 4
    )
    assert (model._sv_index() is not None) == (path == "index")
    expected = _read_lsgkm_explain(oracle)[:, :, 0]
    np.testing.assert_allclose(
        got, expected, rtol=2e-5, atol=5e-6 * np.abs(expected).max()
    )


# Every fixture model on queries of odd lengths from 31 to 151 bp, with N
# stretches (lsgkm_ragged_query.fa)
_LSGKM_QUERY_MODELS = [
    *(f"lsgkm_wgkm_{t}" for t in ("t0", "t1", "t4", "t5")),
    "lsgkm_mode5_est",
    "lsgkm_mode5_rbf",
    *(f"lsgkm_ragged_{t}" for t in ("t2", "t4", "t5")),
]


def _read_lsgkm_explain_rows(path):
    """gkmexplain output as one [4, L_i] array per sequence."""
    import gzip

    with gzip.open(path, "rt") as f:
        rows = [line.rstrip("\n").split("\t") for line in f]
    return [
        np.array([[float(v) for v in pos.split(",")] for pos in r[2].split(";")]).T
        for r in rows
    ]


def _query_seqs():
    from gkmsvm.fasta import read_fasta

    return [s for _, s in read_fasta(str(_FIXTURES / "lsgkm_ragged_query.fa"))]


@pytest.mark.parametrize("name", _LSGKM_QUERY_MODELS)
@pytest.mark.parametrize(
    "lsgkm_mode,mode", [(0, "importance"), (1, "hypothetical"), (4, "perturbation")]
)
def test_matches_lsgkm_gkmexplain_on_mixed_queries(name, lsgkm_mode, mode):
    """Queries of different lengths explained in one padded batch equal the
    kundajelab fork's gkmexplain on each, on every explain path."""
    from gkmsvm.importers.lsgkm import load_lsgkm_model

    oracle = _FIXTURES / f"{name}_query.explain{lsgkm_mode}.txt.gz"
    if not oracle.exists():
        pytest.skip("gkmexplain fixtures not generated")
    model = load_lsgkm_model(_FIXTURES / f"{name}.model.txt.gz")
    seqs = _query_seqs()
    expected = _read_lsgkm_explain_rows(oracle)
    scale = max(np.abs(e).max() for e in expected)
    for table, index in [(True, True), (False, True), (False, False)]:
        model.use_explain_table, model.use_lmer_index = table, index
        model.clear_caches()
        got = np.asarray(gkmexplain(model, seqs, mode=mode, batch_size=8))
        for i, (s, e) in enumerate(zip(seqs, expected)):
            # gkmexplain prints 6 significant digits
            np.testing.assert_allclose(
                got[i, :, : len(s)],
                e,
                rtol=2e-5,
                atol=5e-6 * scale,
                err_msg=f"{table=} {index=} sequence {i}",
            )
            assert (got[i, :, len(s) :] == 0).all()


@pytest.mark.parametrize("name", _LSGKM_QUERY_MODELS)
@pytest.mark.parametrize("path", ["index", "join", "brute"])
def test_matches_lsgkm_mode5_on_mixed_queries(name, path):
    """Mutation impact at each query's central base ((L - 1) / 2), queries of
    different lengths in one call, equals gkmexplain -m 5."""
    from gkmsvm.explain import mutation_impact
    from gkmsvm.importers.lsgkm import load_lsgkm_model

    oracle = _FIXTURES / f"{name}_query.explain5.txt.gz"
    if not oracle.exists():
        pytest.skip("gkmexplain fixtures not generated")
    model = load_lsgkm_model(_FIXTURES / f"{name}.model.txt.gz")
    model.use_lmer_index = path == "index"
    model.kernel.base.use_masked = path != "brute"
    seqs = _query_seqs()
    ref = [s for s in seqs for _ in "ACGT"]
    alt = [
        s[: (len(s) - 1) // 2] + b + s[(len(s) + 1) // 2 :]
        for s in seqs
        for b in "ACGT"
    ]
    got = np.asarray(mutation_impact(model, ref, alt, batch_size=16)).reshape(-1, 4)
    expected = np.stack([e[:, 0] for e in _read_lsgkm_explain_rows(oracle)])
    np.testing.assert_allclose(
        got, expected, rtol=2e-5, atol=5e-6 * np.abs(expected).max()
    )


def test_importance_credits_n_to_a():
    """N packs as A, so its importance goes to A and the total is score - bias."""
    model = _make_model(20, 30, "gkm_esttrunc", {"L": 6, "k": 4, "d": 2})
    s = list(_make_seqs(1, 30, seed=3)[0])
    s[10:14] = "NNNN"
    x = one_hot_encode("".join(s), allow_n=True)[None]
    imp = gkmexplain(model, x, mode="importance")
    assert np.abs(imp[0, 0, 10:14]).sum() > 0 and np.all(imp[0, 1:, 10:14] == 0)
    np.testing.assert_allclose(imp.sum(), model(x)[0, 0] - model.bias, rtol=1e-9)


@pytest.mark.parametrize("name", ["t0_100", "t0_85", "t1_100", "t1_85"])
def test_fixture_models_match_gkmpredict(name):
    """The -t 0/-t 1 fixture models' scores equal gkmpredict's."""
    from gkmsvm.fasta import read_fasta
    from gkmsvm.importers.lsgkm import load_lsgkm_model

    model_file, fasta, prefix = _LSGKM_EXPLAIN_SETS[name]
    pred = _FIXTURES / f"{prefix}.pred.txt"
    if not pred.exists():
        pytest.skip("gkmpredict fixtures not generated")
    model = load_lsgkm_model(_FIXTURES / model_file)
    records = read_fasta(str(_FIXTURES / fasta))
    x = np.stack(
        [one_hot_encode(s, allow_n=True, dtype=np.float64) for _, s in records]
    )
    with open(pred) as f:
        expected = dict(line.split()[:2] for line in f)
    expected = np.array([float(expected[name]) for name, _ in records])
    np.testing.assert_allclose(model(x).ravel(), expected, rtol=2e-5, atol=1e-5)
