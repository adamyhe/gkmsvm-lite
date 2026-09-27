"""Replication tests using real ChIP-seq data (Nanog H1-ESC).

Small subset (100+100 training, 20+20 test) bundled as fixtures.
Reference model scores from the gkmExplain repo's pre-trained
LS-GKM Nanog model (default -t 2 -l 11 -k 7 -d 3).

These tests verify:
  1. Loading and scoring with the reference LS-GKM model
  2. Training from scratch reproduces similar class separation
  3. ISM and GkmExplain produce valid attributions on real data
"""
import gzip
from pathlib import Path

import numpy as np
import pytest

from gkmsvm.codec import one_hot_encode
from gkmsvm.fasta import read_fasta
from gkmsvm.importers.lsgkm import load_lsgkm_model

FIXTURES = Path(__file__).parent / "fixtures"


def _read_fasta_gz(path):
    import tempfile
    with gzip.open(path, "rt") as gz, tempfile.NamedTemporaryFile(
        mode="w", suffix=".fa", delete=False
    ) as tmp:
        tmp.write(gz.read())
        tmp.flush()
        return read_fasta(tmp.name)


def _load_oracle_scores(path):
    scores = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            name, val = line.strip().split("\t")
            scores[name] = float(val)
    return scores


@pytest.fixture(scope="module")
def reference_model():
    return load_lsgkm_model(FIXTURES / "nanog_reference.model.txt.gz")


@pytest.fixture(scope="module")
def train_seqs():
    pos = _read_fasta_gz(FIXTURES / "nanog_pos_train.fa.gz")
    neg = _read_fasta_gz(FIXTURES / "nanog_neg_train.fa.gz")
    return [s for _, s in pos], [s for _, s in neg]


@pytest.fixture(scope="module")
def test_data():
    pos = _read_fasta_gz(FIXTURES / "nanog_pos_test.fa.gz")
    neg = _read_fasta_gz(FIXTURES / "nanog_neg_test.fa.gz")
    pos_x = np.stack([one_hot_encode(s) for _, s in pos])
    neg_x = np.stack([one_hot_encode(s) for _, s in neg])
    pos_names = [n for n, _ in pos]
    neg_names = [n for n, _ in neg]
    return pos_x, neg_x, pos_names, neg_names


@pytest.fixture(scope="module")
def oracle_scores():
    return _load_oracle_scores(FIXTURES / "nanog_oracle_scores.txt.gz")


# ---------------------------------------------------------------------------
# Reference model scoring
# ---------------------------------------------------------------------------


class TestReferenceModelScoring:
    def test_scores_match_oracle(self, reference_model, test_data, oracle_scores):
        pos_x, neg_x, pos_names, neg_names = test_data

        pos_scores = reference_model(pos_x).squeeze(-1)
        neg_scores = reference_model(neg_x).squeeze(-1)

        for i, name in enumerate(pos_names):
            np.testing.assert_allclose(
                pos_scores[i], oracle_scores[name], atol=1e-4,
                err_msg=f"Pos {name}",
            )
        for i, name in enumerate(neg_names):
            np.testing.assert_allclose(
                neg_scores[i], oracle_scores[name], atol=1e-4,
                err_msg=f"Neg {name}",
            )

    def test_positive_scores_higher(self, reference_model, test_data):
        pos_x, neg_x, _, _ = test_data
        pos_mean = reference_model(pos_x).mean()
        neg_mean = reference_model(neg_x).mean()
        assert pos_mean > neg_mean

    def test_rc_invariance(self, reference_model, test_data):
        from gkmsvm.codec import reverse_complement
        pos_x, _, _, _ = test_data
        x = pos_x[:3]
        scores = reference_model(x)
        rc_scores = reference_model(reverse_complement(x))
        np.testing.assert_allclose(scores, rc_scores, atol=1e-5)


# ---------------------------------------------------------------------------
# Training from scratch
# ---------------------------------------------------------------------------


class TestTrainFromScratch:
    @pytest.fixture(scope="class")
    @classmethod
    def trained_model(cls, train_seqs):
        from gkmsvm import train_gkmsvm
        pos, neg = train_seqs
        return train_gkmsvm(
            pos, neg,
            kernel_type="estimated", l=11, k=7, d=3, C=1.0,
            device="cpu",
        )

    def test_has_support_vectors(self, trained_model):
        assert trained_model.num_support_vectors > 0

    def test_separates_classes(self, trained_model, test_data):
        pos_x, neg_x, _, _ = test_data
        pos_mean = trained_model(pos_x).mean()
        neg_mean = trained_model(neg_x).mean()
        assert pos_mean > neg_mean

    def test_smo_separates_classes(self, train_seqs, test_data):
        from gkmsvm import train_gkmsvm
        pos, neg = train_seqs
        model = train_gkmsvm(
            pos, neg,
            kernel_type="estimated", l=11, k=7, d=3, C=1.0,
            solver="smo", device="cpu",
        )
        pos_x, neg_x, _, _ = test_data
        pos_mean = model(pos_x).mean()
        neg_mean = model(neg_x).mean()
        assert pos_mean > neg_mean


# ---------------------------------------------------------------------------
# ISM on real data
# ---------------------------------------------------------------------------


class TestISMReplication:
    def test_ism_ref_base_zero(self, reference_model, test_data):
        from gkmsvm.ism import ism
        pos_x, _, _, _ = test_data
        x = pos_x[:2]
        result = ism(reference_model, x)
        assert result.shape == x.shape
        ref_bases = x.argmax(axis=1)
        for b in range(x.shape[0]):
            for p in range(x.shape[2]):
                assert abs(float(result[b, int(ref_bases[b, p]), p])) < 1e-4

    def test_ism_shape(self, reference_model, test_data):
        from gkmsvm.ism import ism
        pos_x, _, _, _ = test_data
        x = pos_x[:1]
        result = ism(reference_model, x)
        assert result.shape == (1, 4, 200)


# ---------------------------------------------------------------------------
# GkmExplain on real data
# ---------------------------------------------------------------------------


class TestGkmExplainReplication:
    def test_completeness_axiom(self, reference_model, test_data):
        from gkmsvm.explain import gkmexplain
        pos_x, _, _, _ = test_data
        x = pos_x[:2]
        exp = gkmexplain(reference_model, x, mode=0)
        scores = reference_model(x).squeeze(-1)
        exp_sum = (exp * x).sum(axis=(1, 2))
        expected = scores - reference_model.bias
        np.testing.assert_allclose(
            exp_sum, expected.astype(np.float64), atol=1e-4,
        )

    def test_nonref_bases_zero(self, reference_model, test_data):
        from gkmsvm.explain import gkmexplain
        pos_x, _, _, _ = test_data
        x = pos_x[:1]
        exp = gkmexplain(reference_model, x, mode=0)
        non_ref = (1 - x).astype(bool)
        assert (np.abs(exp[non_ref]) < 1e-10).all()

    def test_mode1_produces_output(self, reference_model, test_data):
        from gkmsvm.explain import gkmexplain
        pos_x, _, _, _ = test_data
        x = pos_x[:1]
        exp = gkmexplain(reference_model, x, mode=1)
        assert exp.shape == (1, 4, 200)
        assert not np.allclose(exp, 0)
