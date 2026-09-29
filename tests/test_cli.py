"""Tests for the gkmsvm CLI."""

import random
from pathlib import Path

import numpy as np
import pytest

from gkmsvm.cli import main


@pytest.fixture
def tmp_fasta(tmp_path):
    """Create temp FASTA files for testing."""
    rng = random.Random(42)

    def _write(name, n, length=30):
        path = tmp_path / name
        with open(path, "w") as f:
            for i in range(n):
                seq = "".join(rng.choice("ACGT") for _ in range(length))
                f.write(f">{name}_{i}\n{seq}\n")
        return str(path)

    return _write


class TestPredict:
    def test_predict_to_file(self, tmp_fasta, tmp_path):
        pos = tmp_fasta("pos.fa", 20)
        neg = tmp_fasta("neg.fa", 20)
        model_path = str(tmp_path / "model.npz")
        out_path = str(tmp_path / "scores.tsv")

        main(["train", "-p", pos, "-n", neg, "-o", model_path,
              "-l", "5", "-k", "3", "-t", "direct", "--device", "cpu"])

        test_fa = tmp_fasta("test.fa", 5)
        main(["predict", "-m", model_path, "-i", test_fa,
              "-o", out_path, "--device", "cpu"])

        lines = Path(out_path).read_text().strip().split("\n")
        assert lines[0] == "name\tscore"
        assert len(lines) == 6  # header + 5 sequences

    def test_predict_stdout(self, tmp_fasta, tmp_path, capsys):
        pos = tmp_fasta("pos.fa", 20)
        neg = tmp_fasta("neg.fa", 20)
        model_path = str(tmp_path / "model.npz")

        main(["train", "-p", pos, "-n", neg, "-o", model_path,
              "-l", "5", "-k", "3", "-t", "direct", "--device", "cpu"])

        test_fa = tmp_fasta("test.fa", 3)
        main(["predict", "-m", model_path, "-i", test_fa, "--device", "cpu"])

        out = capsys.readouterr().out
        lines = out.strip().split("\n")
        assert lines[0] == "name\tscore"
        assert len(lines) == 4


class TestTrain:
    def test_train_npz(self, tmp_fasta, tmp_path):
        pos = tmp_fasta("pos.fa", 20)
        neg = tmp_fasta("neg.fa", 20)
        model_path = str(tmp_path / "model.npz")

        main(["train", "-p", pos, "-n", neg, "-o", model_path,
              "-l", "5", "-k", "3", "-t", "direct", "--device", "cpu"])

        assert Path(model_path).exists()
        from gkmsvm.serialization import load_npz
        model = load_npz(model_path)
        assert model.num_support_vectors > 0

    def test_train_lsgkm_format(self, tmp_fasta, tmp_path):
        pos = tmp_fasta("pos.fa", 20)
        neg = tmp_fasta("neg.fa", 20)
        model_path = str(tmp_path / "model.txt")

        main(["train", "-p", pos, "-n", neg, "-o", model_path,
              "-l", "5", "-k", "3", "-t", "direct", "--device", "cpu"])

        assert Path(model_path).exists()
        text = Path(model_path).read_text()
        assert "svm_type" in text
        assert "SV" in text


class TestISM:
    def test_ism_output(self, tmp_fasta, tmp_path):
        pos = tmp_fasta("pos.fa", 20)
        neg = tmp_fasta("neg.fa", 20)
        model_path = str(tmp_path / "model.npz")
        ism_path = str(tmp_path / "ism.npz")

        main(["train", "-p", pos, "-n", neg, "-o", model_path,
              "-l", "5", "-k", "3", "-t", "direct", "--device", "cpu"])

        test_fa = tmp_fasta("test.fa", 3, length=30)
        main(["ism", "-m", model_path, "-i", test_fa,
              "-o", ism_path, "--device", "cpu"])

        data = np.load(ism_path)
        assert data["scores"].shape == (3, 4, 30)
        assert data["one_hot"].shape == (3, 4, 30)
        assert len(data["names"]) == 3


class TestExplain:
    def test_explain_modisco_format(self, tmp_fasta, tmp_path):
        """Output two npz files compatible with modisco motifs -s/-a."""
        pos = tmp_fasta("pos.fa", 20)
        neg = tmp_fasta("neg.fa", 20)
        model_path = str(tmp_path / "model.npz")
        attr_path = str(tmp_path / "attr.npz")
        seq_path = str(tmp_path / "seqs.npz")

        main(["train", "-p", pos, "-n", neg, "-o", model_path,
              "-l", "5", "-k", "3", "-t", "direct", "--device", "cpu"])

        test_fa = tmp_fasta("test.fa", 3, length=30)
        main(["explain", "-m", model_path, "-i", test_fa,
              "-o", attr_path, "-s", seq_path, "--device", "cpu"])

        attr = np.load(attr_path)["arr_0"]
        seqs = np.load(seq_path)["arr_0"]
        assert attr.shape == (3, 4, 30)
        assert seqs.shape == (3, 4, 30)
        assert attr.dtype == np.float32
        assert seqs.sum(axis=1).max() == 1.0  # one-hot


class TestImport:
    def test_import_lsgkm(self, tmp_fasta, tmp_path):
        pos = tmp_fasta("pos.fa", 20)
        neg = tmp_fasta("neg.fa", 20)
        model_path = str(tmp_path / "model.npz")
        txt_path = str(tmp_path / "model.txt")
        reimport_path = str(tmp_path / "reimported.npz")

        main(["train", "-p", pos, "-n", neg, "-o", model_path,
              "-l", "5", "-k", "3", "-t", "direct", "--device", "cpu"])

        from gkmsvm.serialization import load_npz, save_lsgkm
        model = load_npz(model_path)
        save_lsgkm(model, txt_path)

        main(["import", "-i", txt_path, "-o", reimport_path, "-f", "lsgkm"])

        reimported = load_npz(reimport_path)
        assert reimported.num_support_vectors == model.num_support_vectors


class TestToDeltasvm:
    def test_to_deltasvm_roundtrip(self, tmp_fasta, tmp_path):
        pos = tmp_fasta("pos.fa", 20)
        neg = tmp_fasta("neg.fa", 20)
        model_path = str(tmp_path / "model.npz")
        weights_path = str(tmp_path / "weights.tsv")

        main(["train", "-p", pos, "-n", neg, "-o", model_path,
              "-l", "5", "-k", "3", "-t", "direct", "--device", "cpu"])

        main(["to-deltasvm", "-m", model_path, "-o", weights_path,
              "--device", "cpu"])

        assert Path(weights_path).exists()
        lines = [l for l in Path(weights_path).read_text().strip().split("\n")
                 if l and not l.startswith("#")]
        assert len(lines) > 0
        kmer, weight = lines[0].split("\t")
        assert len(kmer) == 5
        float(weight)  # should not raise

    def test_to_deltasvm_scores_correlate(self, tmp_fasta, tmp_path):
        """DeltaSVM scores should correlate well with full SVM scores."""
        pos = tmp_fasta("pos.fa", 20)
        neg = tmp_fasta("neg.fa", 20)
        model_path = str(tmp_path / "model.npz")
        weights_path = str(tmp_path / "weights.tsv")
        svm_scores_path = str(tmp_path / "svm.tsv")
        dsvm_scores_path = str(tmp_path / "dsvm.tsv")

        main(["train", "-p", pos, "-n", neg, "-o", model_path,
              "-l", "5", "-k", "3", "-t", "direct", "--device", "cpu"])

        test_fa = tmp_fasta("test.fa", 10)

        main(["predict", "-m", model_path, "-i", test_fa,
              "-o", svm_scores_path, "--device", "cpu"])

        main(["to-deltasvm", "-m", model_path, "-o", weights_path,
              "--device", "cpu"])

        main(["deltasvm", "-w", weights_path, "-i", test_fa,
              "-l", "5", "-o", dsvm_scores_path, "--no-rc"])

        def read_scores(path):
            scores = []
            with open(path) as f:
                next(f)
                for line in f:
                    scores.append(float(line.strip().split("\t")[1]))
            return np.array(scores)

        svm = read_scores(svm_scores_path)
        dsvm = read_scores(dsvm_scores_path)
        corr = np.corrcoef(svm, dsvm)[0, 1]
        assert corr > 0.95


class TestScoreVariants:
    def test_score_variants_output(self, tmp_fasta, tmp_path):
        pos = tmp_fasta("pos.fa", 20)
        neg = tmp_fasta("neg.fa", 20)
        model_path = str(tmp_path / "model.npz")

        main(["train", "-p", pos, "-n", neg, "-o", model_path,
              "-l", "5", "-k", "3", "-t", "direct", "--device", "cpu"])

        ref_fa = tmp_fasta("ref.fa", 5)
        alt_fa = tmp_fasta("alt.fa", 5)
        out_path = str(tmp_path / "variants.tsv")

        main(["score-variants", "-m", model_path,
              "--ref", ref_fa, "--alt", alt_fa,
              "-o", out_path, "--device", "cpu"])

        lines = Path(out_path).read_text().strip().split("\n")
        assert lines[0] == "name\tdelta"
        assert len(lines) == 6


class TestNoCommand:
    def test_no_command_exits(self):
        with pytest.raises(SystemExit):
            main([])
