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

    def test_trainsvr(self, tmp_fasta, tmp_path):
        seqs = tmp_fasta("seqs.fa", 30)
        labels = tmp_path / "labels.txt"
        labels.write_text("\n".join(str(0.1 * i) for i in range(30)) + "\n")
        model_path = str(tmp_path / "svr.npz")

        main(["trainsvr", "-i", seqs, "--labels", str(labels), "-o", model_path,
              "-l", "5", "-k", "3", "-t", "direct", "--device", "cpu"])

        assert Path(model_path).exists()

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


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("name", ["t1", "t4"])
def test_predict_with_n_matches_gkmpredict(name, tmp_path):
    """FASTA with N stretches: scores equal LS-GKM gkmpredict (N read as A)."""
    model = FIXTURES / f"lsgkm_wgkm_{name}.model.txt.gz"
    fasta = FIXTURES / "lsgkm_wgkm_test100.fa"
    assert "N" in fasta.read_text()
    out = tmp_path / "scores.tsv"
    main(["predict", "-m", str(model), "-i", str(fasta), "-o", str(out), "--device", "cpu"])
    got = dict(line.split("\t") for line in out.read_text().splitlines()[1:])
    with open(FIXTURES / f"lsgkm_wgkm_{name}_test100.pred.txt") as f:
        expected = dict(line.split()[:2] for line in f)
    assert got.keys() == expected.keys()
    np.testing.assert_allclose([float(got[k]) for k in expected],
                               [float(v) for v in expected.values()], rtol=2e-5, atol=1e-5)


def test_train_with_n(tmp_fasta, tmp_path):
    """Training FASTAs may contain N."""
    pos, neg = tmp_path / "pos.fa", tmp_path / "neg.fa"
    rng = random.Random(3)
    for path in (pos, neg):
        seqs = ["".join(rng.choice("ACGT") for _ in range(30)) for _ in range(20)]
        seqs[::4] = [s[:10] + "NNNnn" + s[15:] for s in seqs[::4]]
        path.write_text("".join(f">{path.stem}{i}\n{s}\n" for i, s in enumerate(seqs)))
    model_path = str(tmp_path / "model.npz")
    main(["train", "-p", str(pos), "-n", str(neg), "-o", model_path,
          "-l", "5", "-k", "3", "-d", "2", "--device", "cpu"])
    assert Path(model_path).exists()


def test_allow_other_bases(tmp_path):
    """Non-ACGTN letters stop predict unless --allow-other-bases, which scores them
    as N (A, as gkmpredict does)."""
    model = str(FIXTURES / "lsgkm_wgkm_t1.model.txt.gz")
    seqs = [s for s in FIXTURES.joinpath("lsgkm_wgkm_test100.fa").read_text().split("\n")
            if s and not s.startswith(">")][:6]
    iupac = [s[:20] + "RYKM" + s[24:] for s in seqs]
    as_n = [s[:20] + "NNNN" + s[24:] for s in seqs]
    paths = {}
    for name, ss in (("iupac", iupac), ("n", as_n)):
        paths[name] = tmp_path / f"{name}.fa"
        paths[name].write_text("".join(f">s{i}\n{s}\n" for i, s in enumerate(ss)))
    with pytest.raises(ValueError, match="allow_other"):
        main(["predict", "-m", model, "-i", str(paths["iupac"]), "--device", "cpu"])
    out = {}
    for name, extra in (("iupac", ["--allow-other-bases"]), ("n", [])):
        path = tmp_path / f"{name}.tsv"
        main(["predict", "-m", model, "-i", str(paths[name]), "-o", str(path),
              "--device", "cpu", *extra])
        out[name] = path.read_text()
    assert out["iupac"] == out["n"]
