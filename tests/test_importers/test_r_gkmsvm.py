"""Tests for the R gkmSVM model importer."""
import random

import numpy as np
import pytest

from gkmsvm.codec import one_hot_encode
from gkmsvm.importers.r_gkmsvm import load_r_gkmsvm_model


def _make_seqs(n, length, seed=42):
    rng = random.Random(seed)
    return ["".join(rng.choice("ACGT") for _ in range(length)) for _ in range(n)]


def _write_gkmmodel(path, svs, coefs, rho, l=11, k=7, d=3):
    """Write a .gkmmodel file with #-prefixed headers and FASTA SVs."""
    with open(path, "w") as f:
        f.write(f"#L {l}\n")
        f.write(f"#k {k}\n")
        f.write(f"#d {d}\n")
        f.write(f"#rho {rho}\n")
        f.write(f"#nsv {len(svs)}\n")
        n_pos = sum(1 for c in coefs if c > 0)
        f.write(f"#npos {n_pos}\n")
        f.write(f"#nneg {len(svs) - n_pos}\n")
        for i, (seq, coef) in enumerate(zip(svs, coefs)):
            f.write(f">sv_{i}\t{coef}\n")
            f.write(f"{seq}\n")


def _write_twofile(alpha_path, svseq_path, svs, coefs, rho, l=11, k=7, d=3):
    """Write legacy _svalpha.out + _svseq.fa files."""
    with open(alpha_path, "w") as f:
        f.write(f"#L {l}\n")
        f.write(f"#k {k}\n")
        f.write(f"#d {d}\n")
        f.write(f"#rho {rho}\n")
        for i, coef in enumerate(coefs):
            f.write(f"sv_{i}\t{coef}\n")

    with open(svseq_path, "w") as f:
        for i, seq in enumerate(svs):
            f.write(f">sv_{i}\n{seq}\n")


class TestRGkmsvmGkmmodel:
    def test_load_basic(self, tmp_path):
        svs = _make_seqs(3, 11, seed=1)
        coefs = [0.5, -0.3, 0.2]
        rho = 0.1
        path = tmp_path / "test.gkmmodel"
        _write_gkmmodel(path, svs, coefs, rho, l=5, k=3)

        model = load_r_gkmsvm_model(path)
        assert model.num_support_vectors == 3
        assert model.bias == pytest.approx(0.1)

    def test_bias_is_positive_rho(self, tmp_path):
        svs = _make_seqs(3, 11, seed=2)
        coefs = [0.5, -0.3, 0.2]
        rho = 0.42
        path = tmp_path / "test.gkmmodel"
        _write_gkmmodel(path, svs, coefs, rho, l=5, k=3)

        model = load_r_gkmsvm_model(path)
        assert model.bias == pytest.approx(0.42)

    def test_scores_finite(self, tmp_path):
        svs = _make_seqs(3, 11, seed=3)
        coefs = [0.5, -0.3, 0.2]
        path = tmp_path / "test.gkmmodel"
        _write_gkmmodel(path, svs, coefs, 0.1, l=5, k=3)

        model = load_r_gkmsvm_model(path)
        x = one_hot_encode(_make_seqs(1, 15, seed=100)[0])[np.newaxis]
        assert np.isfinite(model(x)).all()

    def test_kernel_params(self, tmp_path):
        svs = _make_seqs(3, 15, seed=4)
        coefs = [0.5, -0.3, 0.2]
        path = tmp_path / "test.gkmmodel"
        _write_gkmmodel(path, svs, coefs, 0.1, l=8, k=5, d=2)

        model = load_r_gkmsvm_model(path)
        assert model.kernel_params["L"] == 8
        assert model.kernel_params["k"] == 5
        assert model.kernel_params["d"] == 2

    def test_nsv_mismatch_raises(self, tmp_path):
        svs = _make_seqs(3, 11, seed=5)
        coefs = [0.5, -0.3, 0.2]
        path = tmp_path / "bad.gkmmodel"
        with open(path, "w") as f:
            f.write("#rho 0.1\n")
            f.write("#nsv 5\n")
            f.write("#L 5\n")
            f.write("#k 3\n")
            for i, (seq, coef) in enumerate(zip(svs, coefs)):
                f.write(f">sv_{i}\t{coef}\n{seq}\n")

        with pytest.raises(ValueError, match="Expected 5"):
            load_r_gkmsvm_model(path)

    def test_missing_rho_raises(self, tmp_path):
        svs = _make_seqs(3, 11, seed=6)
        coefs = [0.5, -0.3, 0.2]
        path = tmp_path / "bad.gkmmodel"
        with open(path, "w") as f:
            f.write("#L 5\n")
            f.write("#k 3\n")
            for i, (seq, coef) in enumerate(zip(svs, coefs)):
                f.write(f">sv_{i}\t{coef}\n{seq}\n")

        with pytest.raises(ValueError, match="rho"):
            load_r_gkmsvm_model(path)

    def test_multiline_sequence(self, tmp_path):
        svs = _make_seqs(2, 20, seed=7)
        coefs = [0.5, -0.3]
        path = tmp_path / "test.gkmmodel"
        with open(path, "w") as f:
            f.write("#rho 0.1\n#L 5\n#k 3\n#d 3\n")
            for i, (seq, coef) in enumerate(zip(svs, coefs)):
                f.write(f">sv_{i}\t{coef}\n")
                f.write(f"{seq[:10]}\n{seq[10:]}\n")

        model = load_r_gkmsvm_model(path)
        assert model.num_support_vectors == 2


class TestRGkmsvmTwoFile:
    def test_load_explicit_paths(self, tmp_path):
        svs = _make_seqs(3, 11, seed=10)
        coefs = [0.5, -0.3, 0.2]
        rho = 0.15
        alpha_path = tmp_path / "model_svalpha.out"
        svseq_path = tmp_path / "model_svseq.fa"
        _write_twofile(alpha_path, svseq_path, svs, coefs, rho, l=5, k=3)

        model = load_r_gkmsvm_model(alpha_path, svseq_path=svseq_path)
        assert model.num_support_vectors == 3
        assert model.bias == pytest.approx(0.15)

    def test_auto_detect_svseq(self, tmp_path):
        svs = _make_seqs(3, 11, seed=11)
        coefs = [0.5, -0.3, 0.2]
        rho = 0.2
        alpha_path = tmp_path / "mymodel_svalpha.out"
        svseq_path = tmp_path / "mymodel_svseq.fa"
        _write_twofile(alpha_path, svseq_path, svs, coefs, rho, l=5, k=3)

        model = load_r_gkmsvm_model(alpha_path)
        assert model.num_support_vectors == 3

    def test_missing_svseq_raises(self, tmp_path):
        svs = _make_seqs(3, 11, seed=12)
        coefs = [0.5, -0.3, 0.2]
        alpha_path = tmp_path / "mymodel_svalpha.out"
        with open(alpha_path, "w") as f:
            f.write("#rho 0.1\n")
            for i, coef in enumerate(coefs):
                f.write(f"sv_{i}\t{coef}\n")

        with pytest.raises(FileNotFoundError):
            load_r_gkmsvm_model(alpha_path)

    def test_twofile_scores_match_gkmmodel(self, tmp_path):
        svs = _make_seqs(3, 11, seed=13)
        coefs = [0.5, -0.3, 0.2]
        rho = 0.15

        gkmmodel_path = tmp_path / "test.gkmmodel"
        _write_gkmmodel(gkmmodel_path, svs, coefs, rho, l=5, k=3)

        alpha_path = tmp_path / "test_svalpha.out"
        svseq_path = tmp_path / "test_svseq.fa"
        _write_twofile(alpha_path, svseq_path, svs, coefs, rho, l=5, k=3)

        model_unified = load_r_gkmsvm_model(gkmmodel_path)
        model_twofile = load_r_gkmsvm_model(alpha_path, svseq_path=svseq_path)

        x = one_hot_encode(_make_seqs(1, 15, seed=200)[0])[np.newaxis]
        np.testing.assert_allclose(model_unified(x), model_twofile(x), atol=1e-6)

    def test_coef_seq_mismatch_raises(self, tmp_path):
        svs = _make_seqs(3, 11, seed=14)
        alpha_path = tmp_path / "model_svalpha.out"
        svseq_path = tmp_path / "model_svseq.fa"

        with open(alpha_path, "w") as f:
            f.write("#rho 0.1\n#L 5\n#k 3\n")
            f.write("sv_0\t0.5\n")
            f.write("sv_1\t-0.3\n")

        with open(svseq_path, "w") as f:
            for i, seq in enumerate(svs):
                f.write(f">sv_{i}\n{seq}\n")

        with pytest.raises(ValueError, match="does not match"):
            load_r_gkmsvm_model(alpha_path, svseq_path=svseq_path)
