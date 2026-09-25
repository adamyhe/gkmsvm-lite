"""Tests for the classic gkmSVM model importer."""
import random

import pytest
import torch

from gkmsvm.codec import one_hot_encode
from gkmsvm.importers.classic import load_classic_model
from gkmsvm.importers.lsgkm import load_lsgkm_model
from gkmsvm.kernels.direct import DirectGkmKernel


def _make_seqs(n, length, seed=42):
    rng = random.Random(seed)
    return ["".join(rng.choice("ACGT") for _ in range(length)) for _ in range(n)]


def _write_classic_model_embedded(path, svs, coefs, rho, l, k, kernel_type=0):
    """Write a classic model file with embedded SVs."""
    with open(path, "w") as f:
        f.write(f"svm_type c_svc\n")
        f.write(f"kernel_type {kernel_type}\n")
        f.write(f"L {l}\n")
        f.write(f"k {k}\n")
        f.write(f"nr_class 2\n")
        f.write(f"total_sv {len(svs)}\n")
        f.write(f"rho {rho}\n")
        f.write("SV\n")
        for c, s in zip(coefs, svs):
            f.write(f"{c} {s}\n")


def _write_classic_model_twofile(model_path, svseq_path, svs, coefs, rho, l, k, kernel_type=0):
    """Write a classic model as two separate files."""
    with open(model_path, "w") as f:
        f.write(f"svm_type c_svc\n")
        f.write(f"kernel_type {kernel_type}\n")
        f.write(f"L {l}\n")
        f.write(f"k {k}\n")
        f.write(f"nr_class 2\n")
        f.write(f"total_sv {len(svs)}\n")
        f.write(f"rho {rho}\n")
        f.write("SV\n")
        for c in coefs:
            f.write(f"{c}\n")

    with open(svseq_path, "w") as f:
        for i, s in enumerate(svs):
            f.write(f">sv_{i}\n{s}\n")


class TestClassicImporterEmbedded:
    def test_load_basic(self, tmp_path):
        svs = _make_seqs(3, 11, seed=1)
        coefs = [0.5, -0.3, 0.2]
        rho = 0.1
        path = tmp_path / "model.gkmmodel"
        _write_classic_model_embedded(path, svs, coefs, rho, l=5, k=3)

        model = load_classic_model(path)
        assert model.num_support_vectors == 3
        assert model.kernel_type == "gkm_cnt"
        assert model.bias == pytest.approx(0.1)  # +rho, not -rho

    def test_opposite_bias_sign(self, tmp_path):
        """Classic model should have bias = +rho (opposite of LS-GKM's -rho)."""
        svs = _make_seqs(3, 11, seed=2)
        coefs = [0.5, -0.3, 0.2]
        rho = 0.42

        classic_path = tmp_path / "classic.gkmmodel"
        _write_classic_model_embedded(classic_path, svs, coefs, rho, l=5, k=3)

        lsgkm_path = tmp_path / "lsgkm.model.txt"
        with open(lsgkm_path, "w") as f:
            f.write("svm_type c_svc\n")
            f.write("kernel_type gkm_cnt\n")
            f.write("L 5\n")
            f.write("k 3\n")
            f.write("nr_class 2\n")
            f.write(f"total_sv {len(svs)}\n")
            f.write(f"rho {rho}\n")
            f.write("SV\n")
            for c, s in zip(coefs, svs):
                f.write(f"{c} {s}\n")

        classic_model = load_classic_model(classic_path)
        lsgkm_model = load_lsgkm_model(lsgkm_path)

        assert classic_model.bias == pytest.approx(rho)
        assert lsgkm_model.bias == pytest.approx(-rho)
        assert classic_model.bias == pytest.approx(-lsgkm_model.bias)

    def test_scores_differ_by_twice_rho(self, tmp_path):
        """Same model loaded as classic vs LS-GKM should differ by 2*rho."""
        svs = _make_seqs(3, 11, seed=3)
        coefs = [0.5, -0.3, 0.2]
        rho = 0.42

        classic_path = tmp_path / "classic.gkmmodel"
        _write_classic_model_embedded(classic_path, svs, coefs, rho, l=5, k=3)

        lsgkm_path = tmp_path / "lsgkm.model.txt"
        with open(lsgkm_path, "w") as f:
            f.write("svm_type c_svc\n")
            f.write("kernel_type gkm_cnt\n")
            f.write("L 5\n")
            f.write("k 3\n")
            f.write("nr_class 2\n")
            f.write(f"total_sv {len(svs)}\n")
            f.write(f"rho {rho}\n")
            f.write("SV\n")
            for c, s in zip(coefs, svs):
                f.write(f"{c} {s}\n")

        classic = load_classic_model(classic_path)
        lsgkm = load_lsgkm_model(lsgkm_path)

        x = one_hot_encode(_make_seqs(1, 15, seed=100)[0]).unsqueeze(0)
        diff = classic(x).item() - lsgkm(x).item()
        assert diff == pytest.approx(2 * rho, abs=1e-5)

    def test_integer_kernel_type(self, tmp_path):
        """Classic format uses integer kernel types."""
        svs = _make_seqs(3, 11, seed=4)
        coefs = [0.5, -0.3, 0.2]
        path = tmp_path / "model.gkmmodel"
        _write_classic_model_embedded(path, svs, coefs, 0.1, l=5, k=3, kernel_type=0)

        model = load_classic_model(path)
        assert model.kernel_type == "gkm_cnt"

    def test_kernel_type_2(self, tmp_path):
        svs = _make_seqs(3, 11, seed=5)
        coefs = [0.5, -0.3, 0.2]
        path = tmp_path / "model.gkmmodel"
        _write_classic_model_embedded(path, svs, coefs, 0.1, l=5, k=3, kernel_type=2)

        model = load_classic_model(path)
        assert model.kernel_type == "gkm_esttrunc"


class TestClassicImporterTwoFile:
    def test_load_two_file(self, tmp_path):
        svs = _make_seqs(3, 11, seed=10)
        coefs = [0.5, -0.3, 0.2]
        rho = 0.15

        model_path = tmp_path / "model.txt"
        svseq_path = tmp_path / "model.svseq.fa"
        _write_classic_model_twofile(
            model_path, svseq_path, svs, coefs, rho, l=5, k=3
        )

        model = load_classic_model(model_path, svseq_path=svseq_path)
        assert model.num_support_vectors == 3
        assert model.bias == pytest.approx(rho)

    def test_two_file_scores_match_embedded(self, tmp_path):
        svs = _make_seqs(3, 11, seed=11)
        coefs = [0.5, -0.3, 0.2]
        rho = 0.15

        embedded_path = tmp_path / "embedded.gkmmodel"
        _write_classic_model_embedded(embedded_path, svs, coefs, rho, l=5, k=3)

        model_path = tmp_path / "twofile.txt"
        svseq_path = tmp_path / "twofile.svseq.fa"
        _write_classic_model_twofile(
            model_path, svseq_path, svs, coefs, rho, l=5, k=3
        )

        embedded = load_classic_model(embedded_path)
        twofile = load_classic_model(model_path, svseq_path=svseq_path)

        x = one_hot_encode(_make_seqs(1, 15, seed=200)[0]).unsqueeze(0)
        assert torch.allclose(embedded(x), twofile(x), atol=1e-6)

    def test_alphas_without_sv_marker(self, tmp_path):
        """Alpha file without SV marker (just header + coefficients)."""
        svs = _make_seqs(3, 11, seed=12)
        coefs = [0.5, -0.3, 0.2]
        rho = 0.1

        model_path = tmp_path / "model.txt"
        svseq_path = tmp_path / "model.svseq.fa"

        with open(model_path, "w") as f:
            f.write("svm_type c_svc\n")
            f.write("kernel_type 0\n")
            f.write("L 5\n")
            f.write("k 3\n")
            f.write("nr_class 2\n")
            f.write(f"total_sv {len(svs)}\n")
            f.write(f"rho {rho}\n")
            for c in coefs:
                f.write(f"{c}\n")

        with open(svseq_path, "w") as f:
            for i, s in enumerate(svs):
                f.write(f">sv_{i}\n{s}\n")

        model = load_classic_model(model_path, svseq_path=svseq_path)
        assert model.num_support_vectors == 3
        assert model.bias == pytest.approx(rho)


class TestClassicImporterErrors:
    def test_missing_rho(self, tmp_path):
        path = tmp_path / "bad.gkmmodel"
        with open(path, "w") as f:
            f.write("svm_type c_svc\n")
            f.write("kernel_type 0\n")
            f.write("L 5\n")
            f.write("k 3\n")
            f.write("nr_class 2\n")
            f.write("total_sv 1\n")
            f.write("SV\n")
            f.write("0.5 ACGTACGTACG\n")

        with pytest.raises(ValueError, match="rho"):
            load_classic_model(path)

    def test_missing_kernel_type(self, tmp_path):
        path = tmp_path / "bad.gkmmodel"
        with open(path, "w") as f:
            f.write("svm_type c_svc\n")
            f.write("L 5\n")
            f.write("k 3\n")
            f.write("rho 0.1\n")
            f.write("nr_class 2\n")
            f.write("total_sv 1\n")
            f.write("SV\n")
            f.write("0.5 ACGTACGTACG\n")

        with pytest.raises(ValueError, match="kernel_type"):
            load_classic_model(path)

    def test_sv_count_mismatch(self, tmp_path):
        svs = _make_seqs(3, 11, seed=20)
        coefs = [0.5, -0.3, 0.2]
        path = tmp_path / "bad.gkmmodel"
        with open(path, "w") as f:
            f.write("svm_type c_svc\n")
            f.write("kernel_type 0\n")
            f.write("L 5\n")
            f.write("k 3\n")
            f.write("nr_class 2\n")
            f.write("total_sv 5\n")
            f.write("rho 0.1\n")
            f.write("SV\n")
            for c, s in zip(coefs, svs):
                f.write(f"{c} {s}\n")

        with pytest.raises(ValueError, match="Expected 5"):
            load_classic_model(path)

    def test_coef_seq_count_mismatch(self, tmp_path):
        svs = _make_seqs(3, 11, seed=21)
        model_path = tmp_path / "model.txt"
        svseq_path = tmp_path / "model.svseq.fa"

        with open(model_path, "w") as f:
            f.write("svm_type c_svc\n")
            f.write("kernel_type 0\n")
            f.write("L 5\n")
            f.write("k 3\n")
            f.write("nr_class 2\n")
            f.write("rho 0.1\n")
            f.write("SV\n")
            f.write("0.5\n")
            f.write("-0.3\n")

        with open(svseq_path, "w") as f:
            for i, s in enumerate(svs):
                f.write(f">sv_{i}\n{s}\n")

        with pytest.raises(ValueError, match="does not match"):
            load_classic_model(model_path, svseq_path=svseq_path)


class TestClassicImporterKernelModes:
    def test_kernel_type_3_rbf(self, tmp_path):
        svs = _make_seqs(3, 11, seed=30)
        coefs = [0.5, -0.3, 0.2]
        path = tmp_path / "model.gkmmodel"
        with open(path, "w") as f:
            f.write("svm_type c_svc\n")
            f.write("kernel_type 3\n")
            f.write("L 5\n")
            f.write("k 3\n")
            f.write("d 3\n")
            f.write("gamma 2.0\n")
            f.write("nr_class 2\n")
            f.write(f"total_sv {len(svs)}\n")
            f.write("rho 0.1\n")
            f.write("SV\n")
            for c, s in zip(coefs, svs):
                f.write(f"{c} {s}\n")

        model = load_classic_model(path)
        assert model.kernel_type == "gkmrbf"
        x = one_hot_encode(_make_seqs(1, 15, seed=300)[0]).unsqueeze(0)
        score = model(x)
        assert torch.isfinite(score).all()

    def test_kernel_type_4_wgkm(self, tmp_path):
        svs = _make_seqs(3, 11, seed=31)
        coefs = [0.5, -0.3, 0.2]
        path = tmp_path / "model.gkmmodel"
        with open(path, "w") as f:
            f.write("svm_type c_svc\n")
            f.write("kernel_type 4\n")
            f.write("L 5\n")
            f.write("k 3\n")
            f.write("M 3\n")
            f.write("H 1.0\n")
            f.write("nr_class 2\n")
            f.write(f"total_sv {len(svs)}\n")
            f.write("rho 0.1\n")
            f.write("SV\n")
            for c, s in zip(coefs, svs):
                f.write(f"{c} {s}\n")

        model = load_classic_model(path)
        assert model.kernel_type == "wgkm"
        x = one_hot_encode(_make_seqs(1, 15, seed=301)[0]).unsqueeze(0)
        score = model(x)
        assert torch.isfinite(score).all()

    def test_kernel_type_5_wgkmrbf(self, tmp_path):
        svs = _make_seqs(3, 11, seed=32)
        coefs = [0.5, -0.3, 0.2]
        path = tmp_path / "model.gkmmodel"
        with open(path, "w") as f:
            f.write("svm_type c_svc\n")
            f.write("kernel_type 5\n")
            f.write("L 5\n")
            f.write("k 3\n")
            f.write("M 3\n")
            f.write("H 1.0\n")
            f.write("gamma 2.0\n")
            f.write("nr_class 2\n")
            f.write(f"total_sv {len(svs)}\n")
            f.write("rho 0.1\n")
            f.write("SV\n")
            for c, s in zip(coefs, svs):
                f.write(f"{c} {s}\n")

        model = load_classic_model(path)
        assert model.kernel_type == "wgkmrbf"
        x = one_hot_encode(_make_seqs(1, 15, seed=302)[0]).unsqueeze(0)
        score = model(x)
        assert torch.isfinite(score).all()
