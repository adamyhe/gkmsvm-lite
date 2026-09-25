import gzip
import textwrap

import pytest
import torch

from gkmsvm.codec import one_hot_encode
from gkmsvm.importers.lsgkm import load_lsgkm_model, parse_lsgkm_header
from gkmsvm.kernels.direct import DirectGkmKernel


SAMPLE_MODEL = textwrap.dedent("""\
    svm_type c_svc
    kernel_type gkm_cnt
    L 3
    k 2
    d 3
    nr_class 2
    total_sv 3
    rho 0.25
    label 1 -1
    nr_sv 2 1
    SV
    0.5 ACGT
    0.3 AAAA
    -0.8 CCCC
""")

SAMPLE_MODEL_NO_NORC = textwrap.dedent("""\
    svm_type c_svc
    kernel_type gkm_cnt
    L 3
    k 2
    d 3
    nr_class 2
    total_sv 2
    rho -0.1
    SV
    1.0 ACGT
    -0.5 TGCA
""")


@pytest.fixture
def model_path(tmp_path):
    p = tmp_path / "test.model.txt"
    p.write_text(SAMPLE_MODEL)
    return p


@pytest.fixture
def model_path_gz(tmp_path):
    p = tmp_path / "test.model.txt.gz"
    with gzip.open(p, "wt") as f:
        f.write(SAMPLE_MODEL)
    return p


@pytest.fixture
def model_no_norc(tmp_path):
    p = tmp_path / "no_norc.model.txt"
    p.write_text(SAMPLE_MODEL_NO_NORC)
    return p


class TestParseHeader:
    def test_basic(self, model_path):
        h = parse_lsgkm_header(model_path)
        assert h["svm_type"] == "c_svc"
        assert h["kernel_type"] == "gkm_cnt"
        assert h["L"] == 3
        assert h["k"] == 2
        assert h["d"] == 3
        assert h["nr_class"] == 2
        assert h["total_sv"] == 3
        assert h["rho"] == [0.25]
        assert h["label"] == [1, -1]
        assert h["nr_sv"] == [2, 1]

    def test_norc_default(self, model_no_norc):
        h = parse_lsgkm_header(model_no_norc)
        assert h["norc"] == 0

    def test_gzip(self, model_path_gz):
        h = parse_lsgkm_header(model_path_gz)
        assert h["kernel_type"] == "gkm_cnt"
        assert h["total_sv"] == 3


class TestLoadModel:
    def test_basic(self, model_path):
        model = load_lsgkm_model(model_path)
        assert model.num_support_vectors == 3
        assert model.bias == pytest.approx(-0.25)
        assert model.kernel_type == "gkm_cnt"

    def test_coefficients(self, model_path):
        model = load_lsgkm_model(model_path)
        expected = torch.tensor([0.5, 0.3, -0.8])
        assert torch.allclose(model.coefficients, expected)

    def test_gzip(self, model_path_gz):
        model = load_lsgkm_model(model_path_gz)
        assert model.num_support_vectors == 3

    def test_norc_absent_defaults_to_rc(self, model_no_norc):
        model = load_lsgkm_model(model_no_norc)
        assert model.kernel.include_rc is True

    def test_forward_runs(self, model_path):
        model = load_lsgkm_model(model_path)
        query = one_hot_encode("ACGT").unsqueeze(0)
        score = model(query)
        assert score.shape == (1, 1)
        assert torch.isfinite(score).all()

    def test_end_to_end_score(self, model_path):
        """Verify loaded model produces correct scores vs. hand computation."""
        model = load_lsgkm_model(model_path)
        query = one_hot_encode("ACGT").unsqueeze(0)

        # Manually compute: K(query, sv_i) * coef_i summed, then + bias
        kernel = DirectGkmKernel(l=3, k=2, normalize=True, include_rc=True)
        svs = model.support_sequences
        K = kernel.pairwise(query, svs)  # [1, 3]
        coefs = model.coefficients
        manual_score = (K * coefs).sum(dim=1, keepdim=True) + model._bias

        model_score = model(query)
        assert torch.allclose(model_score, manual_score, atol=1e-6)


class TestLoadEstTruncModel:
    def test_load_esttrunc(self, tmp_path):
        p = tmp_path / "trunc.model.txt"
        p.write_text(textwrap.dedent("""\
            svm_type c_svc
            kernel_type gkm_esttrunc
            L 11
            k 7
            d 3
            nr_class 2
            total_sv 2
            rho 0.5
            label 1 -1
            nr_sv 1 1
            SV
            1.0 ACGTACGTACGT
            -0.5 TGCATGCATGCA
        """))
        model = load_lsgkm_model(p)
        assert model.num_support_vectors == 2
        assert model.kernel_type == "gkm_esttrunc"
        assert model.bias == pytest.approx(-0.5)

    def test_esttrunc_forward_runs(self, tmp_path):
        p = tmp_path / "trunc.model.txt"
        p.write_text(textwrap.dedent("""\
            svm_type c_svc
            kernel_type gkm_esttrunc
            L 3
            k 2
            d 1
            nr_class 2
            total_sv 2
            rho 0.1
            SV
            0.8 ACGT
            -0.3 TGCA
        """))
        model = load_lsgkm_model(p)
        query = one_hot_encode("ACGT").unsqueeze(0)
        score = model(query)
        assert score.shape == (1, 1)
        assert torch.isfinite(score).all()


class TestLoadModelErrors:
    def test_unsupported_kernel(self, tmp_path):
        p = tmp_path / "fake.model.txt"
        p.write_text(textwrap.dedent("""\
            svm_type c_svc
            kernel_type totally_fake_kernel
            L 11
            k 7
            d 3
            nr_class 2
            total_sv 1
            rho 0.0
            SV
            1.0 ACGTACGTACGT
        """))
        with pytest.raises(NotImplementedError, match="totally_fake_kernel"):
            load_lsgkm_model(p)

    def test_wrong_sv_count(self, tmp_path):
        p = tmp_path / "bad.model.txt"
        p.write_text(textwrap.dedent("""\
            svm_type c_svc
            kernel_type gkm_cnt
            L 3
            k 2
            d 3
            nr_class 2
            total_sv 5
            rho 0.0
            SV
            1.0 ACGT
        """))
        with pytest.raises(ValueError, match="Expected 5"):
            load_lsgkm_model(p)

    def test_variable_length_svs(self, tmp_path):
        p = tmp_path / "varlen.model.txt"
        p.write_text(textwrap.dedent("""\
            svm_type c_svc
            kernel_type gkm_cnt
            L 3
            k 2
            d 3
            nr_class 2
            total_sv 2
            rho 0.0
            SV
            1.0 ACGT
            -1.0 ACGTAC
        """))
        with pytest.raises(ValueError, match="same length"):
            load_lsgkm_model(p)
