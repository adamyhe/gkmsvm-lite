import random

import numpy as np
import pytest

from gkmsvm.codec import one_hot_encode
from gkmsvm.kernels.direct import DirectGkmKernel
from gkmsvm.svm import GkmSVM


@pytest.fixture
def small_model():
    """A small GkmSVM with 3 support vectors, l=3, k=2."""
    svs = np.stack([
        one_hot_encode("ACGT"),
        one_hot_encode("AAAA"),
        one_hot_encode("CCCC"),
    ])
    coefs = np.array([0.5, -0.3, 0.2], dtype=np.float32)
    return GkmSVM(
        support_sequences=svs,
        coefficients=coefs,
        bias=-0.1,
        kernel_type="gkm_cnt",
        kernel_params={"L": 3, "k": 2, "include_rc": False},
    )


class TestGkmSVMConstruction:
    def test_basic(self, small_model):
        assert small_model.num_support_vectors == 3
        assert small_model.bias == pytest.approx(-0.1)
        assert small_model.kernel_params == {"L": 3, "k": 2, "include_rc": False}

    def test_shape_validation(self):
        with pytest.raises(ValueError, match="\\[S, 4, L\\]"):
            GkmSVM(
                support_sequences=np.zeros((3, 10)),
                coefficients=np.zeros(3),
                bias=0.0,
                kernel_type="gkm_cnt",
                kernel_params={"L": 3, "k": 2, "include_rc": True},
            )

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="must match"):
            GkmSVM(
                support_sequences=np.zeros((3, 4, 10)),
                coefficients=np.zeros(5),
                bias=0.0,
                kernel_type="gkm_cnt",
                kernel_params={"L": 3, "k": 2, "include_rc": True},
            )

    def test_unsupported_kernel(self):
        with pytest.raises(NotImplementedError, match="Unknown kernel type"):
            GkmSVM(
                support_sequences=np.zeros((3, 4, 10)),
                coefficients=np.zeros(3),
                bias=0.0,
                kernel_type="totally_fake_kernel",
                kernel_params={"L": 11, "k": 7, "include_rc": True},
            )


class TestGkmSVMForward:
    def test_output_shape(self, small_model):
        x = one_hot_encode("ACGT")[np.newaxis]
        scores = small_model(x)
        assert scores.shape == (1, 1)

    def test_batch(self, small_model):
        x = np.stack([one_hot_encode("ACGT"), one_hot_encode("AAAA")])
        scores = small_model(x)
        assert scores.shape == (2, 1)

    def test_known_score(self):
        sv = one_hot_encode("AAA")[np.newaxis]
        coef = np.array([1.0], dtype=np.float32)
        bias = 0.0
        model = GkmSVM(
            support_sequences=sv,
            coefficients=coef,
            bias=bias,
            kernel_type="gkm_cnt",
            kernel_params={"L": 3, "k": 2, "include_rc": False},
        )
        query = one_hot_encode("AAA")[np.newaxis]
        score = model(query)
        assert float(score[0, 0]) == pytest.approx(1.0)

    def test_bias_applied(self):
        sv = one_hot_encode("AAA")[np.newaxis]
        coef = np.array([1.0], dtype=np.float32)
        model = GkmSVM(
            support_sequences=sv,
            coefficients=coef,
            bias=-0.5,
            kernel_type="gkm_cnt",
            kernel_params={"L": 3, "k": 2, "include_rc": False},
        )
        query = one_hot_encode("AAA")[np.newaxis]
        score = model(query)
        assert float(score[0, 0]) == pytest.approx(0.5)


class TestGkmSVMChunked:
    def test_chunked_matches_full(self, small_model):
        x = np.stack([one_hot_encode("ACGT"), one_hot_encode("AAAA")])
        full_scores = small_model(x)

        small_model.sv_chunk_size = 2
        chunked_scores = small_model(x)

        np.testing.assert_allclose(full_scores, chunked_scores, atol=1e-6)

    def test_chunk_size_one(self, small_model):
        x = one_hot_encode("ACGT")[np.newaxis]
        full_scores = small_model(x)

        small_model.sv_chunk_size = 1
        chunked_scores = small_model(x)

        np.testing.assert_allclose(full_scores, chunked_scores, atol=1e-6)

    def test_chunked_esttrunc(self):
        rng = random.Random(99)
        seqs = ["".join(rng.choice("ACGT") for _ in range(30)) for _ in range(20)]
        svs = np.stack([one_hot_encode(s) for s in seqs])
        coefs = np.random.RandomState(42).randn(20).astype(np.float32)
        queries = np.stack([one_hot_encode(s) for s in seqs[:5]])

        model = GkmSVM(svs, coefs, -0.1, "gkm_esttrunc",
                        {"L": 11, "k": 7, "d": 3, "include_rc": True})
        full = model(queries)

        for chunk in [3, 7, 10, 20]:
            model.sv_chunk_size = chunk
            chunked = model(queries)
            np.testing.assert_allclose(full, chunked, atol=1e-6,
                err_msg=f"chunk_size={chunk}")


class TestGkmSVMVariants:
    def test_score_variants(self, small_model):
        ref = one_hot_encode("ACGT")[np.newaxis]
        alt = one_hot_encode("AAAA")[np.newaxis]
        delta = small_model.score_variants(ref, alt)
        expected = small_model(alt) - small_model(ref)
        np.testing.assert_allclose(delta, expected)

    def test_same_variant_is_zero(self, small_model):
        seq = one_hot_encode("ACGT")[np.newaxis]
        delta = small_model.score_variants(seq, seq)
        assert float(delta[0, 0]) == pytest.approx(0.0, abs=1e-6)
