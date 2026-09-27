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


class TestToDeltaSVM:
    def test_single_sequence_exact(self):
        """Normalized SVM score can be recovered from DeltaSVM + K(x,x)."""
        rng = random.Random(42)
        seqs = ["".join(rng.choice("ACGT") for _ in range(30)) for _ in range(10)]
        svs = np.stack([one_hot_encode(s) for s in seqs[:5]])
        coefs = np.array([0.5, -0.3, 0.2, -0.1, 0.4], dtype=np.float32)
        model = GkmSVM(
            svs, coefs, bias=-0.05, kernel_type="gkm_cnt",
            kernel_params={"L": 3, "k": 2, "include_rc": True},
        )
        dsvm = model.to_deltasvm()
        assert dsvm.k == dsvm.l == 3
        assert dsvm.weights.shape == (64,)

        for seq in seqs[5:]:
            x = one_hot_encode(seq)[np.newaxis]
            svm_score = model(x).item()
            dsvm_score = dsvm(x).item()
            diag = float(model.kernel._raw_diagonal(x)[0])
            recovered = (dsvm_score - model.bias) / np.sqrt(diag) + model.bias
            assert recovered == pytest.approx(svm_score, abs=1e-5)

    def test_variant_correlation_realistic_l(self):
        """DeltaSVM variant effects correlate near-perfectly for l >= 10."""
        from gkmsvm import train_gkmsvm

        rng = random.Random(0)
        pos = ["".join(rng.choice("ACGT") for _ in range(50)) for _ in range(60)]
        neg = ["".join(rng.choice("ACGT") for _ in range(50)) for _ in range(60)]
        model = train_gkmsvm(pos, neg, l=10, k=7, C=1.0, kernel_type="direct")
        dsvm = model.to_deltasvm()

        test_seqs = ["".join(rng.choice("ACGT") for _ in range(50)) for _ in range(20)]
        svm_d, dsvm_d = [], []
        for seq in test_seqs:
            ref = one_hot_encode(seq)[np.newaxis]
            p = 25
            new_b = "C" if seq[p] != "C" else "A"
            alt = one_hot_encode(seq[:p] + new_b + seq[p + 1:])[np.newaxis]
            svm_d.append((model(alt) - model(ref)).item())
            dsvm_d.append((dsvm(alt) - dsvm(ref)).item())

        corr = np.corrcoef(svm_d, dsvm_d)[0, 1]
        assert corr > 0.99, f"Pearson r={corr:.4f}, expected > 0.99"

    def test_l_too_large(self):
        svs = np.stack([one_hot_encode("A" * 20)])
        coefs = np.array([1.0], dtype=np.float32)
        model = GkmSVM(
            svs, coefs, bias=0.0, kernel_type="gkm_cnt",
            kernel_params={"L": 15, "k": 7, "include_rc": False},
        )
        with pytest.raises(ValueError, match="l <= 14"):
            model.to_deltasvm()
