"""MLX tests for Apple Silicon GPU inference. Skipped when MLX is not available.

Each test computes the result on both CPU and MLX, then asserts they match.
This validates the packed uint32 MLX kernel, device transfers, and MLX code paths.
"""
import random

import numpy as np
import pytest

from gkmsvm.backend import HAS_MLX
from gkmsvm.codec import one_hot_encode
from gkmsvm.svm import GkmSVM

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")

if HAS_MLX:
    import mlx.core as mx
    from gkmsvm.backend import to_mlx, to_cpu


def _make_seqs(n, length, seed=42):
    rng = random.Random(seed)
    return ["".join(rng.choice("ACGT") for _ in range(length)) for _ in range(n)]


def _make_model(n_sv, seqlen, kernel_type, kernel_params, seed=1):
    svs = np.stack([one_hot_encode(s) for s in _make_seqs(n_sv, seqlen, seed)])
    rng = np.random.default_rng(seed)
    coefs = rng.standard_normal(n_sv).astype(np.float32)
    return GkmSVM(svs, coefs, -0.15, kernel_type, kernel_params)


@pytest.fixture
def direct_model():
    return _make_model(5, 15, "gkm_cnt", {"L": 3, "k": 2, "include_rc": True})


@pytest.fixture
def esttrunc_model():
    return _make_model(
        10, 30, "gkm_esttrunc",
        {"L": 11, "k": 7, "d": 3, "include_rc": True}, seed=2,
    )


@pytest.fixture
def esttrunc_model_no_rc_no_norm():
    model = _make_model(
        5, 20, "gkm_esttrunc",
        {"L": 11, "k": 7, "d": 3, "include_rc": False}, seed=3,
    )
    model.kernel.normalize = False
    return model


# ---------------------------------------------------------------------------
# Device transfer
# ---------------------------------------------------------------------------


class TestMLXDeviceTransfer:
    def test_mlx_moves_to_device(self, direct_model):
        direct_model.mlx()
        assert isinstance(direct_model.support_sequences, mx.array)
        assert isinstance(direct_model.coefficients, mx.array)

    def test_cpu_moves_back(self, direct_model):
        direct_model.mlx()
        direct_model.cpu()
        assert isinstance(direct_model.support_sequences, np.ndarray)
        assert isinstance(direct_model.coefficients, np.ndarray)

    def test_mlx_returns_self(self, direct_model):
        result = direct_model.mlx()
        assert result is direct_model


# ---------------------------------------------------------------------------
# Forward pass (scoring)
# ---------------------------------------------------------------------------


class TestMLXScoring:
    def test_direct_scores_match(self, direct_model):
        x = np.stack([one_hot_encode(s) for s in _make_seqs(3, 15, seed=10)])
        cpu_scores = direct_model(x)

        direct_model.mlx()
        mlx_scores = direct_model(to_mlx(x))

        np.testing.assert_allclose(
            to_cpu(mlx_scores), cpu_scores, atol=1e-5,
        )

    def test_esttrunc_scores_match(self, esttrunc_model):
        x = np.stack([one_hot_encode(s) for s in _make_seqs(2, 30, seed=11)])
        cpu_scores = esttrunc_model(x)

        esttrunc_model.mlx()
        mlx_scores = esttrunc_model(to_mlx(x))

        np.testing.assert_allclose(
            to_cpu(mlx_scores), cpu_scores, atol=1e-5,
        )

    def test_esttrunc_no_rc_no_norm(self, esttrunc_model_no_rc_no_norm):
        model = esttrunc_model_no_rc_no_norm
        x = np.stack([one_hot_encode(s) for s in _make_seqs(2, 20, seed=12)])
        cpu_scores = model(x)

        model.mlx()
        mlx_scores = model(to_mlx(x))

        np.testing.assert_allclose(
            to_cpu(mlx_scores), cpu_scores, atol=1e-5,
        )

    def test_chunked_scoring(self, esttrunc_model):
        x = np.stack([one_hot_encode(s) for s in _make_seqs(2, 30, seed=13)])
        cpu_scores = esttrunc_model(x)

        esttrunc_model.mlx()
        esttrunc_model.sv_chunk_size = 3
        mlx_scores = esttrunc_model(to_mlx(x))

        np.testing.assert_allclose(
            to_cpu(mlx_scores), cpu_scores, atol=1e-5,
        )

    def test_single_sequence(self, direct_model):
        x = one_hot_encode("ACGTACGTACGTACG")[np.newaxis]
        cpu_scores = direct_model(x)

        direct_model.mlx()
        mlx_scores = direct_model(to_mlx(x))

        np.testing.assert_allclose(
            to_cpu(mlx_scores), cpu_scores, atol=1e-5,
        )

    def test_rc_invariance(self, esttrunc_model):
        from gkmsvm.codec import reverse_complement

        seq = _make_seqs(1, 30, seed=14)[0]
        x = one_hot_encode(seq)[np.newaxis]
        x_rc = reverse_complement(x)

        esttrunc_model.mlx()
        score = esttrunc_model(to_mlx(x))
        score_rc = esttrunc_model(to_mlx(x_rc))

        np.testing.assert_allclose(
            to_cpu(score), to_cpu(score_rc), atol=1e-5,
        )


# ---------------------------------------------------------------------------
# ISM
# ---------------------------------------------------------------------------


class TestMLXIsm:
    def test_ism_matches_cpu(self, direct_model):
        from gkmsvm.ism import ism

        x = np.stack([one_hot_encode(s) for s in _make_seqs(2, 15, seed=20)])
        cpu_result = ism(direct_model, x)

        direct_model.mlx()
        mlx_result = ism(direct_model, to_mlx(x))

        np.testing.assert_allclose(
            to_cpu(mlx_result), cpu_result, atol=1e-4,
        )

    def test_ism_esttrunc_matches_cpu(self, esttrunc_model):
        from gkmsvm.ism import ism

        x = one_hot_encode(_make_seqs(1, 30, seed=21)[0])[np.newaxis]
        cpu_result = ism(esttrunc_model, x)

        esttrunc_model.mlx()
        mlx_result = ism(esttrunc_model, to_mlx(x))

        np.testing.assert_allclose(
            to_cpu(mlx_result), cpu_result, atol=1e-4,
        )

    def test_ism_ref_base_zero(self, esttrunc_model):
        from gkmsvm.ism import ism

        x = one_hot_encode(_make_seqs(1, 30, seed=22)[0])[np.newaxis]
        esttrunc_model.mlx()
        result = to_cpu(ism(esttrunc_model, to_mlx(x)))
        ref_bases = x.argmax(axis=1)
        for p in range(x.shape[2]):
            assert abs(float(result[0, int(ref_bases[0, p]), p])) < 1e-4


# ---------------------------------------------------------------------------
# GkmExplain (CPU fallback — verifies explain still works when model is on MLX)
# ---------------------------------------------------------------------------


class TestMLXGkmExplain:
    def test_mode0_falls_back_to_cpu(self, esttrunc_model):
        from gkmsvm.explain import gkmexplain

        x = one_hot_encode(_make_seqs(1, 30, seed=30)[0])[np.newaxis]
        cpu_result = gkmexplain(esttrunc_model, x, mode=0)

        esttrunc_model.mlx()
        mlx_result = gkmexplain(esttrunc_model, to_mlx(x), mode=0)

        np.testing.assert_allclose(
            to_cpu(mlx_result), cpu_result, atol=1e-6,
        )

    def test_decomposition_from_mlx(self, esttrunc_model):
        from gkmsvm.explain import gkmexplain

        x = one_hot_encode(_make_seqs(1, 30, seed=34)[0])[np.newaxis]
        cpu_score = esttrunc_model(x).squeeze(-1)

        esttrunc_model.mlx()
        exp = to_cpu(gkmexplain(esttrunc_model, to_mlx(x), mode=0))
        exp_sum = (exp * x).sum(axis=(1, 2))
        expected = cpu_score - esttrunc_model.bias

        np.testing.assert_allclose(
            exp_sum, expected.astype(np.float64), atol=1e-5,
        )


# ---------------------------------------------------------------------------
# Kernel pairwise (lower-level packed kernel tests)
# ---------------------------------------------------------------------------


class TestMLXKernelPairwise:
    def test_pairwise_direct(self):
        from gkmsvm.kernels.direct import DirectGkmKernel

        kernel = DirectGkmKernel(l=3, k=2, include_rc=False, normalize=False)
        seqs_a = np.stack([one_hot_encode(s) for s in _make_seqs(3, 10, seed=40)])
        seqs_b = np.stack([one_hot_encode(s) for s in _make_seqs(4, 10, seed=41)])

        cpu_result = kernel.pairwise(seqs_a, seqs_b)
        mlx_result = kernel.pairwise(to_mlx(seqs_a), to_mlx(seqs_b))

        np.testing.assert_allclose(
            to_cpu(mlx_result), cpu_result, atol=1e-5,
        )

    def test_pairwise_esttrunc(self):
        from gkmsvm.kernels.esttrunc import EstTruncGkmKernel

        kernel = EstTruncGkmKernel(
            l=11, k=7, d=3, include_rc=True, normalize=True,
        )
        seqs_a = np.stack([one_hot_encode(s) for s in _make_seqs(2, 30, seed=42)])
        seqs_b = np.stack([one_hot_encode(s) for s in _make_seqs(3, 30, seed=43)])

        cpu_result = kernel.pairwise(seqs_a, seqs_b)
        mlx_result = kernel.pairwise(to_mlx(seqs_a), to_mlx(seqs_b))

        np.testing.assert_allclose(
            to_cpu(mlx_result), cpu_result, atol=1e-5,
        )

    def test_pairwise_from_indices_packed(self):
        from gkmsvm.kernels.direct import DirectGkmKernel

        kernel = DirectGkmKernel(l=7, k=5, include_rc=False, normalize=False)
        seqs_a = np.stack([one_hot_encode(s) for s in _make_seqs(3, 15, seed=45)])
        seqs_b = np.stack([one_hot_encode(s) for s in _make_seqs(5, 15, seed=46)])

        bx = kernel.base_index_windows(seqs_a)
        by = kernel.base_index_windows(seqs_b)
        cpu_result = kernel.pairwise_from_indices(bx, by)

        mlx_result = kernel.pairwise_from_indices(to_mlx(bx), to_mlx(by))

        np.testing.assert_allclose(
            to_cpu(mlx_result), cpu_result, atol=1e-5,
        )

    def test_diagonal_packed(self):
        from gkmsvm.kernels.direct import DirectGkmKernel

        kernel = DirectGkmKernel(l=7, k=5, include_rc=True, normalize=False)
        seqs = np.stack([one_hot_encode(s) for s in _make_seqs(4, 15, seed=47)])

        cpu_diag = kernel._raw_diagonal(seqs)
        mlx_diag = kernel._raw_diagonal(to_mlx(seqs))

        np.testing.assert_allclose(
            to_cpu(mlx_diag), cpu_diag, atol=1e-5,
        )

    def test_diagonal_esttrunc_packed(self):
        from gkmsvm.kernels.esttrunc import EstTruncGkmKernel

        kernel = EstTruncGkmKernel(
            l=11, k=7, d=3, include_rc=True, normalize=False,
        )
        seqs = np.stack([one_hot_encode(s) for s in _make_seqs(4, 30, seed=48)])

        cpu_diag = kernel._raw_diagonal(seqs)
        mlx_diag = kernel._raw_diagonal(to_mlx(seqs))

        np.testing.assert_allclose(
            to_cpu(mlx_diag), cpu_diag, atol=1e-4,
        )

    def test_self_pairwise_symmetric(self):
        from gkmsvm.kernels.esttrunc import EstTruncGkmKernel

        kernel = EstTruncGkmKernel(
            l=11, k=7, d=3, include_rc=True, normalize=True,
        )
        seqs = np.stack([one_hot_encode(s) for s in _make_seqs(4, 30, seed=44)])

        result = to_cpu(kernel.pairwise(to_mlx(seqs), to_mlx(seqs)))
        np.testing.assert_allclose(result, result.T, atol=1e-5)


# ---------------------------------------------------------------------------
# Variant scoring
# ---------------------------------------------------------------------------


class TestMLXVariantScoring:
    def test_score_variants_matches_cpu(self, esttrunc_model):
        ref = np.stack([one_hot_encode(s) for s in _make_seqs(2, 30, seed=50)])
        alt = np.stack([one_hot_encode(s) for s in _make_seqs(2, 30, seed=51)])
        cpu_delta = esttrunc_model.score_variants(ref, alt)

        esttrunc_model.mlx()
        mlx_delta = esttrunc_model.score_variants(to_mlx(ref), to_mlx(alt))

        np.testing.assert_allclose(
            to_cpu(mlx_delta), cpu_delta, atol=1e-5,
        )

    def test_same_variant_zero(self, esttrunc_model):
        seq = one_hot_encode(_make_seqs(1, 30, seed=52)[0])[np.newaxis]
        esttrunc_model.mlx()
        seq_mlx = to_mlx(seq)
        delta = esttrunc_model.score_variants(seq_mlx, seq_mlx)
        assert abs(float(to_cpu(delta)[0, 0])) < 1e-5


# ---------------------------------------------------------------------------
# Gram matrix
# ---------------------------------------------------------------------------


class TestMLXGram:
    def test_gram_matches_cpu(self):
        from gkmsvm.gram import compute_gram
        from gkmsvm.kernels.esttrunc import EstTruncGkmKernel

        kernel = EstTruncGkmKernel(
            l=11, k=7, d=3, include_rc=True, normalize=True,
        )
        seqs = np.stack([one_hot_encode(s) for s in _make_seqs(6, 25, seed=60)])
        cpu_gram = compute_gram(kernel, seqs)

        mlx_gram = compute_gram(kernel, to_mlx(seqs))

        np.testing.assert_allclose(mlx_gram, cpu_gram, atol=1e-4)

    def test_gram_symmetric(self):
        from gkmsvm.gram import compute_gram
        from gkmsvm.kernels.esttrunc import EstTruncGkmKernel

        kernel = EstTruncGkmKernel(
            l=11, k=7, d=3, include_rc=True, normalize=True,
        )
        seqs = np.stack([one_hot_encode(s) for s in _make_seqs(5, 25, seed=61)])

        gram = compute_gram(kernel, to_mlx(seqs))
        np.testing.assert_allclose(gram, gram.T, atol=1e-10)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _random_seqs_str(n, length, rng):
    return ["".join(rng.choice(list("ACGT")) for _ in range(length)) for _ in range(n)]


class TestMLXTraining:
    def test_train_libsvm_matches_cpu(self):
        from gkmsvm import train_gkmsvm

        rng = np.random.default_rng(42)
        pos = _random_seqs_str(20, 20, rng)
        neg = _random_seqs_str(20, 20, rng)

        cpu_model = train_gkmsvm(
            pos, neg, kernel_type="direct", l=7, k=5, C=1.0,
            device="cpu",
        )
        mlx_model = train_gkmsvm(
            pos, neg, kernel_type="direct", l=7, k=5, C=1.0,
            device="mlx",
        )

        x = np.stack([one_hot_encode(s) for s in _random_seqs_str(5, 20, rng)])
        cpu_scores = cpu_model(x)
        mlx_scores = mlx_model(x)

        np.testing.assert_allclose(mlx_scores, cpu_scores, atol=1e-5)

    def test_train_smo_matches_cpu(self):
        from gkmsvm import train_gkmsvm

        rng = np.random.default_rng(99)
        pos = _random_seqs_str(20, 20, rng)
        neg = _random_seqs_str(20, 20, rng)

        cpu_model = train_gkmsvm(
            pos, neg, kernel_type="direct", l=7, k=5, C=1.0,
            solver="smo", device="cpu",
        )
        mlx_model = train_gkmsvm(
            pos, neg, kernel_type="direct", l=7, k=5, C=1.0,
            solver="smo", device="mlx",
        )

        x = np.stack([one_hot_encode(s) for s in _random_seqs_str(5, 20, rng)])
        cpu_scores = cpu_model(x)
        mlx_scores = mlx_model(x)

        np.testing.assert_allclose(mlx_scores, cpu_scores, atol=1e-4)

    def test_train_svr_matches_cpu(self):
        from gkmsvm import train_gkmsvr

        rng = np.random.default_rng(77)
        seqs = _random_seqs_str(30, 20, rng)
        labels = rng.standard_normal(30)

        cpu_model = train_gkmsvr(
            seqs, labels, kernel_type="direct", l=7, k=5, C=1.0,
            device="cpu",
        )
        mlx_model = train_gkmsvr(
            seqs, labels, kernel_type="direct", l=7, k=5, C=1.0,
            device="mlx",
        )

        x = np.stack([one_hot_encode(s) for s in _random_seqs_str(5, 20, rng)])
        cpu_scores = cpu_model(x)
        mlx_scores = mlx_model(x)

        np.testing.assert_allclose(mlx_scores, cpu_scores, atol=1e-5)

    def test_model_returns_numpy_arrays(self):
        from gkmsvm import train_gkmsvm

        rng = np.random.default_rng(42)
        pos = _random_seqs_str(15, 20, rng)
        neg = _random_seqs_str(15, 20, rng)

        model = train_gkmsvm(
            pos, neg, kernel_type="direct", l=7, k=5, C=1.0,
            device="mlx",
        )

        assert isinstance(model.support_sequences, np.ndarray)
        assert isinstance(model.coefficients, np.ndarray)

    def test_device_auto_selects(self):
        from gkmsvm import train_gkmsvm

        rng = np.random.default_rng(42)
        pos = _random_seqs_str(15, 20, rng)
        neg = _random_seqs_str(15, 20, rng)

        model = train_gkmsvm(
            pos, neg, kernel_type="direct", l=7, k=5, C=1.0,
            device="auto",
        )
        assert model.num_support_vectors > 0
        assert isinstance(model.support_sequences, np.ndarray)
