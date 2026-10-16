"""GPU tests for all operations. Skipped when CuPy is not available.

Each test computes the result on both CPU and GPU, then asserts they match.
This validates the CuPy RawKernels, device transfers, and GPU code paths.
"""
import random

import numpy as np
import pytest

from gkmsvm.backend import HAS_CUPY
from gkmsvm.codec import one_hot_encode
from gkmsvm.svm import GkmSVM

pytestmark = pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")

if HAS_CUPY:
    import cupy as cp


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


class TestDeviceTransfer:
    def test_cuda_moves_to_gpu(self, direct_model):
        direct_model.cuda()
        assert isinstance(direct_model.support_sequences, cp.ndarray)
        assert isinstance(direct_model.coefficients, cp.ndarray)

    def test_cpu_moves_back(self, direct_model):
        direct_model.cuda()
        direct_model.cpu()
        assert isinstance(direct_model.support_sequences, np.ndarray)
        assert isinstance(direct_model.coefficients, np.ndarray)

    def test_cuda_returns_self(self, direct_model):
        result = direct_model.cuda()
        assert result is direct_model


# ---------------------------------------------------------------------------
# Forward pass (scoring)
# ---------------------------------------------------------------------------


class TestGPUScoring:
    def test_direct_scores_match(self, direct_model):
        x = np.stack([one_hot_encode(s) for s in _make_seqs(3, 15, seed=10)])
        cpu_scores = direct_model(x)

        direct_model.cuda()
        gpu_scores = direct_model(cp.asarray(x))

        np.testing.assert_allclose(
            cp.asnumpy(gpu_scores), cpu_scores, rtol=1e-9, atol=1e-12,
        )

    def test_esttrunc_scores_match(self, esttrunc_model):
        x = np.stack([one_hot_encode(s) for s in _make_seqs(2, 30, seed=11)])
        cpu_scores = esttrunc_model(x)

        esttrunc_model.cuda()
        gpu_scores = esttrunc_model(cp.asarray(x))

        np.testing.assert_allclose(
            cp.asnumpy(gpu_scores), cpu_scores, rtol=1e-9, atol=1e-12,
        )

    def test_esttrunc_no_rc_no_norm(self, esttrunc_model_no_rc_no_norm):
        model = esttrunc_model_no_rc_no_norm
        x = np.stack([one_hot_encode(s) for s in _make_seqs(2, 20, seed=12)])
        cpu_scores = model(x)

        model.cuda()
        gpu_scores = model(cp.asarray(x))

        np.testing.assert_allclose(
            cp.asnumpy(gpu_scores), cpu_scores, rtol=1e-9, atol=1e-12,
        )

    def test_chunked_scoring(self, esttrunc_model):
        x = np.stack([one_hot_encode(s) for s in _make_seqs(2, 30, seed=13)])
        cpu_scores = esttrunc_model(x)

        esttrunc_model.cuda()
        esttrunc_model.sv_chunk_size = 3
        gpu_scores = esttrunc_model(cp.asarray(x))

        np.testing.assert_allclose(
            cp.asnumpy(gpu_scores), cpu_scores, rtol=1e-9, atol=1e-12,
        )

    def test_single_sequence(self, direct_model):
        x = one_hot_encode("ACGTACGTACGTACG")[np.newaxis]
        cpu_scores = direct_model(x)

        direct_model.cuda()
        gpu_scores = direct_model(cp.asarray(x))

        np.testing.assert_allclose(
            cp.asnumpy(gpu_scores), cpu_scores, rtol=1e-9, atol=1e-12,
        )

    def test_rc_invariance_on_gpu(self, esttrunc_model):
        from gkmsvm.codec import reverse_complement

        seq = _make_seqs(1, 30, seed=14)[0]
        x = one_hot_encode(seq)[np.newaxis]
        x_rc = reverse_complement(x)

        esttrunc_model.cuda()
        score = esttrunc_model(cp.asarray(x))
        score_rc = esttrunc_model(cp.asarray(x_rc))

        np.testing.assert_allclose(
            cp.asnumpy(score), cp.asnumpy(score_rc), rtol=1e-9, atol=1e-12,
        )


# ---------------------------------------------------------------------------
# ISM
# ---------------------------------------------------------------------------


class TestGPUIsm:
    def test_ism_matches_cpu(self, direct_model):
        from gkmsvm.ism import ism

        x = np.stack([one_hot_encode(s) for s in _make_seqs(2, 15, seed=20)])
        cpu_result = ism(direct_model, x)

        direct_model.cuda()
        gpu_result = ism(direct_model, cp.asarray(x))

        np.testing.assert_allclose(
            cp.asnumpy(gpu_result), cpu_result, rtol=1e-9, atol=1e-12,
        )

    def test_ism_esttrunc_matches_cpu(self, esttrunc_model):
        from gkmsvm.ism import ism

        x = one_hot_encode(_make_seqs(1, 30, seed=21)[0])[np.newaxis]
        cpu_result = ism(esttrunc_model, x)

        esttrunc_model.cuda()
        gpu_result = ism(esttrunc_model, cp.asarray(x))

        np.testing.assert_allclose(
            cp.asnumpy(gpu_result), cpu_result, rtol=1e-9, atol=1e-12,
        )

    def test_ism_ref_base_zero_on_gpu(self, esttrunc_model):
        from gkmsvm.ism import ism

        x = one_hot_encode(_make_seqs(1, 30, seed=22)[0])[np.newaxis]
        esttrunc_model.cuda()
        result = cp.asnumpy(ism(esttrunc_model, cp.asarray(x)))
        ref_bases = x.argmax(axis=1)
        for p in range(x.shape[2]):
            assert abs(float(result[0, int(ref_bases[0, p]), p])) < 1e-5


# ---------------------------------------------------------------------------
# GkmExplain
# ---------------------------------------------------------------------------


class TestGPUGkmExplain:
    def test_mode0_matches_cpu(self, esttrunc_model):
        from gkmsvm.explain import gkmexplain

        x = one_hot_encode(_make_seqs(1, 30, seed=30)[0])[np.newaxis]
        cpu_result = gkmexplain(esttrunc_model, x, mode=0)

        esttrunc_model.cuda()
        gpu_result = gkmexplain(esttrunc_model, cp.asarray(x), mode=0)

        np.testing.assert_allclose(
            cp.asnumpy(gpu_result), cpu_result, rtol=1e-9, atol=1e-12,
        )

    def test_mode1_matches_cpu(self, esttrunc_model):
        from gkmsvm.explain import gkmexplain

        x = one_hot_encode(_make_seqs(1, 30, seed=31)[0])[np.newaxis]
        cpu_result = gkmexplain(esttrunc_model, x, mode=1)

        esttrunc_model.cuda()
        gpu_result = gkmexplain(esttrunc_model, cp.asarray(x), mode=1)

        np.testing.assert_allclose(
            cp.asnumpy(gpu_result), cpu_result, rtol=1e-9, atol=1e-12,
        )

    def test_mode0_direct_kernel(self, direct_model):
        from gkmsvm.explain import gkmexplain

        x = np.stack([one_hot_encode(s) for s in _make_seqs(2, 15, seed=32)])
        cpu_result = gkmexplain(direct_model, x, mode=0)

        direct_model.cuda()
        gpu_result = gkmexplain(direct_model, cp.asarray(x), mode=0)

        np.testing.assert_allclose(
            cp.asnumpy(gpu_result), cpu_result, rtol=1e-9, atol=1e-12,
        )

    def test_mode1_direct_kernel(self, direct_model):
        from gkmsvm.explain import gkmexplain

        x = np.stack([one_hot_encode(s) for s in _make_seqs(2, 15, seed=33)])
        cpu_result = gkmexplain(direct_model, x, mode=1)

        direct_model.cuda()
        gpu_result = gkmexplain(direct_model, cp.asarray(x), mode=1)

        np.testing.assert_allclose(
            cp.asnumpy(gpu_result), cpu_result, rtol=1e-9, atol=1e-12,
        )

    def test_decomposition_on_gpu(self, esttrunc_model):
        """Sum of attribution * one-hot == score - bias, on GPU."""
        from gkmsvm.explain import gkmexplain

        x = one_hot_encode(_make_seqs(1, 30, seed=34)[0])[np.newaxis]
        esttrunc_model.cuda()
        x_gpu = cp.asarray(x)

        exp = gkmexplain(esttrunc_model, x_gpu, mode=0)
        score = esttrunc_model(x_gpu).squeeze(-1)
        exp_sum = (exp * x_gpu).sum(axis=(1, 2))
        expected = score - esttrunc_model.bias

        np.testing.assert_allclose(
            cp.asnumpy(exp_sum),
            cp.asnumpy(expected).astype(np.float64),
            rtol=1e-9, atol=1e-12,
        )

    def test_nonref_bases_zero_on_gpu(self, esttrunc_model):
        from gkmsvm.explain import gkmexplain

        x = one_hot_encode(_make_seqs(1, 30, seed=35)[0])[np.newaxis]
        esttrunc_model.cuda()
        exp = cp.asnumpy(gkmexplain(esttrunc_model, cp.asarray(x), mode=0))
        non_ref = (1 - x).astype(bool)
        assert (np.abs(exp[non_ref]) < 1e-10).all()

    def test_chunked_matches_full_on_gpu(self, esttrunc_model):
        from gkmsvm.explain import gkmexplain

        x = one_hot_encode(_make_seqs(1, 30, seed=36)[0])[np.newaxis]
        esttrunc_model.cuda()
        x_gpu = cp.asarray(x)

        full = gkmexplain(esttrunc_model, x_gpu, mode=0)
        esttrunc_model.sv_chunk_size = 3
        chunked = gkmexplain(esttrunc_model, x_gpu, mode=0)

        np.testing.assert_allclose(
            cp.asnumpy(full), cp.asnumpy(chunked), atol=1e-8,
        )

    def test_mode0_no_rc_no_norm(self, esttrunc_model_no_rc_no_norm):
        from gkmsvm.explain import gkmexplain

        model = esttrunc_model_no_rc_no_norm
        x = one_hot_encode(_make_seqs(1, 20, seed=37)[0])[np.newaxis]
        cpu_result = gkmexplain(model, x, mode=0)

        model.cuda()
        gpu_result = gkmexplain(model, cp.asarray(x), mode=0)

        np.testing.assert_allclose(
            cp.asnumpy(gpu_result), cpu_result, rtol=1e-9, atol=1e-12,
        )


# ---------------------------------------------------------------------------
# Kernel pairwise (lower-level)
# ---------------------------------------------------------------------------


class TestGPUKernelPairwise:
    def test_pairwise_direct(self):
        from gkmsvm.kernels.direct import DirectGkmKernel

        kernel = DirectGkmKernel(l=3, k=2, include_rc=False, normalize=False)
        seqs_a = np.stack([one_hot_encode(s) for s in _make_seqs(3, 10, seed=40)])
        seqs_b = np.stack([one_hot_encode(s) for s in _make_seqs(4, 10, seed=41)])

        cpu_result = kernel.pairwise(seqs_a, seqs_b)

        gpu_result = kernel.pairwise(cp.asarray(seqs_a), cp.asarray(seqs_b))

        np.testing.assert_allclose(
            cp.asnumpy(gpu_result), cpu_result, rtol=1e-9, atol=1e-12,
        )

    def test_pairwise_esttrunc(self):
        from gkmsvm.kernels.esttrunc import EstTruncGkmKernel

        kernel = EstTruncGkmKernel(
            l=11, k=7, d=3, include_rc=True, normalize=True,
        )
        seqs_a = np.stack([one_hot_encode(s) for s in _make_seqs(2, 30, seed=42)])
        seqs_b = np.stack([one_hot_encode(s) for s in _make_seqs(3, 30, seed=43)])

        cpu_result = kernel.pairwise(seqs_a, seqs_b)

        gpu_result = kernel.pairwise(cp.asarray(seqs_a), cp.asarray(seqs_b))

        np.testing.assert_allclose(
            cp.asnumpy(gpu_result), cpu_result, rtol=1e-9, atol=1e-12,
        )

    def test_self_pairwise_symmetric_on_gpu(self):
        from gkmsvm.kernels.esttrunc import EstTruncGkmKernel

        kernel = EstTruncGkmKernel(
            l=11, k=7, d=3, include_rc=True, normalize=True,
        )
        seqs = np.stack([one_hot_encode(s) for s in _make_seqs(4, 30, seed=44)])

        result = kernel.pairwise(cp.asarray(seqs), cp.asarray(seqs))
        result_np = cp.asnumpy(result)

        np.testing.assert_allclose(result_np, result_np.T, atol=1e-10)


# ---------------------------------------------------------------------------
# Variant scoring
# ---------------------------------------------------------------------------


class TestGPUVariantScoring:
    def test_score_variants_matches_cpu(self, esttrunc_model):
        ref = np.stack([one_hot_encode(s) for s in _make_seqs(2, 30, seed=50)])
        alt = np.stack([one_hot_encode(s) for s in _make_seqs(2, 30, seed=51)])
        cpu_delta = esttrunc_model.score_variants(ref, alt)

        esttrunc_model.cuda()
        gpu_delta = esttrunc_model.score_variants(cp.asarray(ref), cp.asarray(alt))

        np.testing.assert_allclose(
            cp.asnumpy(gpu_delta), cpu_delta, rtol=1e-9, atol=1e-12,
        )

    def test_same_variant_zero_on_gpu(self, esttrunc_model):
        seq = one_hot_encode(_make_seqs(1, 30, seed=52)[0])[np.newaxis]
        esttrunc_model.cuda()
        seq_gpu = cp.asarray(seq)
        delta = esttrunc_model.score_variants(seq_gpu, seq_gpu)
        assert abs(float(cp.asnumpy(delta)[0, 0])) < 1e-6


# ---------------------------------------------------------------------------
# Gram matrix
# ---------------------------------------------------------------------------


class TestGPUGram:
    def test_gram_matches_cpu(self):
        from gkmsvm.gram import compute_gram
        from gkmsvm.kernels.esttrunc import EstTruncGkmKernel

        kernel = EstTruncGkmKernel(
            l=11, k=7, d=3, include_rc=True, normalize=True,
        )
        seqs = np.stack([one_hot_encode(s) for s in _make_seqs(6, 25, seed=60)])
        cpu_gram = compute_gram(kernel, seqs)

        gpu_gram = compute_gram(kernel, cp.asarray(seqs))

        np.testing.assert_allclose(
            cp.asnumpy(gpu_gram), cpu_gram, rtol=1e-9, atol=1e-12,
        )

    def test_gram_symmetric_on_gpu(self):
        from gkmsvm.gram import compute_gram
        from gkmsvm.kernels.esttrunc import EstTruncGkmKernel

        kernel = EstTruncGkmKernel(
            l=11, k=7, d=3, include_rc=True, normalize=True,
        )
        seqs = np.stack([one_hot_encode(s) for s in _make_seqs(5, 25, seed=61)])

        gram = cp.asnumpy(compute_gram(kernel, cp.asarray(seqs)))
        np.testing.assert_allclose(gram, gram.T, atol=1e-10)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _random_seqs_str(n, length, rng):
    return ["".join(rng.choice(list("ACGT")) for _ in range(length)) for _ in range(n)]


class TestGPUTraining:
    def test_train_libsvm_scores_match_cpu(self):
        from gkmsvm import train_gkmsvm

        rng = np.random.default_rng(42)
        pos = _random_seqs_str(20, 20, rng)
        neg = _random_seqs_str(20, 20, rng)

        cpu_model = train_gkmsvm(
            pos, neg, kernel_type="direct", l=7, k=5, C=1.0,
            device="cpu",
        )
        gpu_model = train_gkmsvm(
            pos, neg, kernel_type="direct", l=7, k=5, C=1.0,
            device="cuda",
        )

        rng2 = np.random.default_rng(99)
        x = np.stack([one_hot_encode(s)
                      for s in _random_seqs_str(5, 20, rng2)])
        cpu_scores = cpu_model(x)
        gpu_scores = gpu_model(x)

        np.testing.assert_allclose(gpu_scores, cpu_scores, rtol=1e-9, atol=1e-12)

    def test_train_smo_scores_match_cpu(self):
        from gkmsvm import train_gkmsvm

        rng = np.random.default_rng(99)
        pos = _random_seqs_str(20, 20, rng)
        neg = _random_seqs_str(20, 20, rng)

        cpu_model = train_gkmsvm(
            pos, neg, kernel_type="direct", l=7, k=5, C=1.0,
            solver="smo", device="cpu",
        )
        gpu_model = train_gkmsvm(
            pos, neg, kernel_type="direct", l=7, k=5, C=1.0,
            solver="smo", device="cuda",
        )

        rng2 = np.random.default_rng(77)
        x = np.stack([one_hot_encode(s)
                      for s in _random_seqs_str(5, 20, rng2)])
        cpu_scores = cpu_model(x)
        gpu_scores = gpu_model(x)

        np.testing.assert_allclose(gpu_scores, cpu_scores, rtol=1e-9, atol=1e-12)

    def test_train_returns_numpy_model(self):
        from gkmsvm import train_gkmsvm

        rng = np.random.default_rng(42)
        pos = _random_seqs_str(15, 20, rng)
        neg = _random_seqs_str(15, 20, rng)

        model = train_gkmsvm(
            pos, neg, kernel_type="direct", l=7, k=5, C=1.0,
            device="cuda",
        )

        assert isinstance(model.support_sequences, np.ndarray)
        assert isinstance(model.coefficients, np.ndarray)


class TestMaskedGPU:
    """CUDA masked join must equal the CPU masked join and brute force."""

    @pytest.mark.parametrize("which", ["t0", "t1", "t2_l9", "t2_l11"])
    @pytest.mark.parametrize("col_tile", [8192, 7])
    def test_matches_cpu(self, which, col_tile, monkeypatch):
        import gkmsvm.kernels.masked as masked
        from gkmsvm.codec import reverse_complement
        from gkmsvm.kernels.direct import (
            DirectGkmKernel,
            _fused_pairwise_packed_numba,
        )
        from gkmsvm.kernels.esttrunc import EstTruncGkmKernel

        monkeypatch.setattr(masked, "_GPU_COL_TILE", col_tile)
        kernel = {
            "t0": lambda: DirectGkmKernel(8, 5),
            "t1": lambda: EstTruncGkmKernel(9, 6, d=3, truncate=False),
            "t2_l9": lambda: EstTruncGkmKernel(9, 6, d=3),
            "t2_l11": lambda: EstTruncGkmKernel(11, 7, d=3),
        }[which]()
        X = np.stack([one_hot_encode(s) for s in _make_seqs(21, 80, seed=3)])
        X[2, :, 5:45] = X[9, :, 30:70]  # shared subsequence
        Y = np.stack([one_hot_encode(s) for s in _make_seqs(30, 60, seed=4)])

        def pack(A):
            return kernel.pack(A)

        bx, by, by_rc = pack(X), pack(Y), pack(reverse_complement(Y))
        t, l, mm = kernel._mismatch_table, kernel.l, kernel._min_matches
        plan = masked.mask_plan(tuple(t), l, mm)
        ref = (_fused_pairwise_packed_numba(bx, by, t, l, mm)
               + _fused_pairwise_packed_numba(bx, by_rc, t, l, mm))
        got = masked.masked_pairwise_gpu(cp.asarray(bx), cp.asarray(by), plan,
                                         cp.asarray(by_rc))
        np.testing.assert_allclose(cp.asnumpy(got), ref, rtol=1e-12, atol=1e-12)

    def test_model_scores_unchanged(self):
        """GPU scoring through the masked path (>= 16 SVs and queries) matches CPU."""
        model = _make_model(40, 100, "gkm_esttrunc",
                            {"L": 11, "k": 7, "d": 3, "include_rc": True}, seed=6)
        X = np.stack([one_hot_encode(s) for s in _make_seqs(40, 100, seed=5)])
        cpu = np.asarray(model(X)).ravel()
        model.cuda()
        gpu = cp.asnumpy(model(X)).ravel()
        np.testing.assert_allclose(gpu, cpu, rtol=1e-9, atol=1e-12)


class TestLmerTableGPU:
    def test_table_matches_kernel_path(self):
        model = _make_model(40, 100, "gkm_esttrunc",
                            {"L": 11, "k": 7, "d": 3, "include_rc": True}, seed=7)
        X = np.stack([one_hot_encode(s) for s in _make_seqs(30, 120, seed=8)])
        model.use_lmer_table = False
        ref = np.asarray(model(X)).ravel()
        model.use_lmer_table = True
        model.cuda()
        got = cp.asnumpy(model(X)).ravel()
        assert model._lmer_table is not False
        np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)


class TestLmerIndexGPU:
    """Cached l-mer index on the device vs the CPU index."""

    @pytest.mark.parametrize("which", ["t0", "t2_l9", "t2_l11"])
    def test_index_matches_cpu(self, which):
        from gkmsvm.codec import reverse_complement
        from gkmsvm.kernels.direct import DirectGkmKernel
        from gkmsvm.kernels.esttrunc import EstTruncGkmKernel
        from gkmsvm.kernels.lmer_index import LmerIndex
        from gkmsvm.kernels.packed import PAD_WINDOW

        kernel = {
            "t0": lambda: DirectGkmKernel(8, 5),
            "t2_l9": lambda: EstTruncGkmKernel(9, 6, d=3),
            "t2_l11": lambda: EstTruncGkmKernel(11, 7, d=3),
        }[which]()
        X = np.stack([one_hot_encode(s) for s in _make_seqs(21, 80, seed=3)])
        Y = np.stack([one_hot_encode(s) for s in _make_seqs(30, 60, seed=4)])
        X[2, :, 5:45] = Y[9, :, 10:50]  # shared subsequence
        index = LmerIndex(kernel, kernel.pack(Y), kernel.pack(reverse_complement(Y)))
        bx = kernel.pack(X)
        bx[1, :3] = PAD_WINDOW
        ref = index.pairwise(bx)
        got = cp.asnumpy(index.pairwise(cp.asarray(bx)))
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)

    def test_rbf_model_scores_match_cpu(self):
        model = _make_model(40, 60, "gkmrbf", {"L": 9, "k": 6, "d": 3, "gamma": 2.0})
        X = np.stack([one_hot_encode(s) for s in _make_seqs(25, 60, seed=7)])
        cpu = np.asarray(model(X)).ravel()
        assert model._sv_index() is not None
        model.cuda()
        got = cp.asnumpy(model(cp.asarray(X))).ravel()
        np.testing.assert_allclose(got, cpu, rtol=1e-9, atol=1e-12)
