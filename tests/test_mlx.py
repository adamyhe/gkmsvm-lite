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
        10,
        30,
        "gkm_esttrunc",
        {"L": 11, "k": 7, "d": 3, "include_rc": True},
        seed=2,
    )


@pytest.fixture
def esttrunc_model_no_rc_no_norm():
    model = _make_model(
        5,
        20,
        "gkm_esttrunc",
        {"L": 11, "k": 7, "d": 3, "include_rc": False},
        seed=3,
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

    def test_one_hot_decode_mlx(self):
        from gkmsvm.codec import one_hot_decode

        assert one_hot_decode(to_mlx(one_hot_encode("ACGTTGCA"))) == "ACGTTGCA"


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
            to_cpu(mlx_scores),
            cpu_scores,
            atol=1e-5,
        )

    def test_esttrunc_scores_match(self, esttrunc_model):
        x = np.stack([one_hot_encode(s) for s in _make_seqs(2, 30, seed=11)])
        cpu_scores = esttrunc_model(x)

        esttrunc_model.mlx()
        mlx_scores = esttrunc_model(to_mlx(x))

        np.testing.assert_allclose(
            to_cpu(mlx_scores),
            cpu_scores,
            atol=1e-5,
        )

    def test_esttrunc_no_rc_no_norm(self, esttrunc_model_no_rc_no_norm):
        model = esttrunc_model_no_rc_no_norm
        x = np.stack([one_hot_encode(s) for s in _make_seqs(2, 20, seed=12)])
        cpu_scores = model(x)

        model.mlx()
        mlx_scores = model(to_mlx(x))

        np.testing.assert_allclose(
            to_cpu(mlx_scores),
            cpu_scores,
            atol=1e-5,
        )

    def test_chunked_scoring(self, esttrunc_model):
        x = np.stack([one_hot_encode(s) for s in _make_seqs(2, 30, seed=13)])
        cpu_scores = esttrunc_model(x)

        esttrunc_model.mlx()
        esttrunc_model.sv_chunk_size = 3
        mlx_scores = esttrunc_model(to_mlx(x))

        np.testing.assert_allclose(
            to_cpu(mlx_scores),
            cpu_scores,
            atol=1e-5,
        )

    def test_single_sequence(self, direct_model):
        x = one_hot_encode("ACGTACGTACGTACG")[np.newaxis]
        cpu_scores = direct_model(x)

        direct_model.mlx()
        mlx_scores = direct_model(to_mlx(x))

        np.testing.assert_allclose(
            to_cpu(mlx_scores),
            cpu_scores,
            atol=1e-5,
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
            to_cpu(score),
            to_cpu(score_rc),
            atol=1e-5,
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
            to_cpu(mlx_result),
            cpu_result,
            atol=1e-4,
        )

    def test_ism_esttrunc_matches_cpu(self, esttrunc_model):
        from gkmsvm.ism import ism

        x = one_hot_encode(_make_seqs(1, 30, seed=21)[0])[np.newaxis]
        cpu_result = ism(esttrunc_model, x)

        esttrunc_model.mlx()
        mlx_result = ism(esttrunc_model, to_mlx(x))

        np.testing.assert_allclose(
            to_cpu(mlx_result),
            cpu_result,
            atol=1e-4,
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
# GkmExplain (runs on CPU for a model on MLX and returns NumPy)
# ---------------------------------------------------------------------------


class TestMLXGkmExplain:
    def test_mode0_falls_back_to_cpu(self, esttrunc_model):
        from gkmsvm.explain import gkmexplain

        x = one_hot_encode(_make_seqs(1, 30, seed=30)[0])[np.newaxis]
        cpu_result = gkmexplain(esttrunc_model, x, mode=0)

        esttrunc_model.mlx()
        mlx_result = gkmexplain(esttrunc_model, to_mlx(x), mode=0)

        np.testing.assert_allclose(
            to_cpu(mlx_result),
            cpu_result,
            atol=1e-6,
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
            exp_sum,
            expected.astype(np.float64),
            atol=1e-5,
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
            to_cpu(mlx_result),
            cpu_result,
            atol=1e-5,
        )

    def test_pairwise_esttrunc(self):
        from gkmsvm.kernels.esttrunc import EstTruncGkmKernel

        kernel = EstTruncGkmKernel(
            l=11,
            k=7,
            d=3,
            include_rc=True,
            normalize=True,
        )
        seqs_a = np.stack([one_hot_encode(s) for s in _make_seqs(2, 30, seed=42)])
        seqs_b = np.stack([one_hot_encode(s) for s in _make_seqs(3, 30, seed=43)])

        cpu_result = kernel.pairwise(seqs_a, seqs_b)
        mlx_result = kernel.pairwise(to_mlx(seqs_a), to_mlx(seqs_b))

        np.testing.assert_allclose(
            to_cpu(mlx_result),
            cpu_result,
            atol=1e-5,
        )

    def test_raw_packed(self):
        from gkmsvm.kernels.direct import DirectGkmKernel

        kernel = DirectGkmKernel(l=7, k=5, include_rc=False, normalize=False)
        seqs_a = np.stack([one_hot_encode(s) for s in _make_seqs(3, 15, seed=45)])
        seqs_b = np.stack([one_hot_encode(s) for s in _make_seqs(5, 15, seed=46)])

        cpu_result = kernel.raw_packed(kernel.pack(seqs_a), kernel.pack(seqs_b))
        mlx_result = kernel.raw_packed(
            kernel.pack(to_mlx(seqs_a)), kernel.pack(to_mlx(seqs_b))
        )

        np.testing.assert_allclose(
            to_cpu(mlx_result),
            cpu_result,
            atol=1e-5,
        )

    def test_diagonal_packed(self):
        from gkmsvm.kernels.direct import DirectGkmKernel

        kernel = DirectGkmKernel(l=7, k=5, include_rc=True, normalize=False)
        seqs = np.stack([one_hot_encode(s) for s in _make_seqs(4, 15, seed=47)])

        cpu_diag = kernel._raw_diagonal(seqs)
        mlx_diag = kernel._raw_diagonal(to_mlx(seqs))

        np.testing.assert_allclose(
            to_cpu(mlx_diag),
            cpu_diag,
            atol=1e-5,
        )

    def test_diagonal_esttrunc_packed(self):
        from gkmsvm.kernels.esttrunc import EstTruncGkmKernel

        kernel = EstTruncGkmKernel(
            l=11,
            k=7,
            d=3,
            include_rc=True,
            normalize=False,
        )
        seqs = np.stack([one_hot_encode(s) for s in _make_seqs(4, 30, seed=48)])

        cpu_diag = kernel._raw_diagonal(seqs)
        mlx_diag = kernel._raw_diagonal(to_mlx(seqs))

        np.testing.assert_allclose(
            to_cpu(mlx_diag),
            cpu_diag,
            atol=1e-4,
        )

    def test_self_pairwise_symmetric(self):
        from gkmsvm.kernels.esttrunc import EstTruncGkmKernel

        kernel = EstTruncGkmKernel(
            l=11,
            k=7,
            d=3,
            include_rc=True,
            normalize=True,
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
            to_cpu(mlx_delta),
            cpu_delta,
            atol=1e-5,
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
            l=11,
            k=7,
            d=3,
            include_rc=True,
            normalize=True,
        )
        seqs = np.stack([one_hot_encode(s) for s in _make_seqs(6, 25, seed=60)])
        cpu_gram = compute_gram(kernel, seqs)

        mlx_gram = compute_gram(kernel, to_mlx(seqs))

        np.testing.assert_allclose(mlx_gram, cpu_gram, atol=1e-4)

    def test_gram_symmetric(self):
        from gkmsvm.gram import compute_gram
        from gkmsvm.kernels.esttrunc import EstTruncGkmKernel

        kernel = EstTruncGkmKernel(
            l=11,
            k=7,
            d=3,
            include_rc=True,
            normalize=True,
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
            pos,
            neg,
            kernel_type="direct",
            l=7,
            k=5,
            C=1.0,
            device="cpu",
        )
        mlx_model = train_gkmsvm(
            pos,
            neg,
            kernel_type="direct",
            l=7,
            k=5,
            C=1.0,
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
            pos,
            neg,
            kernel_type="direct",
            l=7,
            k=5,
            C=1.0,
            solver="smo",
            device="cpu",
        )
        mlx_model = train_gkmsvm(
            pos,
            neg,
            kernel_type="direct",
            l=7,
            k=5,
            C=1.0,
            solver="smo",
            device="mlx",
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
            seqs,
            labels,
            kernel_type="direct",
            l=7,
            k=5,
            C=1.0,
            device="cpu",
        )
        mlx_model = train_gkmsvr(
            seqs,
            labels,
            kernel_type="direct",
            l=7,
            k=5,
            C=1.0,
            device="mlx",
        )

        x = np.stack([one_hot_encode(s) for s in _random_seqs_str(5, 20, rng)])
        cpu_scores = cpu_model(x)
        mlx_scores = mlx_model(x)

        np.testing.assert_allclose(mlx_scores, cpu_scores, atol=1e-5)

    def test_train_returns_numpy_model(self):
        from gkmsvm import train_gkmsvm

        rng = np.random.default_rng(42)
        pos = _random_seqs_str(15, 20, rng)
        neg = _random_seqs_str(15, 20, rng)

        model = train_gkmsvm(
            pos,
            neg,
            kernel_type="direct",
            l=7,
            k=5,
            C=1.0,
            device="mlx",
        )

        assert not model._on_mlx
        assert isinstance(model.support_sequences, np.ndarray)
        assert isinstance(model.coefficients, np.ndarray)

    def test_device_auto_selects(self, capsys):
        from gkmsvm import train_gkmsvm

        rng = np.random.default_rng(42)
        pos = _random_seqs_str(15, 20, rng)
        neg = _random_seqs_str(15, 20, rng)

        model = train_gkmsvm(
            pos,
            neg,
            kernel_type="direct",
            l=7,
            k=5,
            C=1.0,
            device="auto",
            verbose=True,
        )
        assert model.num_support_vectors > 0
        assert "Moving training data to MLX" in capsys.readouterr().out


class TestMaskedMLX:
    """Metal masked join and l-mer table vs the CPU paths."""

    @pytest.mark.parametrize(
        "kernel_type,params",
        [
            ("gkm_cnt", {"L": 8, "k": 5}),
            ("gkm_esttrunc", {"L": 9, "k": 6, "d": 3}),
            ("wgkm", {"L": 8, "k": 5, "M": 3, "H": 2.0}),
        ],
    )
    @pytest.mark.parametrize("col_tile", [4096, 7])
    def test_join_matches_cpu(self, kernel_type, params, col_tile, monkeypatch):
        import gkmsvm.kernels.masked as masked
        from gkmsvm.codec import reverse_complement

        monkeypatch.setattr(masked, "_METAL_COL_TILE", col_tile)
        kernel = _make_model(20, 60, kernel_type, params).kernel
        X = np.stack([one_hot_encode(s) for s in _make_seqs(21, 80, seed=3)])
        X[2, :, 5:45] = X[9, :, 30:70]
        Y = np.stack([one_hot_encode(s) for s in _make_seqs(30, 60, seed=4)])

        def pack(A):
            return kernel.pack(A)

        plan = kernel._join_plan()
        args = (pack(X), pack(Y), plan, pack(reverse_complement(Y)))
        np.testing.assert_allclose(
            masked.masked_pairwise_mlx(*args),
            masked.masked_pairwise(*args),
            rtol=1e-12,
            atol=1e-12,
        )

    @pytest.mark.parametrize(
        "kernel_type,params",
        [
            ("gkm_esttrunc", {"L": 9, "k": 6, "d": 3}),
            ("wgkm", {"L": 8, "k": 5, "M": 3, "H": 2.0}),
        ],
    )
    @pytest.mark.parametrize("table", [False, True])
    def test_model_scores(self, kernel_type, params, table):
        """MLX scoring (masked join or l-mer table) matches CPU scoring."""
        model = _make_model(40, 60, kernel_type, params, seed=6)
        X = np.stack([one_hot_encode(s) for s in _make_seqs(30, 80, seed=5)])
        model.use_lmer_table = False
        cpu = np.asarray(model(X)).ravel()
        model.use_lmer_table = table
        model.mlx()
        got = np.array(model(mx.array(X))).ravel()
        np.testing.assert_allclose(got, cpu, atol=1e-5)


class TestWeightedMLX:
    """-t 4/5 window weights in Metal: the masked join and l-mer index count
    in 64 bits (a carry word), brute force and self-kernels in float32."""

    @staticmethod
    def _polya_and_random(n, length, seed):
        x = np.zeros((n, 4, length), np.float32)
        x[: n // 2, 0] = 1  # poly-A: every pair matches, counts past 2^32
        rng = np.random.default_rng(seed)
        x[n // 2 :] = np.eye(4, dtype=np.float32)[
            rng.integers(0, 4, (n - n // 2, length))
        ].transpose(0, 2, 1)
        return x

    @pytest.mark.parametrize("padded", [False, True])
    @pytest.mark.parametrize("col_tile", [2048, 7])
    def test_join_is_exact(self, padded, col_tile, monkeypatch):
        import gkmsvm.kernels.masked as masked
        from gkmsvm.kernels.weighted import CenterWeightedGkmKernel

        monkeypatch.setattr(masked, "_METAL_COL_TILE_W", col_tile)
        kernel = CenterWeightedGkmKernel(11, 7, d=3, M=255, H=1e9)  # all 255
        x = self._polya_and_random(18, 1000, 1)
        lengths = np.r_[1000, np.random.default_rng(2).integers(20, 1001, 17)]
        bx, bx_rc = kernel.pack_both(x, lengths if padded else None)
        w, w_rc = kernel.row_weights(bx), kernel.row_weights(bx_rc, rc=True)
        plan = kernel._join_plan()
        got = masked.masked_pairwise_mlx(bx, bx, plan, bx_rc, wx=w, wy=w, wy_rc=w_rc)
        # poly-A against poly-A: every forward pair a full match at weight 255,
        # none on the RC; exact (the CPU's float64 sums are off by ~1e-9)
        n = (lengths[:9] if padded else np.full(9, 1000)) - 11 + 1
        expected = np.outer(n, n) * 255**2 * kernel._mismatch_table[0]
        np.testing.assert_allclose(got[:9, :9], expected, rtol=1e-15)
        ref = masked.masked_pairwise(bx, bx, plan, bx_rc, wx=w, wy=w, wy_rc=w_rc)
        np.testing.assert_allclose(got, ref, rtol=1e-8)

    @pytest.mark.parametrize("padded", [False, True])
    def test_index_matches_cpu(self, padded):
        from gkmsvm.kernels.lmer_index import LmerIndex
        from gkmsvm.kernels.weighted import CenterWeightedGkmKernel

        kernel = CenterWeightedGkmKernel(11, 7, d=3, M=255, H=1e9)
        y = self._polya_and_random(20, 600, 3)
        lengths = np.random.default_rng(4).integers(20, 601, 20) if padded else None
        by, by_rc = kernel.pack_both(y, lengths)
        index = LmerIndex(kernel, by, by_rc)
        ref = index.pairwise(by)
        got = np.array(index.pairwise(to_mlx(by)))
        np.testing.assert_allclose(got, ref, rtol=1e-6)

    @pytest.mark.parametrize("kernel_type", ["wgkm", "wgkmrbf"])
    def test_brute_and_self_kernels(self, kernel_type):
        """On MLX they run in Metal and return MLX arrays."""
        from gkmsvm.svm import as_batch

        kernel = _make_model(
            3, 40, kernel_type, {"L": 8, "k": 5, "M": 20, "H": 6.0, "gamma": 2.0}
        ).kernel.base
        rng = np.random.default_rng(5)
        seqs = [
            "".join(rng.choice(list("ACGT"), rng.integers(10, 70))) for _ in range(9)
        ]
        x, lengths = as_batch(seqs)
        bx, bx_rc = kernel.pack_both(x, lengths)
        mbx, mrc = to_mlx(bx), to_mlx(bx_rc)
        for got, ref in (
            (kernel.raw_diag_packed(mbx, mrc), kernel.raw_diag_packed(bx, bx_rc)),
            (kernel.raw_diag_packed(mbx), kernel.raw_diag_packed(bx)),
            (kernel.raw_packed(mbx, mbx, mrc), kernel.raw_packed(bx, bx, bx_rc)),
        ):
            assert isinstance(got, mx.array)
            np.testing.assert_allclose(np.array(got), ref, rtol=1e-5)


class TestMutantDiagMLX:
    """Mutant self-kernels (ISM's normalization) on the device vs
    _mutant_diag_numba: only the pairs touching a changed window."""

    @pytest.mark.parametrize("kernel_type", ["gkm_cnt", "gkm_esttrunc", "wgkm"])
    @pytest.mark.parametrize("padded", [False, True])
    @pytest.mark.parametrize("include_rc", [True, False])
    def test_matches_cpu(self, kernel_type, padded, include_rc):
        from gkmsvm.ism import _mutants
        from gkmsvm.kernels.direct import DirectGkmKernel
        from gkmsvm.kernels.esttrunc import EstTruncGkmKernel
        from gkmsvm.kernels.weighted import CenterWeightedGkmKernel
        from gkmsvm.svm import as_batch

        cls, kw = {
            "gkm_cnt": (DirectGkmKernel, {}),
            "gkm_esttrunc": (EstTruncGkmKernel, {}),
            "wgkm": (CenterWeightedGkmKernel, {"M": 20, "H": 6.0}),
        }[kernel_type]
        kernel = cls(8, 5, d=2, include_rc=include_rc, **kw)
        rng = np.random.default_rng(7)
        n = rng.integers(10, 70, 9) if padded else np.full(9, 61)
        x, lengths = as_batch(["".join(rng.choice(list("ACGT"), k)) for k in n])
        bx, bx_rc = kernel.pack(x, lengths), None
        if include_rc:
            bx, bx_rc = kernel.pack_both(x, lengths)
        ref = _mutants(kernel, x, bx, bx_rc)[3]
        dbx = to_mlx(bx)
        got = _mutants(kernel, x, dbx, None if bx_rc is None else to_mlx(bx_rc))[3]
        assert isinstance(got, mx.array)
        np.testing.assert_allclose(np.array(got), ref, rtol=1e-5)


class TestIsmIndexMLX:
    """ISM through the l-mer index on the device (LmerIndex.mutants) vs the
    CPU's fused pass."""

    @staticmethod
    def _case(kernel_type, padded):
        from gkmsvm.svm import as_batch

        rng = np.random.default_rng(11)
        params = {"L": 7, "k": 5, "d": 2, "gamma": 2.0, "M": 20, "H": 6.0}
        n = rng.integers(20, 70, 40) if padded else np.full(40, 60)
        sv, sv_len = as_batch(["".join(rng.choice(list("ACGT"), k)) for k in n])
        model = GkmSVM(
            sv, rng.normal(size=40), 0.3, kernel_type, params, sv_lengths=sv_len
        )
        model.use_lmer_table = False
        q = rng.integers(15, 61, 5) if padded else np.full(5, 60)
        x, lengths = as_batch(["".join(rng.choice(list("ACGT"), k)) for k in q])
        return model, x, lengths

    @pytest.mark.parametrize(
        "kernel_type", ["gkmrbf", "wgkmrbf", "gkm_esttrunc", "wgkm"]
    )
    @pytest.mark.parametrize("padded", [False, True])
    @pytest.mark.parametrize("chunked", [False, True])
    def test_matches_cpu(self, kernel_type, padded, chunked, monkeypatch):
        from gkmsvm.ism import ism
        from gkmsvm.kernels import lmer_index

        if chunked:  # several launches, a few items each
            monkeypatch.setattr(lmer_index, "_MUTANTS_SCRATCH_BYTES", 5 * 40 * 8 * 7)
        model, x, lengths = self._case(kernel_type, padded)
        ref = ism(model, x, lengths=lengths)
        model.mlx()
        assert model._sv_index() is not None
        calls = []
        orig = lmer_index.LmerIndex._mutants_gpu
        monkeypatch.setattr(
            lmer_index.LmerIndex,
            "_mutants_gpu",
            lambda self, *a: calls.append(1) or orig(self, *a),
        )
        got = ism(model, to_mlx(x), lengths=lengths)
        assert calls and isinstance(got, mx.array)
        np.testing.assert_allclose(
            np.array(got), ref, rtol=0, atol=1e-5 * np.abs(ref).max()
        )

    def test_dense_index_runs_on_cpu(self, monkeypatch):
        """Above _METAL_MUTANTS_MAX_DENSITY the CPU pass runs; MLX result."""
        from gkmsvm.ism import ism
        from gkmsvm.kernels import lmer_index

        monkeypatch.setattr(lmer_index, "_METAL_MUTANTS_MAX_DENSITY", 0.0)
        monkeypatch.setattr(
            lmer_index.LmerIndex,
            "_mutants_gpu",
            lambda *a, **k: pytest.fail("ran on the GPU"),
        )
        model, x, lengths = self._case("gkmrbf", False)
        ref = ism(model, x, lengths=lengths)
        model.mlx()
        got = ism(model, to_mlx(x), lengths=lengths)
        assert isinstance(got, mx.array)
        np.testing.assert_allclose(
            np.array(got), ref, rtol=0, atol=1e-5 * np.abs(ref).max()
        )

    @pytest.mark.parametrize("kernel_type", ["gkmrbf", "wgkmrbf"])
    @pytest.mark.parametrize("dense", [False, True])
    def test_mutation_impact_matches_cpu(self, kernel_type, dense, monkeypatch):
        """RBF mutation impact through the index: queried in Metal, or (dense)
        the CPU's fused pass; SNPs and indels."""
        from gkmsvm.explain import mutation_impact
        from gkmsvm.kernels import lmer_index

        if dense:
            monkeypatch.setattr(lmer_index, "_METAL_QUERY_MAX_DENSITY", 0.0)
        model, _, _ = self._case(kernel_type, True)
        rng = np.random.default_rng(5)
        ref = ["".join(rng.choice(list("ACGT"), k)) for k in rng.integers(20, 61, 8)]
        alt = [s[:9] + ("A" if s[9] != "A" else "C") + s[10:] for s in ref[:5]]
        alt += [s[:9] + s[11:] for s in ref[5:7]] + [ref[7][:9] + "GT" + ref[7][9:]]
        expected = mutation_impact(model, ref, alt)
        model.mlx()
        assert model._sv_index().queries_on_device("mlx") != dense
        got = mutation_impact(model, ref, alt)
        np.testing.assert_allclose(
            got, expected, rtol=0, atol=1e-5 * np.abs(expected).max()
        )


class TestLmerIndexMLX:
    """Cached l-mer index on the device vs the CPU index."""

    @pytest.mark.parametrize("which", ["t0", "t2_l9", "t2_l11"])
    @pytest.mark.parametrize("dense", [False, True])
    def test_index_matches_cpu(self, which, dense, monkeypatch):
        """dense: above _METAL_QUERY_MAX_DENSITY the query runs on CPU."""
        from gkmsvm.codec import reverse_complement
        from gkmsvm.kernels import lmer_index
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
        if dense:
            monkeypatch.setattr(lmer_index, "_METAL_QUERY_MAX_DENSITY", 0.0)
            monkeypatch.setattr(
                lmer_index, "_metal_query", lambda weighted: pytest.fail("Metal")
            )
        got = index.pairwise(to_mlx(bx))
        assert isinstance(got, mx.array)
        np.testing.assert_allclose(np.array(got), ref, rtol=1e-6, atol=1e-6)

    def test_rbf_model_scores_match_cpu(self):
        model = _make_model(40, 60, "gkmrbf", {"L": 9, "k": 6, "d": 3, "gamma": 2.0})
        X = np.stack([one_hot_encode(s) for s in _make_seqs(25, 60, seed=7)])
        cpu = np.asarray(model(X)).ravel()
        assert model._sv_index() is not None
        model.mlx()
        got = np.array(model(to_mlx(X))).ravel()
        np.testing.assert_allclose(got, cpu, atol=1e-4)


def test_gkmexplain_mlx_reuses_cpu_model():
    """The MLX fallback keeps one CPU copy of the model (its caches survive calls)."""
    import warnings

    from gkmsvm.backend import to_mlx
    from gkmsvm.explain import gkmexplain

    rng = np.random.default_rng(0)
    sv = np.eye(4)[rng.integers(0, 4, (30, 25))].transpose(0, 2, 1).astype(np.float32)
    model = GkmSVM(
        sv,
        rng.normal(size=30).astype(np.float32),
        0.1,
        "gkm_esttrunc",
        {"L": 8, "k": 5, "d": 2},
    )
    x = np.eye(4)[rng.integers(0, 4, (2, 25))].transpose(0, 2, 1).astype(np.float32)
    expected = gkmexplain(model, x, mode="hypothetical")
    model.mlx()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = gkmexplain(model, to_mlx(x), mode="hypothetical")
        cpu_model = model._sv_cache["cpu_model"]
        second = gkmexplain(model, to_mlx(x), mode="importance")
        gkmexplain(model, to_mlx(x), mode="hypothetical")
    # Window partials folded on CPU in float64, so NumPy float64
    assert isinstance(first, np.ndarray) and first.dtype == np.float64
    assert model._sv_cache["cpu_model"] is cpu_model
    # One copy of the neighbor masks on the device, however many calls
    metal = [k for k in cpu_model._sv_cache if k[0] == "explain_lmer_metal"]
    assert len(metal) == 1
    tol = 1e-6 * np.abs(expected).max()  # float32 sums within each window
    np.testing.assert_allclose(first, expected, rtol=0, atol=tol)
    np.testing.assert_allclose(second, expected * x, rtol=0, atol=tol)


@pytest.mark.parametrize(
    "kernel_type",
    ["gkm_cnt", "gkm_estfull", "gkm_esttrunc", "gkmrbf", "wgkm", "wgkmrbf"],
)
@pytest.mark.parametrize("mode", ["hypothetical", "perturbation"])
def test_gkmexplain_metal_matches_cpu(kernel_type, mode):
    """The Metal l-mer path (index path for RBF) on mixed-length queries and
    SVs with N: the CPU's attributions to 1e-6 of the largest, no CPU
    fallback warning, and importance sums to score - bias (non-RBF)."""
    import warnings

    from gkmsvm.explain import gkmexplain

    rng = np.random.default_rng(3)

    def seqs(n, lo, hi):
        out = []
        for i in range(n):
            s = list(rng.choice(list("ACGT"), int(rng.integers(lo, hi + 1))))
            if i % 4 == 0:
                s[3:7] = "NNNN"
            out.append("".join(s))
        return out

    params = {"L": 7, "k": 5, "d": 2, "gamma": 2.0, "M": 20, "H": 6.0}
    from gkmsvm.svm import as_batch

    sv, sv_len = as_batch(seqs(60, 15, 70))
    model = GkmSVM(sv, rng.normal(size=60), 0.3, kernel_type, params, sv_lengths=sv_len)
    model.use_explain_table = False
    q = seqs(9, 9, 80)
    expected = gkmexplain(model, q, mode=mode)
    model.mlx()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a CPU fallback warns
        got = gkmexplain(model, q, mode=mode)
    assert isinstance(got, np.ndarray) and got.dtype == np.float64
    np.testing.assert_allclose(
        got, expected, rtol=0, atol=1e-6 * np.abs(expected).max()
    )
    if mode == "hypothetical" and kernel_type not in ("gkmrbf", "wgkmrbf"):
        imp = gkmexplain(model, q, mode="importance").sum(axis=(1, 2))
        model.cpu()
        score = np.asarray(model(q)).ravel() - model.bias
        np.testing.assert_allclose(imp, score, rtol=0, atol=1e-6 * np.abs(score).max())


@pytest.mark.parametrize(
    "kernel_type",
    ["gkm_cnt", "gkm_esttrunc", "gkmrbf", "wgkm", "wgkmrbf", "gkm_cnt_rbf"],
)
@pytest.mark.parametrize("mode", ["hypothetical", "perturbation"])
def test_gkmexplain_metal_per_sv_matches_cpu(kernel_type, mode):
    """The Metal per-SV kernel (use_lmer_index = False) on mixed-length
    queries and SVs with N: the CPU's attributions to 1e-6 of the largest,
    NumPy float64, and importance sums to score - bias (non-RBF)."""
    from gkmsvm.explain import gkmexplain
    from gkmsvm.svm import as_batch

    rng = np.random.default_rng(4)

    def seqs(n, lo, hi):
        out = []
        for i in range(n):
            s = list(rng.choice(list("ACGT"), int(rng.integers(lo, hi + 1))))
            if i % 4 == 0:
                s[3:7] = "NNNN"
            out.append("".join(s))
        return out

    params = {"L": 7, "k": 5, "d": 2, "gamma": 2.0, "M": 20, "H": 6.0}
    sv, sv_len = as_batch(seqs(60, 15, 70))
    model = GkmSVM(sv, rng.normal(size=60), 0.3, kernel_type, params, sv_lengths=sv_len)
    model.use_explain_table = False
    model.use_lmer_index = False
    model.sv_chunk_size = 25  # several SV chunks
    q = seqs(9, 9, 80)
    expected = gkmexplain(model, q, mode=mode)
    model.mlx()
    got = gkmexplain(model, q, mode=mode)
    assert isinstance(got, np.ndarray) and got.dtype == np.float64
    np.testing.assert_allclose(
        got, expected, rtol=0, atol=1e-6 * np.abs(expected).max()
    )
    if mode == "hypothetical" and "rbf" not in kernel_type:
        imp = gkmexplain(model, q, mode="importance").sum(axis=(1, 2))
        model.cpu()
        score = np.asarray(model(q)).ravel() - model.bias
        np.testing.assert_allclose(imp, score, rtol=0, atol=1e-6 * np.abs(score).max())
