"""Tests for gkm-SVM training."""

import numpy as np
import pytest

from gkmsvm import train_gkmsvm


def _random_seqs_str(n, length, rng, gc_bias=0.5):
    bases = "ACGT"
    seqs = []
    for _ in range(n):
        seq = "".join(rng.choice(list(bases)) for _ in range(length))
        seqs.append(seq)
    return seqs


def _random_onehot(n, length, rng):
    seqs = []
    for _ in range(n):
        idx = rng.integers(0, 4, size=length)
        x = np.zeros((4, length), dtype=np.float32)
        x[idx, np.arange(length)] = 1.0
        seqs.append(x)
    return np.stack(seqs)


class TestTrainGkmsvm:
    def test_basic_training_strings(self):
        rng = np.random.default_rng(42)
        pos = _random_seqs_str(20, 20, rng)
        neg = _random_seqs_str(20, 20, rng)

        model = train_gkmsvm(
            pos, neg, kernel_type="direct", l=7, k=5, C=1.0
        )

        assert model.num_support_vectors > 0
        assert model.kernel_type == "gkm_cnt"

    def test_basic_training_arrays(self):
        rng = np.random.default_rng(42)
        pos = _random_onehot(20, 20, rng)
        neg = _random_onehot(20, 20, rng)

        model = train_gkmsvm(
            pos, neg, kernel_type="estimated", l=7, k=5, d=3, C=1.0
        )

        assert model.num_support_vectors > 0
        assert model.kernel_type == "gkm_esttrunc"

    def test_model_scores_separate_classes(self):
        rng = np.random.default_rng(123)
        pos = ["AAAAAAAAAAAAAAACCCCC"] * 15 + _random_seqs_str(5, 20, rng)
        neg = ["TTTTTTTTTTTTTTTGGGGG"] * 15 + _random_seqs_str(5, 20, rng)

        model = train_gkmsvm(
            pos, neg, kernel_type="direct", l=7, k=5, C=10.0
        )

        from gkmsvm.codec import one_hot_encode
        x_pos = one_hot_encode("AAAAAAAAAAAAAAACCCCC")[None, ...]
        x_neg = one_hot_encode("TTTTTTTTTTTTTTTGGGGG")[None, ...]

        score_pos = model(x_pos).item()
        score_neg = model(x_neg).item()

        assert score_pos > score_neg

    def test_model_callable(self):
        rng = np.random.default_rng(42)
        pos = _random_seqs_str(15, 20, rng)
        neg = _random_seqs_str(15, 20, rng)

        model = train_gkmsvm(
            pos, neg, kernel_type="direct", l=7, k=5, C=1.0
        )

        x = _random_onehot(3, 20, rng)
        scores = model(x)
        assert scores.shape == (3, 1)

    def test_kernel_params_stored(self):
        rng = np.random.default_rng(42)
        pos = _random_seqs_str(15, 20, rng)
        neg = _random_seqs_str(15, 20, rng)

        model = train_gkmsvm(
            pos, neg, kernel_type="estimated", l=7, k=5, d=2, C=1.0
        )

        params = model.kernel_params
        assert params["L"] == 7
        assert params["k"] == 5
        assert params["d"] == 2

    def test_train_and_save_load(self, tmp_path):
        rng = np.random.default_rng(42)
        pos = _random_seqs_str(15, 20, rng)
        neg = _random_seqs_str(15, 20, rng)

        model = train_gkmsvm(
            pos, neg, kernel_type="direct", l=7, k=5, C=1.0
        )

        path = tmp_path / "trained.npz"
        model.save(str(path))

        from gkmsvm import load_model
        loaded = load_model(str(path))

        x = _random_onehot(3, 20, rng)
        np.testing.assert_allclose(loaded(x), model(x), atol=1e-6)

    def test_sv_chunk_size_passed(self):
        rng = np.random.default_rng(42)
        pos = _random_seqs_str(15, 20, rng)
        neg = _random_seqs_str(15, 20, rng)

        model = train_gkmsvm(
            pos, neg, kernel_type="direct", l=7, k=5, C=1.0,
            sv_chunk_size=5,
        )

        assert model.sv_chunk_size == 5

    def test_weighted_kernel_requires_M_H(self):
        rng = np.random.default_rng(42)
        pos = _random_seqs_str(10, 20, rng)
        neg = _random_seqs_str(10, 20, rng)

        with pytest.raises(ValueError, match="M and H are required"):
            train_gkmsvm(
                pos, neg, kernel_type="weighted", l=7, k=5, C=1.0
            )

    def test_empty_seqs_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            train_gkmsvm([], ["ACGT" * 5], kernel_type="direct", l=7, k=5)

    def test_device_cpu_explicit(self):
        rng = np.random.default_rng(42)
        pos = _random_seqs_str(15, 20, rng)
        neg = _random_seqs_str(15, 20, rng)

        model = train_gkmsvm(
            pos, neg, kernel_type="direct", l=7, k=5, C=1.0,
            device="cpu",
        )
        assert model.num_support_vectors > 0
        assert isinstance(model.support_sequences, np.ndarray)

    def test_device_invalid_raises(self):
        rng = np.random.default_rng(42)
        pos = _random_seqs_str(10, 20, rng)
        neg = _random_seqs_str(10, 20, rng)

        with pytest.raises(ValueError, match="Unknown device"):
            train_gkmsvm(
                pos, neg, kernel_type="direct", l=7, k=5,
                device="tpu",
            )

    def test_nystrom_solver(self):
        rng = np.random.default_rng(42)
        pos = _random_seqs_str(30, 20, rng)
        neg = _random_seqs_str(30, 20, rng)

        model = train_gkmsvm(
            pos, neg, kernel_type="direct", l=7, k=5, C=1.0,
            solver="nystrom", n_components=20, device="cpu",
        )
        assert model.num_support_vectors > 0

        x = _random_onehot(5, 20, rng)
        scores = model(x)
        assert scores.shape == (5, 1)

    def test_smo_solver(self):
        rng = np.random.default_rng(42)
        pos = _random_seqs_str(30, 20, rng)
        neg = _random_seqs_str(30, 20, rng)

        model = train_gkmsvm(
            pos, neg, kernel_type="direct", l=7, k=5, C=1.0,
            solver="smo", device="cpu",
        )
        assert model.num_support_vectors > 0

        x = _random_onehot(5, 20, rng)
        scores = model(x)
        assert scores.shape == (5, 1)

    def test_smo_matches_libsvm(self):
        rng = np.random.default_rng(123)
        pos = _random_seqs_str(25, 20, rng)
        neg = _random_seqs_str(25, 20, rng)

        model_gram = train_gkmsvm(
            pos, neg, kernel_type="direct", l=7, k=5, C=1.0,
            solver="libsvm", device="cpu",
        )
        model_smo = train_gkmsvm(
            pos, neg, kernel_type="direct", l=7, k=5, C=1.0,
            solver="smo", device="cpu",
        )

        x = _random_onehot(10, 20, rng)
        scores_gram = model_gram(x)
        scores_smo = model_smo(x)
        np.testing.assert_allclose(scores_smo, scores_gram, atol=0.1)

    def test_csmo_solver_loaded(self):
        """Verify C SMO solver compiles and loads."""
        from gkmsvm.solver import _get_csmo
        lib = _get_csmo()
        assert lib is not None, "C SMO solver failed to compile/load"

    def test_auto_solver_smo_fallback_warns(self):
        """Auto solver warns when falling back to SMO."""
        import warnings
        rng = np.random.default_rng(42)
        pos = _random_seqs_str(15, 20, rng)
        neg = _random_seqs_str(15, 20, rng)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            train_gkmsvm(
                pos, neg, kernel_type="direct", l=7, k=5, C=1.0,
                solver="auto", device="cpu", max_gram_gb=0.0,
            )
        smo_warnings = [x for x in w if "SMO" in str(x.message)]
        assert len(smo_warnings) == 1
        assert "nystrom" in str(smo_warnings[0].message).lower()
        assert "downsampling" in str(smo_warnings[0].message).lower()

    def test_device_auto_works(self):
        rng = np.random.default_rng(42)
        pos = _random_seqs_str(15, 20, rng)
        neg = _random_seqs_str(15, 20, rng)

        model = train_gkmsvm(
            pos, neg, kernel_type="direct", l=7, k=5, C=1.0,
            device="auto",
        )
        assert model.num_support_vectors > 0
