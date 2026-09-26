"""Tests for model serialization (save/load round-trips)."""

import numpy as np
import pytest

from gkmsvm.codec import one_hot_encode
from gkmsvm.importers.lsgkm import load_lsgkm_model
from gkmsvm.serialization import load_model, save_lsgkm, save_npz
from gkmsvm.svm import GkmSVM


def _make_model():
    """Create a small GkmSVM model for testing."""
    rng = np.random.default_rng(42)
    S = 5
    L = 20
    svs = []
    for _ in range(S):
        idx = rng.integers(0, 4, size=L)
        x = np.zeros((4, L), dtype=np.float32)
        x[idx, np.arange(L)] = 1.0
        svs.append(x)
    support_sequences = np.stack(svs)
    coefficients = np.array([0.5, -0.3, 0.8, -0.1, 0.2], dtype=np.float32)
    bias = 0.42

    return GkmSVM(
        support_sequences=support_sequences,
        coefficients=coefficients,
        bias=bias,
        kernel_type="gkm_esttrunc",
        kernel_params={"L": 7, "k": 5, "d": 3, "include_rc": True},
    )


def _test_seqs():
    rng = np.random.default_rng(99)
    seqs = []
    for _ in range(3):
        idx = rng.integers(0, 4, size=20)
        x = np.zeros((4, 20), dtype=np.float32)
        x[idx, np.arange(20)] = 1.0
        seqs.append(x)
    return np.stack(seqs)


class TestNpzRoundTrip:
    def test_save_load(self, tmp_path):
        model = _make_model()
        path = tmp_path / "model.npz"
        model.save(str(path))

        loaded = load_model(str(path))

        assert loaded.kernel_type == model.kernel_type
        assert loaded.bias == pytest.approx(model.bias)
        assert loaded._kernel_params == model._kernel_params
        np.testing.assert_array_equal(
            loaded.support_sequences, model.support_sequences
        )
        np.testing.assert_array_equal(loaded.coefficients, model.coefficients)

    def test_scores_match(self, tmp_path):
        model = _make_model()
        x = _test_seqs()

        scores_before = model(x)

        path = tmp_path / "model.npz"
        model.save(str(path))
        loaded = load_model(str(path))
        scores_after = loaded(x)

        np.testing.assert_allclose(scores_after, scores_before, atol=1e-6)

    def test_format_auto_detect(self, tmp_path):
        model = _make_model()
        path = tmp_path / "model.npz"
        model.save(str(path))
        loaded = load_model(str(path))
        assert loaded.kernel_type == model.kernel_type


class TestLsgkmRoundTrip:
    def test_save_load_txt(self, tmp_path):
        model = _make_model()
        path = tmp_path / "model.txt"
        model.save(str(path), format="lsgkm")

        loaded = load_lsgkm_model(str(path))

        assert loaded.kernel_type == model.kernel_type
        assert loaded.bias == pytest.approx(model.bias, abs=1e-6)
        assert loaded.num_support_vectors == model.num_support_vectors

    def test_save_load_gz(self, tmp_path):
        model = _make_model()
        path = tmp_path / "model.txt.gz"
        model.save(str(path), format="lsgkm")

        loaded = load_lsgkm_model(str(path))

        assert loaded.kernel_type == model.kernel_type
        assert loaded.bias == pytest.approx(model.bias, abs=1e-6)

    def test_scores_match(self, tmp_path):
        model = _make_model()
        x = _test_seqs()
        scores_before = model(x)

        path = tmp_path / "model.txt"
        model.save(str(path), format="lsgkm")
        loaded = load_lsgkm_model(str(path))
        scores_after = loaded(x)

        np.testing.assert_allclose(scores_after, scores_before, atol=1e-5)

    def test_load_model_auto_detect_txt(self, tmp_path):
        model = _make_model()
        path = tmp_path / "model.txt"
        model.save(str(path), format="lsgkm")

        loaded = load_model(str(path))
        assert loaded.kernel_type == model.kernel_type

    def test_rho_sign_convention(self, tmp_path):
        model = _make_model()
        path = tmp_path / "model.txt"
        model.save(str(path), format="lsgkm")

        with open(path) as f:
            for line in f:
                if line.startswith("rho"):
                    rho_val = float(line.split()[1])
                    assert rho_val == pytest.approx(-model.bias)
                    break


class TestSaveMethod:
    def test_format_override(self, tmp_path):
        model = _make_model()
        path = tmp_path / "model.bin"
        model.save(str(path), format="npz")
        loaded = load_model(str(path) + ".npz")
        assert loaded.kernel_type == model.kernel_type

    def test_invalid_format(self, tmp_path):
        model = _make_model()
        with pytest.raises(ValueError, match="Unknown format"):
            model.save(str(tmp_path / "x"), format="pickle")
