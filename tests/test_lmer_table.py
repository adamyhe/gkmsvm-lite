"""The exact l-mer table path must give the same scores as the kernel path."""

import numpy as np
import pytest

from gkmsvm.svm import GkmSVM


def _onehot(n, length, rng):
    idx = rng.integers(0, 4, size=(n, length))
    x = np.zeros((n, 4, length))
    np.put_along_axis(x, idx[:, None, :], 1.0, axis=1)
    return x


CASES = {
    "t0": ("gkm_cnt", {"L": 7, "k": 4}),
    "t1": ("gkm_estfull", {"L": 8, "k": 5, "d": 3}),
    "t2": ("gkm_esttrunc", {"L": 8, "k": 5, "d": 3}),
    "t2_d2": ("gkm_esttrunc", {"L": 9, "k": 6, "d": 2}),
}


def _model(case, include_rc, rng, n_sv=30):
    kernel_type, params = CASES[case]
    sv = _onehot(n_sv, 40, rng)
    coef = rng.normal(size=n_sv)
    return GkmSVM(sv, coef, 0.3, kernel_type=kernel_type,
                  kernel_params={**params, "include_rc": include_rc})


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("include_rc", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
def test_table_matches_kernel(case, include_rc, normalize):
    rng = np.random.default_rng(0)
    model = _model(case, include_rc, rng)
    model.kernel.normalize = normalize
    X = _onehot(25, 50, rng)
    X[3, :, 5:35] = model.support_sequences[2, :, 0:30]  # near-identical to an SV

    model.use_lmer_table = False
    ref = np.asarray(model(X))
    model.use_lmer_table = True
    got = np.asarray(model(X))
    assert model._lmer_table is not False  # the table path ran
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


def test_auto_skips_small_models():
    rng = np.random.default_rng(1)
    model = _model("t2", True, rng, n_sv=20)
    model(_onehot(3, 50, rng))
    assert model._lmer_table is None  # "auto" with < 500 SVs: never built


def test_rbf_uses_kernel_path():
    rng = np.random.default_rng(2)
    sv = _onehot(30, 40, rng)
    model = GkmSVM(sv, rng.normal(size=30), 0.1, kernel_type="gkmrbf",
                   kernel_params={"L": 8, "k": 5, "d": 3, "gamma": 2.0})
    model.use_lmer_table = True
    model(_onehot(4, 50, rng))
    assert model._lmer_table is None  # RBF never builds a table
