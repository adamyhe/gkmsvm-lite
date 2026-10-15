"""Cached l-mer index, and the packed kernel interface for every kernel type."""

import numpy as np
import pytest

from gkmsvm.codec import reverse_complement
from gkmsvm.gram import compute_gram
from gkmsvm.kernels import lmer_index
from gkmsvm.kernels.direct import DirectGkmKernel
from gkmsvm.kernels.esttrunc import EstTruncGkmKernel
from gkmsvm.kernels.packed import PAD_WINDOW
from gkmsvm.kernels.rbf import RbfGkmKernel
from gkmsvm.kernels.weighted import CenterWeightedGkmKernel, CenterWeightedRbfGkmKernel
from gkmsvm.solver import KernelColumnCache
from gkmsvm.svm import GkmSVM


def _onehot(n, length, rng):
    return np.eye(4)[rng.integers(0, 4, (n, length))].transpose(0, 2, 1).copy()


KERNELS = {
    "t0_l8k5": lambda: DirectGkmKernel(8, 5),
    "t1_l9k6": lambda: EstTruncGkmKernel(9, 6, d=3, truncate=False),
    "t2_l9k6d3": lambda: EstTruncGkmKernel(9, 6, d=3),
    "t2_l10k6d2": lambda: EstTruncGkmKernel(10, 6, d=2),
}


@pytest.mark.parametrize("name", KERNELS)
@pytest.mark.parametrize("rc", [True, False])
def test_index_matches_brute(name, rc):
    kernel = KERNELS[name]()
    kernel.use_masked = False
    rng = np.random.default_rng(1)
    X, Y = _onehot(7, 50, rng), _onehot(23, 40, rng)
    X[2, :, 5:35] = Y[4, :, 3:33]  # near-identical stretch
    by = kernel.pack(Y)
    by_rc = kernel.pack(reverse_complement(Y)) if rc else None
    assert lmer_index.applies(kernel)
    index = lmer_index.LmerIndex(kernel, by, by_rc)
    bx = kernel.pack(X)
    np.testing.assert_allclose(index.pairwise(bx), kernel.raw_packed(bx, by, by_rc),
                               rtol=1e-12, atol=1e-12)


def test_index_skips_padding():
    kernel = EstTruncGkmKernel(9, 6, d=3)
    rng = np.random.default_rng(2)
    Y = _onehot(10, 30, rng)
    index = lmer_index.LmerIndex(kernel, *kernel.pack_both(Y))
    bx = kernel.pack(_onehot(3, 20, rng))
    padded = np.concatenate([bx, np.full((3, 5), PAD_WINDOW, dtype=np.uint32)], axis=1)
    np.testing.assert_array_equal(index.pairwise(padded), index.pairwise(bx))


def test_applies():
    assert lmer_index.applies(EstTruncGkmKernel(11, 7, d=3))
    assert not lmer_index.applies(DirectGkmKernel(11, 7))  # 31,714 neighbors per window
    assert not lmer_index.applies(EstTruncGkmKernel(13, 7, d=2))  # l > MAX_L
    assert not lmer_index.applies(CenterWeightedGkmKernel(8, 5, M=3, H=2.0))


MODELS = [
    ("gkm_esttrunc", {"L": 8, "k": 5, "d": 3}),
    ("gkmrbf", {"L": 8, "k": 5, "d": 3, "gamma": 2.0}),
    ("wgkm", {"L": 8, "k": 5, "M": 3, "H": 2.0}),
    ("wgkmrbf", {"L": 8, "k": 5, "M": 3, "H": 2.0, "gamma": 2.0}),
]


def _model(kernel_type, params, rng):
    sv = _onehot(30, 40, rng)
    return GkmSVM(sv, rng.normal(size=30), 0.1, kernel_type, params)


@pytest.mark.parametrize("kernel_type,params", MODELS)
def test_model_scores_match_pairwise(kernel_type, params):
    rng = np.random.default_rng(3)
    model = _model(kernel_type, params, rng)
    x = _onehot(20, 40, rng)
    expected = model.kernel.pairwise(x, model.support_sequences) @ model.coefficients + model.bias
    np.testing.assert_allclose(model(x).ravel(), expected, rtol=1e-10, atol=1e-12)
    model.use_lmer_index = False
    model._sv_cache = {}
    np.testing.assert_allclose(model(x).ravel(), expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("kernel_type,params", MODELS)
def test_mutation_impact_index_vs_join(kernel_type, params):
    rng = np.random.default_rng(4)
    model = _model(kernel_type, params, rng)
    ref = _onehot(18, 40, rng)
    alt = ref.copy()
    alt[:, :, 20] = np.eye(4)[(ref[:, :, 20].argmax(1) + 1) % 4]
    got = model.score_variants(ref, alt, method="gkmexplain")
    model.use_lmer_index = False
    model._sv_cache = {}
    np.testing.assert_allclose(got, model.score_variants(ref, alt, method="gkmexplain"),
                               rtol=1e-10, atol=1e-12)


KERNEL_OBJS = {
    "t2": lambda: EstTruncGkmKernel(8, 5, d=3),
    "t3": lambda: RbfGkmKernel(8, 5, d=3, gamma=2.0),
    "t4": lambda: CenterWeightedGkmKernel(8, 5, M=3, H=2.0),
    "t5": lambda: CenterWeightedRbfGkmKernel(8, 5, M=3, H=2.0, gamma=2.0),
}


@pytest.mark.parametrize("name", KERNEL_OBJS)
def test_gram_and_columns_match_pairwise(name):
    """Gram panels and SMO columns go through the base kernel's packed path."""
    kernel = KERNEL_OBJS[name]()
    rng = np.random.default_rng(5)
    X = _onehot(40, 30, rng)
    expected = kernel.pairwise(X, X)
    np.testing.assert_allclose(compute_gram(kernel, X, chunk_size=16), expected,
                               rtol=1e-10, atol=1e-12)
    cache = KernelColumnCache(kernel, X)
    assert cache._use_fast
    for idx in ([3], list(range(5, 30))):
        cols = cache.compute_columns(idx)
        for i in idx:
            np.testing.assert_allclose(cols[i], expected[i], rtol=1e-10, atol=1e-12)


def test_weighted_with_n_uses_onehot_path():
    """N positions: the packed path would treat N as A, so the one-hot path runs."""
    kernel = CenterWeightedGkmKernel(8, 5, M=3, H=2.0)
    rng = np.random.default_rng(6)
    X = _onehot(20, 30, rng)
    X[3, :, 10] = 0
    assert not kernel.packs(X)
    ref = CenterWeightedGkmKernel(8, 5, M=3, H=2.0)
    ref.use_masked = False
    np.testing.assert_allclose(compute_gram(kernel, X), ref.pairwise(X, X), rtol=1e-10)
