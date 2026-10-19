import math
from pathlib import Path

import numpy as np
import pytest

from gkmsvm.codec import one_hot_encode
from gkmsvm.kernels.esttrunc import _build_estlmer_table
from gkmsvm.kernels.weighted import (
    CenterWeightedGkmKernel,
    CenterWeightedRbfGkmKernel,
    _window_weights,
)
from gkmsvm.svm import GkmSVM

FIXTURES = Path(__file__).parent.parent / "fixtures"
L, K, D, M, H = 8, 5, 2, 20, 6.0
PARAMS = {"L": L, "k": K, "d": D, "M": M, "H": H, "gamma": 2.0}


def _ref_weights(W, M, H):
    """LS-GKM's gkmkernel_new_object, transcribed."""
    center = W // 2
    return [min(M, math.floor(M * math.exp(-1 * math.log(2) * abs(center - i) / H) + 1))
            for i in range(W)]


def _ref_kernel(x, y, l=L, k=K, d=D, M=M, H=H, rc=True):
    """Raw wgkm from the definition, on strings (N read as A, as LS-GKM does)."""
    h = _build_estlmer_table(l, k, True)
    h[d + 1:] = 0
    code = {"A": 0, "C": 1, "G": 2, "T": 3}
    ex, ey = [code.get(c, 0) for c in x], [code.get(c, 0) for c in y]

    def windows(e):
        return [e[i:i + l] for i in range(len(e) - l + 1)]

    wx, wy = _ref_weights(len(ex) - l + 1, M, H), _ref_weights(len(ey) - l + 1, M, H)
    blocks = [(windows(ey), wy)]
    if rc:  # RC window j comes from forward window W-1-j and carries its weight
        blocks.append((windows([3 - b for b in ey[::-1]]), wy[::-1]))
    total = 0.0
    for i, u in enumerate(windows(ex)):
        for ws, ww in blocks:
            for j, v in enumerate(ws):
                total += wx[i] * ww[j] * h[sum(a != b for a, b in zip(u, v))]
    return total


def _seqs(rng, n, length, with_n=True):
    out = []
    for i in range(n):
        s = list(rng.choice(list("ACGT"), length))
        if with_n and i % 3 == 0:
            a = int(rng.integers(0, length - 8))
            s[a:a + 6] = "N" * 6
        out.append("".join(s))
    return out


def _encode(seqs):
    return np.stack([one_hot_encode(s, allow_n=True) for s in seqs])


class TestWindowWeights:
    def test_values(self):
        # center W // 2; floor(M 2^(-dist/H) + 1), capped at M
        np.testing.assert_array_equal(_window_weights(5, 20, 1.0), [6, 11, 20, 11, 6])
        np.testing.assert_array_equal(_window_weights(4, 20, 1.0), [6, 11, 20, 11])

    @pytest.mark.parametrize("W", [1, 2, 33, 34, 290])
    @pytest.mark.parametrize("M,H", [(50, 50.0), (20, 6.0), (1, 3.0), (255, 10.0)])
    def test_matches_lsgkm(self, W, M, H):
        w = _window_weights(W, M, H)
        assert w.dtype == np.uint8
        np.testing.assert_array_equal(w, _ref_weights(W, M, H))
        assert w.min() >= 1 and w.max() <= M

    def test_rejects_bad_parameters(self):
        with pytest.raises(ValueError, match="M"):
            CenterWeightedGkmKernel(8, 5, M=256)
        with pytest.raises(ValueError, match="H"):
            CenterWeightedGkmKernel(8, 5, H=0)


class TestCenterWeightedGkmKernel:
    @pytest.mark.parametrize("include_rc", [True, False])
    @pytest.mark.parametrize("lengths", [(40, 40), (41, 36)])  # odd/even window counts
    def test_raw_matches_definition(self, include_rc, lengths):
        rng = np.random.default_rng(0)
        xs, ys = _seqs(rng, 5, lengths[0]), _seqs(rng, 20, lengths[1])
        kernel = CenterWeightedGkmKernel(L, K, d=D, M=M, H=H, include_rc=include_rc)
        expected = np.array([[_ref_kernel(a, b, rc=include_rc) for b in ys] for a in xs])
        got_masked = kernel._raw_pairwise(_encode(xs), _encode(ys))
        kernel.use_masked = False
        got_brute = kernel._raw_pairwise(_encode(xs), _encode(ys))
        np.testing.assert_allclose(got_masked, expected, rtol=1e-12)
        np.testing.assert_allclose(got_brute, expected, rtol=1e-12)
        diag = kernel._raw_diagonal(_encode(xs))
        np.testing.assert_allclose(diag, [_ref_kernel(a, a, rc=include_rc) for a in xs],
                                   rtol=1e-12)

    def test_n_is_a(self):
        """N scores as A, as in LS-GKM."""
        rng = np.random.default_rng(1)
        xs, ys = _seqs(rng, 4, 41), _seqs(rng, 6, 41)
        kernel = CenterWeightedGkmKernel(L, K, d=D, M=M, H=H)
        as_a = [s.replace("N", "A") for s in xs]
        np.testing.assert_allclose(kernel.pairwise(_encode(xs), _encode(ys)),
                                   kernel.pairwise(_encode(as_a), _encode(ys)), rtol=1e-12)

    def test_gram_matches_definition(self):
        from gkmsvm.gram import compute_gram

        rng = np.random.default_rng(2)
        seqs = _seqs(rng, 40, 41)
        kernel = CenterWeightedGkmKernel(L, K, d=D, M=M, H=H)
        raw = np.array([[_ref_kernel(a, b) for b in seqs] for a in seqs])
        expected = raw / np.sqrt(np.outer(np.diag(raw), np.diag(raw)))
        for chunk in (1000, 16):  # one tile; column panels with diagonal tiles
            np.testing.assert_allclose(compute_gram(kernel, _encode(seqs), chunk_size=chunk),
                                       expected, rtol=1e-12)


@pytest.mark.parametrize("kernel_type", ["wgkm", "wgkmrbf"])
def test_scoring_paths_match_definition(kernel_type):
    """l-mer index, l-mer table, masked join and brute force all give the
    definition's scores."""
    rng = np.random.default_rng(3)
    svs, qs = _seqs(rng, 30, 41), _seqs(rng, 20, 41)
    coef = rng.normal(size=30)
    model = GkmSVM(_encode(svs), coef, 0.3, kernel_type, PARAMS)
    raw = np.array([[_ref_kernel(q, s) for s in svs] for q in qs])
    dq = np.array([_ref_kernel(q, q) for q in qs])
    ds = np.array([_ref_kernel(s, s) for s in svs])
    k = raw / np.sqrt(np.outer(dq, ds))
    if kernel_type == "wgkmrbf":
        k = np.exp(2.0 * (k - 1))
    expected = k @ coef + 0.3
    for index, table, masked in [(True, False, True), (False, True, True),
                                 (False, False, True), (False, False, False)]:
        model.use_lmer_index, model.use_lmer_table = index, table
        model.kernel.base.use_masked = masked
        model.clear_caches()
        np.testing.assert_allclose(np.asarray(model(_encode(qs))).ravel(), expected,
                                   rtol=1e-12, err_msg=f"{index=} {table=} {masked=}")


@pytest.mark.parametrize("kernel_type", ["wgkm", "wgkmrbf"])
@pytest.mark.parametrize("L", [33, 34])  # odd/even window counts
def test_ism_scores_mutants(kernel_type, L):
    """Window-delta ISM through the table, index, masked join and brute force
    equals scoring every mutant."""
    from gkmsvm.ism import ism

    rng = np.random.default_rng(4)
    model = GkmSVM(_encode(_seqs(rng, 30, L)), rng.normal(size=30), 0.1, kernel_type, PARAMS)
    x = _encode(_seqs(rng, 3, L, with_n=False))
    mut = np.broadcast_to(x[:, None, None], (3, 4, L, 4, L)).copy()
    for p in range(L):
        mut[:, :, p, :, p] = np.eye(4)
    expected = model(mut.reshape(-1, 4, L)).reshape(3, 4, L) - model(x)[:, :, None]
    paths = [(True, False, True), (False, False, True), (False, False, False)]
    if kernel_type == "wgkm":
        paths.append((False, True, True))
    for index, table, masked in paths:
        model.use_lmer_index, model.use_lmer_table = index, table
        model.kernel.base.use_masked = masked
        model.clear_caches()
        np.testing.assert_allclose(ism(model, x), expected, rtol=1e-9, atol=1e-12,
                                   err_msg=f"{index=} {table=} {masked=}")


def test_mutant_self_kernels():
    """raw_diag_mutants equals the self-kernel of each mutant."""
    from gkmsvm.ism import _packed_mutations

    rng = np.random.default_rng(7)
    L = 30
    kernel = CenterWeightedGkmKernel(8, 5, d=D, M=M, H=H)
    x = _encode(_seqs(rng, 2, L, with_n=False))
    bx, bx_rc = kernel.pack_both(x)
    W = L - kernel.l + 1
    _, new = _packed_mutations(bx, L, W, kernel.l)
    new_rc = _packed_mutations(bx_rc, L, W, kernel.l)[1][::-1, ::-1]
    got = kernel.raw_diag_mutants(bx, bx_rc, new, new_rc, kernel.raw_diag_packed(bx, bx_rc))
    for v in range(4):
        for p in range(L):
            m = x.copy()
            m[:, :, p] = 0
            m[:, v, p] = 1
            np.testing.assert_allclose(got[v, p], kernel._raw_diagonal(m), rtol=1e-12)


def _ref_explain(x, svs, coef, l=L, d=D):
    """Hypothetical GkmExplain from the definition: each window pair (i, j)
    with m mismatches adds coef / sqrt(K(x, x) K(sv, sv)) wx_i wy_j times
    alpha(m) (match) or kappa(m) (mismatch) at each position, to the SV's base."""
    from gkmsvm.explain import _build_alpha_table, _build_kappa_table

    h = _build_estlmer_table(l, K, True)
    h[d + 1:] = 0
    alpha, kappa = _build_alpha_table(h, l, d), _build_kappa_table(h, l, d)
    code = {"A": 0, "C": 1, "G": 2, "T": 3}
    ex = [code.get(c, 0) for c in x]
    wx = _ref_weights(len(ex) - l + 1, M, H)
    out = np.zeros((4, len(ex)))
    for sv, c in zip(svs, coef):
        ey = [code.get(b, 0) for b in sv]
        wy = _ref_weights(len(ey) - l + 1, M, H)
        ec = c / math.sqrt(_ref_kernel(x, x) * _ref_kernel(sv, sv))
        rc = [3 - b for b in ey[::-1]]
        for seq, ww in ((ey, wy), (rc, wy[::-1])):
            for i in range(len(ex) - l + 1):
                u = ex[i:i + l]
                for j in range(len(seq) - l + 1):
                    v = seq[j:j + l]
                    m = sum(a != b for a, b in zip(u, v))
                    if m > d:
                        continue
                    for k in range(l):
                        t = alpha[m] if u[k] == v[k] else kappa[m]
                        out[v[k], i + k] += ec * wx[i] * ww[j] * t
    return out


# (use_explain_table, use_lmer_index): table, l-mer path, per-SV kernel
_EXPLAIN_PATHS = [(True, True), (False, True), (False, False)]


@pytest.mark.parametrize("kernel_type", ["wgkm", "wgkmrbf"])
@pytest.mark.parametrize("mode", ["hypothetical", "perturbation", "importance"])
def test_gkmexplain_paths_agree(kernel_type, mode):
    """Explain table, l-mer path (index for RBF) and per-SV kernel agree, with
    odd and even window counts."""
    from gkmsvm import gkmexplain

    rng = np.random.default_rng(5)
    for L_ in (33, 34):
        model = GkmSVM(_encode(_seqs(rng, 12, L_)), rng.normal(size=12), 0.1, kernel_type,
                       PARAMS)
        x = _encode(_seqs(rng, 3, L_))
        outs = []
        for table, index in _EXPLAIN_PATHS:
            model.use_explain_table, model.use_lmer_index = table, index
            model.clear_caches()
            outs.append(gkmexplain(model, x, mode=mode))
            if table and kernel_type == "wgkm":
                assert model._explain_tables
        for got in outs[1:]:
            np.testing.assert_allclose(got, outs[0], rtol=1e-10, atol=1e-13)


def test_gkmexplain_matches_definition():
    from gkmsvm import gkmexplain

    rng = np.random.default_rng(8)
    svs, coef = _seqs(rng, 4, 26), rng.normal(size=4)
    model = GkmSVM(_encode(svs), coef, 0.1, "wgkm", PARAMS)
    model.use_lmer_index = False  # per-SV kernel; the others agree with it (above)
    xs = _seqs(rng, 2, 27)
    got = gkmexplain(model, _encode(xs), mode="hypothetical")
    for b, x in enumerate(xs):
        np.testing.assert_allclose(got[b], _ref_explain(x, svs, coef), rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("kernel_type", ["wgkm", "wgkmrbf"])
def test_gkmexplain_completeness(kernel_type):
    """Importance sums to score - bias (RBF: minus exp(-gamma) sum coef, as for -t 3)."""
    from gkmsvm import gkmexplain

    rng = np.random.default_rng(9)
    coef = rng.normal(size=20)
    model = GkmSVM(_encode(_seqs(rng, 20, 41)), coef, 0.3, kernel_type, PARAMS)
    x = _encode(_seqs(rng, 4, 41, with_n=False))
    expected = np.asarray(model(x)).ravel() - 0.3
    if kernel_type == "wgkmrbf":
        expected -= np.exp(-PARAMS["gamma"]) * coef.sum()
    for table, index in _EXPLAIN_PATHS:
        model.use_explain_table, model.use_lmer_index = table, index
        model.clear_caches()
        got = gkmexplain(model, x, mode="importance").sum(axis=(1, 2))
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("query_norm", [True, False])
def test_deltasvm_weights_score_lmers(query_norm):
    """Table weights equal scoring each l-mer through the kernel (one window, weight M)."""
    rng = np.random.default_rng(6)
    model = GkmSVM(_encode(_seqs(rng, 20, 33)), rng.normal(size=20), 0.2, "wgkm",
                   {**PARAMS, "L": 6, "k": 4})
    from_table = model.to_deltasvm(query_norm=query_norm).weights
    model.use_lmer_table = False
    model._lmer_table = False  # no table: score every l-mer
    from_kernel = model.to_deltasvm(query_norm=query_norm).weights
    np.testing.assert_allclose(np.asarray(from_table), np.asarray(from_kernel), rtol=1e-10)


class TestCenterWeightedRbfGkmKernel:
    def test_self_similarity_is_one(self):
        kernel = CenterWeightedRbfGkmKernel(L, K, M=M, H=H, gamma=1.0)
        x = one_hot_encode("ACGTACGTACGTAGGT")[np.newaxis]
        assert float(kernel.pairwise(x, x)[0, 0]) == pytest.approx(1.0)

    def test_gamma_effect(self):
        x = one_hot_encode("ACGTACGTACGTAGGT")[np.newaxis]
        y = one_hot_encode("ACGTACGTACGTAGGA")[np.newaxis]
        low = CenterWeightedRbfGkmKernel(L, K, M=M, H=H, gamma=0.1).pairwise(x, y)
        high = CenterWeightedRbfGkmKernel(L, K, M=M, H=H, gamma=10.0).pairwise(x, y)
        assert float(low[0, 0]) > float(high[0, 0])


def _read_fasta(path):
    names, seqs = [], []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                names.append(line[1:].strip())
                seqs.append("")
            else:
                seqs[-1] += line.strip()
    return names, seqs


@pytest.mark.parametrize("name", ["t4", "t5"])
@pytest.mark.parametrize("length", [100, 85])
def test_matches_gkmpredict(name, length):
    """Scores equal LS-GKM gkmpredict (benchmarks/dev/lsgkm_wgkm_fixtures.py)."""
    from gkmsvm.importers.lsgkm import load_lsgkm_model

    model_path = FIXTURES / f"lsgkm_wgkm_{name}.model.txt.gz"
    pred = FIXTURES / f"lsgkm_wgkm_{name}_test{length}.pred.txt"
    if not (model_path.exists() and pred.exists()):
        pytest.skip("gkmpredict -t 4/5 fixtures not generated")
    model = load_lsgkm_model(model_path)
    names, seqs = _read_fasta(FIXTURES / f"lsgkm_wgkm_test{length}.fa")
    with open(pred) as f:
        expected = dict(line.split()[:2] for line in f)
    expected = np.array([float(expected[n]) for n in names])
    for index in (True, False):
        model.use_lmer_index = index
        model.clear_caches()
        got = np.asarray(model(_encode(seqs))).ravel()
        # gkmpredict prints 6 significant digits
        np.testing.assert_allclose(got, expected, rtol=2e-5, atol=1e-5)
