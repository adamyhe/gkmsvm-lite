"""Sequences of different lengths (docs/design/variable-length.md): padded
batches equal scoring each sequence at its own length."""

import numpy as np
import pytest

from gkmsvm.codec import one_hot_encode
from gkmsvm.kernels.packed import PAD_WINDOW
from gkmsvm.svm import GkmSVM

PARAMS = {"L": 6, "k": 4, "d": 2, "M": 20, "H": 6.0, "gamma": 2.0}
KERNELS = ["gkm_cnt", "estimated_full", "estimated", "gkmrbf", "wgkm", "wgkmrbf"]


def _seqs(rng, n, lo, hi, with_n=True):
    out = []
    for i in range(n):
        s = list(rng.choice(list("ACGT"), int(rng.integers(lo, hi + 1))))
        if with_n and i % 4 == 0 and len(s) > 10:
            s[3:7] = "NNNN"
        out.append("".join(s))
    return out


def _onehot(s):
    return one_hot_encode(s, allow_n=True, dtype=np.float64)[None]


def _model(svs, coef, kernel_type, bias=0.2):
    model = GkmSVM(_onehot(svs[0]), coef[:1], bias, kernel_type, PARAMS)  # kernel only
    from gkmsvm.svm import as_batch

    x, lengths = as_batch(
        [one_hot_encode(s, allow_n=True, dtype=np.float64) for s in svs]
    )
    return GkmSVM(x, coef, bias, kernel_type, PARAMS, sv_lengths=lengths), model.kernel


def _expected(kernel, queries, svs, coef, bias):
    """Score of each query from per-pair kernel values at the sequences' own lengths."""
    k = np.array(
        [
            [float(kernel.pairwise(_onehot(q), _onehot(s))[0, 0]) for s in svs]
            for q in queries
        ]
    )
    assert np.isfinite(k).all()
    # einsum, not @: Accelerate's matmul raises spurious FP warnings on macOS
    return np.einsum("qs,s->q", k, coef) + bias


def test_pack_pads_past_each_length():
    from gkmsvm.kernels.esttrunc import EstTruncGkmKernel

    kernel = EstTruncGkmKernel(6, 4, d=2)
    seqs = ["ACGTACGTAC", "ACGTACG", "ACGTA"]  # 5, 2 and 0 windows
    from gkmsvm.svm import as_batch

    x, lengths = as_batch(seqs)
    fwd, rc = kernel.pack_both(x, lengths)
    pad = fwd == PAD_WINDOW
    np.testing.assert_array_equal(pad.sum(axis=1), [0, 3, 5])
    np.testing.assert_array_equal(rc == PAD_WINDOW, pad[:, ::-1])  # RC: pads first
    for i, s in enumerate(seqs[:2]):  # the real windows equal the unpadded ones
        f1, r1 = kernel.pack_both(_onehot(s))
        np.testing.assert_array_equal(fwd[i, : f1.shape[1]], f1[0])
        np.testing.assert_array_equal(rc[i, -r1.shape[1] :], r1[0])


@pytest.mark.parametrize("kernel_type", KERNELS)
def test_kernel_paths_match_per_sequence(kernel_type):
    """Masked join, brute force and the self-kernels on padded rows and columns."""
    from gkmsvm.svm import as_batch

    rng = np.random.default_rng(0)
    xs, ys = _seqs(rng, 20, 12, 30), _seqs(rng, 25, 9, 33)
    base = GkmSVM(_onehot(xs[0]), np.ones(1), 0.0, kernel_type, PARAMS).kernel.base
    raw = np.array(
        [
            [float(base._raw_pairwise(_onehot(a), _onehot(b))[0, 0]) for b in ys]
            for a in xs
        ]
    )
    diag = np.array([float(base._raw_diagonal(_onehot(a))[0]) for a in xs])
    x, lx = as_batch(xs)
    y, ly = as_batch(ys)
    bx, _ = base.pack_both(x, lx)
    by, by_rc = base.pack_both(y, ly)
    for masked in (True, False):  # 20 x 25 rows go through the masked join
        base.use_masked = masked
        np.testing.assert_allclose(
            base.raw_packed(bx, by, by_rc),
            raw,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"{masked=}",
        )
    base.use_masked = True
    np.testing.assert_allclose(
        base.raw_diag_packed(*base.pack_both(x, lx)), diag, rtol=1e-12
    )


@pytest.mark.parametrize("kernel_type", KERNELS)
def test_model_scores_match_per_sequence(kernel_type):
    """SVs and queries of different lengths, through the l-mer index, the l-mer
    table, the masked join and brute force."""
    rng = np.random.default_rng(1)
    svs, queries = _seqs(rng, 30, 10, 40), _seqs(rng, 20, 8, 45)
    coef = rng.normal(size=30)
    model, kernel = _model(svs, coef, kernel_type)
    assert model.sv_lengths is not None
    expected = _expected(kernel, queries, svs, coef, 0.2)
    paths = [(True, False, True), (False, False, True), (False, False, False)]
    if kernel_type in ("gkm_cnt", "estimated_full", "estimated", "wgkm"):
        paths.append((False, True, True))
    for index, table, masked in paths:
        model.use_lmer_index, model.use_lmer_table = index, table
        model.kernel.base.use_masked = masked
        model.clear_caches()
        got = np.asarray(model(queries)).ravel()
        np.testing.assert_allclose(
            got,
            expected,
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"{index=} {table=} {masked=}",
        )


def test_padded_array_with_lengths_equals_list():
    from gkmsvm.svm import as_batch

    rng = np.random.default_rng(2)
    model, _ = _model(_seqs(rng, 20, 15, 25), rng.normal(size=20), "estimated")
    queries = _seqs(rng, 7, 10, 30)
    x, lengths = as_batch(queries)
    np.testing.assert_array_equal(model(queries), model(x, lengths=lengths))


def test_uniform_lengths_are_not_padded():
    from gkmsvm.svm import as_batch

    x, lengths = as_batch(["ACGTACGTAA", "ACGTACGTCC"])
    assert lengths is None and x.shape == (2, 4, 10)


def test_short_sequences_score_nan():
    rng = np.random.default_rng(3)
    model, _ = _model(_seqs(rng, 20, 15, 25), rng.normal(size=20), "estimated")
    with pytest.warns(UserWarning, match="shorter than l"):
        got = np.asarray(model(["ACG", "ACGTACGTACGTAAC", "ACGTA"])).ravel()
    assert np.isnan(got[[0, 2]]).all() and np.isfinite(got[1])
    np.testing.assert_allclose(got[1], np.asarray(model(["ACGTACGTACGTAAC"])).ravel())


def test_long_sequences_warn():
    rng = np.random.default_rng(4)
    model, _ = _model(_seqs(rng, 20, 15, 25), rng.normal(size=20), "estimated")
    long = "".join(rng.choice(list("ACGT"), 2100))
    with pytest.warns(UserWarning, match="LS-GKM reads only the first 2047"):
        model([long, "ACGTACGTAC"])


def test_padded_input_off_cpu_raises():
    pytest.importorskip("mlx.core")
    rng = np.random.default_rng(5)
    model, _ = _model(_seqs(rng, 20, 15, 25), rng.normal(size=20), "estimated")
    model.mlx()
    with pytest.raises(NotImplementedError, match="CPU only"):
        model(["ACGTACGTACGTAAC", "ACGTACGTACGTAACGT"])


@pytest.mark.parametrize("kernel_type", ["estimated", "gkmrbf", "wgkm"])
def test_fixed_queries_against_mixed_svs(kernel_type):
    """ISM, mutation impact and to_deltasvm on a model whose SVs differ in
    length (queries one length) equal the same computations from per-pair
    kernel values."""
    from gkmsvm.explain import mutation_impact
    from gkmsvm.ism import ism

    rng = np.random.default_rng(6)
    svs = _seqs(rng, 25, 12, 36)
    coef = rng.normal(size=25)
    model, kernel = _model(svs, coef, kernel_type)
    q = _seqs(rng, 1, 20, 20, with_n=False)[0]
    # ISM: every substitution scored from per-pair kernel values
    mutants = [q[:p] + b + q[p + 1 :] for p in range(len(q)) for b in "ACGT"]
    ref = _expected(kernel, [q], svs, coef, 0.2)[0]
    exp_ism = (_expected(kernel, mutants, svs, coef, 0.2) - ref).reshape(len(q), 4).T
    exp_ism[np.array([[b == c for c in q] for b in "ACGT"])] = 0
    np.testing.assert_allclose(
        np.asarray(ism(model, _onehot(q)))[0], exp_ism, rtol=1e-9, atol=1e-12
    )
    # mutation impact equals itself on the same SVs one length at a time is not
    # defined; check it against its definition through per-pair raw kernels
    alt = q[:10] + ("A" if q[10] != "A" else "C") + q[11:]
    base = kernel.base
    k_ref = np.array(
        [float(base._raw_pairwise(_onehot(q), _onehot(s))[0, 0]) for s in svs]
    )
    k_alt = np.array(
        [float(base._raw_pairwise(_onehot(alt), _onehot(s))[0, 0]) for s in svs]
    )
    d_ref = float(base._raw_diagonal(_onehot(q))[0])
    d_sv = np.array([float(base._raw_diagonal(_onehot(s))[0]) for s in svs])
    z = np.sqrt(d_ref * d_sv)
    dk, kn = (k_alt - k_ref) / z, k_ref / z
    if kernel_type == "gkmrbf":
        g = PARAMS["gamma"]
        dk = np.where(
            kn > 0,
            dk * (np.exp(g * (kn - 1)) - np.exp(-g)) / np.where(kn > 0, kn, 1),
            0,
        )
    np.testing.assert_allclose(
        float(mutation_impact(model, _onehot(q), _onehot(alt))[0, 0]),
        float(coef @ dk),
        rtol=1e-9,
        atol=1e-12,
    )
    if kernel_type != "gkmrbf":  # to_deltasvm: each weight is model() on the l-mer
        w = np.asarray(model.to_deltasvm(query_norm=True).weights)
        lmers = ["ACGTAC", "TTTTTT", "GATTAC"]
        idx = [
            sum("ACGT".index(c) * 4 ** (5 - i) for i, c in enumerate(u)) for u in lmers
        ]
        np.testing.assert_allclose(
            w[idx],
            _expected(kernel, lmers, svs, coef, 0.2) - 0.2,
            rtol=1e-9,
            atol=1e-12,
        )


def test_gkmexplain_on_mixed_svs_raises():
    from gkmsvm import gkmexplain

    rng = np.random.default_rng(7)
    model, _ = _model(_seqs(rng, 20, 15, 25), rng.normal(size=20), "estimated")
    with pytest.raises(NotImplementedError, match="stage 4"):
        gkmexplain(model, _onehot("ACGTACGTACGTAACGTA"))


def test_npz_and_lsgkm_round_trip(tmp_path):
    from gkmsvm.importers.lsgkm import load_lsgkm_model
    from gkmsvm.serialization import load_npz

    rng = np.random.default_rng(8)
    model, _ = _model(_seqs(rng, 20, 10, 30), rng.normal(size=20), "wgkm")
    queries = _seqs(rng, 5, 12, 28)
    expected = np.asarray(model(queries))
    model.save(str(tmp_path / "m.npz"))
    model.save(str(tmp_path / "m.model.txt"), fmt="lsgkm")
    for loaded in (
        load_npz(tmp_path / "m.npz"),
        load_lsgkm_model(tmp_path / "m.model.txt"),
    ):
        np.testing.assert_array_equal(loaded.sv_lengths, model.sv_lengths)
        np.testing.assert_allclose(np.asarray(loaded(queries)), expected, rtol=1e-6)


FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("name", ["t2", "t4", "t5"])
def test_matches_gkmpredict(name):
    """An LS-GKM model trained on sequences of different lengths, scoring a
    30-200 bp test set (one sequence shorter than l: NaN, as gkmpredict),
    through every path (benchmarks/dev/lsgkm_wgkm_fixtures.py --ragged)."""
    from gkmsvm.fasta import read_fasta
    from gkmsvm.importers.lsgkm import load_lsgkm_model

    model = load_lsgkm_model(FIXTURES / f"lsgkm_ragged_{name}.model.txt.gz")
    assert model.sv_lengths is not None
    records = read_fasta(str(FIXTURES / "lsgkm_ragged_test.fa"))
    with open(FIXTURES / f"lsgkm_ragged_{name}_test.pred.txt") as f:
        expected = dict(line.split()[:2] for line in f)
    expected = np.array([float(expected[n]) for n, _ in records])
    assert np.isnan(expected).sum() == 1
    paths = [(True, False, True), (False, False, True), (False, False, False)]
    if name == "t2" or name == "t4":
        paths.append((False, True, True))
    for index, table, masked in paths:
        model.use_lmer_index, model.use_lmer_table = index, table
        model.kernel.base.use_masked = masked
        model.clear_caches()
        with pytest.warns(UserWarning, match="shorter than l"):
            got = np.asarray(model([s for _, s in records])).ravel()
        # gkmpredict prints 6 significant digits
        np.testing.assert_allclose(
            got, expected, rtol=2e-5, atol=1e-5, err_msg=f"{index=} {table=} {masked=}"
        )
