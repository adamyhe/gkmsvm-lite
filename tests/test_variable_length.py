"""Sequences of different lengths (docs/design/variable-length.md): padded
batches equal scoring each sequence at its own length."""

import numpy as np
import pytest

from gkmsvm.codec import one_hot_encode
from gkmsvm.kernels.packed import PAD_WINDOW
from gkmsvm.svm import GkmSVM

PARAMS = {"L": 6, "k": 4, "d": 2, "M": 20, "H": 6.0, "gamma": 2.0}
KERNELS = [
    "gkm_cnt",
    "estimated_full",
    "estimated",
    "gkmrbf",
    "wgkm",
    "wgkmrbf",
    "gkm_cnt_rbf",
]


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


@pytest.mark.parametrize(
    "kernel_type", ["gkm_cnt", "estimated", "gkmrbf", "wgkm", "wgkmrbf"]
)
@pytest.mark.parametrize("mode", ["hypothetical", "perturbation", "importance"])
def test_gkmexplain_on_mixed_svs(kernel_type, mode):
    """Attributions are a sum over SVs (an RBF SV's factor depends on its own
    kernel value only), so a model with SVs of different lengths equals the
    sum of one-SV models at each SV's length, on every explain path."""
    from gkmsvm import gkmexplain

    rng = np.random.default_rng(7)
    svs = _seqs(rng, 12, 12, 36)
    coef = rng.normal(size=12)
    model, _ = _model(svs, coef, kernel_type)
    x = np.concatenate([_onehot(q) for q in _seqs(rng, 3, 25, 25)])
    expected = sum(
        np.asarray(
            gkmexplain(
                GkmSVM(_onehot(s), c[None], 0.0, kernel_type, PARAMS), x, mode=mode
            )
        )
        for s, c in zip(svs, coef)
    )
    for table, index in [(True, True), (False, True), (False, False)]:
        model.use_explain_table, model.use_lmer_index = table, index
        model.clear_caches()
        np.testing.assert_allclose(
            gkmexplain(model, x, mode=mode),
            expected,
            rtol=1e-10,
            atol=1e-13,
            err_msg=f"{table=} {index=}",
        )


def _device(device):
    """(function moving an array to device, rtol), or skip."""
    from gkmsvm.backend import HAS_CUPY, to_gpu, to_mlx

    if device == "cuda":
        if not HAS_CUPY:
            pytest.skip("CuPy not installed")
        return to_gpu, 1e-9
    pytest.importorskip("mlx.core")
    return to_mlx, 2e-4  # float32


def _np(a):
    from gkmsvm.backend import to_cpu

    return np.asarray(to_cpu(a), dtype=np.float64)


def _close(got, expected, rtol, msg):
    """Relative to the largest value: estimated_full's signed table cancels to
    ~1e-17 in places."""
    np.testing.assert_allclose(
        _np(got), expected, rtol=rtol, atol=rtol * np.abs(expected).max(), err_msg=msg
    )


@pytest.mark.parametrize("device", ["cuda", "mlx"])
@pytest.mark.parametrize("kernel_type", KERNELS)
def test_padded_kernels_on_device(device, kernel_type):
    """CUDA and Metal kernels skip PAD: padded rows and columns through the
    masked join and brute force, and the self-kernels, equal the CPU's."""
    from gkmsvm.svm import as_batch

    move, rtol = _device(device)

    rng = np.random.default_rng(10)
    base = GkmSVM(_onehot("ACGTACGT"), np.ones(1), 0.0, kernel_type, PARAMS).kernel.base
    x, lx = as_batch(_seqs(rng, 20, 12, 30))
    y, ly = as_batch(_seqs(rng, 25, 9, 33))
    bx, bx_rc = base.pack_both(x, lx)
    by, by_rc = base.pack_both(y, ly)
    for masked in (True, False):
        base.use_masked = masked
        expected = base.raw_packed(bx, by, by_rc)
        got = base.raw_packed(move(bx), move(by), move(by_rc))
        _close(got, expected, rtol, f"{masked=}")
    base.use_masked = True
    _close(
        base.raw_diag_packed(move(bx), move(bx_rc)),
        base.raw_diag_packed(bx, bx_rc),
        rtol,
        "diag",
    )


@pytest.mark.parametrize("device", ["cuda", "mlx"])
@pytest.mark.parametrize("kernel_type", KERNELS)
def test_mixed_lengths_on_device(device, kernel_type):
    """Scoring, ISM, mutation impact, variant scores and GkmExplain of
    mixed-length queries, and to_deltasvm, on a model with mixed-length SVs,
    equal the CPU on every path."""
    from gkmsvm import gkmexplain
    from gkmsvm.explain import mutation_impact
    from gkmsvm.ism import ism

    rng = np.random.default_rng(11)
    svs, queries = _seqs(rng, 30, 10, 40), _seqs(rng, 20, 8, 45)
    model, _ = _model(svs, rng.normal(size=30), kernel_type)
    q = _seqs(rng, 4, 15, 30)
    alt = [s[:7] + ("A" if s[7] != "A" else "C") + s[8:] for s in q[:2]]
    alt += [s[:5] + s[8:] for s in q[2:]]  # deletions
    paths = [(True, False, True), (False, False, True), (False, False, False)]
    if kernel_type in ("gkm_cnt", "estimated_full", "estimated", "wgkm"):
        paths.append((False, True, True))

    def run(m):
        out = {}
        for index, table, masked in paths:
            m.use_lmer_index, m.use_lmer_table = index, table
            m.kernel.base.use_masked = masked
            m.clear_caches()
            key = f"{index=} {table=} {masked=}"
            out[f"score {key}"] = m(queries)
            out[f"ism {key}"] = ism(m, q)
            out[f"impact {key}"] = mutation_impact(m, q, alt)
            out[f"variants {key}"] = m.score_variants(q, alt)
            for mode in ("hypothetical", "perturbation"):
                m.use_explain_table = table
                out[f"explain {mode} {key}"] = gkmexplain(m, q, mode=mode)
        if kernel_type not in ("gkmrbf", "wgkmrbf", "gkm_cnt_rbf"):
            out["deltasvm"] = m.to_deltasvm(query_norm=True).weights
        return {k: _np(v) for k, v in out.items()}

    with np.errstate(all="ignore"):
        expected = run(model)
        _, rtol = _device(device)
        got = run(model.cuda() if device == "cuda" else model.mlx())
    for key in expected:
        _close(got[key], expected[key], rtol, key)


@pytest.mark.parametrize("device", ["cuda", "mlx"])
@pytest.mark.parametrize("kernel_type", ["estimated", "wgkmrbf"])
@pytest.mark.parametrize("solver", ["libsvm", "smo"])
def test_training_on_device(device, kernel_type, solver):
    """Training on sequences of different lengths on CUDA/MLX: the Gram equals
    the CPU's, and the trained model the CPU's to the solver's tolerance."""
    from gkmsvm.gram import compute_gram
    from gkmsvm.svm import as_batch
    from gkmsvm.train import _build_kernel, train_gkmsvm

    move, rtol = _device(device)
    rng = np.random.default_rng(17)
    pos, neg = _seqs(rng, 25, 12, 45), _seqs(rng, 25, 12, 45)
    kernel, _, _ = _build_kernel(kernel_type, 6, 4, 2, 2.0, 20, 6.0, True)
    x, lengths = as_batch(pos + neg)
    gram = compute_gram(kernel, x, lengths=lengths)
    _close(compute_gram(kernel, move(x), lengths=lengths), gram, rtol, "gram")
    kw = dict(kernel_type=kernel_type, l=6, k=4, d=2, M=20, H=6.0, gamma=2.0)
    kw.update(solver=solver, tol=1e-6) if solver == "smo" else kw.update(solver=solver)
    cpu = train_gkmsvm(pos, neg, device="cpu", **kw)
    dev = train_gkmsvm(pos, neg, device=device, **kw)
    q = _seqs(rng, 8, 12, 45)
    # LIBSVM stops at tol 1e-3, and a last-digit difference in the Gram can
    # move it to another point within that (seen on CUDA: 4e-16 in the Gram,
    # 4e-5 in scores); SMO runs at 1e-6. MLX Gram tiles are float32.
    tol = 1e-6 if device == "cuda" and solver == "smo" else 1e-3
    np.testing.assert_allclose(_np(dev(q)), _np(cpu(q)), atol=tol)


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


# -- Training on sequences of different lengths -------------------------------

_RAGGED_PARAMS = {  # benchmarks/dev/lsgkm_wgkm_fixtures.py RAGGED_MODELS
    "t2": dict(kernel_type=2, l=10, k=6, d=3),
    "t4": dict(kernel_type=4, l=10, k=6, d=3, M=20, H=10),
    "t5": dict(kernel_type=5, l=10, k=6, d=3, M=50, H=30, gamma=2.0),
}


def _ragged_training_set():
    from gkmsvm.fasta import read_fasta

    return [
        [s for _, s in read_fasta(str(FIXTURES / f"lsgkm_ragged_{kind}.fa"))]
        for kind in ("pos", "neg")
    ]


def _sv_coefficients(model):
    """{SV sequence at its length: coefficient}."""
    from gkmsvm.codec import one_hot_decode

    L = model.support_sequences.shape[2]
    lengths = (
        model.sv_lengths
        if model.sv_lengths is not None
        else [L] * len(model.coefficients)
    )
    return {
        one_hot_decode(sv[:, :n]): float(c)
        for sv, n, c in zip(model.support_sequences, lengths, model.coefficients)
    }


@pytest.mark.parametrize("kernel_type", KERNELS)
def test_gram_on_mixed_lengths(kernel_type):
    """The self- and cross-Gram (masked-join panels and brute-force tiles) and
    SMO columns of padded sequences equal per-pair kernel values."""
    from gkmsvm.gram import compute_gram
    from gkmsvm.solver import KernelColumnCache
    from gkmsvm.svm import as_batch

    rng = np.random.default_rng(12)
    xs, ys = _seqs(rng, 40, 10, 45), _seqs(rng, 20, 8, 50)
    kernel = GkmSVM(_onehot(xs[0]), np.ones(1), 0.0, kernel_type, PARAMS).kernel
    pair = np.array(
        [
            [float(kernel.pairwise(_onehot(a), _onehot(b))[0, 0]) for b in xs + ys]
            for a in xs
        ]
    )
    x, lx = as_batch(xs)
    y, ly = as_batch(ys)
    for masked in (True, False):
        kernel.base.use_masked = masked
        msg = f"{masked=}"
        np.testing.assert_allclose(
            compute_gram(kernel, x, lengths=lx, chunk_size=17),
            pair[:, :40],
            rtol=1e-12,
            atol=1e-14,
            err_msg=msg,
        )
        np.testing.assert_allclose(
            compute_gram(kernel, x, y, lengths=lx, y_lengths=ly, chunk_size=17),
            pair[:, 40:],
            rtol=1e-12,
            atol=1e-14,
            err_msg=msg,
        )
    kernel.base.use_masked = True
    cols = KernelColumnCache(kernel, x, lx).compute_columns([0, 7, 39])
    for i, col in cols.items():
        np.testing.assert_allclose(col, pair[:, i], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("name", ["t2", "t4", "t5"])
@pytest.mark.parametrize("solver", ["libsvm", "smo"])
def test_training_matches_gkmtrain(name, solver):
    """Trained on gkmtrain's mixed-length (60-140 bp) training sets, the model
    has gkmtrain's support vectors, its coefficients to the solvers' 1e-3
    stopping tolerance, and gkmpredict's scores on a 30-200 bp test set."""
    from gkmsvm.fasta import read_fasta
    from gkmsvm.importers.lsgkm import load_lsgkm_model
    from gkmsvm.train import train_gkmsvm

    pos, neg = _ragged_training_set()
    model = train_gkmsvm(
        pos, neg, C=1.0, solver=solver, device="cpu", **_RAGGED_PARAMS[name]
    )
    assert model.sv_lengths is not None
    ref = _sv_coefficients(
        load_lsgkm_model(FIXTURES / f"lsgkm_ragged_{name}.model.txt.gz")
    )
    got = _sv_coefficients(model)
    assert set(got) == set(ref)
    np.testing.assert_allclose([got[s] for s in ref], list(ref.values()), atol=5e-3)
    records = read_fasta(str(FIXTURES / "lsgkm_ragged_test.fa"))
    with open(FIXTURES / f"lsgkm_ragged_{name}_test.pred.txt") as f:
        pred = dict(line.split()[:2] for line in f)
    expected = np.array([float(pred[n]) for n, _ in records])
    with pytest.warns(UserWarning, match="shorter than l"):
        scores = np.asarray(model([s for _, s in records])).ravel()
    np.testing.assert_allclose(scores, expected, atol=5e-4)


@pytest.mark.parametrize("name", ["t2", "t4", "t5"])
def test_gkmtrain_solution_is_optimal_under_our_kernel(name):
    """gkmtrain's dual solution satisfies the KKT conditions of the problem our
    kernel poses on the mixed-length training set, to its stopping tolerance
    (eps 1e-3): y f(x) = 1 on free SVs, >= 1 off them, <= 1 at C."""
    from gkmsvm.importers.lsgkm import load_lsgkm_model

    pos, neg = _ragged_training_set()
    model = load_lsgkm_model(FIXTURES / f"lsgkm_ragged_{name}.model.txt.gz")
    alpha = {s: abs(c) for s, c in _sv_coefficients(model).items()}
    # SVs hold N as A (it packs as A), so look the training sequences up that way
    a = np.array([alpha.get(s.replace("N", "A"), 0.0) for s in pos + neg])
    assert (a > 0).sum() == model.num_support_vectors
    y = np.r_[np.ones(len(pos)), -np.ones(len(neg))]
    yf = y * np.asarray(model(pos + neg)).ravel()
    free = (a > 1e-6) & (a < 1 - 1e-6)
    assert np.abs(yf[free] - 1).max() < 1e-3
    assert (yf[a == 0] > 1 - 1e-3).all() and (yf[a >= 1 - 1e-6] < 1 + 1e-3).all()


def test_nystrom_and_svr_on_mixed_lengths():
    """Nyström with every sequence a landmark gives the exact Gram's model, and
    SVR on mixed lengths equals sklearn's on per-pair kernel values."""
    from sklearn.svm import SVR

    from gkmsvm.train import train_gkmsvm, train_gkmsvr

    rng = np.random.default_rng(13)
    pos, neg = _seqs(rng, 30, 12, 50), _seqs(rng, 30, 12, 50)
    kw = dict(kernel_type="estimated", l=6, k=4, d=2, device="cpu")
    exact = train_gkmsvm(pos, neg, solver="libsvm", **kw)
    nys = train_gkmsvm(pos, neg, solver="nystrom", n_components=60, **kw)
    q = _seqs(rng, 10, 12, 50)
    np.testing.assert_allclose(np.asarray(nys(q)), np.asarray(exact(q)), atol=1e-6)

    seqs, labels = pos + neg, rng.normal(size=60)
    model = train_gkmsvr(seqs, labels, C=1.0, epsilon=0.1, **kw)
    kernel = model.kernel
    gram = np.array(
        [
            [float(kernel.pairwise(_onehot(a), _onehot(b))[0, 0]) for b in seqs]
            for a in seqs
        ]
    )
    ref = SVR(C=1.0, epsilon=0.1, kernel="precomputed").fit(gram, labels)
    np.testing.assert_allclose(
        np.asarray(model(seqs)).ravel(), ref.predict(gram), rtol=1e-6, atol=1e-6
    )


def test_training_rejects_sequences_shorter_than_l():
    from gkmsvm.train import train_gkmsvm

    with pytest.raises(ValueError, match="at least l = 6"):
        train_gkmsvm(["ACGTACGTAC", "ACG"], ["TTGACCATGA"], l=6, k=4, d=2, device="cpu")


# -- ISM, GkmExplain and variant scores on queries of different lengths -------

_QUERY_PATHS = [(True, False, True), (False, False, True), (False, False, False)]


def _query_paths(kernel_type):
    """(use_lmer_index, use_lmer_table, use_masked) for the kernel's paths."""
    table = kernel_type in ("gkm_cnt", "estimated_full", "estimated", "wgkm")
    return _QUERY_PATHS + ([(False, True, True)] if table else [])


@pytest.mark.parametrize("kernel_type", KERNELS)
def test_ism_on_mixed_queries(kernel_type):
    """ISM of queries of different lengths, padded in one batch, equals each
    query's own, is 0 past its length, and NaN for one shorter than l."""
    from gkmsvm.ism import ism

    rng = np.random.default_rng(14)
    model, _ = _model(_seqs(rng, 30, 10, 40), rng.normal(size=30), kernel_type)
    queries = _seqs(rng, 7, 8, 33)
    for index, table, masked in _query_paths(kernel_type):
        model.use_lmer_index, model.use_lmer_table = index, table
        model.kernel.base.use_masked = masked
        model.clear_caches()
        msg = f"{index=} {table=} {masked=}"
        got = np.asarray(ism(model, queries, batch_size=3))
        for i, q in enumerate(queries):
            np.testing.assert_allclose(
                got[i, :, : len(q)],
                np.asarray(ism(model, _onehot(q)))[0],
                rtol=1e-10,
                atol=1e-13,
                err_msg=msg,
            )
            assert (got[i, :, len(q) :] == 0).all(), msg
    with pytest.warns(
        UserWarning, match="shorter than l = 6: their ISM deltas are NaN"
    ):
        out = np.asarray(ism(model, [queries[0], "ACGT"]))
    assert np.isnan(out[1]).all() and np.isfinite(out[0]).all()


@pytest.mark.parametrize("kernel_type", KERNELS)
@pytest.mark.parametrize("mode", ["hypothetical", "perturbation", "importance"])
def test_gkmexplain_on_mixed_queries(kernel_type, mode):
    """GkmExplain of queries of different lengths in one batch equals each
    query's own on the explain table, l-mer/index and per-SV paths."""
    from gkmsvm import gkmexplain

    rng = np.random.default_rng(15)
    model, _ = _model(_seqs(rng, 25, 10, 40), rng.normal(size=25), kernel_type)
    queries = _seqs(rng, 6, 8, 33)
    for table, index in [(True, True), (False, True), (False, False)]:
        model.use_explain_table, model.use_lmer_index = table, index
        model.clear_caches()
        got = np.asarray(gkmexplain(model, queries, mode=mode, batch_size=4))
        for i, q in enumerate(queries):
            np.testing.assert_allclose(
                got[i, :, : len(q)],
                np.asarray(gkmexplain(model, _onehot(q), mode=mode))[0],
                rtol=1e-10,
                atol=1e-13,
                err_msg=f"{table=} {index=}",
            )
            assert (got[i, :, len(q) :] == 0).all()


@pytest.mark.parametrize("kernel_type", KERNELS)
def test_variant_scores_on_mixed_lengths(kernel_type):
    """score_variants on SNVs and deletions of queries of different lengths:
    "kernel" is score(alt) - score(ref) and "gkmexplain" mutation impact as
    defined from per-pair raw kernel values, through every scoring path."""
    rng = np.random.default_rng(16)
    svs = _seqs(rng, 25, 10, 40)
    coef = rng.normal(size=25)
    model, kernel = _model(svs, coef, kernel_type)
    refs = _seqs(rng, 12, 10, 33)
    alts = [r[:5] + ("A" if r[5] != "A" else "G") + r[6:] for r in refs[:6]]
    alts += [r[:4] + r[7:] for r in refs[6:]]  # deletions
    base = kernel.base
    raw = np.array(
        [
            [float(base._raw_pairwise(_onehot(q), _onehot(s))[0, 0]) for s in svs]
            for q in refs + alts
        ]
    )
    d_ref = np.array([float(base._raw_diagonal(_onehot(q))[0]) for q in refs])
    d_sv = np.array([float(base._raw_diagonal(_onehot(s))[0]) for s in svs])
    z = np.sqrt(d_ref[:, None] * d_sv[None, :])
    kn, dk = raw[:12] / z, (raw[12:] - raw[:12]) / z
    if kernel.base is not kernel:  # RBF
        g = PARAMS["gamma"]
        safe = np.where(kn > 0, kn, 1)
        dk = np.where(kn > 0, dk * (np.exp(g * (kn - 1)) - np.exp(-g)) / safe, 0)
    impact = np.einsum("bs,s->b", dk, coef)
    delta = _expected(kernel, alts, svs, coef, 0.2) - _expected(
        kernel, refs, svs, coef, 0.2
    )
    for index, _, masked in _QUERY_PATHS:
        model.use_lmer_index = index
        model.kernel.base.use_masked = masked
        model.clear_caches()
        for method, expected in (("gkmexplain", impact), ("kernel", delta)):
            np.testing.assert_allclose(
                np.asarray(model.score_variants(refs, alts, method=method)).ravel(),
                expected,
                rtol=1e-9,
                atol=1e-12,
                err_msg=f"{method} {index=} {masked=}",
            )


def test_deltasvm_on_mixed_lengths():
    """DeltaSVM scores each sequence of a mixed-length batch at its own length."""
    rng = np.random.default_rng(18)
    model, _ = _model(_seqs(rng, 20, 10, 30), rng.normal(size=20), "estimated")
    dsvm = model.to_deltasvm()
    queries = _seqs(rng, 8, 8, 40)
    expected = [float(np.asarray(dsvm(_onehot(q)))[0, 0]) for q in queries]
    np.testing.assert_allclose(np.asarray(dsvm(queries)).ravel(), expected, rtol=1e-12)
    from gkmsvm.svm import as_batch

    x, lengths = as_batch(queries)
    np.testing.assert_allclose(
        np.asarray(dsvm(x, lengths=lengths)).ravel(), expected, rtol=1e-12
    )
