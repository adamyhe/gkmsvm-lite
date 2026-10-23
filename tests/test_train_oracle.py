"""train_gkmsvm against gkmtrain: models trained on the same 100 bp sets (a
planted motif in the positives, N stretches in a fifth of each), -t 0 to -t 5
(benchmarks/dev/lsgkm_wgkm_fixtures.py; the mixed-length version is in
test_variable_length.py)."""

import numpy as np
import pytest

from gkmsvm.codec import decode_codes
from gkmsvm.fasta import read_fasta
from gkmsvm.importers.lsgkm import load_lsgkm_model

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"

# gkmtrain's options for each fixture model (-c 1.0; LS-GKM's default d is 3)
PARAMS = {
    "t0": dict(kernel_type=0, l=10, k=6, d=3),
    "t1": dict(kernel_type=1, l=11, k=7, d=3),
    "t2": dict(kernel_type=2, l=10, k=6, d=3),
    "t3": dict(kernel_type=3, l=10, k=6, d=3, gamma=2.0),
    "t4": dict(kernel_type=4, l=11, k=7, d=3, M=20, H=10),
    "t5": dict(kernel_type=5, l=10, k=6, d=3, M=50, H=30, gamma=2.0),
    # gkm_cnt_rbf, trained by the kundajelab fork's gkmtrain
    "t6": dict(kernel_type=6, l=10, k=6, d=3, gamma=2.0),
    # -w: the positives' C is w*C
    "t2w2": dict(kernel_type=2, l=10, k=6, d=3, pos_weight=2.0),
    "t4w05": dict(kernel_type=4, l=11, k=7, d=3, M=20, H=10, pos_weight=0.5),
}


def _seqs(name):
    return [s for _, s in read_fasta(str(FIXTURES / name))]


def _sv_coefficients(model):
    """{SV sequence: coefficient}."""
    return {
        decode_codes(sv): float(c)
        for sv, c in zip(model.support_sequences, model.coefficients)
    }


@pytest.mark.parametrize("name", PARAMS)
@pytest.mark.parametrize("solver", ["libsvm", "smo"])
def test_training_matches_gkmtrain(name, solver):
    """The trained model has gkmtrain's support vectors, its coefficients to
    the solvers' 1e-3 stopping tolerance, and gkmpredict's scores."""
    from gkmsvm.train import train_gkmsvm

    pos, neg = _seqs("lsgkm_wgkm_pos.fa"), _seqs("lsgkm_wgkm_neg.fa")
    model = train_gkmsvm(pos, neg, C=1.0, solver=solver, device="cpu", **PARAMS[name])
    ref = _sv_coefficients(
        load_lsgkm_model(FIXTURES / f"lsgkm_wgkm_{name}.model.txt.gz")
    )
    got = _sv_coefficients(model)
    assert set(got) == set(ref)
    np.testing.assert_allclose([got[s] for s in ref], list(ref.values()), atol=5e-3)
    for length in (100, 85):
        records = read_fasta(str(FIXTURES / f"lsgkm_wgkm_test{length}.fa"))
        with open(FIXTURES / f"lsgkm_wgkm_{name}_test{length}.pred.txt") as f:
            pred = dict(line.split()[:2] for line in f)
        np.testing.assert_allclose(
            np.asarray(model([s for _, s in records])).ravel(),
            [float(pred[n]) for n, _ in records],
            atol=5e-4,
        )


@pytest.mark.parametrize("name", PARAMS)
def test_gkmtrain_solution_is_optimal_under_our_kernel(name):
    """gkmtrain's dual solution meets the KKT conditions of the problem our
    kernel poses on its training set, to its stopping tolerance (eps 1e-3):
    y f(x) = 1 on free SVs, >= 1 off them, <= 1 at C (w C for positives)."""
    pos, neg = _seqs("lsgkm_wgkm_pos.fa"), _seqs("lsgkm_wgkm_neg.fa")
    model = load_lsgkm_model(FIXTURES / f"lsgkm_wgkm_{name}.model.txt.gz")
    alpha = {s: abs(c) for s, c in _sv_coefficients(model).items()}
    # SVs hold N as A (it packs as A), so look the training sequences up that way
    a = np.array([alpha.get(s.replace("N", "A"), 0.0) for s in pos + neg])
    assert (a > 0).sum() == model.num_support_vectors
    y = np.r_[np.ones(len(pos)), -np.ones(len(neg))]
    C = np.where(y > 0, PARAMS[name].get("pos_weight", 1.0), 1.0)
    yf = y * np.asarray(model(pos + neg)).ravel()
    free = (a > 1e-6) & (a < C - 1e-6)
    assert np.abs(yf[free] - 1).max() < 1e-3
    assert (yf[a == 0] > 1 - 1e-3).all() and (yf[a >= C - 1e-6] < 1 + 1e-3).all()


def test_cli_w(tmp_path):
    """gkmsvm train -w is pos_weight."""
    from gkmsvm.cli import main
    from gkmsvm.serialization import load_model

    out = str(tmp_path / "m.npz")
    main(
        [
            "train",
            "-p",
            str(FIXTURES / "lsgkm_wgkm_pos.fa"),
            "-n",
            str(FIXTURES / "lsgkm_wgkm_neg.fa"),
            "-o",
            out,
            *("-t", "2", "-l", "10", "-k", "6", "-d", "3", "-w", "2.0"),
            *("--device", "cpu"),
        ]
    )
    ref = _sv_coefficients(load_lsgkm_model(FIXTURES / "lsgkm_wgkm_t2w2.model.txt.gz"))
    got = _sv_coefficients(load_model(out))
    assert set(got) == set(ref)
    np.testing.assert_allclose([got[s] for s in ref], list(ref.values()), atol=5e-3)


# epsilon-SVR (-y 3 -c 1 -p 0.1) by kundajelab/lsgkm-svr's gkmtrain
SVR_PARAMS = {
    "t2": dict(kernel_type=2, l=10, k=6, d=3),
    "t3": dict(kernel_type=3, l=10, k=6, d=3, gamma=2.0),
    "t4": dict(kernel_type=4, l=10, k=6, d=3, M=20, H=10),
}


def _svr_predictions(name, length):
    records = read_fasta(str(FIXTURES / f"lsgkm_wgkm_test{length}.fa"))
    with open(FIXTURES / f"lsgkm_svr_{name}_test{length}.pred.txt") as f:
        pred = dict(line.split()[:2] for line in f)
    return [s for _, s in records], [float(pred[n]) for n, _ in records]


@pytest.mark.parametrize("name", SVR_PARAMS)
@pytest.mark.parametrize("solver", ["libsvm", "smo"])
def test_svr_matches_lsgkm_svr(name, solver):
    """train_gkmsvr gives lsgkm-svr's support vectors, coefficients to the
    1e-3 stopping tolerance and predictions; its model files load and
    reproduce its gkmpredict."""
    from gkmsvm.train import train_gkmsvr

    seqs = _seqs("lsgkm_svr_train.fa")
    labels = np.loadtxt(FIXTURES / "lsgkm_svr_labels.txt")
    model = train_gkmsvr(
        seqs,
        labels,
        C=1.0,
        epsilon=0.1,
        solver=solver,
        device="cpu",
        **SVR_PARAMS[name],
    )
    imported = load_lsgkm_model(FIXTURES / f"lsgkm_svr_{name}.model.txt.gz")
    ref = _sv_coefficients(imported)
    got = _sv_coefficients(model)
    assert set(got) == set(ref)
    np.testing.assert_allclose([got[s] for s in ref], list(ref.values()), atol=5e-3)
    for length in (100, 85):
        queries, expected = _svr_predictions(name, length)
        # gkmpredict prints 6 significant digits
        np.testing.assert_allclose(
            np.asarray(imported(queries)).ravel(), expected, rtol=2e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(model(queries)).ravel(), expected, atol=5e-4
        )


def test_cli_trainsvr_smo(tmp_path):
    """gkmsvm trainsvr --solver smo gives lsgkm-svr's model."""
    from gkmsvm.cli import main
    from gkmsvm.serialization import load_model

    out = str(tmp_path / "m.npz")
    main(
        [
            "trainsvr",
            "-i",
            str(FIXTURES / "lsgkm_svr_train.fa"),
            "--labels",
            str(FIXTURES / "lsgkm_svr_labels.txt"),
            "-o",
            out,
            *("-t", "2", "-l", "10", "-k", "6", "-d", "3", "--solver", "smo"),
            *("--cache-size", "16", "--device", "cpu"),
        ]
    )
    ref = _sv_coefficients(load_lsgkm_model(FIXTURES / "lsgkm_svr_t2.model.txt.gz"))
    got = _sv_coefficients(load_model(out))
    assert set(got) == set(ref)
    np.testing.assert_allclose([got[s] for s in ref], list(ref.values()), atol=5e-3)
