"""Classic (C++) and R gkmSVM models as their tools wrote them
(benchmarks/dev/gkmsvm_fixtures.py): trained on the 100 bp fixture sets
with N stretches, scored on 100 and 85 bp test sets."""

import re
from pathlib import Path

import numpy as np
import pytest

from gkmsvm.fasta import read_fasta
from gkmsvm.importers.classic import load_classic_model
from gkmsvm.importers.r_gkmsvm import load_r_gkmsvm_model

FIXTURES = Path(__file__).parent.parent / "fixtures"
# (fixture prefix, loader, gkmsvm_kernel -t / R useTgkm); l=10 k=6 d=3
MODELS = [
    ("gkmsvm_classic_t0", "classic", 0),
    ("gkmsvm_classic_t1", "classic", 1),
    ("gkmsvm_classic_t2", "classic", 2),
    ("gkmsvm_r_t1", "r", 1),
    ("gkmsvm_r_t2", "r", 2),
]


def _load(prefix, tool, filter_type):
    alpha = FIXTURES / f"{prefix}_svalpha.out"
    with pytest.warns(UserWarning, match="other than ACGT, dropped"):
        if tool == "r":
            return load_r_gkmsvm_model(alpha, useTgkm=filter_type)
        if filter_type == 1:  # the tools' default kernel
            return load_classic_model(alpha)
        return load_classic_model(alpha, filter_type=filter_type)


def _scores(path):
    return dict((line.split()[0], float(line.split()[1])) for line in open(path))


@pytest.mark.parametrize("prefix, tool, filter_type", MODELS)
@pytest.mark.parametrize("test", ["test100", "test85"])
def test_scores_match_gkmsvm_kernel(prefix, tool, filter_type, test):
    """Scores equal sum_i alpha_i K(x, sv_i) with the tool's gkmsvm_kernel K
    (no bias), for queries read as the tools read them (N dropped)."""
    model = _load(prefix, tool, filter_type)
    assert model.bias == 0.0 and model.sv_lengths is not None  # N dropped
    records = read_fasta(str(FIXTURES / f"lsgkm_wgkm_{test}.fa"))
    queries = [re.sub("[^ACGT]", "", s) for _, s in records]
    expected = _scores(FIXTURES / f"{prefix}_{test}_kernel.txt")
    got = np.asarray(model(queries)).ravel()
    # gkmsvm_kernel prints 7 significant digits
    np.testing.assert_allclose(
        got, [expected[n] for n, _ in records], rtol=1e-5, atol=1e-6
    )
    # gkmsvm_classify itself differs from that sum (its tree and simple
    # algorithms normalize differently from gkmsvm_kernel): by up to ~2%
    classify = _scores(FIXTURES / f"{prefix}_{test}_classify.txt")
    assert np.abs(got - [classify[n] for n, _ in records]).max() < 0.03


def test_cli_import(tmp_path):
    """gkmsvm import takes the kernel the files don't record."""
    from gkmsvm.cli import main
    from gkmsvm.serialization import load_model

    out = tmp_path / "m.npz"
    alpha = FIXTURES / "gkmsvm_classic_t2_svalpha.out"
    with pytest.warns(UserWarning, match="dropped"):
        main(["import", "-f", "classic", "-i", str(alpha), "-o", str(out), "-t", "2"])
    model = load_model(out)
    assert model.kernel_type == "gkm_cnt" and model.kernel.l == 10
    ref = _load("gkmsvm_classic_t2", "classic", 2)
    q = ["ACGTTGCATGCAAGTCCTAGGATCCATGCATGACTGACAA"]
    np.testing.assert_allclose(np.asarray(model(q)), np.asarray(ref(q)), rtol=1e-6)


def _write(tmp_path, alphas, seqs):
    """PREFIX_svalpha.out ({id: alpha}) and PREFIX_svseq.fa ({id: sequence})."""
    alpha = tmp_path / "m_svalpha.out"
    alpha.write_text("".join(f"{i}\t{a:e}\n" for i, a in alphas.items()))
    (tmp_path / "m_svseq.fa").write_text(
        "".join(f">{i}\n{s}\n" for i, s in seqs.items())
    )
    return alpha


def test_svs_matched_by_id(tmp_path):
    """As gkmsvm_classify: each alpha picks its SV by id (extra FASTA records
    are ignored, lowercase is read as uppercase); an id missing from the
    FASTA is an error."""
    from gkmsvm.codec import one_hot_encode

    seqs = {"a": "ACGTTGCAAGGTC", "b": "ttgacgatcgaat", "c": "GGGCCCATATAGC"}
    model = load_classic_model(_write(tmp_path, {"c": 0.5, "b": -0.3}, seqs), L=5, k=3)
    assert model.num_support_vectors == 2 and model.bias == 0.0
    np.testing.assert_array_equal(
        model.support_sequences,
        np.stack([one_hot_encode(seqs[i].upper()) for i in ("c", "b")]),
    )
    np.testing.assert_array_equal(model.coefficients, [0.5, -0.3])
    with pytest.raises(ValueError, match="not in"):
        load_classic_model(_write(tmp_path, {"a": 1.0, "z": 1.0}, seqs), L=5, k=3)


def test_kernel_arguments(tmp_path):
    """The files record no kernel: it comes from the arguments, R's under
    its gkmsvm_kernel names."""
    alpha = _write(tmp_path, {"a": 1.0}, {"a": "ACGTTGCAAGGTCAT"})
    model = load_classic_model(alpha, L=6, k=4, d=2, filter_type=0, include_rc=False)
    assert model.kernel_type == "gkm_estfull"
    assert (model.kernel.l, model.kernel.k, model.kernel.d) == (6, 4, 2)
    assert not model.kernel.include_rc
    r = load_r_gkmsvm_model(alpha, L=6, K=4, maxnmm=2, useTgkm=0, addRC=False)
    assert r.kernel_params == model.kernel_params
    with pytest.raises(NotImplementedError, match="wildcard"):
        load_classic_model(alpha, filter_type=3)


def test_malformed_alpha_file(tmp_path):
    alpha = tmp_path / "m_svalpha.out"
    alpha.write_text("svm_type c_svc\nrho 0.1 0.2\n")
    (tmp_path / "m_svseq.fa").write_text(">a\nACGT\n")
    with pytest.raises(ValueError):
        load_classic_model(alpha)
    with pytest.raises(ValueError, match="give svseq_path"):
        load_classic_model(tmp_path / "alphas.txt")
