"""The fixed-length pipeline: center-cropping, summit windows, CLI --length,
--genome and --variable-length (docs/design/variable-length.md)."""

import random

import numpy as np
import pytest

from gkmsvm.cli import main
from gkmsvm.codec import center_crop, one_hot_encode


def _rand(rng, n):
    return "".join(rng.choice("ACGT") for _ in range(n))


def test_center_crop():
    assert center_crop(["AACCGGTT", "ACGTACGTAC"], 4) == ["CCGG", "TACG"]
    x = np.arange(2 * 4 * 10).reshape(2, 4, 10)
    np.testing.assert_array_equal(center_crop(x, 6), x[:, :, 2:8])
    with pytest.raises(ValueError, match="2 sequences are shorter than 6 bp.*b, c"):
        center_crop(["ACGTACGT", "ACG", "AC"], 6, names=["a", "b", "c"])


@pytest.fixture
def model_path(tmp_path):
    rng = random.Random(0)
    pos = tmp_path / "pos.fa"
    neg = tmp_path / "neg.fa"
    for path in (pos, neg):
        path.write_text("".join(f">s{i}\n{_rand(rng, 40)}\n" for i in range(20)))
    out = str(tmp_path / "model.npz")
    main(
        [
            "train",
            "-p",
            str(pos),
            "-n",
            str(neg),
            "-o",
            out,
            "-l",
            "5",
            "-k",
            "3",
            "-d",
            "2",
            "--device",
            "cpu",
        ]
    )
    return out


def _scores(path):
    lines = open(path).read().splitlines()[1:]
    return {n: float(v) for n, v in (line.split("\t") for line in lines)}


def test_predict_crops_mixed_lengths_to_sv_length(model_path, tmp_path):
    """Records of different lengths are center-cropped to the model's SV
    length (40 bp); a shorter one stops with an error naming it."""
    from gkmsvm.serialization import load_model

    rng = random.Random(1)
    seqs = [_rand(rng, n) for n in (40, 55, 90)]
    fa = tmp_path / "q.fa"
    fa.write_text("".join(f">q{i}\n{s}\n" for i, s in enumerate(seqs)))
    out = tmp_path / "s.tsv"
    main(
        ["predict", "-m", model_path, "-i", str(fa), "-o", str(out), "--device", "cpu"]
    )
    model = load_model(model_path)
    expected = np.asarray(model(center_crop(seqs, 40))).ravel()
    np.testing.assert_allclose(
        list(_scores(out).values()), expected, rtol=1e-6, atol=1e-6
    )

    fa.write_text(fa.read_text() + ">tiny\nACGTACGTAC\n")
    with pytest.raises(SystemExit):
        main(["predict", "-m", model_path, "-i", str(fa), "--device", "cpu"])


def test_predict_uniform_records_unchanged(model_path, tmp_path):
    """Records sharing a length other than the SV length are scored as they
    are (e.g. 51 bp variant contexts)."""
    from gkmsvm.serialization import load_model

    rng = random.Random(2)
    seqs = [_rand(rng, 25) for _ in range(3)]
    fa = tmp_path / "q.fa"
    fa.write_text("".join(f">q{i}\n{s}\n" for i, s in enumerate(seqs)))
    out = tmp_path / "s.tsv"
    main(
        ["predict", "-m", model_path, "-i", str(fa), "-o", str(out), "--device", "cpu"]
    )
    expected = np.asarray(load_model(model_path)(seqs)).ravel()
    np.testing.assert_allclose(
        list(_scores(out).values()), expected, rtol=1e-6, atol=1e-6
    )


def test_predict_variable_length(model_path, tmp_path):
    from gkmsvm.serialization import load_model

    rng = random.Random(3)
    seqs = [_rand(rng, n) for n in (12, 40, 77)]
    fa = tmp_path / "q.fa"
    fa.write_text("".join(f">q{i}\n{s}\n" for i, s in enumerate(seqs)))
    out = tmp_path / "s.tsv"
    main(
        [
            "predict",
            "-m",
            model_path,
            "-i",
            str(fa),
            "-o",
            str(out),
            "--device",
            "cpu",
            "--variable-length",
        ]
    )
    expected = np.asarray(load_model(model_path)(seqs)).ravel()
    np.testing.assert_allclose(
        list(_scores(out).values()), expected, rtol=1e-6, atol=1e-6
    )


def test_predict_from_peaks_and_genome(model_path, tmp_path):
    pytest.importorskip("pyfaidx")
    from gkmsvm.fasta import extract_loci
    from gkmsvm.serialization import load_model

    rng = random.Random(4)
    genome = tmp_path / "g.fa"
    genome.write_text(f">chr1\n{_rand(rng, 500)}\n")
    peaks = tmp_path / "p.narrowPeak"
    peaks.write_text(
        "chr1\t100\t180\tpk1\t0\t.\t1\t1\t1\t30\n"
        "chr1\t200\t230\tpk2\t0\t.\t1\t1\t1\t-1\n"
        "chr1\t480\t499\tpk3\t0\t.\t1\t1\t1\t10\n"  # window past the end: dropped
    )
    out = tmp_path / "s.tsv"
    main(
        [
            "predict",
            "-m",
            model_path,
            "-i",
            str(peaks),
            "--genome",
            str(genome),
            "-o",
            str(out),
            "--device",
            "cpu",
        ]
    )
    scores = _scores(out)
    assert list(scores) == ["pk1", "pk2"]
    X = extract_loci(peaks, genome, in_window=40)[0]
    np.testing.assert_allclose(
        list(scores.values()),
        np.asarray(load_model(model_path)(X)).ravel(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_train_crops_to_length(tmp_path):
    """Training FASTAs of different lengths are center-cropped (--length, else 300)."""
    from gkmsvm.serialization import load_model

    rng = random.Random(5)
    pos, neg = tmp_path / "pos.fa", tmp_path / "neg.fa"
    for path in (pos, neg):
        path.write_text(
            "".join(f">s{i}\n{_rand(rng, rng.randint(50, 80))}\n" for i in range(20))
        )
    out = str(tmp_path / "m.npz")
    main(
        [
            "train",
            "-p",
            str(pos),
            "-n",
            str(neg),
            "-o",
            out,
            "-l",
            "5",
            "-k",
            "3",
            "-d",
            "2",
            "--length",
            "48",
            "--device",
            "cpu",
        ]
    )
    assert load_model(out).support_sequences.shape[-1] == 48
    with pytest.raises(SystemExit):  # default 300 bp: every record is shorter
        main(
            [
                "train",
                "-p",
                str(pos),
                "-n",
                str(neg),
                "-o",
                out,
                "-l",
                "5",
                "-k",
                "3",
                "--device",
                "cpu",
            ]
        )


def test_variable_length_train_ism_explain_variants(tmp_path):
    """--variable-length trains on, and runs ism, explain and score-variants
    on, FASTA records at their own lengths."""
    from gkmsvm import gkmexplain
    from gkmsvm.ism import ism
    from gkmsvm.serialization import load_model
    from gkmsvm.train import train_gkmsvm

    rng = random.Random(6)
    sets = [[_rand(rng, rng.randint(30, 60)) for _ in range(20)] for _ in range(2)]
    pos, neg = tmp_path / "pos.fa", tmp_path / "neg.fa"
    for path, seqs in zip((pos, neg), sets):
        path.write_text("".join(f">s{i}\n{s}\n" for i, s in enumerate(seqs)))
    out = str(tmp_path / "m.npz")
    args = ["-l", "5", "-k", "3", "-d", "2", "--device", "cpu"]
    main(
        ["train", "-p", str(pos), "-n", str(neg), "-o", out, *args, "--variable-length"]
    )
    model = load_model(out)
    assert model.sv_lengths is not None
    expected = train_gkmsvm(*sets, l=5, k=3, d=2, device="cpu")
    q = [_rand(rng, n) for n in (12, 40, 77)]
    np.testing.assert_allclose(
        np.asarray(model(q)), np.asarray(expected(q)), rtol=1e-6, atol=1e-6
    )

    fa = tmp_path / "q.fa"
    fa.write_text("".join(f">q{i}\n{s}\n" for i, s in enumerate(q)))
    vl = ["--device", "cpu", "--variable-length"]
    main(["ism", "-m", out, "-i", str(fa), "-o", str(tmp_path / "ism.npz"), *vl])
    res = np.load(tmp_path / "ism.npz")
    np.testing.assert_array_equal(res["lengths"], [12, 40, 77])
    np.testing.assert_allclose(res["scores"], np.asarray(ism(model, q)), atol=1e-12)

    attr, seqs = tmp_path / "attr.npz", tmp_path / "seqs.npz"
    main(["explain", "-m", out, "-i", str(fa), "-o", str(attr), "-s", str(seqs), *vl])
    np.testing.assert_allclose(
        np.load(attr)["arr_0"],
        np.asarray(gkmexplain(model, q, mode="hypothetical")),
        rtol=1e-6,
        atol=1e-7,
    )

    alt = tmp_path / "alt.fa"
    alts = [s[:4] + s[6:] for s in q]  # deletions
    alt.write_text("".join(f">q{i}\n{s}\n" for i, s in enumerate(alts)))
    tsv = tmp_path / "v.tsv"
    main(
        [
            "score-variants",
            "-m",
            out,
            "--ref",
            str(fa),
            "--alt",
            str(alt),
            "-o",
            str(tsv),
            *vl,
        ]
    )
    np.testing.assert_allclose(
        list(_scores(tsv).values()),
        np.asarray(model(alts)).ravel() - np.asarray(model(q)).ravel(),
        atol=1e-6,
    )
