"""How does LS-GKM treat N? Compare gkmpredict with gkmsvm-lite on inputs with N.

LS-GKM's source encodes any non-ACGT base as A before taking the reverse
complement, so N should behave as A on the forward strand and T on the reverse
one: the same as replacing every N with A. For each model this scores test
sequences with and without N three ways:

  gkmpredict   LS-GKM
  native       load_lsgkm_model + model(), N as gkmsvm-lite treats it
  n_as_a       the same with every N replaced by A in the SVs and queries

and reports the largest |score - gkmpredict| for sequences with N and
without. native ~ gkmpredict means gkmsvm-lite agrees with LS-GKM on N;
n_as_a ~ gkmpredict but native not means the kernel treats N differently.

Models: ENCFF579AOX (-t 2, 4 SVs with N), and small models trained with
gkmtrain -t 2/3/4/5 on random sequences with a planted motif, N stretches in a
fifth of them.

Usage (a host with LS-GKM in the "lsgkm" conda env):
  python benchmarks/dev/lsgkm_n_check.py
  python benchmarks/dev/lsgkm_n_check.py --kernels 4 5 --keep
"""

from __future__ import annotations

import argparse
import gzip
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import lsgkm_binary, write_seqs

from gkmsvm.codec import one_hot_encode
from gkmsvm.importers.lsgkm import load_lsgkm_model

ROOT = Path(__file__).resolve().parents[2]
ENCODE = ROOT / "tests" / "fixtures" / "encode_ENCFF579AOX.model.txt.gz"
MOTIF = "TGACTCAGCA"


def random_seq(rng, n):
    return "".join(rng.choice("ACGT") for _ in range(n))


def add_n(rng, s, block=True):
    """s with a stretch of 5-30 Ns (block) or ~5% scattered Ns."""
    s = list(s)
    if block:
        w = rng.randint(5, 30)
        a = rng.randrange(len(s) - w)
        s[a : a + w] = "N" * w
    else:
        for i in rng.sample(range(len(s)), max(1, len(s) // 20)):
            s[i] = "N"
    return "".join(s)


def test_seqs(rng, n, length):
    """n sequences without N, then n with a block of N, then n with scattered N."""
    clean = [random_seq(rng, length) for _ in range(n)]
    plant = [
        s[: length // 2] + MOTIF + s[length // 2 + len(MOTIF) :]
        for s in clean[: n // 2]
    ]
    clean = plant + clean[n // 2 :]
    return (
        clean
        + [add_n(rng, s) for s in clean]
        + [add_n(rng, s, block=False) for s in clean]
    )


def run(cmd):
    p = subprocess.run(
        cmd, capture_output=True, text=True, errors="replace", check=False
    )
    if p.returncode:
        sys.exit(f"{' '.join(cmd)} failed ({p.returncode}):\n{p.stderr[-2000:]}")


def gkmpredict(binary, model, seqs, workdir, threads):
    fa, out = workdir / "test.fa", workdir / "test.pred.txt"
    write_seqs(fa, seqs, "t")
    cmd = (
        [binary]
        + (["-T", str(threads)] if threads > 1 else [])
        + [str(fa), str(model), str(out)]
    )
    run(cmd)
    with open(out) as f:
        scores = {name: float(score) for name, score, *_ in map(str.split, f)}
    return np.array([scores[f"t{i}"] for i in range(len(seqs))])


def ours(model, seqs, n_as_a):
    if n_as_a:
        sv = model.support_sequences.copy()
        n_cols = sv.sum(axis=1) == 0  # [S, L]
        sv[:, 0, :][n_cols] = 1  # N -> A
        model = type(model)(
            sv, model.coefficients, model.bias, model.kernel_type, model._kernel_params
        )
        seqs = [s.replace("N", "A") for s in seqs]
    x = np.stack([one_hot_encode(s, allow_n=True) for s in seqs])
    return np.asarray(model(x, batch_size=256)).ravel()


def compare(name, model_path, seqs, n_clean, binary, workdir, threads):
    ref = gkmpredict(binary, model_path, seqs, workdir, threads)
    model = load_lsgkm_model(model_path)
    n_sv_n = int((model.support_sequences.sum(axis=1) == 0).any(axis=1).sum())
    for label, n_as_a in (("native", False), ("n_as_a", True)):
        diff = np.abs(ours(model, seqs, n_as_a) - ref)
        print(
            f"{name:14} {n_sv_n:>5} {label:7} {diff[:n_clean].max():>12.2e} "
            f"{diff[n_clean:].max():>12.2e} {np.abs(ref).max():>10.3f}",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--kernels",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5],
        help="gkmtrain -t values to train and compare",
    )
    ap.add_argument("--no-encode", action="store_true", help="skip ENCFF579AOX")
    ap.add_argument("--conda-env", default="lsgkm")
    ap.add_argument("--bin-dir", default=None, help="LS-GKM binaries if not in the env")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--keep", action="store_true", help="keep the work directory")
    args = ap.parse_args()

    train = lsgkm_binary("gkmtrain", args.conda_env, args.bin_dir)
    predict = lsgkm_binary("gkmpredict", args.conda_env, args.bin_dir)
    if not (train and predict):
        sys.exit("gkmtrain/gkmpredict not found (--conda-env / --bin-dir)")
    workdir = Path(tempfile.mkdtemp(prefix="lsgkm_n_"))
    rng = random.Random(0)

    print(
        f"{'model':14} {'N SVs':>5} {'path':7} {'max|d| no N':>12} {'max|d| N':>12} "
        f"{'max|score|':>10}"
    )
    if not args.no_encode:
        n = 20
        encode = workdir / "ENCFF579AOX.model.txt"  # gkmpredict doesn't read gzip
        with gzip.open(ENCODE, "rb") as src, open(encode, "wb") as dst:
            shutil.copyfileobj(src, dst)
        compare(
            "ENCFF579AOX",
            encode,
            test_seqs(rng, n, 300),
            n,
            predict,
            workdir,
            args.threads,
        )

    pos = [random_seq(rng, 100) for _ in range(300)]
    pos = [s[:45] + MOTIF + s[55:] for s in pos]
    neg = [random_seq(rng, 100) for _ in range(300)]
    pos = [add_n(rng, s) if i % 5 == 0 else s for i, s in enumerate(pos)]
    neg = [add_n(rng, s) if i % 5 == 0 else s for i, s in enumerate(neg)]
    write_seqs(workdir / "pos.fa", pos, "p")
    write_seqs(workdir / "neg.fa", neg, "n")
    n = 30
    seqs = test_seqs(rng, n, 100)
    for t in args.kernels:
        prefix = workdir / f"t{t}"
        cmd = [
            train,
            "-t",
            str(t),
            "-l",
            "11",
            "-k",
            "7",
            "-d",
            "3",
            "-g",
            "1.0",
            "-M",
            "50",
            "-H",
            "50",
            "-c",
            "1.0",
            str(workdir / "pos.fa"),
            str(workdir / "neg.fa"),
            str(prefix),
        ]
        run(cmd)
        compare(
            f"gkmtrain -t {t}",
            Path(f"{prefix}.model.txt"),
            seqs,
            n,
            predict,
            workdir,
            args.threads,
        )
    if args.keep:
        print(f"work directory: {workdir}")


if __name__ == "__main__":
    main()
