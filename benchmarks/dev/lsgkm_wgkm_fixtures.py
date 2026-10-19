"""Write gkmpredict oracle fixtures for -t 4 (wgkm) and -t 5 (wgkmrbf).

Trains two small models with gkmtrain on random 100 bp sequences (a planted
motif in the positives, N stretches in a fifth of each set) and scores test
sequences of 100 and 85 bp (even and odd window counts, which place the
center window differently), with and without N, using gkmpredict. Writes to
tests/fixtures:

  lsgkm_wgkm_t4.model.txt.gz   -t 4 -l 11 -k 7 -d 3 -M 20 -H 10
  lsgkm_wgkm_t5.model.txt.gz   -t 5 -l 10 -k 6 -d 3 -M 50 -H 30 -g 2
  lsgkm_wgkm_test100.fa / lsgkm_wgkm_test85.fa
  lsgkm_wgkm_t{4,5}_test{100,85}.pred.txt

Usage (a host with LS-GKM in the "lsgkm" conda env):
  python benchmarks/dev/lsgkm_wgkm_fixtures.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import lsgkm_binary, write_seqs

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
MOTIF = "TGACTCAGCA"
MODELS = {
    "t4": ["-t", "4", "-l", "11", "-k", "7", "-d", "3", "-M", "20", "-H", "10"],
    "t5": ["-t", "5", "-l", "10", "-k", "6", "-d", "3", "-M", "50", "-H", "30", "-g", "2.0"],
}


def random_seq(rng, n):
    return "".join(rng.choice("ACGT") for _ in range(n))


def add_n(rng, s):
    w = rng.randint(5, 20)
    a = rng.randrange(len(s) - w)
    return s[:a] + "N" * w + s[a + w:]


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", check=False)
    if p.returncode:
        sys.exit(f"{' '.join(cmd)} failed ({p.returncode}):\n{p.stderr[-2000:]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--conda-env", default="lsgkm")
    ap.add_argument("--bin-dir", default=None, help="LS-GKM binaries if not in the env")
    args = ap.parse_args()

    train = lsgkm_binary("gkmtrain", args.conda_env, args.bin_dir)
    predict = lsgkm_binary("gkmpredict", args.conda_env, args.bin_dir)
    if not (train and predict):
        sys.exit("gkmtrain/gkmpredict not found (--conda-env / --bin-dir)")
    workdir = Path(tempfile.mkdtemp(prefix="lsgkm_wgkm_"))
    rng = random.Random(0)

    pos = [random_seq(rng, 100) for _ in range(80)]
    pos = [s[:45] + MOTIF + s[55:] for s in pos]
    neg = [random_seq(rng, 100) for _ in range(80)]
    pos = [add_n(rng, s) if i % 5 == 0 else s for i, s in enumerate(pos)]
    neg = [add_n(rng, s) if i % 5 == 0 else s for i, s in enumerate(neg)]
    write_seqs(workdir / "pos.fa", pos, "p")
    write_seqs(workdir / "neg.fa", neg, "n")

    tests = {}
    for length in (100, 85):
        seqs = [random_seq(rng, length) for _ in range(12)]
        mid = length // 2 - len(MOTIF) // 2
        seqs[:4] = [s[:mid] + MOTIF + s[mid + len(MOTIF):] for s in seqs[:4]]
        seqs += [add_n(rng, s) for s in seqs[:8]]
        tests[length] = FIXTURES / f"lsgkm_wgkm_test{length}.fa"
        write_seqs(tests[length], seqs, "t")

    for name, opts in MODELS.items():
        prefix = workdir / name
        run([train, *opts, "-c", "1.0", str(workdir / "pos.fa"), str(workdir / "neg.fa"),
             str(prefix)])
        model = Path(f"{prefix}.model.txt")
        with open(model, "rb") as src, gzip.open(FIXTURES / f"lsgkm_wgkm_{name}.model.txt.gz",
                                                 "wb") as dst:
            shutil.copyfileobj(src, dst)
        for length, fa in tests.items():
            run([predict, str(fa), str(model),
                 str(FIXTURES / f"lsgkm_wgkm_{name}_test{length}.pred.txt")])
    print("wrote:", *sorted(p.name for p in FIXTURES.glob("lsgkm_wgkm_*")), sep="\n  ")


if __name__ == "__main__":
    main()
