"""Write gkmpredict and gkmexplain oracle fixtures for -t 0, -t 1, -t 4 and -t 5.

Trains small models with gkmtrain on random 100 bp sequences (a planted
motif in the positives, N stretches in a fifth of each set) and scores test
sequences of 100 and 85 bp (even and odd window counts, which place the
center window differently), with and without N, using gkmpredict. Writes to
tests/fixtures:

  lsgkm_wgkm_t0.model.txt.gz   -t 0 -l 10 -k 6
  lsgkm_wgkm_t1.model.txt.gz   -t 1 -l 11 -k 7 -d 3
  lsgkm_wgkm_t4.model.txt.gz   -t 4 -l 11 -k 7 -d 3 -M 20 -H 10
  lsgkm_wgkm_t5.model.txt.gz   -t 5 -l 10 -k 6 -d 3 -M 50 -H 30 -g 2
  lsgkm_wgkm_test100.fa / lsgkm_wgkm_test85.fa
  lsgkm_wgkm_t{0,1,4,5}_test{100,85}.pred.txt

(-t 2/-t 3 attributions are checked on the lsgkm_mode5_* models.) --models
trains a subset; the test FASTAs come out the same every run.

With --explain-bin (a build of the kundajelab/lsgkm fork, which has
gkmexplain), also runs gkmexplain on each model and test set in modes 0
(importance), 1 (hypothetical), 4 (perturbation over pairs with up to d+1
mismatches: every pair a substitution changes, which is what
gkmexplain(mode="perturbation") computes) and, on the odd-length set, 5
(central-position mutation impact); and modes 0, 1, 4 for the -t 2/-t 3
models of the mode 5 fixtures:

  lsgkm_wgkm_t{0,1,4,5}_test{100,85}.explain{0,1,4}.txt.gz, ..._test85.explain5.txt.gz
  lsgkm_mode5_{est,rbf}.explain{0,1,4}.txt.gz, lsgkm_mode5_{est,rbf}_test{100,85}.explain*.txt.gz

Usage (a host with LS-GKM in the "lsgkm" conda env):
  python benchmarks/dev/lsgkm_wgkm_fixtures.py [--explain-bin ~/lsgkm-kundaje/bin]
  # gkmexplain outputs only, for the models and FASTAs already in tests/fixtures:
  python benchmarks/dev/lsgkm_wgkm_fixtures.py --explain-only --explain-bin ...
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
EXPLAIN_MODES = (0, 1, 4, 5)
MODELS = {
    "t0": ["-t", "0", "-l", "10", "-k", "6"],
    "t1": ["-t", "1", "-l", "11", "-k", "7", "-d", "3"],
    "t4": ["-t", "4", "-l", "11", "-k", "7", "-d", "3", "-M", "20", "-H", "10"],
    "t5": [
        "-t",
        "5",
        "-l",
        "10",
        "-k",
        "6",
        "-d",
        "3",
        "-M",
        "50",
        "-H",
        "30",
        "-g",
        "2.0",
    ],
}


def random_seq(rng, n):
    return "".join(rng.choice("ACGT") for _ in range(n))


def add_n(rng, s):
    w = rng.randint(5, 20)
    a = rng.randrange(len(s) - w)
    return s[:a] + "N" * w + s[a + w :]


def run(cmd):
    p = subprocess.run(
        cmd, capture_output=True, text=True, errors="replace", check=False
    )
    if p.returncode:
        sys.exit(f"{' '.join(cmd)} failed ({p.returncode}):\n{p.stderr[-2000:]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--conda-env", default="lsgkm")
    ap.add_argument("--bin-dir", default=None, help="LS-GKM binaries if not in the env")
    ap.add_argument(
        "--explain-bin",
        default=None,
        help="directory with the kundajelab/lsgkm fork's gkmexplain",
    )
    ap.add_argument(
        "--models",
        default=",".join(MODELS),
        help=f"comma-separated models to train (default: {','.join(MODELS)})",
    )
    ap.add_argument(
        "--explain-only",
        action="store_true",
        help="only run gkmexplain on the fixture models and FASTAs already written",
    )
    args = ap.parse_args()

    explain = None
    if args.explain_bin:
        explain = Path(args.explain_bin).expanduser() / "gkmexplain"
        if not explain.exists():
            sys.exit(f"{explain} not found")
    if args.explain_only:
        if explain is None:
            sys.exit("--explain-only needs --explain-bin")
        run_explain(explain)
        return

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
        seqs[:4] = [s[:mid] + MOTIF + s[mid + len(MOTIF) :] for s in seqs[:4]]
        seqs += [add_n(rng, s) for s in seqs[:8]]
        tests[length] = FIXTURES / f"lsgkm_wgkm_test{length}.fa"
        write_seqs(tests[length], seqs, "t")

    for name in args.models.split(","):
        opts = MODELS[name]
        prefix = workdir / name
        run(
            [
                train,
                *opts,
                "-c",
                "1.0",
                str(workdir / "pos.fa"),
                str(workdir / "neg.fa"),
                str(prefix),
            ]
        )
        model = Path(f"{prefix}.model.txt")
        with (
            open(model, "rb") as src,
            gzip.open(FIXTURES / f"lsgkm_wgkm_{name}.model.txt.gz", "wb") as dst,
        ):
            shutil.copyfileobj(src, dst)
        for length, fa in tests.items():
            run(
                [
                    predict,
                    str(fa),
                    str(model),
                    str(FIXTURES / f"lsgkm_wgkm_{name}_test{length}.pred.txt"),
                ]
            )
    if explain is not None:
        run_explain(explain)
    print("wrote:", *sorted(p.name for p in FIXTURES.glob("lsgkm_wgkm_*")), sep="\n  ")


def run_explain(explain):
    """gkmexplain on every fixture model and test FASTA, in EXPLAIN_MODES."""
    workdir = Path(tempfile.mkdtemp(prefix="lsgkm_wgkm_explain_"))
    jobs = []  # (model .txt.gz, FASTA, output prefix, modes)
    for name in MODELS:
        if not (FIXTURES / f"lsgkm_wgkm_{name}.model.txt.gz").exists():
            continue
        for fa in sorted(FIXTURES.glob("lsgkm_wgkm_test*.fa")):
            length = int(fa.stem.removeprefix("lsgkm_wgkm_test"))
            # mode 5 explains the central base, so needs an odd length
            modes = [m for m in EXPLAIN_MODES if m != 5 or length % 2]
            jobs.append(
                (
                    f"lsgkm_wgkm_{name}.model.txt.gz",
                    fa,
                    f"lsgkm_wgkm_{name}_test{length}",
                    modes,
                )
            )
    for kind in ("est", "rbf"):  # mode 5 outputs are already lsgkm_mode5_{kind}.txt
        jobs.append(
            (
                f"lsgkm_mode5_{kind}.model.txt.gz",
                FIXTURES / "lsgkm_mode5_test51.fa",
                f"lsgkm_mode5_{kind}",
                [m for m in EXPLAIN_MODES if m != 5],
            )
        )
        # The same -t 2/-t 3 models on the sets with N stretches and both window parities
        for fa in sorted(FIXTURES.glob("lsgkm_wgkm_test*.fa")):
            length = int(fa.stem.removeprefix("lsgkm_wgkm_test"))
            jobs.append(
                (
                    f"lsgkm_mode5_{kind}.model.txt.gz",
                    fa,
                    f"lsgkm_mode5_{kind}_test{length}",
                    [m for m in EXPLAIN_MODES if m != 5 or length % 2],
                )
            )
    for model_gz, fa, prefix, modes in jobs:
        # Plain text, without the "norc 0" line the fork rejects (RC on is its default)
        model = workdir / model_gz.removesuffix(".gz")
        with gzip.open(FIXTURES / model_gz, "rt") as src:
            model.write_text("".join(line for line in src if line.strip() != "norc 0"))
        for mode in modes:
            raw = workdir / f"{prefix}.explain{mode}.txt"
            run([str(explain), "-m", str(mode), str(fa), str(model), str(raw)])
            out = FIXTURES / f"{raw.name}.gz"
            # mtime=0: unchanged outputs give byte-identical files
            out.write_bytes(gzip.compress(raw.read_bytes(), mtime=0))
            print("wrote", out.name)


if __name__ == "__main__":
    main()
