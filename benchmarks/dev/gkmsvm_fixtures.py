"""Write oracle fixtures for the classic (C++) and R gkmSVM importers.

Both tools write a model as two files, PREFIX_svalpha.out (SV id <TAB> signed
alpha) and PREFIX_svseq.fa, with no bias and no kernel parameters; they read
FASTA by dropping every non-ACGT letter. This script trains models with each
tool on the -t 0 to -t 5 fixture training sets (100 bp, N stretches in a fifth;
benchmarks/dev/lsgkm_wgkm_fixtures.py), then scores the 100 and 85 bp test
sets two ways:

  - PREFIX_kernel.txt: sum_i alpha_i K(x, sv_i) with K from the tool's own
    gkmsvm_kernel (the kernel the alphas were fit with): the oracle;
  - PREFIX_classify.txt: gkmsvm_classify's output, which for these settings
    differs from that sum (its tree and simple algorithms normalize
    differently from gkmsvm_kernel); kept to document it.

Models: classic -t 0 (full filter), -t 1 (truncated, the default) and -t 2
(gkm count), l=10 k=6 d=3, trained by gkmsvm_train; R gkmSVM useTgkm=1 and 2,
trained by kernlab through gkmsvm_train. Written to tests/fixtures as
gkmsvm_{classic,r}_t{0,1,2}_{svalpha.out,svseq.fa,test100_kernel.txt,...}.

Usage:
  python benchmarks/dev/gkmsvm_fixtures.py --classic-bin DIR --r-lib DIR
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
L, K, D = 10, 6, 3
CLASSIC_TYPES = (0, 1, 2)
R_TYPES = (1, 2)
TESTS = ("test100", "test85")


def run(cmd, env=None):
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    if p.returncode:
        sys.exit(
            f"{' '.join(map(str, cmd))} failed ({p.returncode}):\n{p.stderr[-2000:]}"
        )
    return p


def read_ids(fasta):
    return [line[1:].split()[0] for line in open(fasta) if line.startswith(">")]


def kernel_scores(kernel_txt, test_fa, svseq_fa, svalpha):
    """sum_i alpha_i K(x, sv_i) for each test sequence, from a gkmsvm_kernel
    matrix over test sequences then SVs (lower triangle, one row per line)."""
    rows = [list(map(float, line.split())) for line in open(kernel_txt)]
    n = len(rows)
    mat = np.zeros((n, n))
    for i, r in enumerate(rows):
        mat[i, : len(r)] = r
    mat = mat + np.tril(mat, -1).T
    tests, svs = read_ids(test_fa), read_ids(svseq_fa)
    assert n == len(tests) + len(svs), (n, len(tests), len(svs))
    alpha = dict(line.split("\t")[:2] for line in open(svalpha) if line.strip())
    a = np.array([float(alpha[s]) for s in svs])
    return dict(zip(tests, mat[: len(tests), len(tests) :] @ a))


def write_scores(path, scores):
    with open(path, "w") as f:
        f.writelines(f"{name}\t{v:.10g}\n" for name, v in scores.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--classic-bin", required=True, help="built gkmsvm (C++) directory")
    ap.add_argument("--r-lib", required=True, help="R library with gkmSVM installed")
    args = ap.parse_args()
    cbin = Path(args.classic_bin).expanduser()
    work = Path(tempfile.mkdtemp(prefix="gkmsvm_fixtures_"))
    pos, neg = FIXTURES / "lsgkm_wgkm_pos.fa", FIXTURES / "lsgkm_wgkm_neg.fa"
    opts = ["-l", str(L), "-k", str(K), "-d", str(D)]

    def score_tests(prefix, kernel_fn, classify_fn):
        svseq = FIXTURES / f"{prefix}_svseq.fa"
        svalpha = FIXTURES / f"{prefix}_svalpha.out"
        for t in TESTS:
            fa = FIXTURES / f"lsgkm_wgkm_{t}.fa"
            kt = work / f"{prefix}_{t}_kernel.out"
            kernel_fn(fa, svseq, kt)
            write_scores(
                FIXTURES / f"{prefix}_{t}_kernel.txt",
                kernel_scores(kt, fa, svseq, svalpha),
            )
            out = FIXTURES / f"{prefix}_{t}_classify.txt"
            classify_fn(fa, svseq, svalpha, out)
            print("wrote", out.name)

    for t in CLASSIC_TYPES:
        prefix = f"gkmsvm_classic_t{t}"
        topts = ["-t", str(t), *opts]
        kern = work / f"{prefix}_train_kernel.out"
        run([cbin / "gkmsvm_kernel", *topts, pos, neg, kern])
        run([cbin / "gkmsvm_train", kern, pos, neg, work / prefix])
        for suffix in ("svalpha.out", "svseq.fa"):
            shutil.copy(work / f"{prefix}_{suffix}", FIXTURES / f"{prefix}_{suffix}")
        score_tests(
            prefix,
            lambda fa, sv, out, topts=topts: run(
                [cbin / "gkmsvm_kernel", *topts, fa, sv, out]
            ),
            lambda fa, sv, al, out, topts=topts: run(
                [cbin / "gkmsvm_classify", *topts, fa, sv, al, out]
            ),
        )

    env = {**os.environ, "R_LIBS_USER": str(Path(args.r_lib).expanduser())}

    def rscript(code):
        run(["Rscript", "-e", f"suppressMessages(library(gkmSVM)); {code}"], env)

    for t in R_TYPES:
        prefix = f"gkmsvm_r_t{t}"
        ropts = f"L={L}, K={K}, maxnmm={D}, useTgkm={t}"
        kern = work / f"{prefix}_train_kernel.out"
        rscript(f'gkmsvm_kernel("{pos}", "{neg}", "{kern}", {ropts})')
        rscript(f'gkmsvm_train("{kern}", "{pos}", "{neg}", "{work / prefix}", C=1)')
        for suffix in ("svalpha.out", "svseq.fa"):
            shutil.copy(work / f"{prefix}_{suffix}", FIXTURES / f"{prefix}_{suffix}")
        score_tests(
            prefix,
            lambda fa, sv, out, ropts=ropts: rscript(
                f'gkmsvm_kernel("{fa}", "{sv}", "{out}", {ropts})'
            ),
            lambda fa, sv, al, out, ropts=ropts: rscript(
                f'gkmsvm_classify("{fa}", svseqfile="{sv}", alphafile="{al}", '
                f'outfile="{out}", svmfnprfx=NA, {ropts})'
            ),
        )


if __name__ == "__main__":
    main()
