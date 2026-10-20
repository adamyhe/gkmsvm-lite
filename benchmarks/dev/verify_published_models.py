"""Check the importers on published models against the tools that wrote them.

  - LS-GKM: ENCODE gkm-SVM models (ENCFF579AOX, ENCFF701NCI), the GkmExplain
    paper's dsQTL -t 2 and -t 3 models and its NANOG model, scored on 300 bp
    random sequences (a third with N stretches) and the NANOG test set,
    against gkmpredict.
  - Classic gkmSVM: the models bundled with the C++ tool (test/test_svmtrain,
    test_svmtrain2), against sum alpha K from its gkmsvm_kernel on the bundled
    test sets (gkmsvm_classify disagrees with gkmsvm_kernel by up to ~2%).
  - deltaSVM: the Lee 2015 supplementary weight tables (GM12878, K562, HepG2),
    on 500 random 19 bp SNVs and deltasvm.pl's own example, against
    deltasvm.pl.

Prints the largest difference per model; each should be at the tool's printed
precision (~5e-6 for gkmpredict, ~1e-6 for the others). Not part of a figure.

Usage:
  python benchmarks/dev/verify_published_models.py --gkmpredict PATH \\
      --classic-bin DIR [--workdir DIR]
"""

from __future__ import annotations

import argparse
import gzip
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gkmsvm_fixtures import kernel_scores

from gkmsvm.fasta import read_fasta
from gkmsvm.importers.classic import load_classic_model
from gkmsvm.importers.deltasvm import load_deltasvm_model
from gkmsvm.importers.lsgkm import load_lsgkm_model

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
GKMEXPLAIN = (
    "https://raw.githubusercontent.com/kundajelab/gkmexplain/master/dsQTL/"
    "gm12878_sequence_sets/{}"
)
LSGKM_MODELS = {
    "ENCFF579AOX (ENCODE)": FIXTURES / "encode_ENCFF579AOX.model.txt.gz",
    "ENCFF701NCI (ENCODE)": "https://www.encodeproject.org/files/ENCFF701NCI/"
    "@@download/ENCFF701NCI.txt.gz",
    "dsQTL -t 2 negset1": GKMEXPLAIN.format(
        "gkmsvm_t2_l10_k6_d3_t16_negset1.model.txt"
    ),
    "dsQTL -t 3 negset1": GKMEXPLAIN.format(
        "gkmsvm_t3_l10_k6_d3_c10_g2_t16_negset1.model.txt"
    ),
    "NANOG (gkmexplain)": FIXTURES / "nanog_reference.model.txt.gz",
}
DELTASVM = "https://www.beerlab.org/deltasvm/downloads/{}"
WEIGHTS = ("gm12878", "k562", "hepg2")


def fetch(src, work: Path) -> Path:
    if isinstance(src, Path):
        return src
    path = work / src.rsplit("/", 1)[1]
    if not path.exists():
        urlretrieve(src, path)
    return path


def plain(path: Path, work: Path) -> Path:
    """path, decompressed into work if gzipped (gkmpredict reads plain text)."""
    if path.suffix != ".gz":
        return path
    out = work / path.name[:-3]
    out.write_bytes(gzip.open(path).read())
    return out


def lsgkm(gkmpredict: str, work: Path):
    rng = random.Random(7)
    seqs = []
    for i in range(150):
        s = "".join(rng.choice("ACGT") for _ in range(300))
        if i % 3 == 0:
            a = rng.randrange(250)
            s = (s[:a] + "N" * rng.randint(5, 40) + s[a + 40 :])[:300]
        seqs.append(s)
    for f in ("nanog_pos_test.fa.gz", "nanog_neg_test.fa.gz"):
        seqs += [
            line.strip()
            for line in gzip.open(FIXTURES / f, "rt")
            if not line.startswith(">")
        ]
    fa = work / "queries.fa"
    fa.write_text("".join(f">q{i}\n{s}\n" for i, s in enumerate(seqs)))
    for name, src in LSGKM_MODELS.items():
        path = fetch(src, work)
        out = work / f"{path.name}.pred"
        subprocess.run(
            [gkmpredict, str(fa), str(plain(path, work)), str(out)],
            capture_output=True,
            check=True,
        )
        expected = {line.split()[0]: float(line.split()[1]) for line in open(out)}
        model = load_lsgkm_model(path)
        got = np.asarray(model(seqs)).ravel()
        e = np.array([expected[f"q{i}"] for i in range(len(seqs))])
        print(
            f"{name:22s} {model.kernel_type:12s} {model.num_support_vectors:6d} SVs  "
            f"max|ours - gkmpredict| {np.abs(got - e).max():.1e}"
        )


def classic(cbin: Path, work: Path):
    test = cbin / "test"
    for prefix in ("test_svmtrain", "test_svmtrain2"):
        alpha, svseq = test / f"{prefix}_svalpha.out", test / f"{prefix}_svseq.fa"
        model = load_classic_model(alpha)  # trained with the tools' defaults
        for kind in ("positives", "negatives"):
            fa = test / f"test_{kind}.fa"
            out = work / f"{prefix}_{kind}.kernel"
            subprocess.run(
                [str(cbin / "gkmsvm_kernel"), "-d", "3", str(fa), str(svseq), str(out)],
                capture_output=True,
                check=True,
            )
            with np.errstate(all="ignore"):  # Accelerate's spurious matmul warnings
                expected = kernel_scores(out, fa, svseq, alpha)
            records = read_fasta(str(fa))
            got = np.asarray(model([s for _, s in records])).ravel()
            e = np.array([expected[n.split()[0]] for n, _ in records])
            print(
                f"classic {prefix:15s} {kind:9s} max|ours - gkmsvm_kernel| "
                f"{np.abs(got - e).max():.1e}"
            )


def deltasvm(work: Path):
    script = fetch(DELTASVM.format("deltasvm_script.tar.gz"), work)
    subprocess.run(["tar", "-xzf", str(script), "-C", str(work)], check=True)
    pl = work / "deltasvm_script"
    rng = random.Random(11)
    refs, alts = [], []
    for _ in range(500):
        s = "".join(rng.choice("ACGT") for _ in range(19))
        b = rng.choice([c for c in "ACGT" if c != s[9]])
        refs.append(s)
        alts.append(s[:9] + b + s[10:])
    sets = {"random": (work / "ref.fa", work / "alt.fa")}
    sets["random"][0].write_text("".join(f">v{i}\n{s}\n" for i, s in enumerate(refs)))
    sets["random"][1].write_text("".join(f">v{i}\n{s}\n" for i, s in enumerate(alts)))
    sets["example"] = (pl / "ref_seq_ex.fa", pl / "alt_seq_ex.fa")
    for cell in WEIGHTS:
        weights = fetch(DELTASVM.format(f"SupplementaryTable_{cell}weights.txt"), work)
        model = load_deltasvm_model(weights, 10)
        for name, (ref, alt) in sets.items():
            out = work / f"{cell}_{name}.txt"
            subprocess.run(
                [
                    "perl",
                    str(pl / "deltasvm.pl"),
                    str(ref),
                    str(alt),
                    str(weights),
                    str(out),
                ],
                capture_output=True,
                check=True,
            )
            expected = [float(line.split()[1]) for line in open(out)]
            r = [s for _, s in read_fasta(str(ref))]
            a = [s for _, s in read_fasta(str(alt))]
            got = np.asarray(model(a)).ravel() - np.asarray(model(r)).ravel()
            print(
                f"deltaSVM {cell:8s} {name:8s} max|ours - deltasvm.pl| "
                f"{np.abs(got - expected).max():.1e}"
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gkmpredict", required=True, help="LS-GKM gkmpredict binary")
    ap.add_argument("--classic-bin", required=True, help="built gkmsvm (C++) directory")
    ap.add_argument("--workdir", default=None, help="downloads and outputs")
    args = ap.parse_args()
    work = Path(args.workdir or tempfile.mkdtemp(prefix="verify_models_"))
    work.mkdir(parents=True, exist_ok=True)
    lsgkm(args.gkmpredict, work)
    classic(Path(args.classic_bin).expanduser(), work)
    deltasvm(work)


if __name__ == "__main__":
    main()
