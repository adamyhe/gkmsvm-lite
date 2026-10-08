"""Benchmark: dsQTL variant effect prediction (Shrikumar et al. 2019, Fig. 8).

Replicates the GkmExplain dsQTL evaluation
(github.com/kundajelab/gkmexplain/tree/master/dsQTL). Per model (one per
negative set), with per-model AUPRC and scores summed across models:
  - deltaSVM: l-mer weights = gkmpredict on each 10-mer (19bp Beer lab seqs)
  - ISM: score(alt) - score(ref) (51bp hg18 context, as in the paper)
  - GkmExplain: mutation impact score, lsgkm ``gkmexplain -m 5`` (51bp)

Parameter sets:
  - l10k6, l11k7: -t 2, trained here
  - l10k6_rbf, l11k7_rbf: -t 3 gkmrbf, gamma=2, C=10, trained here
  - paper_t2, paper_t3: the paper's published lsgkm models (downloaded)

The paper's per-negset AUPRCs are printed for the l=10 k=6 sets.

Pass --replicate-published to also compare against Lee 2015 published
deltaSVM scores (downloads hg19 genome, ~900 MB compressed).

All data auto-downloaded on first run (hg18 genome, ~940 MB compressed, for
51bp context). Trained models cached to benchmarks/models/.

Usage:
  python benchmarks/bench_dsqtl.py --device cuda
  python benchmarks/bench_dsqtl.py --params paper_t3     # paper's models
  python benchmarks/bench_dsqtl.py --params l10k6_rbf    # retrained gkmrbf
  python benchmarks/bench_dsqtl.py --n-negsets 1 --params l10k6  # quick
  python benchmarks/bench_dsqtl.py --replicate-published
  python benchmarks/bench_dsqtl.py --force-train
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import gzip
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

SEQ_DIR = Path(__file__).parent.parent / "examples" / "data" / "gm12878_sequence_sets"
DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"

SEQ_URL = "https://beerlab.org/deltasvm/downloads/gm12878_sequence_sets.tar.gz"
GEO_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE31nnn/GSE31388/suppl/GSE31388_dsQtlTable.txt.gz"
WEIGHTS_URL = (
    "https://beerlab.org/deltasvm/downloads/SupplementaryTable_gm12878weights.txt"
)
LEE2015_URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.1038%2Fng.3331/MediaObjects/41588_2015_BFng3331_MOESM26_ESM.xlsx"
)
HG19_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz"
HG18_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg18/bigZips/hg18.fa.gz"
PAPER_MODEL_URL = (
    "https://raw.githubusercontent.com/kundajelab/gkmexplain/master/dsQTL/"
    "gm12878_sequence_sets/gkmsvm_{tag}_negset{i}.model.txt"
)

PARAM_SETS = {
    "l10k6": {"l": 10, "k": 6, "d": 3, "kernel_type": "estimated"},
    "l11k7": {"l": 11, "k": 7, "d": 3, "kernel_type": "estimated"},
    "l10k6_rbf": {"l": 10, "k": 6, "d": 3, "kernel_type": "rbf", "gamma": 2.0, "C": 10.0},
    "l11k7_rbf": {"l": 11, "k": 7, "d": 3, "kernel_type": "rbf", "gamma": 2.0, "C": 10.0},
    "paper_t2": {"l": 10, "k": 6, "d": 3, "kernel_type": "estimated",
                 "paper": "t2_l10_k6_d3_t16"},
    "paper_t3": {"l": 10, "k": 6, "d": 3, "kernel_type": "rbf", "gamma": 2.0, "C": 10.0,
                 "paper": "t3_l10_k6_d3_c10_g2_t16"},
}

# Shrikumar et al. 2019 per-negset AUPRCs (gkmexplain repo, summarize_auprcs.sh).
PAPER_AUPRC = {
    "GkmExplain": [0.18905, 0.19101, 0.18523, 0.18698, 0.19477],
    "ISM": [0.18808, 0.18968, 0.18433, 0.18569, 0.19432],
    "deltaSVM-gkmrbf": [0.18287, 0.18649, 0.18001, 0.18028, 0.18733],
    "deltaSVM-gkm": [0.17918, 0.18565, 0.17697, 0.17943, 0.18483],
}
PAPER_REF_COLUMNS = {
    "rbf": {"deltaSVM": "deltaSVM-gkmrbf", "ISM": "ISM", "GkmExplain": "GkmExplain"},
    "estimated": {"deltaSVM": "deltaSVM-gkm"},
}

FLANK = 9
CONTEXT_HALF = 25


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_data(replicate_published: bool = False):
    """Download all required data files."""
    import tarfile
    from urllib.request import urlretrieve

    # Beer lab sequences (training + pre-extracted test variants)
    if not SEQ_DIR.exists():
        SEQ_DIR.parent.mkdir(parents=True, exist_ok=True)
        tarball = SEQ_DIR.parent / "gm12878_sequence_sets.tar.gz"
        if not tarball.exists():
            print("Downloading Beer lab sequence sets (14 MB)...")
            urlretrieve(SEQ_URL, tarball)
        print("Extracting...")
        with tarfile.open(tarball) as tar:
            tar.extractall(SEQ_DIR.parent, filter="data")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _download_genome(HG18_URL, "hg18")

    # GEO effect sizes
    geo_path = DATA_DIR / "GSE31388_dsQtlTable.txt.gz"
    if not geo_path.exists():
        print("Downloading GEO GSE31388 dsQTL effect sizes...")
        urlretrieve(GEO_URL, geo_path)

    if not replicate_published:
        return

    # Published deltaSVM weights
    weights_path = DATA_DIR / "gm12878_deltasvm_weights.txt"
    if not weights_path.exists():
        print("Downloading Beer lab deltaSVM weights (16 MB)...")
        urlretrieve(WEIGHTS_URL, weights_path)

    # Lee 2015 variant table
    tsv_path = DATA_DIR / "dsqtl_lee2015.tsv"
    if not tsv_path.exists():
        import openpyxl

        xlsx_path = DATA_DIR / "lee2015_supp.xlsx"
        if not xlsx_path.exists():
            print("Downloading Lee 2015 Supplementary Table 1...")
            urlretrieve(LEE2015_URL, xlsx_path)
        print("Converting XLSX to TSV...")
        wb = openpyxl.load_workbook(xlsx_path, read_only=True)
        ws = wb["SuppTable1"]
        with open(tsv_path, "w") as f:
            f.writelines(
                "\t".join(str(v) if v is not None else "" for v in row) + "\n"
                for row in ws.iter_rows(values_only=True)
            )
        wb.close()

    _download_genome(HG19_URL, "hg19")


def _download_genome(url: str, name: str):
    import shutil
    from urllib.request import urlretrieve

    fasta_path = DATA_DIR / f"{name}.fa"
    if fasta_path.exists():
        return
    gz_path = DATA_DIR / f"{name}.fa.gz"
    if not gz_path.exists():
        print(f"Downloading {name} genome (~900 MB compressed)...")
        urlretrieve(url, gz_path)
    print(f"Decompressing {name}.fa.gz...")
    with gzip.open(gz_path, "rb") as f_in, open(fasta_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_training_data(n_negsets: int = 5):
    from gkmsvm import read_fasta

    pos_seqs = [seq for _, seq in read_fasta(str(SEQ_DIR / "gm12878_shared.fa"))]
    print(f"Training positives: {len(pos_seqs):,} sequences, {len(pos_seqs[0])}bp")

    neg_sets = []
    for i in range(1, n_negsets + 1):
        negs = [
            seq
            for _, seq in read_fasta(str(SEQ_DIR / f"nullseqs_gm12878_shared.{i}.1.fa"))
        ]
        neg_sets.append(negs)
        print(f"  Negative set {i}: {len(negs):,} sequences")

    return pos_seqs, neg_sets


def _context_seqs(major, minor, genome):
    """51bp hg18 context with the Beer lab allele at the center.

    Mirrors prep_fasta_for_ism_and_gkmexplain.sh in the gkmexplain repo.
    """
    ref_seqs, alt_seqs = [], []
    for (name, maj), (_, mnr) in zip(major, minor):
        chrom, rng = name.split(":")
        center = int(rng.split("-")[0]) + FLANK
        ctx = str(genome[chrom][center - CONTEXT_HALF - 1:center + CONTEXT_HALF]).upper()
        left, right = ctx[:CONTEXT_HALF], ctx[CONTEXT_HALF + 1:]
        ref_seqs.append(left + maj[FLANK].upper() + right)
        alt_seqs.append(left + mnr[FLANK].upper() + right)
    return ref_seqs, alt_seqs


def load_test_variants():
    """Returns (19bp ref/alt for deltaSVM, 51bp ref/alt for ISM/GkmExplain)."""
    import pyfaidx

    from gkmsvm import one_hot_encode, read_fasta

    pos_major = list(read_fasta(str(SEQ_DIR / "dsqtl_test_pos.major.fa")))
    pos_minor = list(read_fasta(str(SEQ_DIR / "dsqtl_test_pos.minor.fa")))
    neg_major = list(read_fasta(str(SEQ_DIR / "dsqtl_test_neg.major.fa")))
    neg_minor = list(read_fasta(str(SEQ_DIR / "dsqtl_test_neg.minor.fa")))

    major, minor = pos_major + neg_major, pos_minor + neg_minor
    labels = np.array([1] * len(pos_major) + [0] * len(neg_major))

    def _enc(seqs):
        return np.stack([one_hot_encode(s, dtype=np.float64, allow_n=True) for s in seqs])

    X19 = (_enc([s for _, s in major]), _enc([s for _, s in minor]))
    genome = pyfaidx.Fasta(str(DATA_DIR / "hg18.fa"))
    ref51, alt51 = _context_seqs(major, minor, genome)
    genome.close()
    X51 = (_enc(ref51), _enc(alt51))

    print(
        f"Test variants: {len(labels)} ({labels.sum()} sig + "
        f"{(labels == 0).sum()} ctrl), {X19[0].shape[2]}bp (deltaSVM), "
        f"{X51[0].shape[2]}bp (ISM/GkmExplain)"
    )

    effect_sizes = _load_effect_sizes(pos_major)
    if effect_sizes is not None:
        effect_sizes = np.concatenate([effect_sizes, np.full(len(neg_major), np.nan)])

    return X19, X51, labels, effect_sizes


def _load_effect_sizes(pos_major: list[tuple[str, str]]) -> np.ndarray | None:
    import pandas as pd

    geo_path = DATA_DIR / "GSE31388_dsQtlTable.txt.gz"
    if not geo_path.exists():
        print("  Effect sizes not found, skipping regression metrics")
        return None

    geo = pd.read_csv(geo_path, sep="\t", compression="gzip")
    geo = geo.sort_values("Pr(>|t|)").drop_duplicates(
        subset=["Chr", "SNP"], keep="first"
    )

    n_pos = len(pos_major)
    effects = np.full(n_pos, np.nan)
    for idx, (name, _) in enumerate(pos_major):
        parts = name.split(":")
        if len(parts) != 2:
            continue
        chrom = parts[0]
        coords = parts[1].split("-")
        if len(coords) != 2:
            continue
        try:
            snp_pos = int(coords[0]) + 9
            match = geo[(geo["Chr"] == chrom) & (geo["SNP"] == snp_pos)]
            if len(match) > 0:
                effects[idx] = float(match.iloc[0]["Estimate"])
        except ValueError:
            pass

    n_matched = (~np.isnan(effects)).sum()
    print(f"  Effect sizes matched: {n_matched}/{n_pos}")
    return effects


# ---------------------------------------------------------------------------
# Model training with caching
# ---------------------------------------------------------------------------


def _model_path(pk: str, neg_idx: int) -> Path:
    return MODEL_DIR / f"gm12878_{pk}_neg{neg_idx}.npz"


def _dsvm_path(pk: str, neg_idx: int) -> Path:
    return MODEL_DIR / f"gm12878_{pk}_neg{neg_idx}_dsvm_qn.npz"


def _load_paper_model(tag: str, neg_idx: int):
    from urllib.request import urlretrieve

    from gkmsvm.importers.lsgkm import load_lsgkm_model

    path = MODEL_DIR / f"gkmsvm_{tag}_negset{neg_idx}.model.txt"
    if not path.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading {path.name}...")
        urlretrieve(PAPER_MODEL_URL.format(tag=tag, i=neg_idx), path)
    return load_lsgkm_model(str(path), dtype=np.float64)


def train_models(
    pos_seqs,
    neg_sets,
    params: dict,
    device: str,
    pk: str,
    force_train: bool = False,
    verbose: bool = False,
):
    from gkmsvm import train_gkmsvm
    from gkmsvm.backend import to_cpu
    from gkmsvm.serialization import (
        load_deltasvm_npz,
        load_model,
        save_deltasvm_npz,
        save_npz,
    )

    C = params.get("C", 1.0)
    train_kwargs = {}
    if "gamma" in params:
        train_kwargs["gamma"] = params["gamma"]

    models = []
    dsvms = []

    for i, neg_seqs in enumerate(neg_sets):
        cached = _model_path(pk, i + 1)
        dsvm_cached = _dsvm_path(pk, i + 1)

        if "paper" in params:
            print(f"\n── Model {i + 1}/{len(neg_sets)} ({pk}) ── [published]")
            m = _load_paper_model(params["paper"], i + 1)
        elif cached.exists() and not force_train:
            print(
                f"\n── Model {i + 1}/{len(neg_sets)} "
                f"(l={params['l']} k={params['k']}) ── [cached]"
            )
            m = load_model(str(cached))
        else:
            print(
                f"\n── Model {i + 1}/{len(neg_sets)} "
                f"(l={params['l']} k={params['k']}) ──"
            )
            t0 = time.time()
            m = train_gkmsvm(
                pos_seqs,
                neg_seqs,
                kernel_type=params["kernel_type"],
                l=params["l"],
                k=params["k"],
                d=params["d"],
                C=C,
                solver="auto",
                device=device,
                verbose=verbose,
                **train_kwargs,
            )
            print(f"  {m.num_support_vectors} SVs, {time.time() - t0:.1f}s")
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            m.cpu()
            save_npz(m, str(cached))
            print(f"  Saved {cached}")

        if dsvm_cached.exists() and not force_train:
            print("  DeltaSVM weights: [cached]")
            d = load_deltasvm_npz(str(dsvm_cached))
        else:
            if device == "cuda":
                m.cuda()
            t0 = time.time()
            d = m.to_deltasvm(device=device, query_norm=True, verbose=verbose)
            print(f"  DeltaSVM conversion: {time.time() - t0:.1f}s")
            m.cpu()
            d.cpu()
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            save_deltasvm_npz(d, str(dsvm_cached))
            print(f"  Saved {dsvm_cached}")

        models.append(m)
        dsvms.append(d)

    avg_weights = np.mean([to_cpu(d.weights) for d in dsvms], axis=0)
    avg_dsvm = copy.deepcopy(dsvms[0])
    avg_dsvm.cpu()
    avg_dsvm.weights = avg_weights
    print(f"\nAveraged {len(dsvms)} deltaSVM weight tables")

    return models, dsvms, avg_dsvm


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_deltasvm(dsvm, X_ref, X_alt):
    from gkmsvm.backend import to_cpu

    t0 = time.time()
    scores = to_cpu(dsvm.score_variants(X_ref, X_alt).flatten())
    print(f"  DeltaSVM: {len(scores)} variants in {time.time() - t0:.2f}s")
    return scores


def score_vep(
    model, X_ref, X_alt, device: str, method: str, batch_size: int = 64,
    verbose: bool = False,
):
    from gkmsvm.backend import to_cpu

    if device == "cuda":
        model.cuda()
    elif device == "mlx":
        model.mlx()

    N = len(X_ref)
    t0 = time.time()
    scores = to_cpu(
        model.score_variants(
            X_ref, X_alt, method=method, batch_size=batch_size, verbose=verbose,
        )
    ).flatten()
    elapsed = time.time() - t0

    model.cpu()
    label = "ISM" if method == "kernel" else "GkmExplain"
    print(f"  {label}: {N} variants in {elapsed:.1f}s ({N / elapsed:.0f} variants/s)")
    return scores


# ---------------------------------------------------------------------------
# Published deltaSVM replication
# ---------------------------------------------------------------------------


def load_published_variants(tsv_path: Path, fasta_path: Path) -> dict:
    """Load Lee 2015 variants and extract flanking sequences from hg19."""
    import pyfaidx

    genome = pyfaidx.Fasta(str(fasta_path))
    chroms, positions, ref_seqs, alt_seqs = [], [], [], []
    labels, published_scores, snp_names = [], [], []

    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            chrom = row["chrom_hg19"]
            pos = int(row["pos_hg19"])
            a1, a2 = row["allele1"], row["allele2"]

            start = pos - 1 - FLANK
            end = pos + FLANK
            if start < 0 or chrom not in genome:
                continue
            seq = str(genome[chrom][start:end]).upper()
            if len(seq) != 2 * FLANK + 1 or "N" in seq:
                continue

            ref_base = seq[FLANK]
            if ref_base not in (a1, a2):
                continue

            a1_seq = list(seq)
            a1_seq[FLANK] = a1
            a2_seq = list(seq)
            a2_seq[FLANK] = a2

            chroms.append(chrom)
            positions.append(pos)
            ref_seqs.append("".join(a2_seq))
            alt_seqs.append("".join(a1_seq))
            labels.append(int(row["label"]))
            published_scores.append(float(row["gkm_SVM"]))
            snp_names.append(row["SNPname1"])

    genome.close()
    return {
        "ref_seq": np.array(ref_seqs),
        "alt_seq": np.array(alt_seqs),
        "label": np.array(labels),
        "published_score": np.array(published_scores),
        "snp_name": np.array(snp_names),
    }


def load_effect_sizes_by_snp(gz_path: Path) -> dict[str, float]:
    """Load dsQTL effect sizes keyed by SNP name."""
    effects = {}
    with gzip.open(gz_path, "rt") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            snp_id = f"{row['Chr']}.{row['SNP']}"
            effects[snp_id] = float(row["Estimate"])
    return effects


def replicate_published(variants: dict, effect_sizes: dict, device: str):
    """Score with published deltaSVM weights and compare."""
    from gkmsvm.codec import one_hot_encode
    from gkmsvm.importers.deltasvm import load_deltasvm_model

    print("Loading published deltaSVM weights...")
    model = load_deltasvm_model(
        str(DATA_DIR / "gm12878_deltasvm_weights.txt"),
        l=10,
        include_rc=False,
    )
    if device == "cuda":
        model.cuda()
    elif device == "mlx":
        model.mlx()

    ref_seqs = variants["ref_seq"]
    alt_seqs = variants["alt_seq"]
    n = len(ref_seqs)
    scores = np.zeros(n)

    print(f"Scoring {n} variants with published deltaSVM...")
    t0 = time.perf_counter()
    batch_size = 512
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        ref_b = np.stack([one_hot_encode(s) for s in ref_seqs[start:end]])
        alt_b = np.stack([one_hot_encode(s) for s in alt_seqs[start:end]])
        if device == "cuda":
            from gkmsvm.backend import to_cpu, to_gpu

            ref_b, alt_b = to_gpu(ref_b), to_gpu(alt_b)
            delta = model.score_variants(ref_b, alt_b)
            scores[start:end] = to_cpu(delta.squeeze(1))
        elif device == "mlx":
            from gkmsvm.backend import to_mlx

            ref_b, alt_b = to_mlx(ref_b), to_mlx(alt_b)
            delta = model.score_variants(ref_b, alt_b)
            scores[start:end] = np.asarray(delta.squeeze(1))
        else:
            delta = model.score_variants(ref_b, alt_b)
            scores[start:end] = delta.squeeze(1)

    elapsed = time.perf_counter() - t0

    published = variants["published_score"]
    max_diff = np.max(np.abs(scores - published))
    corr = np.corrcoef(scores, published)[0, 1]
    print(f"  {n} variants in {elapsed:.2f}s ({n / elapsed:.0f} variants/s)")
    print(f"  Max diff from published: {max_diff:.2e}")
    print(f"  Correlation with published: {corr:.6f}")

    labels = variants["label"]
    binary = (labels == 1).astype(int)
    ap = average_precision_score(binary, np.abs(scores))

    snp_names = variants["snp_name"]
    pos_scores, pos_effects = [], []
    for i, name in enumerate(snp_names):
        if name in effect_sizes and labels[i] == 1:
            pos_scores.append(scores[i])
            pos_effects.append(effect_sizes[name])
    r_pos = np.corrcoef(pos_scores, pos_effects)[0, 1]

    print(f"  Classification AP: {ap:.4f}  (published: 0.19)")
    print(f"  Effect size r (sig): {r_pos:.4f}  (published: 0.73)")

    return {"ap": ap, "pearson_r": r_pos, "max_diff": max_diff}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    scores: np.ndarray, labels: np.ndarray, effect_sizes: np.ndarray | None, method: str
) -> dict:
    abs_scores = np.abs(scores)
    auroc = roc_auc_score(labels, abs_scores)
    auprc = average_precision_score(labels, abs_scores)

    result = {"method": method, "auroc": auroc, "auprc": auprc}

    if effect_sizes is not None:
        sig_mask = (labels == 1) & (~np.isnan(effect_sizes))
        if sig_mask.sum() >= 10:
            sig_scores = scores[sig_mask]
            sig_effects = effect_sizes[sig_mask]
            r_p, _ = pearsonr(sig_scores, sig_effects)
            r_s, _ = spearmanr(sig_scores, sig_effects)
            result["pearson"] = r_p
            result["spearman"] = r_s
            result["n_sig"] = int(sig_mask.sum())

    return result


def print_results(results: list[dict], param_label: str):
    print(f"\n{'=' * 70}")
    print(f"Results: {param_label}")
    print(f"{'=' * 70}")
    print(
        f"  {'Method':<25s} {'AUROC':>8s} {'AUPRC':>8s} "
        f"{'Pearson':>8s} {'Spearman':>8s}"
    )
    print(f"  {'-' * 25} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
    for r in results:
        pear = f"{r['pearson']:.4f}" if "pearson" in r else "N/A"
        spear = f"{r['spearman']:.4f}" if "spearman" in r else "N/A"
        print(
            f"  {r['method']:<25s} {r['auroc']:>8.4f} {r['auprc']:>8.4f} "
            f"{pear:>8s} {spear:>8s}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "mlx", "auto"]
    )
    parser.add_argument(
        "--params",
        default="all",
        choices=list(PARAM_SETS.keys()) + ["all"],
        help="Parameter set to train (default: all)",
    )
    parser.add_argument(
        "--n-negsets", type=int, default=5, help="Number of negative sets (default: 5)"
    )
    parser.add_argument(
        "--replicate-published",
        action="store_true",
        help="Compare against Lee 2015 published scores (downloads hg19, ~900 MB)",
    )
    parser.add_argument(
        "--force-train", action="store_true", help="Retrain even if cached models exist"
    )
    parser.add_argument("--output", default=None, help="Save results to TSV file")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show progress bars during scoring and training")
    args = parser.parse_args()

    if args.device == "auto":
        from gkmsvm.backend import HAS_CUPY, HAS_MLX

        if HAS_CUPY:
            args.device = "cuda"
        elif HAS_MLX:
            args.device = "mlx"
        else:
            args.device = "cpu"
    print(f"Device: {args.device}")

    download_data(replicate_published=args.replicate_published)

    # --- Published replication (optional) ---
    if args.replicate_published:
        print("\n" + "=" * 70)
        print("Published DeltaSVM Replication (Lee et al. 2015)")
        print("=" * 70)
        pub_variants = load_published_variants(
            DATA_DIR / "dsqtl_lee2015.tsv", DATA_DIR / "hg19.fa"
        )
        pub_effects = load_effect_sizes_by_snp(DATA_DIR / "GSE31388_dsQtlTable.txt.gz")
        print(
            f"  {len(pub_variants['label'])} variants, {len(pub_effects)} effect sizes"
        )
        replicate_published(pub_variants, pub_effects, args.device)

    # --- Train and evaluate ---
    print("\nLoading data...")
    pos_seqs, neg_sets = load_training_data(n_negsets=args.n_negsets)
    X19, X51, labels, effect_sizes = load_test_variants()

    param_keys = list(PARAM_SETS.keys()) if args.params == "all" else [args.params]

    all_results = []

    for pk in param_keys:
        params = PARAM_SETS[pk]
        is_rbf = params["kernel_type"] == "rbf"
        print(f"\n{'=' * 70}")
        print(
            f"Training: l={params['l']} k={params['k']} d={params['d']} "
            f"-t {params['kernel_type']}"
            + (f" gamma={params['gamma']}" if is_rbf else "")
            + f" ({args.n_negsets} neg sets)"
        )
        print(f"{'=' * 70}")

        models, dsvms, avg_dsvm = train_models(
            pos_seqs, neg_sets, params, args.device, pk=pk,
            force_train=args.force_train, verbose=args.verbose,
        )

        results = []
        X19_ref, X19_alt = X19
        X51_ref, X51_alt = X51
        methods = ("deltaSVM", "ISM", "GkmExplain")
        per_model = {name: np.zeros((len(models), len(labels))) for name in methods}

        for mi in range(len(models)):
            print(f"\n── Evaluating model {mi + 1}/{len(models)} ──")
            per_model["deltaSVM"][mi] = score_deltasvm(dsvms[mi], X19_ref, X19_alt)
            per_model["ISM"][mi] = score_vep(
                models[mi], X51_ref, X51_alt, args.device, "kernel",
                verbose=args.verbose,
            )
            per_model["GkmExplain"][mi] = score_vep(
                models[mi], X51_ref, X51_alt, args.device, "gkmexplain",
                verbose=args.verbose,
            )

        paper_cols = (
            PAPER_REF_COLUMNS[params["kernel_type"]]
            if (params["l"], params["k"]) == (10, 6) else {}
        )
        header = "".join(f"  {name:>11s}" for name in methods)
        header += "".join(f"  {'paper ' + name:>17s}" for name in paper_cols)
        print(f"\n  Per-model AUPRC ({pk}):")
        print(f"  {'Model':>5s}{header}")
        for mi in range(len(models)):
            row = "".join(
                f"  {average_precision_score(labels, np.abs(per_model[name][mi])):>11.4f}"
                for name in methods
            )
            row += "".join(
                f"  {PAPER_AUPRC[ref][mi]:>17.4f}" for ref in paper_cols.values()
            )
            print(f"  {mi + 1:>5d}{row}")

        print("\nScoring with deltaSVM (averaged weights)...")
        avg_dsvm_scores = score_deltasvm(avg_dsvm, X19_ref, X19_alt)
        results.append(evaluate(avg_dsvm_scores, labels, effect_sizes, f"deltaSVM-avg ({pk})"))
        for name in methods:
            results.append(evaluate(
                per_model[name].sum(axis=0), labels, effect_sizes, f"{name}-sum ({pk})",
            ))

        print_results(results, pk)
        all_results.extend(results)

        del models, dsvms, avg_dsvm
        gc.collect()

    if len(param_keys) > 1:
        print_results(all_results, "all parameter sets")

    if args.output:
        import pandas as pd

        pd.DataFrame(all_results).to_csv(args.output, sep="\t", index=False)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
