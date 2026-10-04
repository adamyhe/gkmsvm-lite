"""Benchmark: dsQTL variant effect prediction.

Trains gkm-SVMs on GM12878 DNase-seq peaks (Beer lab data) and evaluates
three variant effect prediction methods:
  - DeltaSVM: k-mer weight linear approximation (averaged across neg sets)
  - Kernel VEP: score(alt) - score(ref)
  - GkmExplain VEP: hypothetical importance at variant position

Supports two parameter settings:
  - l=10, k=6, d=3 (Lee 2015 original, matching published deltaSVM)
  - l=11, k=7, d=3 (LS-GKM defaults, matching ENCODE models)

Pass --replicate-published to also compare against Lee 2015 published
deltaSVM scores (downloads hg19 genome, ~900 MB compressed).

All data auto-downloaded on first run. Trained models cached to
benchmarks/models/ and reloaded on subsequent runs.

Usage:
  python benchmarks/bench_dsqtl.py --device cuda
  python benchmarks/bench_dsqtl.py --device cuda --params l10k6
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
WEIGHTS_URL = "https://beerlab.org/deltasvm/downloads/SupplementaryTable_gm12878weights.txt"
LEE2015_URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.1038%2Fng.3331/MediaObjects/41588_2015_BFng3331_MOESM26_ESM.xlsx"
)
HG19_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz"

PARAM_SETS = {
    "l10k6": {"l": 10, "k": 6, "d": 3, "kernel_type": "estimated"},
    "l11k7": {"l": 11, "k": 7, "d": 3, "kernel_type": "estimated"},
}

FLANK = 9


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
            for row in ws.iter_rows(values_only=True):
                f.write("\t".join(
                    str(v) if v is not None else "" for v in row) + "\n")
        wb.close()

    # hg19 genome
    import shutil
    fasta_path = DATA_DIR / "hg19.fa"
    if not fasta_path.exists():
        gz_path = DATA_DIR / "hg19.fa.gz"
        if not gz_path.exists():
            print("Downloading hg19 genome (~900 MB compressed)...")
            urlretrieve(HG19_URL, gz_path)
        print("Decompressing hg19.fa.gz...")
        with gzip.open(gz_path, "rb") as f_in, \
                open(fasta_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data(n_negsets: int = 5):
    from gkmsvm import read_fasta

    pos_seqs = [seq for _, seq in read_fasta(
        str(SEQ_DIR / "gm12878_shared.fa"))]
    print(f"Training positives: {len(pos_seqs):,} sequences, "
          f"{len(pos_seqs[0])}bp")

    neg_sets = []
    for i in range(1, n_negsets + 1):
        negs = [seq for _, seq in read_fasta(
            str(SEQ_DIR / f"nullseqs_gm12878_shared.{i}.1.fa"))]
        neg_sets.append(negs)
        print(f"  Negative set {i}: {len(negs):,} sequences")

    return pos_seqs, neg_sets


def load_test_variants():
    from gkmsvm import read_fasta, one_hot_encode

    pos_major = list(read_fasta(str(SEQ_DIR / "dsqtl_test_pos.major.fa")))
    pos_minor = list(read_fasta(str(SEQ_DIR / "dsqtl_test_pos.minor.fa")))
    neg_major = list(read_fasta(str(SEQ_DIR / "dsqtl_test_neg.major.fa")))
    neg_minor = list(read_fasta(str(SEQ_DIR / "dsqtl_test_neg.minor.fa")))

    ref_seqs = [s for _, s in pos_major] + [s for _, s in neg_major]
    alt_seqs = [s for _, s in pos_minor] + [s for _, s in neg_minor]
    labels = np.array([1] * len(pos_major) + [0] * len(neg_major))

    X_ref = np.stack([one_hot_encode(s) for s in ref_seqs])
    X_alt = np.stack([one_hot_encode(s) for s in alt_seqs])

    print(f"Test variants: {len(labels)} ({labels.sum()} sig + "
          f"{(labels == 0).sum()} ctrl), {len(ref_seqs[0])}bp")

    effect_sizes = _load_effect_sizes(pos_major)
    if effect_sizes is not None:
        effect_sizes = np.concatenate(
            [effect_sizes, np.full(len(neg_major), np.nan)])

    return X_ref, X_alt, labels, effect_sizes


def _load_effect_sizes(pos_major: list[tuple[str, str]]) -> np.ndarray | None:
    import pandas as pd

    geo_path = DATA_DIR / "GSE31388_dsQtlTable.txt.gz"
    if not geo_path.exists():
        print("  Effect sizes not found, skipping regression metrics")
        return None

    geo = pd.read_csv(geo_path, sep="\t", compression="gzip")
    geo = geo.sort_values("Pr(>|t|)").drop_duplicates(
        subset=["Chr", "SNP"], keep="first")

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
    return MODEL_DIR / f"gm12878_{pk}_neg{neg_idx}_dsvm.npz"


def train_models(pos_seqs, neg_sets, params: dict, device: str,
                 pk: str, force_train: bool = False, C: float = 1.0):
    from gkmsvm import train_gkmsvm
    from gkmsvm.serialization import (
        save_npz, load_model, save_deltasvm_npz, load_deltasvm_npz,
    )
    from gkmsvm.backend import to_cpu

    models = []
    dsvms = []

    for i, neg_seqs in enumerate(neg_sets):
        cached = _model_path(pk, i + 1)
        dsvm_cached = _dsvm_path(pk, i + 1)

        if cached.exists() and not force_train:
            print(f"\n── Model {i+1}/{len(neg_sets)} "
                  f"(l={params['l']} k={params['k']}) ── [cached]")
            m = load_model(str(cached))
        else:
            print(f"\n── Model {i+1}/{len(neg_sets)} "
                  f"(l={params['l']} k={params['k']}) ──")
            t0 = time.time()
            m = train_gkmsvm(
                pos_seqs, neg_seqs,
                kernel_type=params["kernel_type"],
                l=params["l"], k=params["k"], d=params["d"],
                C=C, solver="auto",
                device=device, verbose=True,
            )
            print(f"  {m.num_support_vectors} SVs, {time.time() - t0:.1f}s")
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            m.cpu()
            save_npz(m, str(cached))
            print(f"  Saved {cached}")

        if dsvm_cached.exists() and not force_train:
            print(f"  DeltaSVM weights: [cached]")
            d = load_deltasvm_npz(str(dsvm_cached))
        else:
            if device == "cuda":
                m.cuda()
            t0 = time.time()
            d = m.to_deltasvm(device=device, verbose=True)
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

    return models, avg_dsvm


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_deltasvm(dsvm, X_ref, X_alt):
    from gkmsvm.backend import to_cpu
    t0 = time.time()
    scores = to_cpu(dsvm.score_variants(X_ref, X_alt).flatten())
    print(f"  DeltaSVM: {len(scores)} variants in {time.time() - t0:.2f}s")
    return scores


def score_kernel(model, X_ref, X_alt, device: str, batch_size: int = 64):
    from gkmsvm.backend import to_cpu

    if device == "cuda":
        model.cuda()

    N = len(X_ref)
    scores = np.zeros(N, dtype=np.float64)

    t0 = time.time()
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ref_b = model._match_device(X_ref[start:end])
        alt_b = model._match_device(X_alt[start:end])
        s_ref = model(ref_b, verbose=False).flatten()
        s_alt = model(alt_b, verbose=False).flatten()
        scores[start:end] = to_cpu(s_alt - s_ref)

    elapsed = time.time() - t0
    model.cpu()
    print(f"  Kernel VEP: {N} variants in {elapsed:.1f}s "
          f"({N * 2 / elapsed:.0f} seqs/s)")
    return scores


def score_gkmexplain(model, X_ref, X_alt, device: str,
                     batch_size: int = 32):
    from gkmsvm.backend import to_cpu

    if device == "cuda":
        model.cuda()

    N = len(X_ref)
    scores = np.zeros(N, dtype=np.float64)

    t0 = time.time()
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ref_b = model._match_device(X_ref[start:end])
        alt_b = model._match_device(X_alt[start:end])
        s = model.score_variants(ref_b, alt_b, method="gkmexplain",
                                 batch_size=end - start, verbose=False)
        scores[start:end] = to_cpu(s).flatten()

    elapsed = time.time() - t0
    model.cpu()
    print(f"  GkmExplain VEP: {N} variants in {elapsed:.1f}s "
          f"({N / elapsed:.0f} variants/s)")
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


def replicate_published(variants: dict, effect_sizes: dict,
                        device: str):
    """Score with published deltaSVM weights and compare."""
    from gkmsvm.codec import one_hot_encode
    from gkmsvm.importers.deltasvm import load_deltasvm_model

    print("Loading published deltaSVM weights...")
    model = load_deltasvm_model(
        str(DATA_DIR / "gm12878_deltasvm_weights.txt"),
        l=10, include_rc=False,
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
            from gkmsvm.backend import to_gpu, to_cpu
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
    print(f"  {n} variants in {elapsed:.2f}s "
          f"({n / elapsed:.0f} variants/s)")
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

def evaluate(scores: np.ndarray, labels: np.ndarray,
             effect_sizes: np.ndarray | None, method: str) -> dict:
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
    print(f"  {'Method':<25s} {'AUROC':>8s} {'AUPRC':>8s} "
          f"{'Pearson':>8s} {'Spearman':>8s}")
    print(f"  {'-' * 25} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
    for r in results:
        pear = f"{r['pearson']:.4f}" if "pearson" in r else "N/A"
        spear = f"{r['spearman']:.4f}" if "spearman" in r else "N/A"
        print(f"  {r['method']:<25s} {r['auroc']:>8.4f} {r['auprc']:>8.4f} "
              f"{pear:>8s} {spear:>8s}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "cuda", "mlx", "auto"])
    parser.add_argument("--params", default="both",
                        choices=["l10k6", "l11k7", "both"],
                        help="Parameter set to train (default: both)")
    parser.add_argument("--n-negsets", type=int, default=5,
                        help="Number of negative sets (default: 5)")
    parser.add_argument("--replicate-published", action="store_true",
                        help="Compare against Lee 2015 published scores "
                             "(downloads hg19, ~900 MB)")
    parser.add_argument("--force-train", action="store_true",
                        help="Retrain even if cached models exist")
    parser.add_argument("--output", default=None,
                        help="Save results to TSV file")
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
            DATA_DIR / "dsqtl_lee2015.tsv", DATA_DIR / "hg19.fa")
        pub_effects = load_effect_sizes_by_snp(
            DATA_DIR / "GSE31388_dsQtlTable.txt.gz")
        print(f"  {len(pub_variants['label'])} variants, "
              f"{len(pub_effects)} effect sizes")
        replicate_published(pub_variants, pub_effects, args.device)

    # --- Train and evaluate ---
    print("\nLoading data...")
    pos_seqs, neg_sets = load_training_data(n_negsets=args.n_negsets)
    X_ref, X_alt, labels, effect_sizes = load_test_variants()

    param_keys = (["l10k6", "l11k7"] if args.params == "both"
                  else [args.params])

    all_results = []

    for pk in param_keys:
        params = PARAM_SETS[pk]
        print(f"\n{'=' * 70}")
        print(f"Training: l={params['l']} k={params['k']} d={params['d']} "
              f"({args.n_negsets} neg sets)")
        print(f"{'=' * 70}")

        models, avg_dsvm = train_models(
            pos_seqs, neg_sets, params, args.device,
            pk=pk, force_train=args.force_train)

        results = []

        print("\nScoring with deltaSVM (averaged weights)...")
        dsvm_scores = score_deltasvm(avg_dsvm, X_ref, X_alt)
        results.append(evaluate(dsvm_scores, labels, effect_sizes,
                                f"deltaSVM ({pk})"))

        print("\nScoring with kernel VEP (model 1)...")
        kernel_scores = score_kernel(
            models[0], X_ref, X_alt, args.device)
        results.append(evaluate(kernel_scores, labels, effect_sizes,
                                f"kernel ({pk})"))

        explain_device = args.device if args.device != "mlx" else "cpu"
        if explain_device != args.device:
            print("\nGkmExplain unsupported on MLX, falling back to CPU...")
        print(f"\nScoring with GkmExplain VEP (model 1)...")
        explain_scores = score_gkmexplain(
            models[0], X_ref, X_alt, explain_device)
        results.append(evaluate(explain_scores, labels, effect_sizes,
                                f"gkmexplain ({pk})"))

        corr_dk = np.corrcoef(dsvm_scores, kernel_scores)[0, 1]
        corr_de = np.corrcoef(dsvm_scores, explain_scores)[0, 1]
        print(f"\n  Score correlations (all variants):")
        print(f"    deltaSVM vs kernel:     r = {corr_dk:.4f}")
        print(f"    deltaSVM vs gkmexplain: r = {corr_de:.4f}")

        print_results(results, pk)
        all_results.extend(results)

        del models, avg_dsvm
        gc.collect()

    if len(param_keys) > 1:
        print_results(all_results, "all parameter sets")

    if args.output:
        import pandas as pd
        pd.DataFrame(all_results).to_csv(args.output, sep="\t", index=False)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
