#!/usr/bin/env python3
"""Run the full dsQTL deltaSVM pipeline via the gkmsvm CLI + Python evaluation.

Reproduces the deltasvm_dsqtl.ipynb benchmark end-to-end:
  1. Train 5 gkm-SVMs (one per negative set)
  2. Convert each to deltaSVM weights
  3. Average weights across models
  4. Score dsQTL test variants
  5. Evaluate classification (AUROC/AUPRC) and regression (Spearman ρ)
  6. Save plots and model outputs

Usage:
    cd examples/
    python run_dsqtl_cli.py                  # auto-detect GPU
    python run_dsqtl_cli.py --device cpu     # force CPU
    python run_dsqtl_cli.py --n-models 1     # quick single-model test

Requires data from the notebook's download step (examples/data/).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

DATA_DIR = Path("data")
SEQ_DIR = DATA_DIR / "gm12878_sequence_sets"
PUB_WEIGHTS = DATA_DIR / "SupplementaryTable_gm12878weights.txt"
GEO_TABLE = DATA_DIR / "GSE31388_dsQtlTable.txt.gz"
OUT_DIR = Path("output_dsqtl")

KERNEL_L = 10
KERNEL_K = 6
KERNEL_D = 3
C_PARAM = 1.0


def download_data():
    """Download data if not present."""
    import tarfile
    from urllib.request import urlretrieve

    DATA_DIR.mkdir(exist_ok=True)

    tarball = DATA_DIR / "gm12878_sequence_sets.tar.gz"
    if not SEQ_DIR.exists():
        if not tarball.exists():
            print("Downloading Beer lab sequence sets (14 MB)...")
            urlretrieve(
                "https://beerlab.org/deltasvm/downloads/gm12878_sequence_sets.tar.gz",
                tarball,
            )
        print("Extracting...")
        with tarfile.open(tarball) as tar:
            tar.extractall(DATA_DIR)

    if not PUB_WEIGHTS.exists():
        print("Downloading published deltaSVM weights (16 MB)...")
        urlretrieve(
            "https://beerlab.org/deltasvm/downloads/SupplementaryTable_gm12878weights.txt",
            PUB_WEIGHTS,
        )

    if not GEO_TABLE.exists():
        print("Downloading GEO GSE31388 dsQTL effect sizes...")
        urlretrieve(
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE31nnn/GSE31388/suppl/GSE31388_dsQtlTable.txt.gz",
            GEO_TABLE,
        )

    print(f"Data ready: {SEQ_DIR}")


def run_cli(*args):
    """Run gkmsvm CLI command."""
    from gkmsvm.cli import main as cli_main
    argv = list(args)
    print(f"  $ gkmsvm {' '.join(argv)}")
    cli_main(argv)


def train_models(n_models: int, device: str) -> list[Path]:
    """Train n gkm-SVMs and return model paths."""
    model_paths = []
    pos_fa = str(SEQ_DIR / "gm12878_shared.fa")

    for i in range(1, n_models + 1):
        model_path = OUT_DIR / f"model_{i}.npz"
        neg_fa = str(SEQ_DIR / f"nullseqs_gm12878_shared.{i}.1.fa")

        if model_path.exists():
            print(f"  Model {i}: {model_path} (cached)")
            model_paths.append(model_path)
            continue

        print(f"\n── Training model {i}/{n_models} ──")
        t0 = time.time()
        run_cli(
            "-v", "train",
            "-p", pos_fa, "-n", neg_fa,
            "-o", str(model_path),
            "-t", "estimated",
            "-l", str(KERNEL_L), "-k", str(KERNEL_K), "-d", str(KERNEL_D),
            "-C", str(C_PARAM),
            "--solver", "auto",
            "--device", device,
        )
        print(f"  Trained in {time.time() - t0:.1f}s")
        model_paths.append(model_path)

    return model_paths


def convert_to_deltasvm(model_paths: list[Path]) -> list[Path]:
    """Convert models to deltaSVM weight tables."""
    weight_paths = []

    for model_path in model_paths:
        weight_path = model_path.with_suffix(".deltasvm.tsv")
        if weight_path.exists():
            print(f"  Weights: {weight_path} (cached)")
            weight_paths.append(weight_path)
            continue

        print(f"\n── Converting {model_path.name} to deltaSVM ──")
        t0 = time.time()
        run_cli(
            "-v", "to-deltasvm",
            "-m", str(model_path),
            "-o", str(weight_path),
            "--device", "auto",
        )
        print(f"  Converted in {time.time() - t0:.1f}s")
        weight_paths.append(weight_path)

    return weight_paths


def average_weights(weight_paths: list[Path]) -> Path:
    """Average deltaSVM weight tables and write combined file."""
    from gkmsvm.deltasvm import _index_to_kmer

    avg_path = OUT_DIR / "averaged_deltasvm.tsv"

    all_weights = []
    for wp in weight_paths:
        w = np.zeros(4**KERNEL_L, dtype=np.float64)
        with open(wp) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                kmer, weight = parts[0], float(parts[1])
                from gkmsvm.deltasvm import _kmer_to_index
                w[_kmer_to_index(kmer)] = weight
        all_weights.append(w)

    avg = np.mean(all_weights, axis=0).astype(np.float32)
    n_nonzero = int((avg != 0).sum())

    with open(avg_path, "w") as f:
        f.write("# include_rc=false (RC already in weights)\n")
        for idx in range(len(avg)):
            if avg[idx] != 0.0:
                f.write(f"{_index_to_kmer(idx, KERNEL_L)}\t{avg[idx]:.8g}\n")

    print(f"\nAveraged {len(weight_paths)} models → {avg_path}")
    print(f"  {n_nonzero:,}/{len(avg):,} non-zero weights")
    print(f"  Range: [{avg.min():.4f}, {avg.max():.4f}]")

    return avg_path


def score_variants(avg_weights_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score dsQTL test variants and return (deltas, labels, effects, published_deltas)."""
    import pandas as pd
    from gkmsvm import one_hot_encode, read_fasta, load_deltasvm_model
    from gkmsvm.backend import to_cpu

    pos_major = read_fasta(str(SEQ_DIR / "dsqtl_test_pos.major.fa"))
    pos_minor = read_fasta(str(SEQ_DIR / "dsqtl_test_pos.minor.fa"))
    neg_major = read_fasta(str(SEQ_DIR / "dsqtl_test_neg.major.fa"))
    neg_minor = read_fasta(str(SEQ_DIR / "dsqtl_test_neg.minor.fa"))

    ref_seqs = [s for _, s in pos_major] + [s for _, s in neg_major]
    alt_seqs = [s for _, s in pos_minor] + [s for _, s in neg_minor]
    labels = np.array([1] * len(pos_major) + [0] * len(neg_major))

    print(f"\nScoring {len(ref_seqs)} variants ({(labels == 1).sum()} sig + {(labels == 0).sum()} ctrl)")

    X_ref = np.stack([one_hot_encode(s) for s in ref_seqs])
    X_alt = np.stack([one_hot_encode(s) for s in alt_seqs])

    dsvm = load_deltasvm_model(str(avg_weights_path), KERNEL_L)
    our_deltas = to_cpu(dsvm.score_variants(X_ref, X_alt).flatten())

    pub_dsvm = load_deltasvm_model(str(PUB_WEIGHTS), KERNEL_L)
    pub_deltas = to_cpu(pub_dsvm.score_variants(X_ref, X_alt).flatten())

    r_scores = np.corrcoef(our_deltas, pub_deltas)[0, 1]
    print(f"  Our vs published score correlation: r = {r_scores:.4f}")

    # Load effect sizes
    geo = pd.read_csv(GEO_TABLE, sep="\t", compression="gzip")
    geo = geo.sort_values("Pr(>|t|)").drop_duplicates(subset=["Chr", "SNP"], keep="first")

    effects = np.full(len(ref_seqs), np.nan)
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

    return our_deltas, labels, effects, pub_deltas


def evaluate_and_plot(
    our_deltas: np.ndarray,
    labels: np.ndarray,
    effects: np.ndarray,
    pub_deltas: np.ndarray,
):
    """Evaluate metrics and save plots."""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        roc_curve, precision_recall_curve,
    )
    from scipy.stats import spearmanr, pearsonr
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    abs_ours = np.abs(our_deltas)
    abs_pub = np.abs(pub_deltas)
    baseline = labels.mean()

    auroc = roc_auc_score(labels, abs_ours)
    auprc = average_precision_score(labels, abs_ours)
    auroc_pub = roc_auc_score(labels, abs_pub)
    auprc_pub = average_precision_score(labels, abs_pub)

    print(f"\n{'─' * 55}")
    print(f"dsQTL Classification Results")
    print(f"{'─' * 55}")
    print(f"{'':35s} {'AUROC':>8s}  {'AUPRC':>8s}")
    print(f"{'─' * 55}")
    print(f"{'gkmsvm-lite (retrained)':35s} {auroc:8.4f}  {auprc:8.4f}")
    print(f"{'Published (Lee 2015)':35s} {auroc_pub:8.4f}  {auprc_pub:8.4f}")
    print(f"{'─' * 55}")
    print(f"{'Baseline rate':35s} {'':>8s}  {baseline:8.4f}")

    sig_mask = (labels == 1) & (~np.isnan(effects))
    sig_deltas = our_deltas[sig_mask]
    sig_effects = effects[sig_mask]
    sig_pub = pub_deltas[sig_mask]

    rho, p_rho = spearmanr(sig_deltas, sig_effects)
    rho_pub, p_pub = spearmanr(sig_pub, sig_effects)
    r_val, _ = pearsonr(sig_deltas, sig_effects)

    print(f"\n{'─' * 55}")
    print(f"dsQTL Regression (n={len(sig_deltas)} significant variants)")
    print(f"{'─' * 55}")
    print(f"{'':35s} {'Spearman':>10s}  {'p-value':>10s}")
    print(f"{'─' * 55}")
    print(f"{'gkmsvm-lite (retrained)':35s} {rho:10.4f}  {p_rho:10.2e}")
    print(f"{'Published (Lee 2015)':35s} {rho_pub:10.4f}  {p_pub:10.2e}")
    print(f"{'─' * 55}")

    # Save metrics
    metrics_path = OUT_DIR / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("dsQTL DeltaSVM Benchmark Results\n")
        f.write(f"{'=' * 55}\n\n")
        f.write(f"Classification:\n")
        f.write(f"  gkmsvm-lite  AUROC={auroc:.4f}  AUPRC={auprc:.4f}\n")
        f.write(f"  Lee 2015     AUROC={auroc_pub:.4f}  AUPRC={auprc_pub:.4f}\n")
        f.write(f"  Baseline rate: {baseline:.4f}\n\n")
        f.write(f"Regression (n={len(sig_deltas)}):\n")
        f.write(f"  gkmsvm-lite  Spearman={rho:.4f}  Pearson={r_val:.4f}\n")
        f.write(f"  Lee 2015     Spearman={rho_pub:.4f}\n\n")
        f.write(f"Published reference (Pampari et al. 2024, Fig. 6):\n")
        f.write(f"  gkm-SVM deltaSVM (DNase 68M):  AUPRC = 0.19\n")
        f.write(f"  Enformer (published):           AUPRC = 0.33\n")
        f.write(f"  ChromBPNet (DNase 68M):         AUPRC = 0.43\n")
        f.write(f"  ChromBPNet (ATAC 572M):         AUPRC = 0.54\n")
    print(f"\nMetrics saved: {metrics_path}")

    # ── Figure 1: ROC + PR + score distribution ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    for scores, label, color in [
        (abs_ours, "gkmsvm-lite", "steelblue"),
        (abs_pub, "Lee 2015", "coral"),
    ]:
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{label} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[1]
    for scores, label, color in [
        (abs_ours, "gkmsvm-lite", "steelblue"),
        (abs_pub, "Lee 2015", "coral"),
    ]:
        prec, rec, _ = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)
        ax.plot(rec, prec, color=color, linewidth=2, label=f"{label} (AP={ap:.3f})")
    ax.axhline(baseline, color="k", linestyle="--", alpha=0.3, linewidth=1,
               label=f"Random ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[2]
    ax.hist(abs_ours[labels == 0], bins=50, alpha=0.6, label="Non-significant",
            color="gray", density=True)
    ax.hist(abs_ours[labels == 1], bins=50, alpha=0.6, label="dsQTL (sig.)",
            color="steelblue", density=True)
    ax.set_xlabel("|DeltaSVM score|")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution")
    ax.legend()

    plt.tight_layout()
    fig.savefig(OUT_DIR / "dsqtl_classification.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT_DIR / "dsqtl_classification.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_DIR}/dsqtl_classification.png")

    # ── Figure 2: Regression + score comparison ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.scatter(sig_effects, sig_deltas, alpha=0.4, s=12, color="steelblue",
               rasterized=True)
    z = np.polyfit(sig_effects, sig_deltas, 1)
    x_line = np.linspace(sig_effects.min(), sig_effects.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "r-", alpha=0.6, linewidth=1.5)
    ax.set_xlabel("Measured effect size (beta)")
    ax.set_ylabel("gkmsvm-lite delta-score")
    ax.set_title(f"gkmsvm-lite (rho={rho:.3f}, n={len(sig_deltas)})")
    ax.axhline(0, color="grey", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="grey", linewidth=0.5, alpha=0.5)

    ax = axes[1]
    ax.scatter(sig_effects, sig_pub, alpha=0.4, s=12, color="coral",
               rasterized=True)
    z_pub = np.polyfit(sig_effects, sig_pub, 1)
    ax.plot(x_line, np.polyval(z_pub, x_line), "r-", alpha=0.6, linewidth=1.5)
    ax.set_xlabel("Measured effect size (beta)")
    ax.set_ylabel("Published delta-score (Lee 2015)")
    ax.set_title(f"Published (rho={rho_pub:.3f}, n={len(sig_pub)})")
    ax.axhline(0, color="grey", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="grey", linewidth=0.5, alpha=0.5)

    ax = axes[2]
    r_vs = np.corrcoef(our_deltas, pub_deltas)[0, 1]
    ax.scatter(pub_deltas, our_deltas, alpha=0.1, s=3, color="gray",
               rasterized=True)
    ax.scatter(sig_pub, sig_deltas, alpha=0.5, s=12, color="steelblue",
               label="Significant dsQTLs", rasterized=True)
    lo = min(pub_deltas.min(), our_deltas.min())
    hi = max(pub_deltas.max(), our_deltas.max())
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3)
    ax.set_xlabel("Published delta-score (Lee 2015)")
    ax.set_ylabel("gkmsvm-lite delta-score")
    ax.set_title(f"Score correlation (r={r_vs:.4f})")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "dsqtl_regression.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT_DIR / "dsqtl_regression.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_DIR}/dsqtl_regression.png")

    # ── Figure 3: Weight comparison ──
    from gkmsvm import load_deltasvm_model
    from gkmsvm.backend import to_cpu

    avg_dsvm = load_deltasvm_model(str(OUT_DIR / "averaged_deltasvm.tsv"), KERNEL_L)
    pub_dsvm = load_deltasvm_model(str(PUB_WEIGHTS), KERNEL_L)
    avg_w = to_cpu(avg_dsvm.weights)
    pub_w = to_cpu(pub_dsvm.weights)

    mask = (avg_w != 0) | (pub_w != 0)
    r_w = np.corrcoef(avg_w[mask], pub_w[mask])[0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.scatter(pub_w[mask], avg_w[mask], alpha=0.3, s=5, color="steelblue",
               rasterized=True)
    lim = max(abs(pub_w[mask]).max(), abs(avg_w[mask]).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3)
    ax.set_xlabel("Published weight (Lee 2015)")
    ax.set_ylabel("gkmsvm-lite weight")
    ax.set_title(f"k-mer weight comparison (r={r_w:.4f})")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    ax = axes[1]
    residuals = avg_w[mask] - pub_w[mask]
    ax.hist(residuals, bins=100, color="coral", alpha=0.8, edgecolor="none")
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Weight difference (ours - published)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residuals (mean={residuals.mean():.4f}, std={residuals.std():.4f})")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "dsqtl_weights.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT_DIR / "dsqtl_weights.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_DIR}/dsqtl_weights.png")

    # Save scores
    scores_path = OUT_DIR / "variant_scores.tsv"
    np.savetxt(scores_path,
               np.column_stack([our_deltas, pub_deltas, labels, effects]),
               delimiter="\t",
               header="our_delta\tpub_delta\tlabel\teffect_size",
               comments="")
    print(f"Saved: {scores_path}")


def main():
    parser = argparse.ArgumentParser(description="dsQTL deltaSVM CLI pipeline")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mlx"])
    parser.add_argument("--n-models", type=int, default=5, help="Number of neg-set models (default: 5)")
    parser.add_argument("--skip-training", action="store_true", help="Skip training, use cached models")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    download_data()

    if not args.skip_training:
        model_paths = train_models(args.n_models, args.device)
        weight_paths = convert_to_deltasvm(model_paths)
        avg_path = average_weights(weight_paths)
    else:
        avg_path = OUT_DIR / "averaged_deltasvm.tsv"
        if not avg_path.exists():
            print(f"Error: {avg_path} not found. Run without --skip-training first.")
            sys.exit(1)

    our_deltas, labels, effects, pub_deltas = score_variants(avg_path)
    evaluate_and_plot(our_deltas, labels, effects, pub_deltas)

    print(f"\nAll outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
