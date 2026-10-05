"""Replicate Nanog H1-ESC gkm-SVM model from the gkmExplain paper.

Reproduces the model training, scoring, GkmExplain attribution, and
TF-MoDISco motif discovery from Shrikumar et al. (2019) using the
Nanog H1-ESC ChIP-seq dataset.

Reference data from: https://github.com/kundajelab/gkmexplain
- Nanog_H1ESC/: ChIP-seq sequences + pre-trained LS-GKM model
- Default LS-GKM parameters: -t 2 -l 11 -k 7 -d 3 -C 1.0

This script:
  1. Downloads the full Nanog dataset (4,687 pos + 5,017 neg train)
  2. Loads the pre-trained reference model (8,873 SVs)
  3. Scores test sequences and evaluates AUC
  4. Trains from scratch (libsvm + SMO) and compares
  5. Runs GkmExplain on all 960 positive test sequences
  6. Compares attributions against reference gkmexplain scores
  7. Runs TF-MoDISco on attributions and reports discovered motifs

Data requirements:
    No external data needed — downloads from GitHub on first run.
    Cached in benchmarks/data/nanog/ for subsequent runs.

Usage:
    python benchmarks/replicate_nanog.py                 # full replication
    python benchmarks/replicate_nanog.py --skip-train    # scoring + attribution only
    python benchmarks/replicate_nanog.py --skip-modisco  # skip TF-MoDISco
    python benchmarks/replicate_nanog.py --device cuda   # GPU acceleration
"""

from __future__ import annotations

import argparse
import gzip
import time
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data" / "nanog"

GKMEXPLAIN_REPO = (
    "https://raw.githubusercontent.com/kundajelab/gkmexplain/master/Nanog_H1ESC"
)
MODEL_STORAGE = (
    "https://raw.githubusercontent.com/AvantiShri/model_storage/aae0902/gkmexplain"
)

FILES = {
    "pos_train": f"{GKMEXPLAIN_REPO}/positives_train.fa",
    "neg_train": f"{GKMEXPLAIN_REPO}/negatives_train.fa",
    "pos_test": f"{GKMEXPLAIN_REPO}/positives_test.fa",
    "neg_test": f"{GKMEXPLAIN_REPO}/negatives_test.fa",
    "model": f"{GKMEXPLAIN_REPO}/lsgkm_defaultsettings_t2.model.txt",
    "ref_hyp_scores": f"{MODEL_STORAGE}/gkmexplain_positives_hypimpscores.txt.gz",
    "ref_imp_scores": f"{MODEL_STORAGE}/gkmexplain_positives_impscores.txt.gz",
}


def _download(url, dest):
    if dest.exists():
        return
    print(f"  Downloading {dest.name}")
    urllib.request.urlretrieve(url, dest)


def download_data(skip_ref_scores=False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for key, url in FILES.items():
        if skip_ref_scores and key.startswith("ref_"):
            continue
        dest = DATA_DIR / Path(url).name
        _download(url, dest)
    print(f"  Data cached in {DATA_DIR}")


def read_fasta(path):
    from gkmsvm.fasta import read_fasta as _read_fasta
    return _read_fasta(path)


def _parse_lsgkm_explain_scores(path):
    """Parse LS-GKM gkmexplain output format.

    Each line: name<tab>seq<tab>scores
    where scores = semicolon-separated positions,
    each position = comma-separated channel values (A,C,G,T).
    Returns [N, 4, L] array (transposed from file's [N, L, 4]).
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    records = []
    with opener(path, "rt") as f:
        for line in f:
            parts = line.rstrip().split("\t")
            positions = parts[2].split(";")
            mat = np.array([[float(v) for v in pos.split(",")] for pos in positions])
            records.append(mat.T)
    return np.stack(records)


def _to_device(arr, device):
    if device == "cuda":
        from gkmsvm.backend import to_gpu
        return to_gpu(arr)
    elif device == "mlx":
        from gkmsvm.backend import to_mlx
        return to_mlx(arr)
    return arr


def _to_numpy(arr, device):
    if device == "cuda":
        from gkmsvm.backend import to_cpu
        return to_cpu(arr)
    return np.asarray(arr)


def _gkmexplain_batched(model, x, mode, batch_size=50, device="cpu", verbose=True):
    """Run GkmExplain in batches to avoid GPU OOM."""
    from gkmsvm.explain import gkmexplain
    results = []
    for i in range(0, len(x), batch_size):
        batch_np = x[i:i + batch_size]
        batch = _to_device(batch_np, device)
        exp = gkmexplain(model, batch, mode=mode, verbose=False)
        results.append(_to_numpy(exp, device))
    if verbose:
        print(f"    {len(x)} seqs, batch_size={batch_size}")
    return np.concatenate(results)


def main():
    parser = argparse.ArgumentParser(description="Nanog H1-ESC replication")
    parser.add_argument(
        "--skip-train", action="store_true", help="Skip training from scratch"
    )
    parser.add_argument(
        "--skip-modisco", action="store_true", help="Skip TF-MoDISco analysis"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mlx", "auto"],
        help="Device for scoring/inference (cuda moves model to GPU)",
    )
    parser.add_argument(
        "--solver",
        default="auto",
        choices=["auto", "smo", "libsvm"],
        help="Solver for training",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="Batch size for GkmExplain (lower if GPU OOM)",
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show progress bars during scoring and attribution")
    args = parser.parse_args()

    print("=" * 70)
    print("Nanog H1-ESC gkm-SVM Replication")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Download data
    # ------------------------------------------------------------------
    print("\n[1/7] Downloading data...")
    download_data()

    # ------------------------------------------------------------------
    # 2. Load reference model
    # ------------------------------------------------------------------
    print("\n[2/7] Loading reference model...")
    from gkmsvm.importers.lsgkm import load_lsgkm_model

    model_path = DATA_DIR / "lsgkm_defaultsettings_t2.model.txt"
    ref_model = load_lsgkm_model(model_path)
    print(f"  Reference model: {ref_model.num_support_vectors} SVs")
    print(f"  Kernel: {ref_model.kernel_type}, params: {ref_model.kernel_params}")

    if args.device == "cuda":
        ref_model.cuda()
        print("  Moved to CUDA GPU")
    elif args.device == "mlx":
        ref_model.mlx()
        print("  Moved to MLX")

    # ------------------------------------------------------------------
    # 3. Score test sequences
    # ------------------------------------------------------------------
    print("\n[3/7] Scoring test sequences...")
    from gkmsvm.codec import one_hot_encode

    pos_test = read_fasta(DATA_DIR / "positives_test.fa")
    neg_test = read_fasta(DATA_DIR / "negatives_test.fa")

    pos_x_np = np.stack([one_hot_encode(s) for _, s in pos_test])
    neg_x_np = np.stack([one_hot_encode(s) for _, s in neg_test])

    pos_x = _to_device(pos_x_np, args.device)
    neg_x = _to_device(neg_x_np, args.device)

    t0 = time.perf_counter()
    pos_scores = _to_numpy(ref_model(pos_x, verbose=args.verbose).squeeze(-1), args.device)
    neg_scores = _to_numpy(ref_model(neg_x, verbose=args.verbose).squeeze(-1), args.device)
    t_score = time.perf_counter() - t0

    print(f"  Scored {len(pos_scores) + len(neg_scores)} sequences in {t_score:.2f}s")
    print(f"  Pos mean: {pos_scores.mean():.4f} +/- {pos_scores.std():.4f}")
    print(f"  Neg mean: {neg_scores.mean():.4f} +/- {neg_scores.std():.4f}")
    print(f"  Separation: {pos_scores.mean() - neg_scores.mean():.4f}")

    from sklearn.metrics import roc_auc_score
    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    auc = roc_auc_score(all_labels, all_scores)
    print(f"  Test AUC: {auc:.4f}")

    # ------------------------------------------------------------------
    # 4. Train from scratch
    # ------------------------------------------------------------------
    if not args.skip_train:
        print("\n[4/7] Training from scratch...")
        from gkmsvm import train_gkmsvm

        pos_train = read_fasta(DATA_DIR / "positives_train.fa")
        neg_train = read_fasta(DATA_DIR / "negatives_train.fa")

        pos_seqs = [s for _, s in pos_train]
        neg_seqs = [s for _, s in neg_train]
        print(f"  Training: {len(pos_seqs)} pos + {len(neg_seqs)} neg sequences")

        t0 = time.perf_counter()
        our_model = train_gkmsvm(
            pos_seqs, neg_seqs,
            kernel_type="estimated", l=11, k=7, d=3, C=1.0,
            solver=args.solver,
            device=args.device if args.device != "auto" else "auto",
            verbose=args.verbose,
        )
        t_train = time.perf_counter() - t0
        print(f"  Trained in {t_train:.1f}s — {our_model.num_support_vectors} SVs")

        our_pos = our_model(pos_x_np).squeeze(-1)
        our_neg = our_model(neg_x_np).squeeze(-1)
        our_auc = roc_auc_score(all_labels, np.concatenate([our_pos, our_neg]))
        print(f"  Our test AUC: {our_auc:.4f} (ref: {auc:.4f})")

        our_all = np.concatenate([our_pos, our_neg])
        corr = np.corrcoef(all_scores, our_all)[0, 1]
        print(f"  Score correlation with reference: {corr:.4f}")
    else:
        print("\n[4/7] Skipping training (--skip-train)")

    # ------------------------------------------------------------------
    # 5. GkmExplain on all positive test sequences
    # ------------------------------------------------------------------
    print(f"\n[5/7] Running GkmExplain on {len(pos_x_np)} positive test seqs...")

    # GkmExplain: CUDA has fp64 RawKernel, MLX needs CPU fallback
    explain_device = args.device if args.device != "mlx" else "cpu"
    if explain_device == "mlx":
        exp_model = load_lsgkm_model(model_path)
    else:
        exp_model = ref_model

    # Mode 1: hypothetical contribution scores (mode 0 = mode 1 × OHE)
    t0 = time.perf_counter()
    hyp_scores = _gkmexplain_batched(
        exp_model, pos_x_np, mode=1,
        batch_size=args.batch_size, device=explain_device,
        verbose=args.verbose,
    )
    t_explain = time.perf_counter() - t0
    print(f"  GkmExplain: {len(pos_x_np)} seqs in {t_explain:.1f}s "
          f"({len(pos_x_np)/t_explain:.1f} seq/s)")
    imp_scores = hyp_scores * pos_x_np

    # Completeness check on full set
    ref_full_scores = _to_numpy(
        ref_model(pos_x, verbose=args.verbose).squeeze(-1), args.device
    )
    exp_sums = (imp_scores * pos_x_np).sum(axis=(1, 2))
    expected = ref_full_scores - ref_model.bias
    max_error = np.max(np.abs(exp_sums - expected.astype(np.float64)))
    print(f"  Completeness max error: {max_error:.2e}")

    # ------------------------------------------------------------------
    # 6. Compare against reference gkmexplain scores
    # ------------------------------------------------------------------
    print("\n[6/7] Comparing against reference gkmexplain scores...")
    ref_hyp_path = DATA_DIR / "gkmexplain_positives_hypimpscores.txt.gz"
    ref_imp_path = DATA_DIR / "gkmexplain_positives_impscores.txt.gz"

    if ref_hyp_path.exists() and ref_imp_path.exists():
        ref_hyp = _parse_lsgkm_explain_scores(ref_hyp_path)
        ref_imp = _parse_lsgkm_explain_scores(ref_imp_path)
        print(f"  Reference shapes: hyp={ref_hyp.shape}, imp={ref_imp.shape}")

        n_compare = min(len(hyp_scores), len(ref_hyp))
        our_hyp_flat = hyp_scores[:n_compare].ravel()
        ref_hyp_flat = ref_hyp[:n_compare].ravel()
        hyp_corr = np.corrcoef(our_hyp_flat, ref_hyp_flat)[0, 1]
        hyp_mae = np.mean(np.abs(our_hyp_flat - ref_hyp_flat))
        print(f"  Hypothetical scores — corr: {hyp_corr:.6f}, MAE: {hyp_mae:.2e}")

        our_imp_flat = imp_scores[:n_compare].ravel()
        ref_imp_flat = ref_imp[:n_compare].ravel()
        imp_corr = np.corrcoef(our_imp_flat, ref_imp_flat)[0, 1]
        imp_mae = np.mean(np.abs(our_imp_flat - ref_imp_flat))
        print(f"  Importance scores   — corr: {imp_corr:.6f}, MAE: {imp_mae:.2e}")

        nonzero = ref_imp_flat != 0
        imp_corr_nz = np.corrcoef(our_imp_flat[nonzero], ref_imp_flat[nonzero])[0, 1]
        print(f"  Importance (nonzero only) — corr: {imp_corr_nz:.6f}")
    else:
        print("  Reference scores not found — skipping comparison")

    # ------------------------------------------------------------------
    # 7. TF-MoDISco motif discovery
    # ------------------------------------------------------------------
    if not args.skip_modisco:
        print(f"\n[7/7] Running TF-MoDISco on {len(pos_x_np)} sequences...")
        try:
            from modiscolite import tfmodisco
        except ImportError:
            print("  modisco-lite not installed. Install with: "
                  "uv pip install --system modisco-lite")
            print("  Skipping TF-MoDISco.")
        else:
            # modisco expects [N, L, 4] — transpose from our [N, 4, L]
            onehot_NL4 = pos_x_np.transpose(0, 2, 1)
            hyp_NL4 = hyp_scores.transpose(0, 2, 1)

            t0 = time.perf_counter()
            pos_patterns, neg_patterns = tfmodisco.TFMoDISco(
                onehot_NL4, hyp_NL4,
                verbose=True,
            )
            t_modisco = time.perf_counter() - t0
            print(f"  TF-MoDISco completed in {t_modisco:.1f}s")

            for label, patterns in [("Positive", pos_patterns),
                                    ("Negative", neg_patterns)]:
                if patterns is None:
                    print(f"\n  {label} patterns: none found")
                    continue
                print(f"\n  {label} patterns: {len(patterns)}")
                for i, pattern in enumerate(patterns):
                    n_seqlets = len(pattern.seqlets)
                    ic = _information_content(pattern.sequence)
                    consensus = _consensus_from_ppm(pattern.sequence)
                    print(f"    Pattern {i}: {n_seqlets} seqlets, "
                          f"IC={ic:.1f} bits, consensus={consensus}")

            out_path = DATA_DIR / "modisco_results.h5"
            from modiscolite import io as modisco_io
            modisco_io.save_hdf5(out_path, pos_patterns, neg_patterns,
                                 window_size=21)
            print(f"\n  Results saved to {out_path}")
    else:
        print("\n[7/7] Skipping TF-MoDISco (--skip-modisco)")

    print("\n" + "=" * 70)
    print("Replication complete.")
    print("=" * 70)


def _information_content(ppm):
    """Total information content of a PPM [L, 4]."""
    ppm = np.clip(ppm, 1e-10, 1.0)
    ppm = ppm / ppm.sum(axis=1, keepdims=True)
    entropy = -np.sum(ppm * np.log2(ppm), axis=1)
    ic = 2.0 - entropy
    return float(ic.sum())


def _consensus_from_ppm(ppm):
    """Consensus sequence from a PPM [L, 4]."""
    bases = "ACGT"
    return "".join(bases[i] for i in ppm.argmax(axis=1))


if __name__ == "__main__":
    main()
