"""Train a gkm-SVM on GM12878 ATAC peaks and evaluate on African caQTLs.

Trains on Beer lab GM12878 DNase-seq peaks (same training data as the
pretrained SVC models in bench_dsqtl.py), then evaluates VEP on DART-Eval
Task 5 African caQTLs. Only Task 5 data is downloaded from Synapse.

Data:
  - Training: examples/data/gm12878_sequence_sets/ (Beer lab, 14 MB)
  - Task 5 H5: benchmarks/data/dart-eval/task_5_variant_effect_prediction/data.h5
  - Task 5 TSV: benchmarks/data/dart-eval/task_5_variant_effect_prediction/
                input_data/Afr.CaQTLS.tsv

Usage:
  python benchmarks/train_caqtl_model.py --device cuda
  python benchmarks/train_caqtl_model.py --device cuda --model benchmarks/models/gm12878_l11k7_neg1.npz
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parent.parent
SEQ_DIR = REPO / "examples" / "data" / "gm12878_sequence_sets"
SEQ_URL = "https://beerlab.org/deltasvm/downloads/gm12878_sequence_sets.tar.gz"

DART_SEQ_LEN = 2114
MODEL_DIR = Path(__file__).parent / "models"


def _h5_to_channels_first(seqs_h5: np.ndarray) -> np.ndarray:
    """Convert [N, L, 4] uint8 (DART-Eval H5) → [N, 4, L] float64."""
    return seqs_h5.astype(np.float64).transpose(0, 2, 1)


def _center_crop(seqs: np.ndarray, target_len: int) -> np.ndarray:
    L = seqs.shape[2]
    if L <= target_len:
        return seqs
    start = (L - target_len) // 2
    return seqs[:, :, start:start + target_len]


def download_training_data():
    """Download Beer lab GM12878 sequences if not present."""
    if SEQ_DIR.exists():
        return
    import tarfile
    from urllib.request import urlretrieve

    SEQ_DIR.parent.mkdir(parents=True, exist_ok=True)
    tarball = SEQ_DIR.parent / "gm12878_sequence_sets.tar.gz"
    if not tarball.exists():
        print("Downloading Beer lab sequence sets (14 MB)...")
        urlretrieve(SEQ_URL, tarball)
    print("Extracting...")
    with tarfile.open(tarball) as tar:
        tar.extractall(SEQ_DIR.parent, filter="data")


def load_training_data():
    """Load GM12878 pos/neg sequences from Beer lab data."""
    from gkmsvm import read_fasta

    pos_seqs = [seq for _, seq in read_fasta(
        str(SEQ_DIR / "gm12878_shared.fa"))]
    neg_seqs = [seq for _, seq in read_fasta(
        str(SEQ_DIR / "nullseqs_gm12878_shared.1.1.fa"))]

    print(f"  Training: {len(pos_seqs)} pos + {len(neg_seqs)} neg, "
          f"{len(pos_seqs[0])}bp")
    return pos_seqs, neg_seqs


def train_model(pos_seqs, neg_seqs, l, k, d, C, device, solver="auto",
                verbose=False):
    from gkmsvm.train import train_gkmsvm

    print(f"\n  Training gkm-SVM (l={l}, k={k}, d={d}, C={C})...")

    t0 = time.perf_counter()
    model = train_gkmsvm(
        pos_seqs, neg_seqs,
        l=l, k=k, d=d,
        C=C,
        kernel_type="estimated",
        device=device,
        solver=solver,
        verbose=verbose,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Trained in {elapsed:.1f}s: {model.num_support_vectors} SVs")
    return model


def load_caqtl_data(work_dir: str, sv_len: int):
    """Load caQTL variants, pre-filtered to IsUsed & in_peaks."""
    import h5py

    tsv_path = os.path.join(
        work_dir,
        "task_5_variant_effect_prediction/input_data/Afr.CaQTLS.tsv")
    h5_path = os.path.join(
        work_dir,
        "task_5_variant_effect_prediction/data.h5")

    if not os.path.exists(tsv_path) or not os.path.exists(h5_path):
        print(f"Error: missing caQTL data files", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(tsv_path, sep="\t")
    mask = (df["IsUsed"] == True) & (df["in_peaks"] == True)
    keep_idx = np.where(mask.values)[0]
    print(f"  caQTL variants: {len(keep_idx)}/{len(df)} pass filter "
          f"(IsUsed & in_peaks)")

    with h5py.File(h5_path, "r") as f:
        grp = f["Afr.CaQTLS.tsv"]
        idx = np.sort(keep_idx)
        a1_seqs = _h5_to_channels_first(grp["allele_1_seqs"][idx])
        a2_seqs = _h5_to_channels_first(grp["allele_2_seqs"][idx])

    a1_seqs = _center_crop(a1_seqs, sv_len)
    a2_seqs = _center_crop(a2_seqs, sv_len)
    print(f"  Center-cropped {DART_SEQ_LEN} bp → {sv_len} bp")

    df_filtered = df.iloc[keep_idx].reset_index(drop=True)
    return df_filtered, a1_seqs, a2_seqs


def score_vep(model, a1_seqs, a2_seqs, batch_size, device, verbose=True):
    return model.score_variants(
        a1_seqs, a2_seqs, batch_size=batch_size, verbose=verbose,
    ).flatten()


def evaluate_caqtl(df, scores):
    sig = df[df["label"] == 1]
    ctrl = df[df["label"] == 0]
    n_sig = len(sig)
    n_ctrl = len(ctrl)

    labels = np.concatenate([np.zeros(n_ctrl), np.ones(n_sig)])
    abs_scores = np.concatenate([np.abs(scores[ctrl.index]),
                                 np.abs(scores[sig.index])])

    auroc = roc_auc_score(labels, abs_scores)
    auprc = average_precision_score(labels, abs_scores)

    beta_col = "Beta" if "Beta" in df.columns else "beta"
    sig_with_beta = sig.dropna(subset=[beta_col])
    result = {
        "n_sig": n_sig,
        "n_ctrl": n_ctrl,
        "auroc": auroc,
        "auprc": auprc,
    }

    if len(sig_with_beta) > 10:
        r_p, _ = pearsonr(scores[sig_with_beta.index],
                          sig_with_beta[beta_col])
        r_s, _ = spearmanr(scores[sig_with_beta.index],
                           sig_with_beta[beta_col])
        result["pearson"] = r_p
        result["spearman"] = r_s

    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--model", default=None,
                        help="Pre-trained model file (skip training).")
    parser.add_argument("--l", type=int, default=11, dest="l_param")
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--d", type=int, default=3, dest="d_param")
    parser.add_argument("--C", type=float, default=1.0, dest="C_param")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--solver", default="auto",
                        choices=["auto", "nystrom", "libsvm"],
                        help="Training solver (default: auto).")
    parser.add_argument("--output", default=None,
                        help="Save results to TSV file.")
    parser.add_argument("--work-dir", default=None,
                        help="DART-Eval data directory (overrides DART_WORK_DIR).")
    parser.add_argument("--force-train", action="store_true",
                        help="Retrain even if cached model exists.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    work_dir = (args.work_dir or os.environ.get("DART_WORK_DIR", "")
                or str(Path(__file__).parent / "data" / "dart-eval"))

    print("=" * 60)
    print("caQTL Benchmark: Train GM12878 ATAC → Score African caQTLs")
    print("=" * 60)

    if args.model:
        from gkmsvm.serialization import load_model
        print(f"\nLoading pre-trained model: {args.model}")
        model = load_model(args.model)
        if args.device == "cuda":
            model.cuda()
    else:
        from gkmsvm.serialization import save_npz, load_model

        cached = (MODEL_DIR /
                  f"caqtl_svc_l{args.l_param}k{args.k}d{args.d_param}"
                  f"_C{args.C_param}.npz")

        if cached.exists() and not args.force_train:
            print(f"\nLoading cached model: {cached}")
            model = load_model(str(cached))
            if args.device == "cuda":
                model.cuda()
        else:
            download_training_data()
            print(f"\n--- Loading training data ---")
            pos_seqs, neg_seqs = load_training_data()
            model = train_model(pos_seqs, neg_seqs,
                                l=args.l_param, k=args.k, d=args.d_param,
                                C=args.C_param, device=args.device,
                                solver=args.solver, verbose=args.verbose)
            del pos_seqs, neg_seqs
            gc.collect()

            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.cpu()
            save_npz(model, str(cached))
            print(f"  Saved model to {cached}")
            if args.device == "cuda":
                model.cuda()

    sv_len = model.support_sequences.shape[2]
    print(f"\n  Model: {model.num_support_vectors} SVs, "
          f"L={model.kernel.l}, k={model.kernel.k}, "
          f"SV length: {sv_len} bp")

    # Download Task 5 only when needed for evaluation
    h5_task5 = os.path.join(work_dir,
                            "task_5_variant_effect_prediction/data.h5")
    if not os.path.exists(h5_task5):
        from dart_download import download_dart_data
        download_dart_data(work_dir, tasks=("task_5",))

    print(f"\n--- Loading caQTL data ---")
    df, a1_seqs, a2_seqs = load_caqtl_data(work_dir, sv_len)

    print(f"\n--- Scoring caQTL variants (kernel VEP) ---")
    t0 = time.perf_counter()
    scores = score_vep(model, a1_seqs, a2_seqs, args.batch_size,
                       args.device, verbose=args.verbose)
    elapsed = time.perf_counter() - t0
    print(f"  Scored {len(scores)} variants in {elapsed:.1f}s")

    del a1_seqs, a2_seqs
    gc.collect()

    print(f"\n--- caQTL Evaluation ---")
    metrics = evaluate_caqtl(df, scores)
    print(f"  Variants: {metrics['n_sig']} sig, {metrics['n_ctrl']} ctrl")
    print(f"  AUROC:    {metrics['auroc']:.4f}")
    print(f"  AUPRC:    {metrics['auprc']:.4f}")
    if "pearson" in metrics:
        print(f"  Pearson:  {metrics['pearson']:.4f}")
        print(f"  Spearman: {metrics['spearman']:.4f}")

    if args.output:
        out_df = pd.DataFrame([metrics])
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        out_df.to_csv(args.output, sep="\t", index=False)
        print(f"\n  Saved results to {args.output}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
