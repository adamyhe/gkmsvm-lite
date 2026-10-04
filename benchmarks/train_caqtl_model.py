"""Train a gkm-SVM on GM12878 ATAC peaks and evaluate on African caQTLs.

Uses DART-Eval Task 4 training data (GM12878 IDR peaks vs nonpeaks) to train
a gkm-SVM, then evaluates VEP on Task 5 African caQTLs. This tests whether
a matched chromatin model can predict causal QTL variants.

The DART-Eval Task 4 H5 contains ~51K IDR peaks and ~95K nonpeaks for GM12878
training. We subsample to balanced pos/neg sets for SVM training (gkm-SVMs
are trained on balanced data by convention), then score the ~85K filtered
caQTL variants with kernel VEP.

Data:
  - Task 4 H5: benchmarks/data/dart-eval/task_4_chromatin_activity/data.h5
  - Task 5 H5: benchmarks/data/dart-eval/task_5_variant_effect_prediction/data.h5
  - Task 5 TSV: benchmarks/data/dart-eval/task_5_variant_effect_prediction/
                input_data/Afr.CaQTLS.tsv

Usage:
  python benchmarks/train_caqtl_model.py --device cuda
  python benchmarks/train_caqtl_model.py --device cuda --max-train-seqs 10000  # quick
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

DART_WORK_DIR = os.environ.get("DART_WORK_DIR", "benchmarks/data/dart-eval")
CROP = 557
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


def load_training_data(max_seqs: int | None, target_len: int) -> tuple:
    """Load GM12878 ATAC IDR peaks + nonpeaks from Task 4 H5."""
    import h5py

    h5_path = os.path.join(DART_WORK_DIR,
                           "task_4_chromatin_activity/data.h5")
    if not os.path.exists(h5_path):
        print(f"Error: {h5_path} not found", file=sys.stderr)
        sys.exit(1)

    with h5py.File(h5_path, "r") as f:
        train = f["GM12878"]["train"]

        n_pos = train["idr_peaks"]["seqs"].shape[0]
        n_neg = train["nonpeaks"]["seqs"].shape[0]
        print(f"  Available: {n_pos} IDR peaks, {n_neg} nonpeaks")

        n_use = min(n_pos, n_neg)
        if max_seqs:
            n_use = min(n_use, max_seqs // 2)
        print(f"  Using: {n_use} pos + {n_use} neg = {2 * n_use} sequences")

        rng = np.random.default_rng(42)
        pos_idx = np.sort(rng.choice(n_pos, n_use, replace=False))
        neg_idx = np.sort(rng.choice(n_neg, n_use, replace=False))

        print(f"  Loading positive sequences...")
        pos_seqs = _h5_to_channels_first(train["idr_peaks"]["seqs"][pos_idx])
        print(f"  Loading negative sequences...")
        neg_seqs = _h5_to_channels_first(train["nonpeaks"]["seqs"][neg_idx])

    pos_seqs = _center_crop(pos_seqs, target_len)
    neg_seqs = _center_crop(neg_seqs, target_len)
    print(f"  Center-cropped {DART_SEQ_LEN} bp → {target_len} bp")

    return pos_seqs, neg_seqs


def train_model(pos_seqs, neg_seqs, l, k, d, C, device, solver="auto"):
    """Train a gkm-SVM model."""
    from gkmsvm.train import train_gkmsvm

    print(f"\n  Training gkm-SVM (l={l}, k={k}, d={d}, C={C})...")
    print(f"  Training set: {len(pos_seqs)} pos + {len(neg_seqs)} neg")

    t0 = time.perf_counter()
    model = train_gkmsvm(
        pos_seqs, neg_seqs,
        l=l, k=k, d=d,
        C=C,
        kernel_type="estimated",
        device=device,
        solver=solver,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Trained in {elapsed:.1f}s: {model.num_support_vectors} SVs")
    return model


def load_caqtl_data(sv_len: int):
    """Load caQTL variants, pre-filtered to IsUsed & in_peaks."""
    import h5py

    tsv_path = os.path.join(
        DART_WORK_DIR,
        "task_5_variant_effect_prediction/input_data/Afr.CaQTLS.tsv")
    h5_path = os.path.join(
        DART_WORK_DIR,
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
    """Compute VEP scores: score(alt) - score(ref)."""
    from gkmsvm.backend import to_cpu

    N = a1_seqs.shape[0]
    scores = np.zeros(N, dtype=np.float64)

    chunks = range(0, N, batch_size)
    if verbose:
        from tqdm import tqdm
        chunks = tqdm(chunks, desc="VEP scoring",
                      total=(N + batch_size - 1) // batch_size)

    for start in chunks:
        end = min(start + batch_size, N)
        a1_b = model._match_device(a1_seqs[start:end])
        a2_b = model._match_device(a2_seqs[start:end])
        s1 = to_cpu(model(a1_b, verbose=False).flatten())
        s2 = to_cpu(model(a2_b, verbose=False).flatten())
        scores[start:end] = s2 - s1

    return scores


def evaluate_caqtl(df, scores):
    """Compute DART-Eval caQTL metrics."""
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
    parser.add_argument("--max-train-seqs", type=int, default=None,
                        help="Max training sequences (subsample for speed).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--solver", default="auto",
                        help="Training solver (default: auto).")
    parser.add_argument("--output", default=None,
                        help="Save results to TSV file.")
    parser.add_argument("--force-train", action="store_true",
                        help="Retrain even if cached model exists.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    global DART_WORK_DIR
    if os.environ.get("DART_WORK_DIR"):
        DART_WORK_DIR = os.environ["DART_WORK_DIR"]
    if not DART_WORK_DIR:
        DART_WORK_DIR = str(Path(__file__).parent / "data" / "dart-eval")

    # Auto-download from Synapse if H5 files are missing
    h5_task4 = os.path.join(DART_WORK_DIR,
                            "task_4_chromatin_activity/data.h5")
    h5_task5 = os.path.join(DART_WORK_DIR,
                            "task_5_variant_effect_prediction/data.h5")
    if not os.path.exists(h5_task4) or not os.path.exists(h5_task5):
        from dart_download import download_dart_data
        download_dart_data(DART_WORK_DIR, tasks=("task_4", "task_5"))

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
            sv_len = 300
            print(f"\n--- Loading training data ---")
            pos_seqs, neg_seqs = load_training_data(args.max_train_seqs,
                                                    sv_len)
            model = train_model(pos_seqs, neg_seqs,
                                l=args.l_param, k=args.k, d=args.d_param,
                                C=args.C_param, device=args.device,
                                solver=args.solver)
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

    print(f"\n--- Loading caQTL data ---")
    df, a1_seqs, a2_seqs = load_caqtl_data(sv_len)

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
