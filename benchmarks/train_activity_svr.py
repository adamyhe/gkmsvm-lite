"""Train gkm-SVR for DART-Eval Task 4: cell type activity regression.

Trains epsilon-SVR on ATAC-seq peak sequences with log1p(counts) as
regression targets. Evaluates both regression (Pearson/Spearman of
predicted vs actual counts) and classification (AUROC/AUPRC of peak
vs nonpeak) on held-out test chromosomes.

Trained models are cached to benchmarks/models/ and reloaded on
subsequent runs. Pass --force-train to retrain.

Data:
  DART-Eval Task 4 H5 with one-hot sequences and counts per cell line.
  Set DART_WORK_DIR or pass --work-dir.

    synapse get -r syn60581044 --downloadLocation $DART_WORK_DIR/refs
    synapse get -r syn60581041 --downloadLocation $DART_WORK_DIR/task_4_chromatin_activity

Usage:
  python benchmarks/train_activity_svr.py --device cuda
  python benchmarks/train_activity_svr.py --device cuda --cell-lines GM12878 K562
  python benchmarks/train_activity_svr.py --force-train
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

CELL_LINES = ["GM12878", "H1ESC", "HEPG2", "IMR90", "K562"]
DART_SEQ_LEN = 2114
MODEL_DIR = Path(__file__).parent / "models"


def _model_path(cell_line: str, l: int, k: int, d: int,
                C: float, epsilon: float) -> Path:
    return MODEL_DIR / f"{cell_line}_svr_l{l}k{k}d{d}_C{C}_eps{epsilon}.npz"


def _h5_to_channels_first(seqs_h5: np.ndarray) -> np.ndarray:
    return seqs_h5.astype(np.float64).transpose(0, 2, 1)


def _center_crop(seqs: np.ndarray, target_len: int) -> np.ndarray:
    L = seqs.shape[2]
    if L <= target_len:
        return seqs
    start = (L - target_len) // 2
    return seqs[:, :, start:start + target_len]


def _score_batched(model, seqs: np.ndarray, batch_size: int,
                   verbose: bool = True) -> np.ndarray:
    from gkmsvm.backend import to_cpu

    N = seqs.shape[0]
    scores = np.zeros(N, dtype=np.float64)
    chunks = range(0, N, batch_size)
    if verbose:
        from tqdm import tqdm
        chunks = tqdm(chunks, desc="scoring",
                      total=(N + batch_size - 1) // batch_size)
    for start in chunks:
        end = min(start + batch_size, N)
        xb = model._match_device(seqs[start:end])
        scores[start:end] = to_cpu(model(xb, verbose=False).flatten())
    return scores


def load_train_data(h5_path: str, cell_line: str, target_len: int,
                    max_seqs: int | None):
    import h5py

    with h5py.File(h5_path, "r") as f:
        if cell_line not in f:
            print(f"  SKIP {cell_line}: not in H5", file=sys.stderr)
            return None, None, None
        train = f[cell_line]["train"]

        if "peaks" not in train or "counts" not in train["peaks"]:
            print(f"  SKIP {cell_line}: no peak counts in train split",
                  file=sys.stderr)
            return None, None, None

        n_peaks = train["peaks"]["seqs"].shape[0]
        n_nonpeaks = train["nonpeaks"]["seqs"].shape[0]
        print(f"  Available: {n_peaks} peaks, {n_nonpeaks} nonpeaks")

        if max_seqs and max_seqs < n_peaks:
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(n_peaks, max_seqs, replace=False))
            seqs = _h5_to_channels_first(train["peaks"]["seqs"][idx])
            counts = train["peaks"]["counts"][idx]
        else:
            seqs = _h5_to_channels_first(train["peaks"]["seqs"][:])
            counts = train["peaks"]["counts"][:]

    seqs = _center_crop(seqs, target_len)
    targets = np.log1p(counts).astype(np.float64)
    print(f"  Training: {len(seqs)} peaks, {target_len} bp")
    print(f"  Target range: [{targets.min():.2f}, {targets.max():.2f}] "
          f"(log1p counts)")
    return seqs, targets, counts


def load_test_data(h5_path: str, cell_line: str, target_len: int):
    import h5py

    with h5py.File(h5_path, "r") as f:
        if cell_line not in f:
            return None
        test = f[cell_line]["test"]
        result = {}
        for key in ("peaks", "idr_peaks", "nonpeaks"):
            if key in test:
                entry = {"seqs": _center_crop(
                    _h5_to_channels_first(test[key]["seqs"][:]),
                    target_len)}
                if "counts" in test[key]:
                    entry["counts"] = test[key]["counts"][:]
                result[key] = entry
    return result


def train_svr(seqs, targets, l, k, d, C, epsilon, device, verbose):
    from gkmsvm.train import train_gkmsvr

    print(f"\n  Training gkm-SVR (l={l}, k={k}, d={d}, C={C}, "
          f"epsilon={epsilon})...")
    t0 = time.perf_counter()
    model = train_gkmsvr(
        seqs, targets,
        l=l, k=k, d=d,
        C=C, epsilon=epsilon,
        kernel_type="estimated",
        device=device,
        verbose=verbose,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Trained in {elapsed:.1f}s: {model.num_support_vectors} SVs")
    return model


def get_or_train_model(cell_line, seqs, targets, l, k, d, C, epsilon,
                       device, force_train, verbose):
    from gkmsvm.serialization import save_npz, load_model

    path = _model_path(cell_line, l, k, d, C, epsilon)

    if path.exists() and not force_train:
        print(f"  Loading cached model: {path}")
        model = load_model(str(path))
        if device == "cuda":
            model.cuda()
        return model

    model = train_svr(seqs, targets, l, k, d, C, epsilon, device, verbose)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.cpu()
    save_npz(model, str(path))
    print(f"  Saved model to {path}")
    if device == "cuda":
        model.cuda()
    return model


def evaluate_cell_line(model, test_data, batch_size, device, verbose):
    results = {}

    if device == "cuda":
        model.cuda()

    if "peaks" in test_data and "counts" in test_data["peaks"]:
        peak_seqs = test_data["peaks"]["seqs"]
        counts = test_data["peaks"]["counts"]
        targets = np.log1p(counts).astype(np.float64)

        print(f"  Scoring {len(peak_seqs)} test peaks...")
        preds = _score_batched(model, peak_seqs, batch_size, verbose)

        r_p, _ = pearsonr(preds, targets)
        r_s, _ = spearmanr(preds, targets)
        mse = np.mean((preds - targets) ** 2)
        results["regression"] = {
            "pearson": r_p,
            "spearman": r_s,
            "mse": mse,
            "n": len(preds),
        }
        print(f"  Regression: Pearson={r_p:.4f}, Spearman={r_s:.4f}, "
              f"MSE={mse:.4f}")

    if "idr_peaks" in test_data and "nonpeaks" in test_data:
        idr_seqs = test_data["idr_peaks"]["seqs"]
        nonpeak_seqs = test_data["nonpeaks"]["seqs"]

        print(f"  Scoring {len(idr_seqs)} IDR peaks + "
              f"{len(nonpeak_seqs)} nonpeaks...")
        idr_scores = _score_batched(model, idr_seqs, batch_size, verbose)
        nonpeak_scores = _score_batched(model, nonpeak_seqs, batch_size,
                                        verbose)

        labels = np.concatenate([np.ones(len(idr_scores)),
                                 np.zeros(len(nonpeak_scores))])
        all_scores = np.concatenate([idr_scores, nonpeak_scores])
        auroc = roc_auc_score(labels, all_scores)
        auprc = average_precision_score(labels, all_scores)

        results["classification"] = {
            "auroc": auroc,
            "auprc": auprc,
            "n_idr": len(idr_scores),
            "n_nonpeaks": len(nonpeak_scores),
        }
        print(f"  Classification: AUROC={auroc:.4f}, AUPRC={auprc:.4f}")

    model.cpu()
    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--cell-lines", nargs="+", default=CELL_LINES,
                        choices=CELL_LINES)
    parser.add_argument("--l", type=int, default=11, dest="l_param")
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--d", type=int, default=3, dest="d_param")
    parser.add_argument("--C", type=float, default=1.0, dest="C_param")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--target-len", type=int, default=300,
                        help="Crop sequences to this length (default: 300).")
    parser.add_argument("--max-train-seqs", type=int, default=None,
                        help="Subsample training peaks for speed.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--work-dir", default=None,
                        help="DART-Eval data directory (overrides DART_WORK_DIR).")
    parser.add_argument("--force-train", action="store_true",
                        help="Retrain even if cached model exists.")
    parser.add_argument("--output", default=None,
                        help="Save results summary to TSV file.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    work_dir = (args.work_dir or os.environ.get("DART_WORK_DIR", "")
                or str(Path(__file__).parent / "data" / "dart-eval"))

    h5_path = os.path.join(work_dir, "task_4_chromatin_activity/data.h5")
    if not os.path.exists(h5_path):
        from dart_download import download_dart_data
        download_dart_data(work_dir, tasks=("task_4",))

    print("DART-Eval Task 4: Cell Type Activity SVR")
    print(f"  Data: {h5_path}")
    print(f"  Device: {args.device}")
    print(f"  Params: l={args.l_param}, k={args.k}, d={args.d_param}, "
          f"C={args.C_param}, epsilon={args.epsilon}")

    all_results = []

    for cell_line in args.cell_lines:
        print(f"\n{'=' * 60}")
        print(f"{cell_line}")
        print(f"{'=' * 60}")

        seqs, targets, _ = load_train_data(
            h5_path, cell_line, args.target_len, args.max_train_seqs)
        if seqs is None:
            continue

        model = get_or_train_model(
            cell_line, seqs, targets,
            args.l_param, args.k, args.d_param,
            args.C_param, args.epsilon,
            args.device, args.force_train, args.verbose)

        del seqs, targets
        import gc; gc.collect()

        print(f"\n  --- Test evaluation ---")
        test_data = load_test_data(h5_path, cell_line, args.target_len)
        if test_data is None:
            print(f"  SKIP {cell_line}: no test data")
            continue

        results = evaluate_cell_line(
            model, test_data, args.batch_size, args.device, args.verbose)
        results["cell_line"] = cell_line
        all_results.append(results)

        del model, test_data
        import gc; gc.collect()

    # Summary
    print(f"\n{'=' * 70}")
    print("Summary: gkm-SVR Activity Regression")
    print(f"{'=' * 70}")

    has_reg = any("regression" in r for r in all_results)
    has_cls = any("classification" in r for r in all_results)

    if has_reg:
        print(f"\n  Regression (predicted vs actual log1p counts)")
        print(f"  {'Cell line':<12} {'Pearson':>8} {'Spearman':>8} "
              f"{'MSE':>8} {'N':>8}")
        print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for r in all_results:
            if "regression" not in r:
                continue
            reg = r["regression"]
            print(f"  {r['cell_line']:<12} {reg['pearson']:>8.4f} "
                  f"{reg['spearman']:>8.4f} {reg['mse']:>8.4f} "
                  f"{reg['n']:>8}")

    if has_cls:
        print(f"\n  Classification (IDR peak vs nonpeak)")
        print(f"  {'Cell line':<12} {'AUROC':>8} {'AUPRC':>8} "
              f"{'IDR':>8} {'Nonpeak':>8}")
        print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for r in all_results:
            if "classification" not in r:
                continue
            cls = r["classification"]
            print(f"  {r['cell_line']:<12} {cls['auroc']:>8.4f} "
                  f"{cls['auprc']:>8.4f} {cls['n_idr']:>8} "
                  f"{cls['n_nonpeaks']:>8}")

    if args.output and all_results:
        import pandas as pd
        rows = []
        for r in all_results:
            row = {"cell_line": r["cell_line"]}
            if "regression" in r:
                row.update({f"reg_{k}": v
                            for k, v in r["regression"].items()})
            if "classification" in r:
                row.update({f"cls_{k}": v
                            for k, v in r["classification"].items()})
            rows.append(row)
        pd.DataFrame(rows).to_csv(args.output, sep="\t", index=False)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
