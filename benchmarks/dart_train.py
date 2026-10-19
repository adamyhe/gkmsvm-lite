"""Retrain the DART-Eval Task 4 gkm-SVM models on DART-Eval's training chromosomes.

The ENCODE gkm-SVM models that dart_eval.py scores hold out no test set, so
they have seen peaks on DART-Eval's test chromosomes. This trains one model per
cell line on DART-Eval's training split only (data.h5 <cell>/train: chr1-4,
7-9, 11-13, 15-17, 19, X and Y, as for DART-Eval's models), following the
ENCODE recipe where it can: -t 2 l=11 k=7 d=3 C=1 on 300 bp around the summit,
top 10k peaks against 1x negatives.

  Positives: the n_pos training IDR peaks with the most reads.
  Negatives: n_neg random training background regions (DART-Eval's GC-matched
             nonpeaks).

Unlike ENCODE's models, promoters are kept (DART-Eval's test peaks include
them) and the model is one SVM, not five CV folds concatenated.

Writes the model (.npz, for dart_eval.py -m) and a one-row TSV with the
training set sizes, SV count and training time.

Usage:
  python benchmarks/dart_train.py --cell-line GM12878 --device cuda \\
      --model-output data/models/retrained/GM12878.npz \\
      --output results/dart_retrained_train_GM12878.tsv
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _common import write_tsv
from dart_eval import CELL_LINES, DART_SEQ_LEN


def load_train_split(
    work_dir: str, cell_line: str, n_pos: int, n_neg: int, seq_len: int, seed: int = 0
):
    """Top n_pos IDR peaks by counts and n_neg random nonpeaks, [N, 4, seq_len]."""
    import h5py

    lo = (DART_SEQ_LEN - seq_len) // 2  # same window as dart_eval._center_crop
    path = os.path.join(work_dir, "task_4_chromatin_activity/data.h5")
    with h5py.File(path, "r") as f:
        train = f[cell_line]["train"]
        counts = train["idr_peaks"]["counts"][:]
        pos_idx = np.sort(np.argsort(-counts, kind="stable")[:n_pos])
        n_bg = train["nonpeaks"]["seqs"].shape[0]
        neg_idx = np.sort(
            np.random.default_rng(seed).choice(n_bg, min(n_neg, n_bg), replace=False)
        )
        pos = train["idr_peaks"]["seqs"][:, lo : lo + seq_len][pos_idx]
        neg = train["nonpeaks"]["seqs"][:, lo : lo + seq_len][neg_idx]
    # DART-Eval stores [N, L, 4] uint8
    return (
        pos.transpose(0, 2, 1).astype(np.float32),
        neg.transpose(0, 2, 1).astype(np.float32),
        len(counts),
        n_bg,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cell-line", required=True, choices=CELL_LINES)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--work-dir",
        default=None,
        help="DART-Eval data (default: $DART_WORK_DIR or data/dart-eval)",
    )
    parser.add_argument("--n-pos", type=int, default=10_000)
    parser.add_argument("--n-neg", type=int, default=10_000)
    parser.add_argument("--seq-len", type=int, default=300)
    parser.add_argument("--model-output", required=True, help="Model path (.npz)")
    parser.add_argument("--output", default=None, help="Training summary TSV")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    from gkmsvm import train_gkmsvm
    from gkmsvm.serialization import save_npz

    work_dir = (
        args.work_dir
        or os.environ.get("DART_WORK_DIR")
        or os.path.join(os.path.dirname(__file__), "data", "dart-eval")
    )
    pos, neg, n_idr, n_bg = load_train_split(
        work_dir, args.cell_line, args.n_pos, args.n_neg, args.seq_len
    )
    print(
        f"{args.cell_line}: {len(pos)} of {n_idr} training IDR peaks, "
        f"{len(neg)} of {n_bg} training nonpeaks, {args.seq_len} bp"
    )

    params = dict(kernel_type="estimated", l=11, k=7, d=3, C=1.0)
    t0 = time.perf_counter()
    model = train_gkmsvm(pos, neg, **params, device=args.device, verbose=args.verbose)
    train_s = time.perf_counter() - t0
    print(f"  {model.num_support_vectors} SVs, {train_s:.1f}s")

    model.cpu()
    Path(args.model_output).parent.mkdir(parents=True, exist_ok=True)
    save_npz(model, args.model_output)
    print(f"Saved {args.model_output}")

    if args.output:
        row = {
            "cell_line": args.cell_line,
            "n_pos": len(pos),
            "n_neg": len(neg),
            "n_idr_train": n_idr,
            "n_nonpeaks_train": n_bg,
            "n_sv": model.num_support_vectors,
            "train_s": train_s,
        }
        write_tsv(
            args.output,
            pd.DataFrame([row]),
            "gkm-SVM retrained on DART-Eval's Task 4 training chromosomes",
            args.device,
            model=os.path.basename(args.model_output),
            seq_len=args.seq_len,
            **params,
        )


if __name__ == "__main__":
    main()
