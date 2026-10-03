#!/usr/bin/env python3
"""Benchmark: SMO solver vs Gram+sklearn on the Nanog dataset.

Compares training time, number of support vectors, and prediction
agreement between the column-cached SMO solver and the precomputed
Gram matrix + sklearn SVC solver.

Dataset: Nanog ChIP-seq from GkmExplain (Lee 2016)
  Full:    4,687 pos + 5,017 neg training (200bp)
  Fixture: 100 pos + 100 neg (200bp, test subset)

Usage:
    python benchmarks/bench_smo_vs_gram.py                # full dataset, auto GPU
    python benchmarks/bench_smo_vs_gram.py --device cpu    # force CPU
    python benchmarks/bench_smo_vs_gram.py --fixture       # use small test fixture
"""

from __future__ import annotations

import argparse
import gzip
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures"
BENCHMARKS = REPO / "benchmarks" / "data" / "nanog"


def read_fasta(path: Path) -> list[str]:
    """Read plain or gzipped FASTA, return list of sequences."""
    from gkmsvm import read_fasta as _read_fasta

    if path.name.endswith(".gz"):
        with gzip.open(path, "rt") as gz, tempfile.NamedTemporaryFile(
            mode="w", suffix=".fa", delete=False
        ) as tmp:
            tmp.write(gz.read())
            tmp.flush()
            return [seq for _, seq in _read_fasta(tmp.name)]
    else:
        return [seq for _, seq in _read_fasta(str(path))]


def train_and_time(pos, neg, solver, device, verbose=True, **kwargs):
    from gkmsvm import train_gkmsvm

    t0 = time.time()
    model = train_gkmsvm(
        pos, neg,
        kernel_type="estimated",
        l=11, k=7, d=3,
        C=1.0,
        solver=solver,
        device=device,
        verbose=verbose,
        **kwargs,
    )
    elapsed = time.time() - t0
    return model, elapsed


def evaluate(model, test_pos, test_neg, device):
    from gkmsvm import one_hot_encode
    from gkmsvm.backend import to_cpu

    X_test = np.stack([one_hot_encode(s) for s in test_pos + test_neg])
    y_test = np.array([1] * len(test_pos) + [-1] * len(test_neg))

    if device == "cuda":
        model.cuda()
    scores = to_cpu(model(model._match_device(X_test)).flatten())
    model.cpu()

    preds = np.sign(scores)
    acc = (preds == y_test).mean()

    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(y_test, scores)
    return scores, acc, auroc


def main():
    parser = argparse.ArgumentParser(description="SMO vs Gram+sklearn benchmark")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mlx"])
    parser.add_argument("--fixture", action="store_true",
                        help="Use small test fixture (N=200) instead of full dataset")
    parser.add_argument("--cache-size", type=int, default=512,
                        help="SMO kernel column cache size in MB (default: 512)")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        from gkmsvm.backend import HAS_CUPY, HAS_MLX
        if HAS_CUPY:
            device = "cuda"
        elif HAS_MLX:
            device = "mlx"
        else:
            device = "cpu"
    print(f"Device: {device}")

    if args.fixture:
        print("\nLoading Nanog fixture (small subset)...")
        pos_train = read_fasta(FIXTURES / "nanog_pos_train.fa.gz")
        neg_train = read_fasta(FIXTURES / "nanog_neg_train.fa.gz")
        pos_test = read_fasta(FIXTURES / "nanog_pos_test.fa.gz")
        neg_test = read_fasta(FIXTURES / "nanog_neg_test.fa.gz")
    else:
        print("\nLoading full Nanog dataset...")
        pos_train = read_fasta(BENCHMARKS / "positives_train.fa")
        neg_train = read_fasta(BENCHMARKS / "negatives_train.fa")
        pos_test = read_fasta(BENCHMARKS / "positives_test.fa")
        neg_test = read_fasta(BENCHMARKS / "negatives_test.fa")

    N = len(pos_train) + len(neg_train)
    gram_gb = N * N * 8 / 1024**3
    print(f"  Train: {len(pos_train)} pos + {len(neg_train)} neg = {N} sequences")
    print(f"  Test:  {len(pos_test)} pos + {len(neg_test)} neg")
    print(f"  Sequence length: {len(pos_train[0])}bp")
    print(f"  Gram matrix size: {N}x{N} = {gram_gb:.3f} GB")

    # ── Train with Gram + sklearn ──
    print(f"\n{'=' * 60}")
    print("Gram + sklearn SVC (precomputed kernel)")
    print(f"{'=' * 60}")
    model_gram, t_gram = train_and_time(
        pos_train, neg_train, solver="libsvm", device=device,
    )
    scores_gram, acc_gram, auroc_gram = evaluate(
        model_gram, pos_test, neg_test, device,
    )
    print(f"  Time:    {t_gram:.1f}s")
    print(f"  SVs:     {model_gram.num_support_vectors}")
    print(f"  Bias:    {model_gram.bias:.4f}")
    print(f"  Accuracy: {acc_gram:.4f}")
    print(f"  AUROC:   {auroc_gram:.4f}")

    # ── Train with SMO ──
    print(f"\n{'=' * 60}")
    print(f"Column-cached SMO (WSS3 + shrinking, cache={args.cache_size} MB)")
    print(f"{'=' * 60}")
    model_smo, t_smo = train_and_time(
        pos_train, neg_train, solver="smo", device=device,
        cache_size=args.cache_size,
    )
    scores_smo, acc_smo, auroc_smo = evaluate(
        model_smo, pos_test, neg_test, device,
    )
    print(f"  Time:    {t_smo:.1f}s")
    print(f"  SVs:     {model_smo.num_support_vectors}")
    print(f"  Bias:    {model_smo.bias:.4f}")
    print(f"  Accuracy: {acc_smo:.4f}")
    print(f"  AUROC:   {auroc_smo:.4f}")

    # ── Comparison ──
    print(f"\n{'=' * 60}")
    print("Comparison")
    print(f"{'=' * 60}")

    r = np.corrcoef(scores_gram, scores_smo)[0, 1]
    max_diff = np.max(np.abs(scores_gram - scores_smo))

    print(f"  Score correlation:  r = {r:.6f}")
    print(f"  Max score diff:     {max_diff:.4f}")
    print(f"  Bias diff:          {abs(model_gram.bias - model_smo.bias):.4f}")
    print(f"  SV count:           {model_gram.num_support_vectors} (Gram) vs {model_smo.num_support_vectors} (SMO)")

    print(f"\n{'─' * 60}")
    print(f"{'':25s} {'Gram+sklearn':>15s}  {'SMO':>15s}")
    print(f"{'─' * 60}")
    print(f"{'Training time':25s} {t_gram:>14.1f}s  {t_smo:>14.1f}s")
    print(f"{'Support vectors':25s} {model_gram.num_support_vectors:>15d}  {model_smo.num_support_vectors:>15d}")
    print(f"{'Bias':25s} {model_gram.bias:>15.4f}  {model_smo.bias:>15.4f}")
    print(f"{'Test accuracy':25s} {acc_gram:>15.4f}  {acc_smo:>15.4f}")
    print(f"{'Test AUROC':25s} {auroc_gram:>15.4f}  {auroc_smo:>15.4f}")
    print(f"{'Score correlation':25s} {'':>15s}  {r:>15.6f}")
    print(f"{'─' * 60}")

    speedup = t_gram / t_smo if t_smo > 0 else float("inf")
    if speedup > 1:
        print(f"\nSMO is {speedup:.1f}x faster than Gram+sklearn")
    else:
        print(f"\nGram+sklearn is {1/speedup:.1f}x faster than SMO")


if __name__ == "__main__":
    main()
