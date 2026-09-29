#!/usr/bin/env python3
"""Benchmark: Gram+sklearn training on GM12878 dsQTL (N=44,768).

At N=44,768 the Gram matrix is 14.9 GB — too large for most GPUs but
fits in system RAM. The solver="auto" logic routes to the tiled path:
Gram tiles are computed on GPU and accumulated into a CPU numpy array,
then sklearn's LIBSVM solver fits on CPU.

This benchmark measures training time breakdown:
  1. Gram matrix computation (GPU-tiled or CPU Numba)
  2. sklearn SVC solve (always CPU)
  3. Test-set evaluation

For comparison with a GPU-in-memory Gram path, see bench_smo_vs_gram.py
(Nanog, N=9,704, Gram=0.7 GB fits on GPU).

Dataset: GM12878 dsQTL (Lee et al. 2015)
  Train: 22,384 pos + 22,384 neg = 44,768 sequences (300bp)
  Test:  574 pos + 27,735 neg (dsQTL variants)
  Kernel: -t estimated -l 10 -k 6 -d 3

Usage:
    python bench_tiled_gram_vs_smo.py                # auto GPU
    python bench_tiled_gram_vs_smo.py --device cpu    # CPU only
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "examples" / "data" / "gm12878_sequence_sets"

KERNEL_PARAMS = dict(kernel_type="estimated", l=10, k=6, d=3, C=1.0)


def load_sequences():
    from gkmsvm import read_fasta

    pos = [seq for _, seq in read_fasta(str(DATA / "gm12878_shared.fa"))]
    neg = [seq for _, seq in read_fasta(str(DATA / "nullseqs_gm12878_shared.1.1.fa"))]
    return pos, neg


def load_test_sequences():
    from gkmsvm import read_fasta

    pos_major = [seq for _, seq in read_fasta(str(DATA / "dsqtl_test_pos.major.fa"))]
    neg_major = [seq for _, seq in read_fasta(str(DATA / "dsqtl_test_neg.major.fa"))]
    return pos_major, neg_major


def train_timed(pos, neg, device, verbose=True):
    """Train with Gram+sklearn, return model and timing breakdown."""
    from gkmsvm import train_gkmsvm

    t_total_start = time.time()
    model = train_gkmsvm(
        pos, neg,
        solver="libsvm",
        device=device,
        verbose=verbose,
        **KERNEL_PARAMS,
    )
    t_total = time.time() - t_total_start
    return model, t_total


def evaluate(model, test_pos, test_neg, device):
    from gkmsvm import one_hot_encode
    from gkmsvm.backend import to_cpu

    t0 = time.time()
    X_test = np.stack([one_hot_encode(s) for s in test_pos + test_neg])
    y_test = np.array([1] * len(test_pos) + [-1] * len(test_neg))

    if device in ("cuda", "gpu"):
        model.cuda()
    scores = to_cpu(model(model._match_device(X_test), verbose=True).flatten())
    model.cpu()
    t_eval = time.time() - t0

    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(y_test, scores)
    return scores, auroc, t_eval


def main():
    parser = argparse.ArgumentParser(
        description="Gram+sklearn training benchmark on GM12878 (N=44,768)"
    )
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mlx"])
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

    if device in ("cuda", "gpu"):
        import cupy as cp
        free, total = cp.cuda.Device().mem_info
        print(f"GPU memory: {total / 1024**3:.1f} GB total, {free / 1024**3:.1f} GB free")

    import psutil
    ram = psutil.virtual_memory()
    print(f"System RAM: {ram.total / 1024**3:.1f} GB total, {ram.available / 1024**3:.1f} GB available")

    print("\nLoading GM12878 dsQTL dataset...")
    pos_train, neg_train = load_sequences()
    pos_test, neg_test = load_test_sequences()
    N = len(pos_train) + len(neg_train)
    gram_gb = N * N * 8 / 1024**3
    seqlen = len(pos_train[0])
    print(f"  Train: {len(pos_train)} pos + {len(neg_train)} neg = {N} sequences")
    print(f"  Test:  {len(pos_test)} pos + {len(neg_test)} neg")
    print(f"  Sequence length: {seqlen}bp")
    print(f"  Kernel: -t estimated -l {KERNEL_PARAMS['l']} "
          f"-k {KERNEL_PARAMS['k']} -d {KERNEL_PARAMS['d']}")
    print(f"  Gram matrix: {N}x{N} = {gram_gb:.1f} GB")

    if device in ("cuda", "gpu") and gram_gb * 1024**3 > free:
        print(f"  → Gram exceeds GPU memory — using tiled path "
              f"(GPU compute, CPU storage)")

    # ── Train ──
    print(f"\n{'=' * 60}")
    print("Training: Gram + sklearn SVC")
    print(f"{'=' * 60}")

    model, t_train = train_timed(pos_train, neg_train, device)

    print(f"\n  Total training time: {t_train:.1f}s ({t_train / 60:.1f} min)")
    print(f"  Support vectors: {model.num_support_vectors} / {N} "
          f"({model.num_support_vectors / N * 100:.0f}%)")
    print(f"  Bias: {model.bias:.4f}")

    # ── Evaluate ──
    print(f"\n{'=' * 60}")
    print("Evaluation on dsQTL test set")
    print(f"{'=' * 60}")

    scores, auroc, t_eval = evaluate(model, pos_test, neg_test, device)
    print(f"\n  Evaluation time: {t_eval:.1f}s")
    print(f"  Test AUROC: {auroc:.4f}")
    print(f"  Test sequences: {len(pos_test) + len(neg_test)}")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"  Dataset:           GM12878 dsQTL (N={N}, {seqlen}bp)")
    print(f"  Kernel:            -t estimated -l 10 -k 6 -d 3")
    print(f"  Device:            {device}")
    print(f"  Gram matrix:       {gram_gb:.1f} GB")
    print(f"  Training time:     {t_train:.1f}s ({t_train / 60:.1f} min)")
    print(f"  Support vectors:   {model.num_support_vectors}")
    print(f"  Test AUROC:        {auroc:.4f}")
    print(f"  Evaluation time:   {t_eval:.1f}s")


if __name__ == "__main__":
    main()
