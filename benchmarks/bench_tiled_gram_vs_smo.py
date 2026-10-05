#!/usr/bin/env python3
"""Benchmark: Tiled Gram vs SMO on GM12878 dsQTL (N=44,768).

At N=44,768 the Gram matrix is 14.9 GB — too large for most GPUs but
fits in system RAM. Compares two training paths:

  Gram+sklearn: Gram tiles computed on GPU and accumulated into a CPU
    numpy array, then sklearn's LIBSVM solver fits on CPU.

  Column-cached SMO: WSS3 working set selection with LRU-cached kernel
    columns. Memory is O(cache_size × N), not O(N²).

For the smaller Nanog dataset (N=9,704, Gram fits on GPU), see
bench_smo_vs_gram.py.

Dataset: GM12878 dsQTL (Lee et al. 2015)
  Train: 22,384 pos + 22,384 neg = 44,768 sequences (300bp)
  Test:  574 pos + 27,735 neg (dsQTL variants)
  Kernel: -t estimated -l 10 -k 6 -d 3

Usage:
    python benchmarks/bench_tiled_gram_vs_smo.py                # auto GPU
    python benchmarks/bench_tiled_gram_vs_smo.py --device cpu    # CPU only
    python benchmarks/bench_tiled_gram_vs_smo.py --cache-size 1024
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "examples" / "data" / "gm12878_sequence_sets"

KERNEL_PARAMS = dict(kernel_type="estimated", l=10, k=6, d=3, C=1.0)

SEQ_URL = "https://beerlab.org/deltasvm/downloads/gm12878_sequence_sets.tar.gz"


def download_data():
    """Download Beer lab GM12878 sequences if not present."""
    if DATA.exists():
        return
    import tarfile
    from urllib.request import urlretrieve

    DATA.parent.mkdir(parents=True, exist_ok=True)
    tarball = DATA.parent / "gm12878_sequence_sets.tar.gz"
    if not tarball.exists():
        print("Downloading Beer lab sequence sets (14 MB)...")
        urlretrieve(SEQ_URL, tarball)
    print("Extracting...")
    with tarfile.open(tarball) as tar:
        tar.extractall(DATA.parent, filter="data")


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


def train_timed(pos, neg, solver, device, cache_size=512, verbose=True):
    from gkmsvm import train_gkmsvm

    kwargs = dict(**KERNEL_PARAMS, solver=solver, device=device,
                  verbose=verbose)
    if solver == "smo":
        kwargs["cache_size"] = cache_size

    t0 = time.time()
    model = train_gkmsvm(pos, neg, **kwargs)
    return model, time.time() - t0


def evaluate(model, test_pos, test_neg, device, batch_size=512, verbose=False):
    from gkmsvm import one_hot_encode
    from gkmsvm.backend import to_cpu

    t0 = time.time()
    X_test = np.stack([one_hot_encode(s) for s in test_pos + test_neg])
    y_test = np.array([1] * len(test_pos) + [-1] * len(test_neg))

    if device in ("cuda", "gpu"):
        model.cuda()

    N = len(X_test)
    scores = np.zeros(N, dtype=np.float64)
    chunks = range(0, N, batch_size)
    if verbose:
        from tqdm import tqdm
        chunks = tqdm(chunks, desc="Scoring",
                      total=(N + batch_size - 1) // batch_size)
    for start in chunks:
        end = min(start + batch_size, N)
        batch = model._match_device(X_test[start:end])
        scores[start:end] = to_cpu(model(batch, verbose=False).flatten())

    model.cpu()
    t_eval = time.time() - t0

    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(y_test, scores)
    return scores, auroc, t_eval


def main():
    parser = argparse.ArgumentParser(
        description="Tiled Gram vs SMO training on GM12878 (N=44,768)"
    )
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mlx"])
    parser.add_argument("--cache-size", type=int, default=512,
                        help="SMO kernel column cache size in MB (default: 512)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show progress bars during training and scoring")
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
        print(f"GPU memory: {total / 1024**3:.1f} GB total, "
              f"{free / 1024**3:.1f} GB free")

    import psutil
    ram = psutil.virtual_memory()
    print(f"System RAM: {ram.total / 1024**3:.1f} GB total, "
          f"{ram.available / 1024**3:.1f} GB available")

    download_data()
    print("\nLoading GM12878 dsQTL dataset...")
    pos_train, neg_train = load_sequences()
    pos_test, neg_test = load_test_sequences()
    N = len(pos_train) + len(neg_train)
    gram_gb = N * N * 8 / 1024**3
    seqlen = len(pos_train[0])
    print(f"  Train: {len(pos_train)} pos + {len(neg_train)} neg = {N}")
    print(f"  Test:  {len(pos_test)} pos + {len(neg_test)} neg")
    print(f"  Sequence length: {seqlen}bp")
    print(f"  Kernel: -t estimated -l {KERNEL_PARAMS['l']} "
          f"-k {KERNEL_PARAMS['k']} -d {KERNEL_PARAMS['d']}")
    print(f"  Gram matrix: {N}x{N} = {gram_gb:.1f} GB")

    if device in ("cuda", "gpu") and gram_gb * 1024**3 > free:
        print(f"  → Gram exceeds GPU memory — using tiled path "
              f"(GPU compute, CPU storage)")

    # ── Train with Gram + sklearn ──
    print(f"\n{'=' * 60}")
    print("Gram + sklearn SVC (precomputed kernel)")
    print(f"{'=' * 60}")

    model_gram, t_gram = train_timed(
        pos_train, neg_train, solver="libsvm", device=device,
        verbose=args.verbose)
    scores_gram, auroc_gram, t_eval_gram = evaluate(
        model_gram, pos_test, neg_test, device, verbose=args.verbose)

    print(f"  Time:    {t_gram:.1f}s ({t_gram / 60:.1f} min)")
    print(f"  SVs:     {model_gram.num_support_vectors}")
    print(f"  Bias:    {model_gram.bias:.4f}")
    print(f"  AUROC:   {auroc_gram:.4f}")

    # ── Train with SMO ──
    print(f"\n{'=' * 60}")
    print(f"Column-cached SMO (WSS3 + shrinking, "
          f"cache={args.cache_size} MB)")
    print(f"{'=' * 60}")

    model_smo, t_smo = train_timed(
        pos_train, neg_train, solver="smo", device=device,
        cache_size=args.cache_size, verbose=args.verbose)
    scores_smo, auroc_smo, t_eval_smo = evaluate(
        model_smo, pos_test, neg_test, device, verbose=args.verbose)

    print(f"  Time:    {t_smo:.1f}s ({t_smo / 60:.1f} min)")
    print(f"  SVs:     {model_smo.num_support_vectors}")
    print(f"  Bias:    {model_smo.bias:.4f}")
    print(f"  AUROC:   {auroc_smo:.4f}")

    # ── Comparison ──
    print(f"\n{'=' * 60}")
    print("Comparison")
    print(f"{'=' * 60}")

    r = np.corrcoef(scores_gram, scores_smo)[0, 1]
    max_diff = np.max(np.abs(scores_gram - scores_smo))

    print(f"  Score correlation:  r = {r:.6f}")
    print(f"  Max score diff:     {max_diff:.4f}")
    print(f"  Bias diff:          "
          f"{abs(model_gram.bias - model_smo.bias):.4f}")

    print(f"\n{'─' * 60}")
    print(f"{'':25s} {'Gram+sklearn':>15s}  {'SMO':>15s}")
    print(f"{'─' * 60}")
    print(f"{'Training time':25s} {t_gram:>14.1f}s  {t_smo:>14.1f}s")
    print(f"{'Support vectors':25s} "
          f"{model_gram.num_support_vectors:>15d}  "
          f"{model_smo.num_support_vectors:>15d}")
    print(f"{'Bias':25s} {model_gram.bias:>15.4f}  "
          f"{model_smo.bias:>15.4f}")
    print(f"{'Test AUROC':25s} {auroc_gram:>15.4f}  "
          f"{auroc_smo:>15.4f}")
    print(f"{'Score correlation':25s} {'':>15s}  {r:>15.6f}")
    print(f"{'─' * 60}")

    speedup = t_gram / t_smo if t_smo > 0 else float("inf")
    if speedup > 1:
        print(f"\nSMO is {speedup:.1f}x faster than Gram+sklearn")
    else:
        print(f"\nGram+sklearn is {1/speedup:.1f}x faster than SMO")


if __name__ == "__main__":
    main()
