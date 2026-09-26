"""Benchmark: SMO solver vs sklearn (LIBSVM) precomputed Gram.

Compares wall-clock time, support vector count, and score agreement
across dataset sizes. Both solvers use the same gkm kernel with the
packed uint32 fast path.

Usage:
    python benchmarks/solver_bench.py
    python benchmarks/solver_bench.py --sizes 100,200,500,1000,2000
    python benchmarks/solver_bench.py --kernel estimated --sizes 100,500,1000
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def biased_onehot(n, length, rng, bias_base=0):
    x = np.zeros((n, 4, length), dtype=np.float64)
    probs = [0.1, 0.1, 0.1, 0.1]
    probs[bias_base] = 0.7
    for i in range(n):
        idx = rng.choice(4, size=length, p=probs)
        x[i, idx, np.arange(length)] = 1.0
    return x


def random_onehot(n, length, rng):
    x = np.zeros((n, 4, length), dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, 4, size=length)
        x[i, idx, np.arange(length)] = 1.0
    return x


def run_benchmark(sizes, kernel_type, l, k, d, C, seq_len, cache_size):
    from sklearn.svm import SVC

    from gkmsvm.gram import compute_gram
    from gkmsvm.solver import KernelColumnCache, smo_solve
    from gkmsvm.svm import KERNEL_BUILDERS, resolve_kernel_type

    kernel_type = resolve_kernel_type(kernel_type)
    kernel_params = {"L": l, "k": k, "include_rc": True}
    if kernel_type in ("gkm_esttrunc", "gkm_estfull", "gkmrbf"):
        kernel_params["d"] = d
    kernel = KERNEL_BUILDERS[kernel_type](kernel_params)

    print(f"Kernel: {kernel_type}  l={l} k={k} d={d}  C={C}  L={seq_len}")
    print(f"SMO cache: {cache_size} columns")
    print()
    print(
        f"{'N':>6s}  "
        f"{'Gram':>7s}  {'sk_solv':>7s}  {'sk_tot':>7s}  "
        f"{'SMO':>7s}  {'Ratio':>6s}  "
        f"{'SV_sk':>5s}  {'SV_smo':>6s}  "
        f"{'MaxΔ':>8s}  {'MeanΔ':>8s}"
    )
    print("-" * 88)

    for N in sizes:
        rng = np.random.default_rng(42 + N)
        n_pos = N // 2
        n_neg = N - n_pos
        pos = biased_onehot(n_pos, seq_len, rng, bias_base=0)
        neg = biased_onehot(n_neg, seq_len, rng, bias_base=3)
        X = np.concatenate([pos, neg])
        y = np.array([1.0] * n_pos + [-1.0] * n_neg)

        test_seqs = random_onehot(20, seq_len, np.random.default_rng(99))

        # --- sklearn: Gram + LIBSVM ---
        t0 = time.perf_counter()
        gram = compute_gram(kernel, X, chunk_size=1000)
        t_gram = time.perf_counter() - t0

        t0 = time.perf_counter()
        clf = SVC(kernel="precomputed", C=C)
        clf.fit(gram, y)
        t_sk_solve = time.perf_counter() - t0
        t_sk_total = t_gram + t_sk_solve

        n_sv_sk = len(clf.support_)
        coef_sk = np.zeros(N)
        coef_sk[clf.support_] = clf.dual_coef_[0]
        bias_sk = float(clf.intercept_[0])

        scores_sk = np.array([
            float(np.dot(coef_sk, kernel.pairwise(t[None], X)[0]) + bias_sk)
            for t in test_seqs
        ])
        del gram

        # --- SMO ---
        t0 = time.perf_counter()
        coef_smo, bias_smo = smo_solve(
            kernel, X, y, C=C, tol=1e-3, cache_size=cache_size,
        )
        t_smo = time.perf_counter() - t0

        n_sv_smo = int(np.sum(np.abs(coef_smo) > 1e-10))
        scores_smo = np.array([
            float(np.dot(coef_smo, kernel.pairwise(t[None], X)[0]) + bias_smo)
            for t in test_seqs
        ])

        max_delta = float(np.max(np.abs(scores_smo - scores_sk)))
        mean_delta = float(np.mean(np.abs(scores_smo - scores_sk)))
        ratio = t_sk_total / max(t_smo, 1e-6)

        print(
            f"{N:>6d}  "
            f"{t_gram:>7.3f}  {t_sk_solve:>7.3f}  {t_sk_total:>7.3f}  "
            f"{t_smo:>7.3f}  {ratio:>5.2f}x  "
            f"{n_sv_sk:>5d}  {n_sv_smo:>6d}  "
            f"{max_delta:>8.4f}  {mean_delta:>8.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark SMO vs sklearn solver")
    parser.add_argument(
        "--sizes", default="100,200,500,1000,2000",
        help="Comma-separated dataset sizes (total N, split 50/50)",
    )
    parser.add_argument("--kernel", default="direct", help="Kernel type")
    parser.add_argument("--l", type=int, default=7, help="Window length")
    parser.add_argument("--k", type=int, default=5, help="Informative positions")
    parser.add_argument("--d", type=int, default=3, help="Max mismatch depth")
    parser.add_argument("--C", type=float, default=1.0, help="SVM C parameter")
    parser.add_argument("--seq-len", type=int, default=20, help="Sequence length")
    parser.add_argument("--cache-size", type=int, default=256, help="SMO cache columns")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    run_benchmark(
        sizes, args.kernel, args.l, args.k, args.d, args.C,
        args.seq_len, args.cache_size,
    )


if __name__ == "__main__":
    main()
