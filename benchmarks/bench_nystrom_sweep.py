#!/usr/bin/env python3
"""Nyström landmark formula sweep: accuracy vs exact training for SVC and SVR.

The Nyström solver uses n_components = min(N, max(floor, int(sqrt(N) * mult)))
landmarks. This benchmark sweeps the multiplier to find the accuracy/speed
tradeoff and inform the default formula.

Trains exact (precomputed Gram) reference models, then sweeps multiplier
values and measures score correlation against the exact model on a held-out
test set. Reports wall-clock time, SV count, and Pearson/Spearman correlation
for each (N, multiplier) combination.

SVR targets are the exact SVC model's decision values on training
sequences — a meaningful continuous target correlated with the kernel.

Data:
  examples/data/gm12878_sequence_sets/ (Beer lab GM12878 dsQTL, 14 MB)

Usage:
  python benchmarks/bench_nystrom_sweep.py
  python benchmarks/bench_nystrom_sweep.py --device cuda
  python benchmarks/bench_nystrom_sweep.py --sizes 1000,2000,5000
  python benchmarks/bench_nystrom_sweep.py --multipliers 2,5,10,15,20,30
  python benchmarks/bench_nystrom_sweep.py --floor 500
  python benchmarks/bench_nystrom_sweep.py --plot
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "examples" / "data" / "gm12878_sequence_sets"
DATA_URL = "https://beerlab.org/deltasvm/downloads/gm12878_sequence_sets.tar.gz"

DEFAULT_SIZES = [1000, 2000, 5000, 10000]
DEFAULT_MULTIPLIERS = [2, 5, 10, 15, 20, 30]
DEFAULT_FLOOR = 100
N_TEST = 1000


def landmark_count(N: int, mult: float, floor: int) -> int:
    return min(N, max(floor, int(np.sqrt(N) * mult)))


# ── Data helpers ──────────────────────────────────────────────────


def download_data():
    if DATA.exists():
        return
    import tarfile
    from urllib.request import urlretrieve

    DATA.parent.mkdir(parents=True, exist_ok=True)
    tarball = DATA.parent / "gm12878_sequence_sets.tar.gz"
    if not tarball.exists():
        print("Downloading Beer lab sequence sets (14 MB)...")
        urlretrieve(DATA_URL, tarball)
    print("Extracting...")
    with tarfile.open(tarball) as tar:
        tar.extractall(DATA.parent, filter="data")


def load_dsqtl_seqs() -> tuple[list[str], list[str]]:
    from gkmsvm import read_fasta

    pos = [seq for _, seq in read_fasta(str(DATA / "gm12878_shared.fa"))]
    neg = [seq for _, seq in read_fasta(
        str(DATA / "nullseqs_gm12878_shared.1.1.fa"))]
    return pos, neg


def subsample(seqs: list[str], n: int, rng) -> list[str]:
    idx = rng.choice(len(seqs), size=min(n, len(seqs)), replace=False)
    return [seqs[i] for i in sorted(idx)]


# ── Training wrappers ────────────────────────────────────────────


def train_exact_svc(pos_seqs, neg_seqs, l, k, d, device):
    from gkmsvm import train_gkmsvm

    t0 = time.perf_counter()
    model = train_gkmsvm(
        pos_seqs, neg_seqs,
        kernel_type="estimated", l=l, k=k, d=d,
        C=1.0, solver="libsvm", device=device, verbose=False,
    )
    elapsed = time.perf_counter() - t0
    return model, elapsed


def train_nystrom_svc(pos_seqs, neg_seqs, l, k, d, n_components, device):
    from gkmsvm import train_gkmsvm

    t0 = time.perf_counter()
    model = train_gkmsvm(
        pos_seqs, neg_seqs,
        kernel_type="estimated", l=l, k=k, d=d,
        C=1.0, solver="nystrom", n_components=n_components,
        device=device, verbose=False,
    )
    elapsed = time.perf_counter() - t0
    return model, elapsed


def train_exact_svr(seqs, targets, l, k, d, device):
    from gkmsvm import train_gkmsvr

    t0 = time.perf_counter()
    model = train_gkmsvr(
        seqs, targets,
        kernel_type="estimated", l=l, k=k, d=d,
        C=1.0, epsilon=0.1, solver="libsvm", device=device, verbose=False,
    )
    elapsed = time.perf_counter() - t0
    return model, elapsed


def train_nystrom_svr(seqs, targets, l, k, d, n_components, device):
    from gkmsvm import train_gkmsvr

    t0 = time.perf_counter()
    model = train_gkmsvr(
        seqs, targets,
        kernel_type="estimated", l=l, k=k, d=d,
        C=1.0, epsilon=0.1, solver="nystrom", n_components=n_components,
        device=device, verbose=False,
    )
    elapsed = time.perf_counter() - t0
    return model, elapsed


def score_model(model, X_test):
    from gkmsvm.backend import to_cpu

    X = model._match_device(X_test)
    scores = to_cpu(model(X, verbose=False).flatten())
    model.cpu()
    return scores


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Nyström landmark sweep: accuracy vs exact training",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mlx"])
    parser.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)),
                        help="Comma-separated training set sizes")
    parser.add_argument("--multipliers",
                        default=",".join(map(str, DEFAULT_MULTIPLIERS)),
                        help="Comma-separated sqrt(N) multipliers to sweep")
    parser.add_argument("--floor", type=int, default=DEFAULT_FLOOR,
                        help="Minimum landmark count floor (default: 100)")
    parser.add_argument("-l", type=int, default=11, help="L-mer length")
    parser.add_argument("-k", type=int, default=7, help="Informative positions")
    parser.add_argument("-d", type=int, default=3, help="Max mismatches")
    parser.add_argument("--n-test", type=int, default=N_TEST,
                        help="Held-out test sequences (default: 1000)")
    parser.add_argument("--output",
                        default="benchmarks/results/nystrom_sweep.tsv",
                        help="Output TSV path")
    parser.add_argument("--plot", action="store_true", help="Save figure")
    args = parser.parse_args()

    download_data()
    if not DATA.exists():
        print(f"Error: training data not found at {DATA}", file=sys.stderr)
        sys.exit(1)

    sizes = [int(s) for s in args.sizes.split(",")]
    multipliers = [float(s) for s in args.multipliers.split(",")]

    print("Nyström Formula Sweep")
    print(f"  Formula:   n_components = min(N, max({args.floor}, "
          f"int(sqrt(N) * mult)))")
    print(f"  Kernel:    -t 2 -l {args.l} -k {args.k} -d {args.d}")
    print(f"  Device:    {args.device}")
    print(f"  Sizes:     {sizes}")
    print(f"  Mults:     {multipliers}")
    print(f"  Floor:     {args.floor}")

    print("\nLoading GM12878 dsQTL sequences...")
    all_pos, all_neg = load_dsqtl_seqs()
    seq_len = len(all_pos[0])
    print(f"  {len(all_pos)} pos + {len(all_neg)} neg, {seq_len}bp")

    rng = np.random.default_rng(42)
    test_seqs = (subsample(all_pos, args.n_test // 2, rng)
                 + subsample(all_neg, args.n_test // 2, rng))

    from gkmsvm import one_hot_encode
    X_test = np.stack([one_hot_encode(s) for s in test_seqs])
    print(f"  Test set: {len(test_seqs)} sequences\n")

    svc_results = []
    svr_results = []

    for N in sizes:
        n_half = N // 2
        if n_half > len(all_pos) or n_half > len(all_neg):
            print(f"  SKIP N={N}: exceeds available sequences")
            continue

        gram_gb = N * N * 8 / 1024**3
        print(f"{'=' * 70}")
        print(f"N = {N}  ({n_half} pos + {n_half} neg)  "
              f"Gram = {gram_gb:.2f} GB")
        print(f"{'=' * 70}")

        rng_sub = np.random.default_rng(100 + N)
        pos_sub = subsample(all_pos, n_half, rng_sub)
        neg_sub = subsample(all_neg, n_half, rng_sub)

        # ── SVC ──────────────────────────────────────────────────

        print(f"\n  --- SVC ---")

        print(f"  Training exact (Gram) reference...")
        model_exact, t_exact = train_exact_svc(
            pos_sub, neg_sub, args.l, args.k, args.d, args.device,
        )
        scores_exact = score_model(model_exact, X_test)
        n_sv_exact = model_exact.num_support_vectors
        print(f"    {t_exact:.1f}s  ({n_sv_exact} SVs)")

        svc_results.append({
            "N": N, "mult": None, "n_landmarks": N, "solver": "exact",
            "time_s": t_exact, "n_sv": n_sv_exact,
            "pearson": 1.0, "spearman": 1.0,
        })

        all_train_seqs = pos_sub + neg_sub
        X_train = np.stack([one_hot_encode(s) for s in all_train_seqs])
        svr_targets = score_model(model_exact, X_train)

        for mult in multipliers:
            n_lm = landmark_count(N, mult, args.floor)
            if n_lm >= N:
                print(f"  mult={mult:g} → n={n_lm} >= N={N}, skip")
                continue

            print(f"  Nyström (mult={mult:g}, n_components={n_lm})...")
            model_nys, t_nys = train_nystrom_svc(
                pos_sub, neg_sub, args.l, args.k, args.d, n_lm, args.device,
            )
            scores_nys = score_model(model_nys, X_test)
            n_sv_nys = model_nys.num_support_vectors

            r_p = pearsonr(scores_exact, scores_nys)[0]
            r_s = spearmanr(scores_exact, scores_nys)[0]
            print(f"    {t_nys:.1f}s  ({n_sv_nys} SVs)  "
                  f"r={r_p:.4f}  rho={r_s:.4f}")

            svc_results.append({
                "N": N, "mult": mult, "n_landmarks": n_lm,
                "solver": "nystrom",
                "time_s": t_nys, "n_sv": n_sv_nys,
                "pearson": r_p, "spearman": r_s,
            })

            del model_nys
            gc.collect()

        # ── SVR ──────────────────────────────────────────────────

        print(f"\n  --- SVR (targets = exact SVC decision values) ---")

        print(f"  Training exact (Gram) reference...")
        model_svr_exact, t_svr_exact = train_exact_svr(
            all_train_seqs, svr_targets, args.l, args.k, args.d, args.device,
        )
        scores_svr_exact = score_model(model_svr_exact, X_test)
        n_sv_svr_exact = model_svr_exact.num_support_vectors
        print(f"    {t_svr_exact:.1f}s  ({n_sv_svr_exact} SVs)")

        svr_results.append({
            "N": N, "mult": None, "n_landmarks": N, "solver": "exact",
            "time_s": t_svr_exact, "n_sv": n_sv_svr_exact,
            "pearson": 1.0, "spearman": 1.0,
        })

        for mult in multipliers:
            n_lm = landmark_count(N, mult, args.floor)
            if n_lm >= N:
                continue

            print(f"  Nyström (mult={mult:g}, n_components={n_lm})...")
            model_svr_nys, t_svr_nys = train_nystrom_svr(
                all_train_seqs, svr_targets, args.l, args.k, args.d,
                n_lm, args.device,
            )
            scores_svr_nys = score_model(model_svr_nys, X_test)
            n_sv_svr_nys = model_svr_nys.num_support_vectors

            r_p = pearsonr(scores_svr_exact, scores_svr_nys)[0]
            r_s = spearmanr(scores_svr_exact, scores_svr_nys)[0]
            print(f"    {t_svr_nys:.1f}s  ({n_sv_svr_nys} SVs)  "
                  f"r={r_p:.4f}  rho={r_s:.4f}")

            svr_results.append({
                "N": N, "mult": mult, "n_landmarks": n_lm,
                "solver": "nystrom",
                "time_s": t_svr_nys, "n_sv": n_sv_svr_nys,
                "pearson": r_p, "spearman": r_s,
            })

            del model_svr_nys
            gc.collect()

        del model_exact, model_svr_exact, X_train, svr_targets
        gc.collect()
        print()

    # ── Summary tables ────────────────────────────────────────────

    for label, results in [("SVC", svc_results), ("SVR", svr_results)]:
        print(f"\n{'=' * 88}")
        print(f"NYSTRÖM FORMULA SWEEP — {label}")
        print(f"{'=' * 88}")
        hdr = (f"{'N':>7}  {'mult':>5}  {'n_comp':>6}  {'solver':>7}  "
               f"{'time':>8}  {'n_sv':>6}  {'pearson':>8}  {'spearman':>8}")
        print(hdr)
        print("-" * len(hdr))
        for r in results:
            mult_str = f"{r['mult']:g}" if r["mult"] is not None else "-"
            print(f"{r['N']:>7d}  {mult_str:>5}  {r['n_landmarks']:>6d}  "
                  f"{r['solver']:>7}  {r['time_s']:>7.1f}s  {r['n_sv']:>6d}  "
                  f"{r['pearson']:>8.4f}  {r['spearman']:>8.4f}")

    # ── Save TSV ──────────────────────────────────────────────────

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write(f"# Nyström formula sweep\n")
        f.write(f"# Formula: n_components = min(N, max({args.floor}, "
                f"int(sqrt(N) * mult)))\n")
        f.write(f"# Kernel: -t 2 -l {args.l} -k {args.k} -d {args.d}\n")
        f.write(f"# Device: {args.device}\n\n")

        for label, results in [("SVC", svc_results), ("SVR", svr_results)]:
            f.write(f"## {label}\n")
            f.write("N\tmult\tn_components\tsolver\ttime_s\tn_sv\t"
                    "pearson_vs_exact\tspearman_vs_exact\n")
            for r in results:
                mult_str = f"{r['mult']:g}" if r["mult"] is not None else "NA"
                vals = [
                    str(r["N"]), mult_str, str(r["n_landmarks"]),
                    r["solver"], f"{r['time_s']:.2f}", str(r["n_sv"]),
                    f"{r['pearson']:.6f}", f"{r['spearman']:.6f}",
                ]
                f.write("\t".join(vals) + "\n")
            f.write("\n")

    print(f"\nResults saved: {out_path}")

    if args.plot:
        plot_results(svc_results, svr_results, args, out_path)


# ── Plotting ─────────────────────────────────────────────────────


def plot_results(svc_results, svr_results, args, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 10))

    for col, (label, results) in enumerate(
        [("SVC", svc_results), ("SVR", svr_results)]
    ):
        nystrom_only = [r for r in results if r["solver"] == "nystrom"]
        if not nystrom_only:
            continue

        unique_N = sorted(set(r["N"] for r in nystrom_only))

        # Top: Pearson correlation vs multiplier
        ax = axes[0][col]
        for i, N in enumerate(unique_N):
            subset = [r for r in nystrom_only if r["N"] == N]
            mults = [r["mult"] for r in subset]
            rs = [r["pearson"] for r in subset]
            ax.plot(mults, rs, "o-", color=colors[i % len(colors)],
                    label=f"N={N:,}", linewidth=2, markersize=5)

        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Multiplier (n = sqrt(N) × mult)")
        ax.set_ylabel("Pearson r vs exact")
        ax.set_title(f"{label}: Score Correlation vs Exact")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=max(0.9, ax.get_ylim()[0]))

        # Bottom: Training time vs multiplier
        ax = axes[1][col]
        for i, N in enumerate(unique_N):
            subset = [r for r in nystrom_only if r["N"] == N]
            mults = [r["mult"] for r in subset]
            ts = [r["time_s"] for r in subset]
            ax.plot(mults, ts, "o-", color=colors[i % len(colors)],
                    label=f"N={N:,}", linewidth=2, markersize=5)

            exact = [r for r in results
                     if r["solver"] == "exact" and r["N"] == N]
            if exact:
                ax.axhline(y=exact[0]["time_s"], color=colors[i % len(colors)],
                           linestyle=":", alpha=0.5)

        ax.set_xlabel("Multiplier (n = sqrt(N) × mult)")
        ax.set_ylabel("Training time (s)")
        ax.set_title(f"{label}: Training Time")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png = out_path.with_suffix(".png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {png}")


if __name__ == "__main__":
    main()
