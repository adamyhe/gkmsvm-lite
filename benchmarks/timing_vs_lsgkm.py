#!/usr/bin/env python3
"""Timing comparison: gkmsvm-lite vs LS-GKM C (gkmtrain / gkmpredict).

Measures wall-clock training and inference time across dataset sizes.
Both implementations use the same kernel (-t 2 gkm_esttrunc, l=11 k=7 d=3).

Setup:
    conda create -n lsgkm -c bioconda -c conda-forge ls-gkm -y

Usage:
    python benchmarks/timing_vs_lsgkm.py                          # CPU only
    python benchmarks/timing_vs_lsgkm.py --device cuda             # include GPU
    python benchmarks/timing_vs_lsgkm.py --sizes 500,1000,5000
    python benchmarks/timing_vs_lsgkm.py --setup                   # create env first
    python benchmarks/timing_vs_lsgkm.py --plot                    # save figure
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "examples" / "data" / "gm12878_sequence_sets"

DEFAULT_SIZES = [500, 1000, 2000, 5000, 10000]
N_TEST = 1000
INFER_REPEATS = 3


# ── Conda env helpers ──────────────────────────────────────────────


def setup_conda_env(name: str) -> None:
    print(f"Creating conda env '{name}' with ls-gkm from bioconda...")
    subprocess.run(
        ["conda", "create", "-n", name, "-c", "bioconda", "-c", "conda-forge",
         "ls-gkm", "-y"],
        check=True,
    )
    print(f"Done. Env '{name}' ready.")


def find_binary(env_name: str, name: str) -> str | None:
    result = subprocess.run(
        ["conda", "run", "-n", env_name, "which", name],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


# ── Data helpers ───────────────────────────────────────────────────


def load_dsqtl_seqs() -> tuple[list[str], list[str]]:
    from gkmsvm import read_fasta

    pos = [seq for _, seq in read_fasta(str(DATA / "gm12878_shared.fa"))]
    neg = [seq for _, seq in read_fasta(str(DATA / "nullseqs_gm12878_shared.1.1.fa"))]
    return pos, neg


def write_fasta(seqs: list[str], path: Path, prefix: str = "s") -> None:
    with open(path, "w") as f:
        for i, seq in enumerate(seqs):
            f.write(f">{prefix}{i}\n{seq}\n")


def subsample(seqs: list[str], n: int, rng) -> list[str]:
    idx = rng.choice(len(seqs), size=min(n, len(seqs)), replace=False)
    return [seqs[i] for i in sorted(idx)]


# ── LS-GKM C wrappers ─────────────────────────────────────────────


def run_gkmtrain(
    binary: str, pos_fa: Path, neg_fa: Path, prefix: Path,
    l: int, k: int, d: int, threads: int,
) -> float | None:
    cmd = [
        binary, "-t", "2", "-l", str(l), "-k", str(k), "-d", str(d),
        "-T", str(threads), str(pos_fa), str(neg_fa), str(prefix),
    ]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if r.returncode != 0:
        print(f"    gkmtrain FAILED: {r.stderr[:300]}", file=sys.stderr)
        return None
    return elapsed


def run_gkmpredict(
    binary: str, test_fa: Path, model_path: Path, output_path: Path,
) -> tuple[float | None, np.ndarray | None]:
    cmd = [binary, str(test_fa), str(model_path), str(output_path)]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if r.returncode != 0:
        print(f"    gkmpredict FAILED: {r.stderr[:300]}", file=sys.stderr)
        return None, None
    scores = []
    with open(output_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            scores.append(float(parts[-1]))
    return elapsed, np.array(scores)


def count_svs_in_model(path: Path) -> int:
    n = 0
    in_svs = False
    with open(path) as f:
        for line in f:
            if line.startswith("SV"):
                in_svs = True
                continue
            if in_svs and line.strip():
                n += 1
    return n


# ── gkmsvm-lite wrappers ──────────────────────────────────────────


def train_lite(
    pos_seqs: list[str], neg_seqs: list[str],
    l: int, k: int, d: int, device: str,
):
    from gkmsvm import train_gkmsvm

    t0 = time.perf_counter()
    model = train_gkmsvm(
        pos_seqs, neg_seqs,
        kernel_type="estimated", l=l, k=k, d=d,
        C=1.0, solver="auto", device=device, verbose=False,
    )
    elapsed = time.perf_counter() - t0
    return model, elapsed


def predict_lite(
    model, X_test: np.ndarray, device: str,
) -> tuple[np.ndarray, float]:
    from gkmsvm.backend import to_cpu

    if device == "cuda":
        model.cuda()
    elif device == "mlx":
        model.mlx()

    X = model._match_device(X_test)
    _ = model(X[:1])  # warmup (JIT / GPU transfer)

    t0 = time.perf_counter()
    scores = to_cpu(model(X).flatten())
    elapsed = time.perf_counter() - t0

    model.cpu()
    return scores, elapsed


# ── Plotting ───────────────────────────────────────────────────────


def plot_results(train_results, infer_results, args, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Training time ──
    ax = axes[0]
    Ns = [r["N"] for r in train_results]

    t_c = [r["t_lsgkm"] for r in train_results]
    valid = [(n, t) for n, t in zip(Ns, t_c) if t is not None]
    if valid:
        ax.plot(*zip(*valid), "o-", color="gray", label="LS-GKM C",
                linewidth=2, markersize=6)

    t_cpu = [r["t_lite_cpu"] for r in train_results]
    ax.plot(Ns, t_cpu, "s-", color="steelblue", label="gkmsvm-lite CPU",
            linewidth=2, markersize=6)

    t_gpu = [r["t_lite_gpu"] for r in train_results]
    valid = [(n, t) for n, t in zip(Ns, t_gpu) if t is not None]
    if valid:
        ax.plot(*zip(*valid), "^-", color="coral",
                label=f"gkmsvm-lite {args.device.upper()}", linewidth=2, markersize=6)

    ax.set_xlabel("Training set size (N)")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Training Time")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Inference throughput ──
    ax = axes[1]
    n_svs = [r["n_sv"] for r in infer_results]
    n_test = infer_results[0]["n_test"]

    t_c = [r["t_lsgkm"] for r in infer_results]
    valid = [(s, n_test / t) for s, t in zip(n_svs, t_c) if t is not None]
    if valid:
        ax.plot(*zip(*valid), "o-", color="gray", label="LS-GKM C",
                linewidth=2, markersize=6)

    t_cpu = [r["t_lite_cpu"] for r in infer_results]
    ax.plot(n_svs, [n_test / t for t in t_cpu], "s-", color="steelblue",
            label="gkmsvm-lite CPU", linewidth=2, markersize=6)

    t_gpu = [r["t_lite_gpu"] for r in infer_results]
    valid = [(s, n_test / t) for s, t in zip(n_svs, t_gpu) if t is not None]
    if valid:
        ax.plot(*zip(*valid), "^-", color="coral",
                label=f"gkmsvm-lite {args.device.upper()}", linewidth=2, markersize=6)

    ax.set_xlabel("Number of support vectors")
    ax.set_ylabel("Throughput (sequences/s)")
    ax.set_title("Inference Throughput")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png = out_path.with_suffix(".png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {png}")


# ── Main ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Timing benchmark: gkmsvm-lite vs LS-GKM C",
    )
    parser.add_argument("--conda-env", default="lsgkm",
                        help="Conda env containing ls-gkm (default: lsgkm)")
    parser.add_argument("--setup", action="store_true",
                        help="Create the conda env before benchmarking")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mlx"],
                        help="Device for gkmsvm-lite GPU column (default: cpu)")
    parser.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)),
                        help="Comma-separated training set sizes (total N)")
    parser.add_argument("--n-test", type=int, default=N_TEST,
                        help="Test sequences for inference (default: 1000)")
    parser.add_argument("--repeats", type=int, default=INFER_REPEATS,
                        help="Inference timing repeats, reports median (default: 3)")
    parser.add_argument("-l", type=int, default=11, help="L-mer length")
    parser.add_argument("-k", type=int, default=7, help="Informative positions")
    parser.add_argument("-d", type=int, default=3, help="Max mismatches")
    parser.add_argument("--threads", type=int, default=os.cpu_count(),
                        help="Threads for gkmtrain -T (default: all cores)")
    parser.add_argument("--output", default="benchmarks/results/timing_vs_lsgkm.tsv",
                        help="Output TSV path")
    parser.add_argument("--plot", action="store_true", help="Save timing plot")
    args = parser.parse_args()

    if args.setup:
        setup_conda_env(args.conda_env)

    gkmtrain_bin = find_binary(args.conda_env, "gkmtrain")
    gkmpredict_bin = find_binary(args.conda_env, "gkmpredict")

    if not gkmtrain_bin or not gkmpredict_bin:
        print(f"Error: gkmtrain/gkmpredict not found in conda env '{args.conda_env}'.")
        print(f"Create it with:  conda create -n {args.conda_env} "
              f"-c bioconda -c conda-forge ls-gkm -y")
        print(f"Or run:  python {__file__} --setup")
        sys.exit(1)

    print(f"LS-GKM C:          {gkmtrain_bin}")
    print(f"gkmsvm-lite device: {args.device}")
    print(f"Kernel:             -t 2 -l {args.l} -k {args.k} -d {args.d}")
    print(f"gkmtrain threads:   {args.threads}")
    print(f"Inference repeats:  {args.repeats}")

    if not DATA.exists():
        print(f"\nError: training data not found at {DATA}")
        print("Download with:  python examples/run_dsqtl_cli.py (runs download step)")
        sys.exit(1)

    sizes = [int(s) for s in args.sizes.split(",")]

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

    train_results = []
    infer_results = []

    with tempfile.TemporaryDirectory(prefix="gkmsvm_bench_") as tmpdir:
        tmpdir = Path(tmpdir)
        test_fa = tmpdir / "test.fa"
        write_fasta(test_seqs, test_fa, prefix="test")

        for N in sizes:
            n_half = N // 2
            if n_half > len(all_pos) or n_half > len(all_neg):
                print(f"  SKIP N={N}: exceeds available sequences")
                continue

            gram_gb = N * N * 8 / 1024**3
            print(f"{'=' * 65}")
            print(f"N = {N}  ({n_half} pos + {n_half} neg)  "
                  f"Gram = {gram_gb:.2f} GB")
            print(f"{'=' * 65}")

            rng_sub = np.random.default_rng(100 + N)
            pos_sub = subsample(all_pos, n_half, rng_sub)
            neg_sub = subsample(all_neg, n_half, rng_sub)

            pos_fa = tmpdir / f"pos_{N}.fa"
            neg_fa = tmpdir / f"neg_{N}.fa"
            write_fasta(pos_sub, pos_fa, prefix="pos")
            write_fasta(neg_sub, neg_fa, prefix="neg")

            # ── Training ──────────────────────────────────────────

            print(f"  Training LS-GKM C ({args.threads} threads)...")
            prefix_c = tmpdir / f"lsgkm_{N}"
            t_train_c = run_gkmtrain(
                gkmtrain_bin, pos_fa, neg_fa, prefix_c,
                args.l, args.k, args.d, args.threads,
            )
            model_c_path = Path(f"{prefix_c}.model.txt")
            n_sv_c = count_svs_in_model(model_c_path) if t_train_c else 0
            if t_train_c is not None:
                print(f"    {t_train_c:.1f}s  ({n_sv_c} SVs)")

            print(f"  Training gkmsvm-lite (CPU)...")
            model_lite_cpu, t_train_cpu = train_lite(
                pos_sub, neg_sub, args.l, args.k, args.d, "cpu",
            )
            n_sv_lite = model_lite_cpu.num_support_vectors
            print(f"    {t_train_cpu:.1f}s  ({n_sv_lite} SVs)")

            t_train_gpu = None
            if args.device != "cpu":
                print(f"  Training gkmsvm-lite ({args.device})...")
                model_lite_gpu, t_train_gpu = train_lite(
                    pos_sub, neg_sub, args.l, args.k, args.d, args.device,
                )
                print(f"    {t_train_gpu:.1f}s  "
                      f"({model_lite_gpu.num_support_vectors} SVs)")

            train_results.append({
                "N": N, "seq_len": seq_len, "gram_gb": gram_gb,
                "t_lsgkm": t_train_c, "t_lite_cpu": t_train_cpu,
                "t_lite_gpu": t_train_gpu,
                "n_sv_lsgkm": n_sv_c, "n_sv_lite": n_sv_lite,
            })

            # ── Inference ─────────────────────────────────────────

            print(f"  Inference ({len(test_seqs)} seqs, "
                  f"median of {args.repeats} runs):")

            # LS-GKM C
            t_infer_c = None
            scores_c = None
            if t_train_c is not None:
                times_c = []
                for rep in range(args.repeats):
                    out_c = tmpdir / f"pred_c_{N}_{rep}.txt"
                    t, sc = run_gkmpredict(
                        gkmpredict_bin, test_fa, model_c_path, out_c,
                    )
                    if t is not None:
                        times_c.append(t)
                        scores_c = sc
                if times_c:
                    t_infer_c = float(np.median(times_c))
                    print(f"    LS-GKM C:        {t_infer_c:.2f}s  "
                          f"({len(test_seqs) / t_infer_c:.0f} seq/s)")

            # gkmsvm-lite CPU
            times_cpu = []
            for rep in range(args.repeats):
                sc_cpu, t = predict_lite(model_lite_cpu, X_test, "cpu")
                times_cpu.append(t)
            t_infer_cpu = float(np.median(times_cpu))
            print(f"    lite CPU:        {t_infer_cpu:.2f}s  "
                  f"({len(test_seqs) / t_infer_cpu:.0f} seq/s)")

            # gkmsvm-lite GPU
            t_infer_gpu = None
            if args.device != "cpu":
                times_gpu = []
                m_gpu = model_lite_gpu if t_train_gpu else model_lite_cpu
                for rep in range(args.repeats):
                    sc_gpu, t = predict_lite(m_gpu, X_test, args.device)
                    times_gpu.append(t)
                t_infer_gpu = float(np.median(times_gpu))
                print(f"    lite {args.device.upper():4s}:        "
                      f"{t_infer_gpu:.2f}s  "
                      f"({len(test_seqs) / t_infer_gpu:.0f} seq/s)")

            # Score sanity check (independently trained models)
            if scores_c is not None and len(scores_c) == len(sc_cpu):
                r = np.corrcoef(scores_c, sc_cpu)[0, 1]
                print(f"    Score correlation (C vs lite): r = {r:.4f}")

            infer_results.append({
                "N_train": N, "n_sv": n_sv_lite,
                "n_test": len(test_seqs),
                "t_lsgkm": t_infer_c, "t_lite_cpu": t_infer_cpu,
                "t_lite_gpu": t_infer_gpu,
            })

            print()

    # ── Summary tables ─────────────────────────────────────────────

    gpu = args.device != "cpu"

    print(f"{'=' * 78}")
    print("TRAINING TIME")
    print(f"{'=' * 78}")
    hdr = f"{'N':>7}  {'Gram':>6}  {'LS-GKM C':>10}  {'lite CPU':>10}"
    if gpu:
        hdr += f"  {'lite GPU':>10}  {'C/GPU':>6}"
    hdr += f"  {'SV(C)':>6}  {'SV(lite)':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in train_results:
        line = f"{r['N']:>7d}  {r['gram_gb']:>5.2f}G"
        line += f"  {r['t_lsgkm']:>9.1f}s" if r["t_lsgkm"] else f"  {'N/A':>10}"
        line += f"  {r['t_lite_cpu']:>9.1f}s"
        if gpu:
            if r["t_lite_gpu"] is not None:
                line += f"  {r['t_lite_gpu']:>9.1f}s"
                if r["t_lsgkm"]:
                    line += f"  {r['t_lsgkm'] / r['t_lite_gpu']:>5.1f}x"
                else:
                    line += f"  {'':>6}"
            else:
                line += f"  {'N/A':>10}  {'':>6}"
        line += f"  {r['n_sv_lsgkm']:>6d}  {r['n_sv_lite']:>8d}"
        print(line)

    print(f"\n{'=' * 78}")
    print("INFERENCE TIME")
    print(f"{'=' * 78}")
    hdr = f"{'N_train':>7}  {'N_SV':>6}  {'N_test':>6}  {'LS-GKM C':>10}  {'lite CPU':>10}"
    if gpu:
        hdr += f"  {'lite GPU':>10}  {'C/GPU':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in infer_results:
        line = f"{r['N_train']:>7d}  {r['n_sv']:>6d}  {r['n_test']:>6d}"
        line += f"  {r['t_lsgkm']:>9.2f}s" if r["t_lsgkm"] else f"  {'N/A':>10}"
        line += f"  {r['t_lite_cpu']:>9.2f}s"
        if gpu:
            if r["t_lite_gpu"] is not None:
                line += f"  {r['t_lite_gpu']:>9.2f}s"
                if r["t_lsgkm"]:
                    line += f"  {r['t_lsgkm'] / r['t_lite_gpu']:>5.1f}x"
                else:
                    line += f"  {'':>6}"
            else:
                line += f"  {'N/A':>10}  {'':>6}"
        print(line)

    # ── Save TSV ───────────────────────────────────────────────────

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write(f"# gkmsvm-lite vs LS-GKM C timing comparison\n")
        f.write(f"# Kernel: -t 2 -l {args.l} -k {args.k} -d {args.d}\n")
        f.write(f"# Threads (gkmtrain -T): {args.threads}\n")
        f.write(f"# Device (gkmsvm-lite): {args.device}\n")
        f.write(f"# Test sequences: {args.n_test}\n")
        f.write(f"# Inference repeats: {args.repeats}\n\n")

        f.write("## Training\n")
        f.write("N\tseq_len\tgram_gb\tt_lsgkm_train\tt_lite_cpu_train\t"
                "t_lite_gpu_train\tn_sv_lsgkm\tn_sv_lite\n")
        for r in train_results:
            vals = [
                str(r["N"]), str(r["seq_len"]), f"{r['gram_gb']:.3f}",
                f"{r['t_lsgkm']:.2f}" if r["t_lsgkm"] else "NA",
                f"{r['t_lite_cpu']:.2f}",
                f"{r['t_lite_gpu']:.2f}" if r["t_lite_gpu"] else "NA",
                str(r["n_sv_lsgkm"]), str(r["n_sv_lite"]),
            ]
            f.write("\t".join(vals) + "\n")

        f.write("\n## Inference\n")
        f.write("N_train\tn_sv\tn_test\tt_lsgkm_infer\t"
                "t_lite_cpu_infer\tt_lite_gpu_infer\n")
        for r in infer_results:
            vals = [
                str(r["N_train"]), str(r["n_sv"]), str(r["n_test"]),
                f"{r['t_lsgkm']:.3f}" if r["t_lsgkm"] else "NA",
                f"{r['t_lite_cpu']:.3f}",
                f"{r['t_lite_gpu']:.3f}" if r["t_lite_gpu"] else "NA",
            ]
            f.write("\t".join(vals) + "\n")

    print(f"\nResults saved: {out_path}")

    if args.plot:
        plot_results(train_results, infer_results, args, out_path)


if __name__ == "__main__":
    main()
