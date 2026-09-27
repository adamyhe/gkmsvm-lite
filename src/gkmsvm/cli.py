"""Command-line interface for gkmsvm-lite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _add_device_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device", default="auto",
        choices=["auto", "cpu", "cuda", "mlx"],
        help="Compute device (default: auto).",
    )


def _add_kernel_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-t", "--kernel-type", default="estimated",
        help="Kernel type: 0-5, or name (default: estimated).",
    )
    parser.add_argument("-l", type=int, default=11, help="L-mer length (default: 11).")
    parser.add_argument("-k", type=int, default=7, help="K informative positions (default: 7).")
    parser.add_argument("-d", type=int, default=3, help="Max mismatches (default: 3).")
    parser.add_argument("--no-rc", action="store_true", help="Disable reverse complement.")
    parser.add_argument("--gamma", type=float, default=1.0, help="RBF gamma (for -t 3, 5).")
    parser.add_argument("-M", type=int, default=None, help="Center-weight window (for -t 4, 5).")
    parser.add_argument("-H", type=float, default=None, help="Center-weight decay (for -t 4, 5).")


def _load_model(path: str, device: str = "cpu"):
    from gkmsvm.serialization import load_model
    return load_model(path, device=device)


def _read_seqs(path: str) -> list[tuple[str, str]]:
    from gkmsvm.fasta import read_fasta
    return read_fasta(path)


def _encode_seqs(records: list[tuple[str, str]]) -> np.ndarray:
    from gkmsvm.codec import one_hot_encode
    return np.stack([one_hot_encode(seq) for name, seq in records])


def cmd_predict(args: argparse.Namespace) -> None:
    from gkmsvm.fasta import read_fasta

    model = _load_model(args.model, device=args.device)

    records = read_fasta(args.input)
    names = [name for name, seq in records]
    X = _encode_seqs(records)

    batch = args.batch_size
    all_scores = []
    chunks = range(0, len(X), batch)
    if args.verbose and len(X) > batch:
        from tqdm import tqdm
        chunks = tqdm(chunks, desc="predict", total=(len(X) + batch - 1) // batch)

    for start in chunks:
        xb = model._match_device(X[start:start + batch])
        scores = model(xb, verbose=False).flatten()
        from gkmsvm.backend import to_cpu
        all_scores.append(to_cpu(scores))

    scores = np.concatenate(all_scores)

    out = sys.stdout if args.output is None else open(args.output, "w")
    try:
        out.write("name\tscore\n")
        for name, score in zip(names, scores):
            out.write(f"{name}\t{score:.6f}\n")
    finally:
        if out is not sys.stdout:
            out.close()

    if args.output:
        print(f"Wrote {len(scores)} scores to {args.output}", file=sys.stderr)


def cmd_train(args: argparse.Namespace) -> None:
    from gkmsvm.fasta import read_fasta
    from gkmsvm.train import train_gkmsvm

    pos_records = read_fasta(args.positives)
    neg_records = read_fasta(args.negatives)
    pos_seqs = [seq for _, seq in pos_records]
    neg_seqs = [seq for _, seq in neg_records]

    model = train_gkmsvm(
        pos_seqs, neg_seqs,
        kernel_type=args.kernel_type,
        l=args.l, k=args.k, d=args.d,
        C=args.C,
        gamma=args.gamma,
        M=args.M, H=args.H,
        include_rc=not args.no_rc,
        solver=args.solver,
        cache_size=args.cache_size,
        device=args.device,
        verbose=args.verbose,
    )

    output = Path(args.output)
    if output.suffix == ".npz":
        from gkmsvm.serialization import save_npz
        save_npz(model, output)
    else:
        from gkmsvm.serialization import save_lsgkm
        save_lsgkm(model, output)

    print(
        f"Saved model to {output} "
        f"({model.num_support_vectors} SVs, bias={model.bias:.4f})",
        file=sys.stderr,
    )


def cmd_trainsvr(args: argparse.Namespace) -> None:
    from gkmsvm.fasta import read_fasta
    from gkmsvm.train import train_gkmsvr

    records = read_fasta(args.input)
    seqs = [seq for _, seq in records]

    labels_path = Path(args.labels)
    labels = np.loadtxt(labels_path, dtype=np.float64)
    if labels.shape[0] != len(seqs):
        print(
            f"Error: {len(seqs)} sequences but {labels.shape[0]} labels",
            file=sys.stderr,
        )
        sys.exit(1)

    model = train_gkmsvr(
        seqs, labels,
        kernel_type=args.kernel_type,
        l=args.l, k=args.k, d=args.d,
        C=args.C,
        epsilon=args.epsilon,
        gamma=args.gamma,
        M=args.M, H=args.H,
        include_rc=not args.no_rc,
        device=args.device,
        verbose=args.verbose,
    )

    output = Path(args.output)
    if output.suffix == ".npz":
        from gkmsvm.serialization import save_npz
        save_npz(model, output)
    else:
        from gkmsvm.serialization import save_lsgkm
        save_lsgkm(model, output)

    print(
        f"Saved model to {output} "
        f"({model.num_support_vectors} SVs, bias={model.bias:.4f})",
        file=sys.stderr,
    )


def cmd_ism(args: argparse.Namespace) -> None:
    from gkmsvm.backend import to_cpu
    from gkmsvm.ism import ism

    model = _load_model(args.model, device=args.device)
    records = _read_seqs(args.input)
    names = np.array([name for name, _ in records])
    X = _encode_seqs(records)
    one_hot = X.copy()

    X = model._match_device(X)
    result = ism(model, X, verbose=args.verbose)
    result = to_cpu(result)

    np.savez(args.output, scores=result, one_hot=one_hot, names=names)
    print(
        f"Wrote ISM scores {result.shape} to {args.output}",
        file=sys.stderr,
    )


def cmd_explain(args: argparse.Namespace) -> None:
    from gkmsvm.backend import to_cpu
    from gkmsvm.explain import gkmexplain

    model = _load_model(args.model, device=args.device)
    records = _read_seqs(args.input)
    X = _encode_seqs(records)
    one_hot = X.astype(np.float32)

    X = model._match_device(X)
    hyp = to_cpu(gkmexplain(model, X, mode=1, verbose=args.verbose))

    # Two files: sequences and attributions, each [B, 4, L] as arr_0.
    # Directly usable with: modisco motifs -s seqs.npz -a attr.npz ...
    seq_path = args.sequences_output
    attr_path = args.output

    np.savez(attr_path, hyp.astype(np.float32))
    np.savez(seq_path, one_hot)

    print(
        f"Wrote attributions {hyp.shape} to {attr_path}",
        file=sys.stderr,
    )
    print(
        f"Wrote sequences {one_hot.shape} to {seq_path}",
        file=sys.stderr,
    )
    print(
        f"Run: modisco motifs -s {seq_path} -a {attr_path} "
        f"-n 2000 -o modisco_results.h5",
        file=sys.stderr,
    )


def cmd_deltasvm(args: argparse.Namespace) -> None:
    from gkmsvm.importers.deltasvm import load_deltasvm_model

    dsvm = load_deltasvm_model(
        args.weights, args.l,
        include_rc=not args.no_rc,
        device=args.device,
    )

    records = _read_seqs(args.input)
    names = [name for name, seq in records]
    X = _encode_seqs(records)
    X = dsvm._match_device(X)

    from gkmsvm.backend import to_cpu
    scores = to_cpu(dsvm(X, verbose=args.verbose).flatten())

    out = sys.stdout if args.output is None else open(args.output, "w")
    try:
        out.write("name\tscore\n")
        for name, score in zip(names, scores):
            out.write(f"{name}\t{score:.6f}\n")
    finally:
        if out is not sys.stdout:
            out.close()


def cmd_import(args: argparse.Namespace) -> None:
    fmt = args.format
    inp = args.input
    output = Path(args.output)

    if fmt == "lsgkm":
        from gkmsvm.importers.lsgkm import load_lsgkm_model
        model = load_lsgkm_model(inp, device="cpu")
    elif fmt == "classic":
        from gkmsvm.importers.classic import load_classic_model
        model = load_classic_model(inp, device="cpu")
    elif fmt == "r_gkmsvm":
        from gkmsvm.importers.r_gkmsvm import load_r_gkmsvm_model
        model = load_r_gkmsvm_model(inp, device="cpu")
    else:
        print(f"Unknown format: {fmt}", file=sys.stderr)
        sys.exit(1)

    from gkmsvm.serialization import save_npz
    save_npz(model, output)
    print(
        f"Imported {fmt} model → {output} "
        f"({model.num_support_vectors} SVs)",
        file=sys.stderr,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gkmsvm",
        description="Gapped k-mer SVM for DNA sequence analysis.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show progress.",
    )
    sub = parser.add_subparsers(dest="command")

    # ── predict ──────────────────────────────────────────────────────
    p = sub.add_parser("predict", help="Score sequences with a trained model.")
    p.add_argument("-m", "--model", required=True, help="Model file (.npz or .model.txt).")
    p.add_argument("-i", "--input", required=True, help="Input FASTA file.")
    p.add_argument("-o", "--output", default=None, help="Output TSV (default: stdout).")
    p.add_argument("--batch-size", type=int, default=100, help="Batch size (default: 100).")
    _add_device_arg(p)

    # ── train ────────────────────────────────────────────────────────
    p = sub.add_parser("train", help="Train a gkm-SVM classifier.")
    p.add_argument("-p", "--positives", required=True, help="Positive sequences FASTA.")
    p.add_argument("-n", "--negatives", required=True, help="Negative sequences FASTA.")
    p.add_argument("-o", "--output", required=True, help="Output model (.npz or .model.txt).")
    p.add_argument("-C", type=float, default=1.0, help="Regularization (default: 1.0).")
    p.add_argument(
        "--solver", default="auto", choices=["auto", "smo", "libsvm"],
        help="Solver (default: auto).",
    )
    p.add_argument("--cache-size", type=int, default=256, help="SMO cache columns (default: 256).")
    _add_kernel_args(p)
    _add_device_arg(p)

    # ── trainsvr ─────────────────────────────────────────────────────
    p = sub.add_parser("trainsvr", help="Train a gkm-SVR regressor.")
    p.add_argument("-i", "--input", required=True, help="Sequences FASTA.")
    p.add_argument("--labels", required=True, help="Labels file (one value per line).")
    p.add_argument("-o", "--output", required=True, help="Output model (.npz or .model.txt).")
    p.add_argument("-C", type=float, default=1.0, help="Regularization (default: 1.0).")
    p.add_argument("--epsilon", type=float, default=0.1, help="SVR epsilon (default: 0.1).")
    _add_kernel_args(p)
    _add_device_arg(p)

    # ── ism ──────────────────────────────────────────────────────────
    p = sub.add_parser("ism", help="In-silico mutagenesis.")
    p.add_argument("-m", "--model", required=True, help="Model file.")
    p.add_argument("-i", "--input", required=True, help="Input FASTA.")
    p.add_argument("-o", "--output", required=True, help="Output .npz file (scores + names).")
    _add_device_arg(p)

    # ── explain ──────────────────────────────────────────────────────
    p = sub.add_parser(
        "explain", help="GkmExplain attribution scores.",
        epilog="Outputs two npz files for modisco: "
        "modisco motifs -s SEQS -a ATTR -n 2000 -o results.h5",
    )
    p.add_argument("-m", "--model", required=True, help="Model file.")
    p.add_argument("-i", "--input", required=True, help="Input FASTA.")
    p.add_argument(
        "-o", "--output", required=True,
        help="Attributions .npz file (hypothetical, [B,4,L] as arr_0).",
    )
    p.add_argument(
        "-s", "--sequences-output", required=True,
        help="Sequences .npz file (one-hot, [B,4,L] as arr_0).",
    )
    _add_device_arg(p)

    # ── deltasvm ─────────────────────────────────────────────────────
    p = sub.add_parser("deltasvm", help="Score sequences with deltaSVM weights.")
    p.add_argument("-w", "--weights", required=True, help="DeltaSVM weight file.")
    p.add_argument("-i", "--input", required=True, help="Input FASTA.")
    p.add_argument("-o", "--output", default=None, help="Output TSV (default: stdout).")
    p.add_argument("-l", type=int, required=True, help="L-mer window length.")
    p.add_argument("--no-rc", action="store_true", help="Disable reverse complement.")
    _add_device_arg(p)

    # ── import ───────────────────────────────────────────────────────
    p = sub.add_parser("import", help="Import external model format to .npz.")
    p.add_argument("-i", "--input", required=True, help="Input model file.")
    p.add_argument("-o", "--output", required=True, help="Output .npz file.")
    p.add_argument(
        "-f", "--format", required=True,
        choices=["lsgkm", "classic", "r_gkmsvm"],
        help="Source format.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "predict": cmd_predict,
        "train": cmd_train,
        "trainsvr": cmd_trainsvr,
        "ism": cmd_ism,
        "explain": cmd_explain,
        "deltasvm": cmd_deltasvm,
        "import": cmd_import,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
