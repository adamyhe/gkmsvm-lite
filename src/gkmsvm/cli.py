"""Command-line interface for gkmsvm-lite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _add_device_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mlx"],
        help="Compute device (default: auto).",
    )


# Window length for training when --length isn't given: the ENCODE gkm-SVM
# and DART-Eval convention
TRAIN_LENGTH = 300


def _add_input_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input FASTA (BED/narrowPeak with --genome).",
    )


def _add_input_args(
    parser: argparse.ArgumentParser,
    *,
    length_default: str,
    genome: bool = True,
    variable_length: bool = False,
) -> None:
    """Options for reading sequences: --genome, --length, --variable-length,
    --allow-other-bases (see _load_inputs)."""
    if genome:
        parser.add_argument(
            "--genome",
            default=None,
            help="Genome FASTA: read the inputs as BED/narrowPeak files and take "
            "--length bp around each summit (narrowPeak column 10, else the "
            "midpoint). Windows past a chromosome end are dropped.",
        )
    parser.add_argument(
        "--length",
        type=int,
        default=None,
        help="Window length: FASTA records of different lengths (or longer than "
        "this) are center-cropped to it; a shorter record is an error. Default: "
        f"{length_default}.",
    )
    if variable_length:
        parser.add_argument(
            "--variable-length",
            action="store_true",
            help="Use FASTA records at their own lengths, as LS-GKM does "
            "(compatibility), instead of cropping.",
        )
    parser.add_argument(
        "--allow-other-bases",
        action="store_true",
        help="Read letters other than ACGTN (IUPAC codes, ...) as N, scored as A as "
        "LS-GKM does, instead of stopping with an error. N is always allowed.",
    )


def _fail(msg: str):
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def _model_length(model) -> int | None:
    """The model's SV length, or None when its SVs differ in length."""
    return None if model.sv_lengths is not None else model.support_sequences.shape[2]


def _load_inputs(args: argparse.Namespace, path: str, default_length: int | None):
    """(names, X [B, 4, L], lengths or None, mask over BED lines or None).

    With --genome, path is a BED/narrowPeak file: windows of --length (else
    default_length) around each summit. Otherwise a FASTA: used as is when its
    records share a length (and --length doesn't say otherwise), center-cropped
    to --length (else default_length) when they differ; with
    --variable-length, padded with each record's length.
    """
    from gkmsvm.codec import center_crop, one_hot_encode
    from gkmsvm.fasta import read_fasta

    if getattr(args, "genome", None):
        from gkmsvm.fasta import extract_loci, read_loci

        length = args.length or default_length
        if length is None:
            _fail("give --length for windows from --genome")
        loci = read_loci(path)
        X, mask = extract_loci(loci, args.genome, in_window=length, return_mask=True)
        if (~mask).any():
            print(
                f"Dropped {int((~mask).sum())} of {len(mask)} loci (chromosome missing "
                "from the genome, or window past its end)",
                file=sys.stderr,
            )
        return list(loci["name"][mask]), X, None, mask

    records = read_fasta(path)
    names = [name for name, _ in records]
    seqs = [seq for _, seq in records]

    def encode(s):
        # N (and with --allow-other-bases any letter) encodes as all-zero,
        # scored as A, as in LS-GKM
        return one_hot_encode(s, allow_n=True, allow_other=args.allow_other_bases)

    if getattr(args, "variable_length", False):
        from gkmsvm.svm import as_batch

        X, lengths = as_batch([encode(s) for s in seqs])
        return names, X, lengths, None
    lens = {len(s) for s in seqs}
    target = args.length
    if target is None and len(lens) > 1:
        target = default_length
        if target is None:
            _fail(
                f"{path}: records range from {min(lens)} to {max(lens)} bp; give "
                "--length to center-crop them"
            )
    if target is not None and lens != {target}:
        try:
            seqs = center_crop(seqs, target, names)
        except ValueError as e:
            hint = (
                "; extract full windows from the genome with --genome"
                if hasattr(args, "genome")
                else ""
            )
            if hasattr(args, "variable_length"):
                hint += ", or use records at their own lengths with --variable-length"
            _fail(f"{path}: {e}{hint}")
    return names, np.stack([encode(s) for s in seqs]), None, None


def _as_list(X, lengths):
    """[4, L_i] arrays of X's rows at their lengths (X itself when uniform)."""
    return X if lengths is None else [x[:, :n] for x, n in zip(X, lengths)]


def _add_kernel_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-t",
        "--kernel-type",
        default="estimated",
        help="Kernel type: 0-6, or name (default: estimated).",
    )
    parser.add_argument("-l", type=int, default=11, help="L-mer length (default: 11).")
    parser.add_argument(
        "-k", type=int, default=7, help="K informative positions (default: 7)."
    )
    parser.add_argument("-d", type=int, default=3, help="Max mismatches (default: 3).")
    parser.add_argument(
        "--no-rc", action="store_true", help="Disable reverse complement."
    )
    parser.add_argument(
        "--gamma", type=float, default=1.0, help="RBF gamma (for -t 3, 5)."
    )
    parser.add_argument(
        "-M",
        type=int,
        default=None,
        help="Center window weight, 1-255 (for -t 4, 5; default: 50).",
    )
    parser.add_argument(
        "-H",
        type=float,
        default=None,
        help="Window weight half-life in bp (for -t 4, 5; default: 50).",
    )


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    """Options train and trainsvr share."""
    parser.add_argument(
        "-o", "--output", required=True, help="Output model (.npz or .model.txt)."
    )
    parser.add_argument(
        "-C", type=float, default=1.0, help="Regularization (default: 1.0)."
    )
    parser.add_argument(
        "--solver",
        default="auto",
        choices=["auto", "libsvm", "smo", "nystrom"],
        help="Solver (default: auto: libsvm if the Gram fits in RAM, else smo).",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=None,
        help="SMO kernel-column cache size (default: sized from available RAM).",
    )
    _add_kernel_args(parser)
    _add_device_arg(parser)
    _add_input_args(
        parser,
        length_default="300 (when records differ in length)",
        variable_length=True,
    )


def _load_model(path: str, device: str = "cpu"):
    from gkmsvm.serialization import load_model

    return load_model(path, device=device)


def _write_tsv(path: str | None, columns: tuple[str, str], names, values) -> None:
    """Two-column TSV to `path`, or stdout when None."""
    import contextlib

    with contextlib.nullcontext(sys.stdout) if path is None else open(path, "w") as out:
        out.write("\t".join(columns) + "\n")
        for name, value in zip(names, values):
            out.write(f"{name}\t{value:.6f}\n")


def _save_model(model, path: str) -> None:
    """Save as .npz, or LS-GKM text for any other extension."""
    model.save(path, fmt="npz" if Path(path).suffix == ".npz" else "lsgkm")
    print(
        f"Saved model to {path} "
        f"({model.num_support_vectors} SVs, bias={model.bias:.4f})",
        file=sys.stderr,
    )


def cmd_predict(args: argparse.Namespace) -> None:
    model = _load_model(args.model, device=args.device)
    names, X, lengths, _ = _load_inputs(args, args.input, _model_length(model))

    from gkmsvm.backend import to_cpu

    scores = to_cpu(
        model(X, lengths=lengths, batch_size=args.batch_size, verbose=args.verbose)
    ).ravel()
    _write_tsv(args.output, ("name", "score"), names, scores)
    if args.output:
        print(f"Wrote {len(scores)} scores to {args.output}", file=sys.stderr)


def _train_kwargs(args: argparse.Namespace) -> dict:
    """The keyword arguments train and trainsvr pass to the trainer."""
    return {
        "kernel_type": args.kernel_type,
        "l": args.l,
        "k": args.k,
        "d": args.d,
        "C": args.C,
        "gamma": args.gamma,
        "M": args.M,
        "H": args.H,
        "include_rc": not args.no_rc,
        "solver": args.solver,
        "cache_size": args.cache_size,
        "device": args.device,
        "allow_other_bases": args.allow_other_bases,
        "verbose": args.verbose,
    }


def cmd_train(args: argparse.Namespace) -> None:
    from gkmsvm.train import train_gkmsvm

    length = args.length or TRAIN_LENGTH
    _, pos, pos_len, _ = _load_inputs(args, args.positives, length)
    _, neg, neg_len, _ = _load_inputs(args, args.negatives, length)
    if not args.variable_length and pos.shape[2] != neg.shape[2]:
        _fail(
            f"positives are {pos.shape[2]} bp and negatives {neg.shape[2]} bp; give "
            "--length to crop both"
        )

    model = train_gkmsvm(
        _as_list(pos, pos_len),
        _as_list(neg, neg_len),
        pos_weight=args.w,
        **_train_kwargs(args),
    )
    _save_model(model, args.output)


def cmd_trainsvr(args: argparse.Namespace) -> None:
    from gkmsvm.train import train_gkmsvr

    _, X, lengths, mask = _load_inputs(args, args.input, args.length or TRAIN_LENGTH)
    labels = np.loadtxt(Path(args.labels), dtype=np.float64)
    n_in = len(mask) if mask is not None else X.shape[0]
    if labels.shape[0] != n_in:
        _fail(f"{n_in} sequences but {labels.shape[0]} labels")
    if mask is not None:
        labels = labels[mask]  # labels of the loci kept

    model = train_gkmsvr(
        _as_list(X, lengths),
        labels,
        epsilon=args.epsilon,
        **_train_kwargs(args),
    )
    _save_model(model, args.output)


def cmd_ism(args: argparse.Namespace) -> None:
    from gkmsvm.backend import to_cpu
    from gkmsvm.ism import ism

    model = _load_model(args.model, device=args.device)
    names, X, lengths, _ = _load_inputs(args, args.input, _model_length(model))
    names = np.array(names)
    one_hot = X.copy()

    result = ism(model, X, lengths=lengths, verbose=args.verbose)
    result = to_cpu(result)

    # With --variable-length, rows are padded past their lengths
    extra = {} if lengths is None else {"lengths": lengths}
    np.savez(args.output, scores=result, one_hot=one_hot, names=names, **extra)
    print(
        f"Wrote ISM scores {result.shape} to {args.output}",
        file=sys.stderr,
    )


def cmd_explain(args: argparse.Namespace) -> None:
    from gkmsvm.backend import to_cpu
    from gkmsvm.explain import gkmexplain

    model = _load_model(args.model, device=args.device)
    _, X, lengths, _ = _load_inputs(args, args.input, _model_length(model))
    one_hot = X.astype(np.float32)

    hyp = to_cpu(
        gkmexplain(model, X, mode="hypothetical", lengths=lengths, verbose=args.verbose)
    )

    # Two files: sequences and attributions, each [B, 4, L] as arr_0.
    # Directly usable with: modisco motifs -s seqs.npz -a attr.npz ...
    # (with --variable-length, rows are zero past their lengths, also saved)
    seq_path = args.sequences_output
    attr_path = args.output

    extra = {} if lengths is None else {"lengths": lengths}
    np.savez(attr_path, hyp.astype(np.float32), **extra)
    np.savez(seq_path, one_hot, **extra)

    print(
        f"Wrote attributions {hyp.shape} to {attr_path}",
        file=sys.stderr,
    )
    print(
        f"Wrote sequences {one_hot.shape} to {seq_path}",
        file=sys.stderr,
    )
    if lengths is not None:
        print(
            "Records differ in length: rows are zero past their lengths (saved as "
            "'lengths'); TF-MoDISco expects one length (crop with --length)",
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
        args.weights,
        args.l,
        include_rc=not args.no_rc,
        device=args.device,
    )

    names, X, lengths, _ = _load_inputs(args, args.input, None)

    from gkmsvm.backend import to_cpu

    scores = to_cpu(dsvm(X, lengths=lengths, verbose=args.verbose).flatten())
    _write_tsv(args.output, ("name", "score"), names, scores)


def cmd_to_deltasvm(args: argparse.Namespace) -> None:
    model = _load_model(args.model, device=args.device)
    dsvm = model.to_deltasvm(
        device="cpu", query_norm=args.query_norm, verbose=args.verbose
    )

    output = Path(args.output)
    _save_deltasvm_weights(dsvm, output)

    print(
        f"Converted to DeltaSVM ({dsvm.num_kmers:,}/{dsvm.weights.shape[0]:,} "
        "non-zero weights) "
        f"→ {output}",
        file=sys.stderr,
    )


def _save_deltasvm_weights(dsvm, path: Path) -> None:
    """Write DeltaSVM weights in kmer<TAB>weight format."""
    from gkmsvm.backend import to_cpu
    from gkmsvm.deltasvm import _index_to_kmer

    weights = to_cpu(dsvm.weights)
    k = dsvm.k
    with open(path, "w") as f:
        if not dsvm.include_rc:
            f.write("# include_rc=false (RC already in weights)\n")
        for idx in range(weights.shape[0]):
            w = float(weights[idx])
            if w != 0.0:
                f.write(f"{_index_to_kmer(idx, k)}\t{w:.8g}\n")


def cmd_score_variants(args: argparse.Namespace) -> None:
    from gkmsvm.backend import to_cpu

    model = _load_model(args.model, device=args.device)

    ref_names, X_ref, ref_len, _ = _load_inputs(args, args.ref, _model_length(model))
    _, X_alt, alt_len, _ = _load_inputs(args, args.alt, _model_length(model))
    if X_ref.shape[0] != X_alt.shape[0] or (
        not args.variable_length and X_ref.shape != X_alt.shape
    ):
        _fail(
            f"{X_ref.shape[0]} ref sequences of {X_ref.shape[2]} bp but "
            f"{X_alt.shape[0]} alt sequences of {X_alt.shape[2]} bp"
        )

    deltas = to_cpu(
        model.score_variants(
            X_ref,
            X_alt,
            method=args.method,
            ref_lengths=ref_len,
            alt_lengths=alt_len,
            verbose=args.verbose,
        )
    ).ravel()
    _write_tsv(args.output, ("name", "delta"), ref_names, deltas)


def cmd_import(args: argparse.Namespace) -> None:
    fmt = args.format
    inp = args.input
    output = Path(args.output)

    if fmt == "lsgkm":
        from gkmsvm.importers.lsgkm import load_lsgkm_model

        model = load_lsgkm_model(inp, device="cpu")
    else:
        # classic, r_gkmsvm: gkmsvm_train's files (C++ and R write the same),
        # which don't record the kernel
        from gkmsvm.importers.classic import load_classic_model

        model = load_classic_model(
            inp,
            args.svseq,
            L=args.l,
            k=args.k,
            d=args.d,
            filter_type=args.filter_type,
            include_rc=not args.no_rc,
            device="cpu",
        )

    from gkmsvm.serialization import save_npz

    save_npz(model, output)
    print(
        f"Imported {fmt} model → {output} ({model.num_support_vectors} SVs)",
        file=sys.stderr,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gkmsvm",
        description="Gapped k-mer SVM for DNA sequence analysis.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show progress.",
    )
    sub = parser.add_subparsers(dest="command")

    # ── predict ──────────────────────────────────────────────────────
    p = sub.add_parser("predict", help="Score sequences with a trained model.")
    p.add_argument(
        "-m", "--model", required=True, help="Model file (.npz or .model.txt)."
    )
    _add_input_arg(p)
    p.add_argument("-o", "--output", default=None, help="Output TSV (default: stdout).")
    p.add_argument(
        "--batch-size", type=int, default=100, help="Batch size (default: 100)."
    )
    _add_device_arg(p)
    _add_input_args(p, length_default="the model's SV length", variable_length=True)

    # ── train ────────────────────────────────────────────────────────
    p = sub.add_parser("train", help="Train a gkm-SVM classifier.")
    p.add_argument(
        "-p",
        "--positives",
        required=True,
        help="Positive sequences FASTA (BED/narrowPeak with --genome).",
    )
    p.add_argument(
        "-n",
        "--negatives",
        required=True,
        help="Negative sequences FASTA (BED/narrowPeak with --genome).",
    )
    p.add_argument(
        "-w",
        type=float,
        default=1.0,
        help="Positives' C is w*C (as LS-GKM -w; default: 1.0).",
    )
    _add_train_args(p)

    # ── trainsvr ─────────────────────────────────────────────────────
    p = sub.add_parser("trainsvr", help="Train a gkm-SVR regressor.")
    p.add_argument(
        "-i",
        "--input",
        required=True,
        help="Sequences FASTA (BED/narrowPeak with --genome).",
    )
    p.add_argument("--labels", required=True, help="Labels file (one value per line).")
    p.add_argument(
        "--epsilon", type=float, default=0.1, help="SVR epsilon (default: 0.1)."
    )
    _add_train_args(p)

    # ── ism ──────────────────────────────────────────────────────────
    p = sub.add_parser("ism", help="In-silico mutagenesis.")
    p.add_argument("-m", "--model", required=True, help="Model file.")
    _add_input_arg(p)
    p.add_argument(
        "-o", "--output", required=True, help="Output .npz file (scores + names)."
    )
    _add_device_arg(p)
    _add_input_args(
        p,
        length_default="the model's SV length (when records differ in length)",
        variable_length=True,
    )

    # ── explain ──────────────────────────────────────────────────────
    p = sub.add_parser(
        "explain",
        help="GkmExplain attribution scores.",
        epilog="Outputs two npz files for modisco: "
        "modisco motifs -s SEQS -a ATTR -n 2000 -o results.h5",
    )
    p.add_argument("-m", "--model", required=True, help="Model file.")
    _add_input_arg(p)
    p.add_argument(
        "-o",
        "--output",
        required=True,
        help="Attributions .npz file (hypothetical, [B,4,L] as arr_0).",
    )
    p.add_argument(
        "-s",
        "--sequences-output",
        required=True,
        help="Sequences .npz file (one-hot, [B,4,L] as arr_0).",
    )
    _add_device_arg(p)
    _add_input_args(
        p,
        length_default="the model's SV length (when records differ in length)",
        variable_length=True,
    )

    # ── deltasvm ─────────────────────────────────────────────────────
    p = sub.add_parser("deltasvm", help="Score sequences with deltaSVM weights.")
    p.add_argument("-w", "--weights", required=True, help="DeltaSVM weight file.")
    _add_input_arg(p)
    p.add_argument("-o", "--output", default=None, help="Output TSV (default: stdout).")
    p.add_argument("-l", type=int, required=True, help="L-mer window length.")
    p.add_argument("--no-rc", action="store_true", help="Disable reverse complement.")
    _add_device_arg(p)
    _add_input_args(
        p,
        length_default="none: records must share a length, or give --length",
        variable_length=True,
    )

    # ── to-deltasvm ──────────────────────────────────────────────────
    p = sub.add_parser(
        "to-deltasvm",
        help="Convert a trained model to DeltaSVM weights.",
    )
    p.add_argument(
        "-m", "--model", required=True, help="Model file (.npz or .model.txt)."
    )
    p.add_argument("-o", "--output", required=True, help="Output DeltaSVM weight file.")
    p.add_argument(
        "--query-norm",
        action="store_true",
        help="Keep each l-mer's normalization (weights = gkmpredict on the "
        "l-mer minus bias, as lsgkm deltaSVM weights; implied for RBF).",
    )
    _add_device_arg(p)

    # ── score-variants ──────────────────────────────────────────────
    p = sub.add_parser(
        "score-variants",
        help="Score variant effects (kernel score delta or GkmExplain mutation impact).",
    )
    p.add_argument("-m", "--model", required=True, help="Model file.")
    p.add_argument("--ref", required=True, help="Reference sequences FASTA.")
    p.add_argument("--alt", required=True, help="Alternate sequences FASTA.")
    p.add_argument("-o", "--output", default=None, help="Output TSV (default: stdout).")
    p.add_argument(
        "--method",
        default="kernel",
        choices=["kernel", "gkmexplain"],
        help="kernel: score(alt) - score(ref); gkmexplain: mutation impact "
        "(lsgkm gkmexplain -m 5). Default: kernel.",
    )
    _add_device_arg(p)
    _add_input_args(
        p,
        length_default="the model's SV length (when records differ in length)",
        genome=False,
        variable_length=True,
    )

    # ── import ───────────────────────────────────────────────────────
    p = sub.add_parser("import", help="Import external model format to .npz.")
    p.add_argument("-i", "--input", required=True, help="Input model file.")
    p.add_argument("-o", "--output", required=True, help="Output .npz file.")
    p.add_argument(
        "-f",
        "--format",
        required=True,
        choices=["lsgkm", "classic", "r_gkmsvm"],
        help="Source format.",
    )
    # gkmsvm_train's PREFIX_svalpha.out + PREFIX_svseq.fa record no kernel
    g = p.add_argument_group(
        "classic/R gkmSVM PREFIX_svalpha.out models (the kernel gkmsvm_kernel used)"
    )
    g.add_argument(
        "--svseq", default=None, help="PREFIX_svseq.fa (default: next to the input)."
    )
    g.add_argument("-l", type=int, default=10, help="Word length (default: 10).")
    g.add_argument(
        "-k", type=int, default=6, help="Informative positions (default: 6)."
    )
    g.add_argument("-d", type=int, default=3, help="Max mismatches (default: 3).")
    g.add_argument(
        "-t",
        "--filter-type",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="gkmsvm_kernel -t (R useTgkm): 0 full, 1 truncated (default), "
        "2 gkm counts.",
    )
    g.add_argument(
        "--no-rc", action="store_true", help="Without reverse complements (-R)."
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
        "to-deltasvm": cmd_to_deltasvm,
        "score-variants": cmd_score_variants,
        "import": cmd_import,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
