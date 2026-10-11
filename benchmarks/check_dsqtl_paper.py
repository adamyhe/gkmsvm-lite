"""Per-variant check against the GkmExplain paper's published dsQTL outputs.

Loads the paper's lsgkm models and compares our scores, variant by variant,
to the outputs in github.com/kundajelab/gkmexplain (dsQTL/.../dsqtl_analysis):
  - deltaSVM (deltasvm_out/), -t 2 and -t 3
  - ISM (ism_out/) and GkmExplain mode 5 (gkmexplain_out/), -t 3 only

deltaSVM is computed three ways to localize any mismatch:
  table     — to_deltasvm(query_norm=True) on --device, scored on CPU
  direct    — gkmpredict on each 10-mer the variants use, model() on --device
  direct-cpu — the same on CPU, for a random subset of those 10-mers
  deltaSVM.pl — the table with poly-A/T zeroed, as the Beer lab script scores

Usage:
  python benchmarks/check_dsqtl_paper.py --device cuda
  python benchmarks/check_dsqtl_paper.py --device cuda --negsets 1 --tags t3
"""

from __future__ import annotations

import argparse
import time
from urllib.request import urlretrieve

import numpy as np
from sklearn.metrics import average_precision_score

import bench_dsqtl as bench

PAPER_TAGS = {"t2": "t2_l10_k6_d3_t16", "t3": "t3_l10_k6_d3_c10_g2_t16"}
PAPER_OUT_URL = (
    "https://raw.githubusercontent.com/kundajelab/gkmexplain/master/dsQTL/"
    "gm12878_sequence_sets/dsqtl_analysis/{path}"
)
PAPER_OUT_DIR = bench.DATA_DIR / "gkmexplain_paper_outputs"
L = 10


def _paper_file(path: str) -> list[tuple[str, float]]:
    local = PAPER_OUT_DIR / path
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading {path}...")
        urlretrieve(PAPER_OUT_URL.format(path=path), local)
    rows = [line.rstrip("\n").split("\t") for line in open(local)]
    return [(r[0], float(r[1])) for r in rows]


def _paper_scores(kind: str, tag: str, negset: int, names19: list[str]) -> np.ndarray:
    full = f"gkmsvm_{tag}_negset{negset}.txt"
    if kind == "deltaSVM":
        files = [f"deltasvm_out/positives_{full}", f"deltasvm_out/negatives_{full}"]
    elif kind == "ISM":
        files = [f"ism_out/ism_diff_pos_51_{full}", f"ism_out/ism_diff_neg_51_{full}"]
    else:
        files = [f"gkmexplain_out/positives_51_{full}", f"gkmexplain_out/negatives_51_{full}"]
    rows = _paper_file(files[0]) + _paper_file(files[1])
    names = [n for n, _ in rows]
    if kind != "deltaSVM":
        # 51bp files are keyed by the context window; map back to the 19bp name.
        names = [_name51_to_19(n) for n in names]
    if names != names19:
        raise RuntimeError(f"{kind} {tag} negset{negset}: variant order differs from Beer lab files")
    return np.array([v for _, v in rows])


def _name51_to_19(name: str) -> str:
    chrom, rng = name.split(":")
    start = int(rng.split("-")[0]) + bench.CONTEXT_HALF + 1 - bench.FLANK
    return f"{chrom}:{start}-{start + 2 * bench.FLANK}"


def _lmer_index(X: np.ndarray) -> np.ndarray:
    """[N, 4, Lx] one-hot → [N, W] l-mer indices (first base most significant)."""
    b = X.argmax(axis=1).astype(np.int64)
    W = X.shape[2] - L + 1
    powers = 4 ** np.arange(L - 1, -1, -1, dtype=np.int64)
    return np.stack([(b[:, w:w + L] * powers).sum(axis=1) for w in range(W)], axis=1)


def _lmer_onehot(idx: np.ndarray) -> np.ndarray:
    powers = 4 ** np.arange(L - 1, -1, -1, dtype=np.int64)
    bases = (idx[:, None] // powers[None, :]) % 4
    x = np.zeros((len(idx), 4, L), dtype=np.float64)
    x[np.arange(len(idx))[:, None], bases, np.arange(L)[None, :]] = 1.0
    return x


def _lmer_weights(model, idx: np.ndarray, device: str) -> np.ndarray:
    from gkmsvm.backend import to_cpu

    if device == "cuda":
        model.cuda()
    w = to_cpu(model(_lmer_onehot(idx), batch_size=8192)).ravel() - float(model.bias)
    model.cpu()
    return w


def _compare(rows: list, label: str, ours: np.ndarray, ref: np.ndarray, labels: np.ndarray):
    err = np.abs(ours - ref)
    rows.append({
        "check": label,
        "max_abs_err": float(err.max()),
        "max_rel_err": float((err / np.maximum(np.abs(ref), 1e-3)).max()),
        "corr": float(np.corrcoef(ours, ref)[0, 1]),
        "auprc_ours": float(average_precision_score(labels, np.abs(ours))) if labels is not None else np.nan,
        "auprc_ref": float(average_precision_score(labels, np.abs(ref))) if labels is not None else np.nan,
    })
    r = rows[-1]
    print(f"  {label:<38s} max|err| {r['max_abs_err']:.3e}  corr {r['corr']:.8f}"
          + (f"  AUPRC ours {r['auprc_ours']:.5f} ref {r['auprc_ref']:.5f}" if labels is not None else ""))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--negsets", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--tags", nargs="+", default=["t3", "t2"], choices=list(PAPER_TAGS))
    parser.add_argument("--n-cpu-lmers", type=int, default=2000,
                        help="10-mers checked against a CPU forward pass (default: 2000)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default=None, help="Save checks to TSV")
    args = parser.parse_args()

    from gkmsvm import read_fasta

    bench.download_data()
    X19, X51, labels, _ = bench.load_test_variants()
    names19 = (
        [n for n, _ in read_fasta(str(bench.SEQ_DIR / "dsqtl_test_pos.major.fa"))]
        + [n for n, _ in read_fasta(str(bench.SEQ_DIR / "dsqtl_test_neg.major.fa"))]
    )
    ref_idx, alt_idx = _lmer_index(X19[0]), _lmer_index(X19[1])
    uniq, inv = np.unique(np.concatenate([ref_idx, alt_idx], axis=1), return_inverse=True)
    inv = inv.reshape(len(ref_idx), -1)
    W = ref_idx.shape[1]
    print(f"{len(uniq):,} distinct 10-mers across {len(labels):,} variants")
    rng = np.random.default_rng(0)
    cpu_sel = rng.choice(len(uniq), size=min(args.n_cpu_lmers, len(uniq)), replace=False)

    rows = []
    for t in args.tags:
        tag = PAPER_TAGS[t]
        for negset in args.negsets:
            print(f"\n=== {tag} negset{negset} ===")
            model = bench._load_paper_model(tag, negset)
            paper_dsvm = _paper_scores("deltaSVM", tag, negset, names19)

            t0 = time.time()
            if args.device == "cuda":
                model.cuda()
            dsvm = model.to_deltasvm(device="cpu", query_norm=True)
            model.cpu()
            dsvm.cpu()
            print(f"  to_deltasvm: {time.time() - t0:.0f}s")
            table = np.asarray(dsvm.weights, dtype=np.float64)
            table_scores = dsvm.score_variants(X19[0], X19[1]).ravel()

            w_direct = _lmer_weights(model, uniq, args.device)
            direct_scores = (w_direct[inv[:, W:]] - w_direct[inv[:, :W]]).sum(axis=1)
            w_cpu = _lmer_weights(model, uniq[cpu_sel], "cpu")

            p = f"{t} neg{negset}"
            _compare(rows, f"{p} weights table vs direct", table[uniq], w_direct, None)
            _compare(rows, f"{p} weights direct vs cpu", w_direct[cpu_sel], w_cpu, None)
            _compare(rows, f"{p} weights table vs cpu", table[uniq[cpu_sel]], w_cpu, None)
            _compare(rows, f"{p} deltaSVM table vs paper", table_scores, paper_dsvm, labels)
            _compare(rows, f"{p} deltaSVM direct vs paper", direct_scores, paper_dsvm, labels)
            pl_scores = bench.deltasvm_pl(dsvm).score_variants(X19[0], X19[1]).ravel()
            _compare(rows, f"{p} deltaSVM.pl vs paper", pl_scores, paper_dsvm, labels)

            if t == "t3":
                for kind, method in (("ISM", "kernel"), ("GkmExplain", "gkmexplain")):
                    if args.device == "cuda":
                        model.cuda()
                    from gkmsvm.backend import to_cpu
                    ours = to_cpu(model.score_variants(
                        X51[0], X51[1], method=method, batch_size=args.batch_size,
                    )).ravel()
                    model.cpu()
                    _compare(rows, f"{p} {kind} vs paper", ours,
                             _paper_scores(kind, tag, negset, names19), labels)
            del model, dsvm

    if args.output:
        import pandas as pd

        pd.DataFrame(rows).to_csv(args.output, sep="\t", index=False)
        print(f"\nSaved checks to {args.output}")


if __name__ == "__main__":
    main()
