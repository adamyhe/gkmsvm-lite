"""Train dsQTL models and compare VEP scoring methods.

Replicates the Lee 2015 deltaSVM pipeline and extends it to compare:
  - DeltaSVM (k-mer weight linear approximation)
  - Full kernel SVM scoring: score(alt) - score(ref)
  - GkmExplain VEP: hypothetical importance at variant position

Trains gkm-SVMs on GM12878 DNase-seq peaks (Beer lab data) with two
parameter settings:
  - l=10, k=6, d=3 (Lee 2015 original, matching published deltaSVM)
  - l=11, k=7, d=3 (LS-GKM defaults, matching ENCODE models)

Data requirements:
  examples/data/gm12878_sequence_sets/
    gm12878_shared.fa                   # 22,384 positive sequences
    nullseqs_gm12878_shared.{1..5}.1.fa # 5 GC-matched negative sets
    dsqtl_test_{pos,neg}.{major,minor}.fa # test variant sequences

  benchmarks/data/
    GSE31388_dsQtlTable.txt.gz          # effect sizes (optional)

Usage:
  python benchmarks/train_dsqtl_models.py --device cuda
  python benchmarks/train_dsqtl_models.py --device cuda --params l10k6
  python benchmarks/train_dsqtl_models.py --n-negsets 1 --params l10k6  # quick
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import gzip
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

SEQ_DIR = Path(__file__).parent.parent / "examples" / "data" / "gm12878_sequence_sets"
DATA_DIR = Path(__file__).parent / "data"


PARAM_SETS = {
    "l10k6": {"l": 10, "k": 6, "d": 3, "kernel_type": "estimated"},
    "l11k7": {"l": 11, "k": 7, "d": 3, "kernel_type": "estimated"},
}


def load_training_data(n_negsets: int = 5):
    from gkmsvm import read_fasta

    pos_seqs = [seq for _, seq in read_fasta(str(SEQ_DIR / "gm12878_shared.fa"))]
    print(f"Training positives: {len(pos_seqs):,} sequences, {len(pos_seqs[0])}bp")

    neg_sets = []
    for i in range(1, n_negsets + 1):
        negs = [seq for _, seq in read_fasta(
            str(SEQ_DIR / f"nullseqs_gm12878_shared.{i}.1.fa"))]
        neg_sets.append(negs)
        print(f"  Negative set {i}: {len(negs):,} sequences")

    return pos_seqs, neg_sets


def load_test_variants():
    from gkmsvm import read_fasta, one_hot_encode

    pos_major = list(read_fasta(str(SEQ_DIR / "dsqtl_test_pos.major.fa")))
    pos_minor = list(read_fasta(str(SEQ_DIR / "dsqtl_test_pos.minor.fa")))
    neg_major = list(read_fasta(str(SEQ_DIR / "dsqtl_test_neg.major.fa")))
    neg_minor = list(read_fasta(str(SEQ_DIR / "dsqtl_test_neg.minor.fa")))

    ref_seqs = [s for _, s in pos_major] + [s for _, s in neg_major]
    alt_seqs = [s for _, s in pos_minor] + [s for _, s in neg_minor]
    labels = np.array([1] * len(pos_major) + [0] * len(neg_major))

    X_ref = np.stack([one_hot_encode(s) for s in ref_seqs])
    X_alt = np.stack([one_hot_encode(s) for s in alt_seqs])

    print(f"Test variants: {len(labels)} ({labels.sum()} sig + "
          f"{(labels == 0).sum()} ctrl), {len(ref_seqs[0])}bp")

    effect_sizes = _load_effect_sizes(pos_major)

    return X_ref, X_alt, labels, effect_sizes


def _load_effect_sizes(pos_major: list[tuple[str, str]]) -> np.ndarray | None:
    import pandas as pd

    geo_path = DATA_DIR / "GSE31388_dsQtlTable.txt.gz"
    if not geo_path.exists():
        print("  Effect sizes not found, skipping regression metrics")
        return None

    geo = pd.read_csv(geo_path, sep="\t", compression="gzip")
    geo = geo.sort_values("Pr(>|t|)").drop_duplicates(
        subset=["Chr", "SNP"], keep="first")

    n_pos = len(pos_major)
    effects = np.full(n_pos, np.nan)
    for idx, (name, _) in enumerate(pos_major):
        parts = name.split(":")
        if len(parts) != 2:
            continue
        chrom = parts[0]
        coords = parts[1].split("-")
        if len(coords) != 2:
            continue
        try:
            snp_pos = int(coords[0]) + 9
            match = geo[(geo["Chr"] == chrom) & (geo["SNP"] == snp_pos)]
            if len(match) > 0:
                effects[idx] = float(match.iloc[0]["Estimate"])
        except ValueError:
            pass

    n_matched = (~np.isnan(effects)).sum()
    print(f"  Effect sizes matched: {n_matched}/{n_pos}")
    return effects


def train_models(pos_seqs, neg_sets, params: dict, device: str, C: float = 1.0):
    from gkmsvm import train_gkmsvm
    from gkmsvm.backend import to_cpu

    models = []
    dsvms = []

    for i, neg_seqs in enumerate(neg_sets):
        print(f"\n── Model {i+1}/{len(neg_sets)} "
              f"(l={params['l']} k={params['k']}) ──")
        t0 = time.time()
        m = train_gkmsvm(
            pos_seqs, neg_seqs,
            kernel_type=params["kernel_type"],
            l=params["l"], k=params["k"], d=params["d"],
            C=C, solver="auto",
            device=device, verbose=True,
        )
        print(f"  {m.num_support_vectors} SVs, {time.time() - t0:.1f}s")

        if device == "cuda":
            m.cuda()

        t0 = time.time()
        d = m.to_deltasvm(device=device, verbose=True)
        print(f"  DeltaSVM conversion: {time.time() - t0:.1f}s")

        m.cpu()
        models.append(m)
        dsvms.append(d)

    avg_weights = np.mean([to_cpu(d.weights) for d in dsvms], axis=0)
    avg_dsvm = copy.deepcopy(dsvms[0])
    avg_dsvm.cpu()
    avg_dsvm.weights = avg_weights
    print(f"\nAveraged {len(dsvms)} deltaSVM weight tables")

    return models, avg_dsvm


def score_deltasvm(dsvm, X_ref, X_alt):
    from gkmsvm.backend import to_cpu
    t0 = time.time()
    scores = to_cpu(dsvm.score_variants(X_ref, X_alt).flatten())
    print(f"  DeltaSVM: {len(scores)} variants in {time.time() - t0:.2f}s")
    return scores


def score_kernel(model, X_ref, X_alt, device: str, batch_size: int = 64):
    from gkmsvm.backend import to_cpu

    if device == "cuda":
        model.cuda()

    N = len(X_ref)
    scores = np.zeros(N, dtype=np.float64)

    t0 = time.time()
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ref_b = model._match_device(X_ref[start:end])
        alt_b = model._match_device(X_alt[start:end])
        s_ref = model(ref_b, verbose=False).flatten()
        s_alt = model(alt_b, verbose=False).flatten()
        scores[start:end] = to_cpu(s_alt - s_ref)

    elapsed = time.time() - t0
    model.cpu()
    print(f"  Kernel VEP: {N} variants in {elapsed:.1f}s "
          f"({N * 2 / elapsed:.0f} seqs/s)")
    return scores


def score_gkmexplain(model, X_ref, X_alt, device: str, batch_size: int = 32):
    from gkmsvm.backend import to_cpu

    if device == "cuda":
        model.cuda()

    N = len(X_ref)
    scores = np.zeros(N, dtype=np.float64)

    t0 = time.time()
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ref_b = model._match_device(X_ref[start:end])
        alt_b = model._match_device(X_alt[start:end])
        s = model.score_variants(ref_b, alt_b, method="gkmexplain",
                                 batch_size=end - start, verbose=False)
        scores[start:end] = to_cpu(s).flatten()

    elapsed = time.time() - t0
    model.cpu()
    print(f"  GkmExplain VEP: {N} variants in {elapsed:.1f}s "
          f"({N / elapsed:.0f} variants/s)")
    return scores


def evaluate(scores: np.ndarray, labels: np.ndarray,
             effect_sizes: np.ndarray | None, method: str) -> dict:
    abs_scores = np.abs(scores)
    auroc = roc_auc_score(labels, abs_scores)
    auprc = average_precision_score(labels, abs_scores)

    result = {"method": method, "auroc": auroc, "auprc": auprc}

    if effect_sizes is not None:
        sig_mask = (labels == 1) & (~np.isnan(effect_sizes))
        if sig_mask.sum() >= 10:
            sig_scores = scores[sig_mask]
            sig_effects = effect_sizes[sig_mask]
            r_p, _ = pearsonr(sig_scores, sig_effects)
            r_s, _ = spearmanr(sig_scores, sig_effects)
            result["pearson"] = r_p
            result["spearman"] = r_s
            result["n_sig"] = int(sig_mask.sum())

    return result


def print_results(results: list[dict], param_label: str):
    print(f"\n{'=' * 70}")
    print(f"Results: {param_label}")
    print(f"{'=' * 70}")
    print(f"  {'Method':<25s} {'AUROC':>8s} {'AUPRC':>8s} "
          f"{'Pearson':>8s} {'Spearman':>8s}")
    print(f"  {'-' * 25} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
    for r in results:
        pear = f"{r['pearson']:.4f}" if "pearson" in r else "N/A"
        spear = f"{r['spearman']:.4f}" if "spearman" in r else "N/A"
        print(f"  {r['method']:<25s} {r['auroc']:>8.4f} {r['auprc']:>8.4f} "
              f"{pear:>8s} {spear:>8s}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--params", default="both",
                        choices=["l10k6", "l11k7", "both"],
                        help="Parameter set to train (default: both)")
    parser.add_argument("--n-negsets", type=int, default=5,
                        help="Number of negative sets to train on (default: 5)")
    parser.add_argument("--output", default=None,
                        help="Save results to TSV file")
    parser.add_argument("--save-models", default=None,
                        help="Directory to save trained models")
    args = parser.parse_args()

    print("Loading data...")
    pos_seqs, neg_sets = load_training_data(n_negsets=args.n_negsets)
    X_ref, X_alt, labels, effect_sizes = load_test_variants()

    param_keys = (["l10k6", "l11k7"] if args.params == "both"
                  else [args.params])

    all_results = []

    for pk in param_keys:
        params = PARAM_SETS[pk]
        print(f"\n{'=' * 70}")
        print(f"Training: l={params['l']} k={params['k']} d={params['d']} "
              f"({args.n_negsets} neg sets)")
        print(f"{'=' * 70}")

        models, avg_dsvm = train_models(
            pos_seqs, neg_sets, params, args.device)

        if args.save_models:
            from gkmsvm.serialization import save_model
            save_dir = Path(args.save_models)
            save_dir.mkdir(parents=True, exist_ok=True)
            for i, m in enumerate(models):
                path = save_dir / f"gm12878_{pk}_neg{i+1}.npz"
                save_model(m, str(path))
                print(f"  Saved {path}")

        results = []

        print("\nScoring with deltaSVM (averaged weights)...")
        dsvm_scores = score_deltasvm(avg_dsvm, X_ref, X_alt)
        results.append(evaluate(dsvm_scores, labels, effect_sizes,
                                f"deltaSVM ({pk})"))

        print("\nScoring with kernel VEP (model 1)...")
        kernel_scores = score_kernel(
            models[0], X_ref, X_alt, args.device)
        results.append(evaluate(kernel_scores, labels, effect_sizes,
                                f"kernel ({pk})"))

        print("\nScoring with GkmExplain VEP (model 1)...")
        explain_scores = score_gkmexplain(
            models[0], X_ref, X_alt, args.device)
        results.append(evaluate(explain_scores, labels, effect_sizes,
                                f"gkmexplain ({pk})"))

        corr_dk = np.corrcoef(dsvm_scores, kernel_scores)[0, 1]
        corr_de = np.corrcoef(dsvm_scores, explain_scores)[0, 1]
        print(f"\n  Score correlations (all variants):")
        print(f"    deltaSVM vs kernel:     r = {corr_dk:.4f}")
        print(f"    deltaSVM vs gkmexplain: r = {corr_de:.4f}")

        print_results(results, pk)
        all_results.extend(results)

        del models, avg_dsvm
        gc.collect()

    if len(param_keys) > 1:
        print_results(all_results, "all parameter sets")

    if args.output:
        import pandas as pd
        pd.DataFrame(all_results).to_csv(args.output, sep="\t", index=False)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
