"""GkmExplain completeness check (Fig. S4).

Importance-mode attributions sum to the decision value minus the bias:
sum(gkmexplain(x)) = f(x) - b. Scores the Nanog reference lsgkm model
(GkmExplain paper, -t 2 l=11 k=7 d=3) on its 40 held-out test sequences plus
random 200bp sequences, and records both sides of the identity per sequence.

f(x) is always the float64 CPU score, so on CUDA or MLX the error is the
device's attributions against the exact target. MLX computes in float32; its
own score is recorded too (score_minus_bias_device), which separates the
attributions' float32 error from the scores'.

Usage:
  python benchmarks/completeness.py
  python benchmarks/completeness.py --device cuda \\
      --output benchmarks/results/completeness.tsv
  python benchmarks/completeness.py --device mlx
"""

from __future__ import annotations

import argparse

import numpy as np

from _common import REPO, read_seqs, to_device, write_tsv

FIXTURES = REPO / "tests" / "fixtures"
SEQ_LEN = 200


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mlx"])
    parser.add_argument(
        "--n-random",
        type=int,
        default=200,
        help="Random 200bp sequences added to the test set",
    )
    parser.add_argument(
        "--output", default=None, help="Save per-sequence values to TSV"
    )
    args = parser.parse_args()

    import pandas as pd
    from gkmsvm import gkmexplain, one_hot_encode
    from gkmsvm.backend import to_cpu
    from gkmsvm.importers.lsgkm import load_lsgkm_model

    path = str(FIXTURES / "nanog_reference.model.txt.gz")
    cpu_model = load_lsgkm_model(path)  # float64 target
    model = to_device(load_lsgkm_model(path), args.device)

    pos = read_seqs(FIXTURES / "nanog_pos_test.fa.gz")
    neg = read_seqs(FIXTURES / "nanog_neg_test.fa.gz")
    rng = np.random.default_rng(0)
    rand = ["".join(rng.choice(list("ACGT"), SEQ_LEN)) for _ in range(args.n_random)]
    seqs = pos + neg + rand
    groups = ["positive"] * len(pos) + ["negative"] * len(neg) + ["random"] * len(rand)

    X = np.stack([one_hot_encode(s) for s in seqs])
    score = np.asarray(to_cpu(cpu_model(X, batch_size=64)), dtype=np.float64).ravel()
    score_dev = np.asarray(to_cpu(model(X, batch_size=64)), dtype=np.float64).ravel()
    attr = to_cpu(gkmexplain(model, X, mode="importance", batch_size=50))
    attr_sum = np.asarray(attr, dtype=np.float64).sum(axis=(1, 2))
    expected = score - float(cpu_model.bias)

    err = np.abs(attr_sum - expected)
    print(f"{len(seqs)} sequences, {model.num_support_vectors} SVs")
    print(f"max |sum(attr) - (f(x) - b)| = {err.max():.3e}")
    print(f"max relative error            = {(err / np.abs(expected).max()).max():.3e}")
    if args.device != "cpu":
        print(f"max |f_{args.device}(x) - f_cpu(x)|      = {np.abs(score_dev - score).max():.3e}")

    if args.output:
        df = pd.DataFrame(
            {
                "group": groups,
                "score_minus_bias": expected,
                "attribution_sum": attr_sum,
                "score_minus_bias_device": score_dev - float(cpu_model.bias),
            }
        )
        write_tsv(
            args.output,
            df,
            "GkmExplain completeness, Nanog reference model",
            args.device,
            model="nanog_reference.model.txt.gz",
        )


if __name__ == "__main__":
    main()
