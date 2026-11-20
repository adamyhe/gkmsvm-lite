"""RBF GkmExplain's l-mer path: per-row V against index sums per visit.

For RBF models the l-mer path either sums each neighbor's index entries
(per-query SV weights) at every visit, or builds each row's V once
(`explain.cpu._dense_V`, one pass over the index) and takes the non-RBF V
path. `explain.lmer._use_dense_V` picks by cost; this times both ways (and
reports the pick) on random subsets of the GkmExplain paper's dsQTL -t 3 model's
SVs (l=10, d=3; 32,715 SVs, ~18 entries per l-mer) at several query
lengths, on CPU and MLX, and checks they agree. The constants in
`explain.lmer` (_DENSE_V_S, _VISIT_S) were fit from its output.

Usage (from the repo root; MLX columns need Apple Silicon):
  python benchmarks/dev/explain_dense_v.py
  python benchmarks/dev/explain_dense_v.py --sv-sizes 2000,all --lengths 19,300 --devices cpu
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dsqtl_auprc as bench  # noqa: E402
from _common import read_seqs, set_cpu_threads  # noqa: E402

TAG = "t3_l10_k6_d3_c10_g2_t16"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sv-sizes", default="500,2000,8000,all",
                    help="Comma-separated SV subset sizes ('all' = the whole model)")
    ap.add_argument("--lengths", default="19,51,101,300", help="Query lengths (bp)")
    ap.add_argument("--rows", type=int, default=64, help="Queries per call")
    ap.add_argument("--modes", default="hypothetical,perturbation")
    ap.add_argument("--devices", default="cpu,mlx")
    ap.add_argument("--threads", type=int, default=10, help="CPU threads")
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

    from gkmsvm import one_hot_encode
    from gkmsvm.explain import gkmexplain
    from gkmsvm.explain import lmer
    from gkmsvm.svm import GkmSVM

    set_cpu_threads(args.threads)
    full = bench._load_paper_model(TAG, 1)
    dhs = [s for s in read_seqs(bench.SEQ_DIR / "gm12878_shared.fa") if len(s) >= 300]
    lengths = [int(n) for n in args.lengths.split(",")]
    X = {n: np.stack([one_hot_encode(s[150 - n // 2:][:n]) for s in dhs[: args.rows]])
         for n in lengths}

    def best(model, x, mode, dense):
        lmer._DENSE_V = dense
        out = gkmexplain(model, x, mode=mode, batch_size=50)  # compiles, caches
        ts = []
        for _ in range(args.repeats):
            t = time.perf_counter()
            gkmexplain(model, x, mode=mode, batch_size=50)
            ts.append(time.perf_counter() - t)
        return min(ts), np.asarray(out, dtype=np.float64)

    # The pick _use_dense_V makes for a 50-row batch
    picks = []
    use = lmer._use_dense_V
    lmer._use_dense_V = lambda *a: picks.append(use(*a)) or picks[-1]

    rng = np.random.default_rng(0)
    S_full = full.num_support_vectors
    sizes = [S_full if s == "all" else min(int(s), S_full) for s in args.sv_sizes.split(",")]
    for S in sizes:
        idx = np.arange(S_full) if S == S_full else np.sort(rng.choice(S_full, S, replace=False))
        index = None
        for dev in args.devices.split(","):
            model = GkmSVM(
                full.support_sequences[idx],
                np.asarray(full.coefficients)[idx],
                full.bias,
                full.kernel_type,
                full._kernel_params,
                sv_lengths=None if full.sv_lengths is None else full.sv_lengths[idx],
            )
            if dev == "mlx":
                model.mlx()
            if index is None:
                index = model._sv_index()
                print(f"\n## {S:,} SVs: {index.ids.size / (len(index.offsets) - 1):.2f} "
                      "index entries per l-mer", flush=True)
            for mode in args.modes.split(","):
                for n in lengths:
                    t_idx, o_idx = best(model, X[n], mode, False)
                    t_den, o_den = best(model, X[n], mode, True)
                    picks.clear()
                    lmer._DENSE_V = None
                    gkmexplain(model, X[n][:50], mode=mode, batch_size=50)
                    pick = "dense" if picks[0] else "index"
                    rel = np.abs(o_den - o_idx).max() / max(np.abs(o_idx).max(), 1e-30)
                    # Within 10%: a tie at this run-to-run noise
                    ok = (pick == "dense") == (t_den < t_idx) or abs(t_idx / t_den - 1) < 0.1
                    print(f"  {dev:3s} {mode:12s} {n:4d} bp  index {t_idx * 1e3:8.1f} ms  "
                          f"dense {t_den * 1e3:8.1f} ms  ({t_idx / t_den:5.2f}x)  "
                          f"auto picks {pick}{'' if ok else '  <-- the slower one'}  "
                          f"max rel diff {rel:.1e}", flush=True)
    lmer._DENSE_V = None


if __name__ == "__main__":
    main()
