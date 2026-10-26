"""How much of each CUDA kernel's time is float64 arithmetic? Profile with ncu.

Runs each GPU operation the timing panels use once, inside an NVTX range, on
fig. 1C's model (the GkmExplain paper's -t 3 l=10 k=6 d=3, negset 1) and the
dsQTL 51bp variants:

  gram          compute_gram on random 300bp sequences (l-mer index)
  score_index   model(x), l-mer index (lmer_index_query: float64 atomics)
  score_join    model(x) with use_lmer_index=False (masked join)
  score_brute   model(x) on < 16 rows without the index (brute force)
  ism           ism(model, x)
  impact        mutation_impact(model, ref, alt) (fig. 1C's GkmExplain rows)
  explain       gkmexplain as called (CUDA explain_lmer kernel for l <= 12)
  explain_sv    gkmexplain with the l-mer path and explain table off, so the
                per-SV CUDA fused_explain kernel runs

Without ncu it prints each operation's wall time. Under ncu, the FP64 pipe
utilization, issued instruction mix and L2 atomic traffic per kernel say
whether FP64 throughput matters. Run ncu once per op: ISM and scoring launch
hundreds of kernels, so one --launch-count shared by every op runs out early.

Usage (GPU node):
  python benchmarks/dev/profile_fp64.py
  nsys profile --trace=cuda,nvtx -o prof python benchmarks/dev/profile_fp64.py
  ncu --nvtx --nvtx-include "regex:#1/" --launch-count 400 --csv --page raw \
      --metrics gpu__time_duration.sum,sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sectors_op_atom.sum,lts__t_sectors_op_red.sum \
      python benchmarks/dev/profile_fp64.py --ops ism > ncu_ism.csv

Call a process python directly under nsys/ncu: `conda run` reads their
options (--trace) as its own.

Result (L40S, FP64 1/64 of FP32, at ae1ca73 from #43; one warm call each):

  op            wall    main kernel (share of GPU time)   FP64-bound share
  ism           3.62 s  lmer_index_mutants (99%)          0%   (atomics; SM 3%)
  score_join    1.07 s  masked_join (76%)                 0%
  explain_sv    0.84 s  fused_explain (99%)               0%   (memory 78%)
  score_index   0.44 s  lmer_index_query (91%)            4%   (atomics, memory 74%)
  impact        0.35 s  lmer_index_query (92%)            3%
  gram          0.21 s  lmer_index_query (77%)            23%  (fused_cross_diagonal)
  explain       0.02 s  explain_lmer (53%)                55%
  score_brute   0.01 s  fused_pairwise_idx (96%)          96%

FP64-bound: kernels with the FP64 pipe >= 50% busy. The long operations are
bound by scattered atomics and memory traffic, so the L40S's low FP64 rate
only shows in the short ones.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OPS = ["gram", "score_index", "score_join", "score_brute", "ism", "impact", "explain",
       "explain_sv"]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ops", default=",".join(OPS))
    p.add_argument("--n-gram", type=int, default=2000)
    p.add_argument("--n-score", type=int, default=5000)
    p.add_argument("--n-ism", type=int, default=200)
    p.add_argument("--n-impact", type=int, default=2000)
    p.add_argument("--n-explain", type=int, default=100)
    p.add_argument("--repeats", type=int, default=2,
                   help="Runs per op; the first compiles and fills caches")
    args = p.parse_args()

    import cupy as cp
    from cupy.cuda import nvtx

    import dsqtl_auprc as bench
    from gkmsvm import explain, ism, mutation_impact, one_hot_encode
    from gkmsvm.gram import compute_gram

    model = bench._load_paper_model("t3_l10_k6_d3_c10_g2_t16", 1).cuda()
    _, (ref51, alt51), _, _ = bench.load_test_variants()
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGT"), 300)) for _ in range(args.n_gram)]
    x_gram = np.stack([one_hot_encode(s) for s in seqs])
    print(f"model: {model.num_support_vectors} SVs, kernel {model.kernel_type}; "
          f"GPU {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")

    def run_explain_sv():
        saved, table = explain._explain_lmer, model.use_explain_table
        explain._explain_lmer = lambda *a, **k: None
        model.use_explain_table = False
        try:
            return explain.gkmexplain(model, cp.asarray(ref51[:args.n_explain]))
        finally:
            explain._explain_lmer, model.use_explain_table = saved, table

    def run_join(n):
        model.use_lmer_index = False
        try:
            return model(cp.asarray(ref51[:n]))
        finally:
            model.use_lmer_index = True

    ops = {
        "gram": lambda: compute_gram(model.kernel, cp.asarray(x_gram)),
        "score_index": lambda: model(cp.asarray(ref51[:args.n_score])),
        "score_join": lambda: run_join(args.n_score),
        "score_brute": lambda: run_join(8),
        "ism": lambda: ism(model, cp.asarray(ref51[:args.n_ism])),
        "impact": lambda: mutation_impact(model, cp.asarray(ref51[:args.n_impact]),
                                          cp.asarray(alt51[:args.n_impact])),
        "explain": lambda: explain.gkmexplain(model, cp.asarray(ref51[:args.n_explain])),
        "explain_sv": run_explain_sv,
    }
    for name in args.ops.split(","):
        for r in range(args.repeats):
            nvtx.RangePush(f"{name}#{r}")
            t = time.perf_counter()
            ops[name]()
            cp.cuda.Device().synchronize()
            dt = time.perf_counter() - t
            nvtx.RangePop()
            print(f"{name}#{r}: {dt:.3f} s", flush=True)


if __name__ == "__main__":
    main()
