"""MLX against the CPU on l-mer indexes of different densities (pre-release check).

The Metal paths through the SV l-mer index (fused query + reduce, mutation
impact, ISM, the RBF GkmExplain l-mer path) scatter each query's hits over
the index entries, so their speed against the CPU depends on how many
entries an l-mer has. This keeps random subsets of the GkmExplain paper's
dsQTL -t 3 model's support vectors (l=10, d=3; 32,715 SVs, about 18 entries
per non-empty l-mer) and times each path on CPU and MLX, checking that MLX
agrees with the CPU to float32 precision.

M5, 10 CPU threads (2026-11-04), MLX vs CPU:

  SVs     entries/l-mer  predict  variants  impact  ISM    hypothetical
  500     1.3            2.3x     2.7x      3.4x    2.9x   1.5x
  2,000   2.1            3.3x     3.7x      4.4x    4.0x   2.1x
  8,000   5.4            3.0x     3.2x      3.5x    3.3x   2.0x
  32,715  18.7           2.8x     2.8x      2.3x    1.8x   1.3x

A ratio below 1 on a release candidate is a regression to look into.

Usage (from the repo root, on Apple Silicon with MLX):
  python benchmarks/dev/mlx_index_density.py
  python benchmarks/dev/mlx_index_density.py --sv-sizes 500,32715 --threads 10
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
    ap.add_argument("--threads", type=int, default=10, help="CPU threads")
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

    import mlx.core as mx

    from gkmsvm import one_hot_encode
    from gkmsvm.explain import gkmexplain
    from gkmsvm.ism import ism
    from gkmsvm.svm import GkmSVM

    set_cpu_threads(args.threads)
    full = bench._load_paper_model(TAG, 1)
    dhs = [s[:300] for s in read_seqs(bench.SEQ_DIR / "gm12878_shared.fa") if len(s) >= 300]
    X = np.stack([one_hot_encode(s) for s in dhs[:1000]])
    _, X51, _, _ = bench.load_test_variants()
    ref, alt = X51[0][:4000], X51[1][:4000]

    def run(f):
        r = f()
        if isinstance(r, mx.array):
            mx.eval(r)
        return r

    def best(f):
        out = run(f)  # compiles kernels, builds caches
        ts = []
        for _ in range(args.repeats):
            t = time.perf_counter()
            run(f)
            ts.append(time.perf_counter() - t)
        return min(ts), np.asarray(out, dtype=np.float64)

    ops = [
        ("predict 1000 x 300 bp", lambda m, d: m(d(X))),
        ("variants kernel 4000", lambda m, d: m.score_variants(d(ref), d(alt), method="kernel")),
        ("variants gkmexplain 4000",
         lambda m, d: m.score_variants(d(ref), d(alt), method="gkmexplain")),
        ("ISM 16 x 300 bp", lambda m, d: ism(m, d(X[:16]))),
        ("hypothetical 64 x 300 bp", lambda m, d: gkmexplain(m, d(X[:64]), mode="hypothetical")),
    ]
    rng = np.random.default_rng(0)
    S_full = full.num_support_vectors
    sizes = [S_full if s == "all" else min(int(s), S_full) for s in args.sv_sizes.split(",")]
    worst = np.inf
    for S in sizes:
        idx = np.arange(S_full) if S == S_full else np.sort(rng.choice(S_full, S, replace=False))

        def model():
            return GkmSVM(
                full.support_sequences[idx],
                np.asarray(full.coefficients)[idx],
                full.bias,
                full.kernel_type,
                full._kernel_params,
                sv_lengths=None if full.sv_lengths is None else full.sv_lengths[idx],
            )

        cpu, gpu = model(), model().mlx()
        index = cpu._sv_index()
        filled = np.diff(index.offsets) > 0
        print(f"\n## {S:,} SVs: {index.ids.size:,} index entries, "
              f"{index.ids.size / max(1, filled.sum()):.1f} per non-empty l-mer", flush=True)
        for name, f in ops:
            tc, oc = best(lambda: f(cpu, lambda a: a))
            tg, og = best(lambda: f(gpu, mx.array))
            rel = np.abs(og - oc).max() / max(np.abs(oc).max(), 1e-30)
            worst = min(worst, tc / tg)
            print(f"  {name:26s} cpu {tc * 1e3:9.1f} ms  mlx {tg * 1e3:9.1f} ms  "
                  f"{tc / tg:5.2f}x  max rel diff {rel:.1e}", flush=True)
    print(f"\nSlowest MLX path: {worst:.2f}x the CPU"
          + ("  <-- below 1: check for a regression" if worst < 1 else ""))


if __name__ == "__main__":
    main()
