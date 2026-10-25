"""compute_gram's self-Gram through the l-mer index vs the masked join.

compute_gram goes through the l-mer index whenever it applies
(kernels/lmer_index.py): one index over all N sequences, queried in row
blocks, each row only from the diagonal on. The masked join (column panels,
the column index rebuilt for every panel) is what it replaced, and what
kernels the index can't serve still use. This times both on the same N,
forcing the join by making lmer_index.applies return False.

For each N, on GM12878 DHS and null sequences (300 bp), -t 2 l=11 k=7 d=3 by
default. max_abs_diff compares the two Grams on --check-rows random rows.
Kernels are JIT-compiled on a small N first. Both runs reuse one [N, N]
float64 buffer (12.8 GB at N=40k).

Usage (from the repo root):
  python benchmarks/dev/gram_index_vs_masked.py --device cpu --threads 16
  python benchmarks/dev/gram_index_vs_masked.py --device cuda --sizes 2000,10000,40000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import RESULTS, SEQ_DIR, gpu_sync, read_seqs, set_cpu_threads, write_tsv

DEFAULT_SIZES = [1000, 2000, 5000, 10000, 20000, 40000]
SETS = ["gm12878_shared.fa"] + [
    f"nullseqs_gm12878_shared.{i}.1.fa" for i in range(1, 6)
]


def load_onehot(n_max: int, device: str, seed: int):
    from gkmsvm import one_hot_encode
    from gkmsvm.backend import to_gpu

    seqs = []
    for name in SETS:
        if len(seqs) >= n_max:
            break
        seqs += read_seqs(SEQ_DIR / name)
    if len(seqs) < n_max:
        sys.exit(f"--sizes asks for {n_max} sequences; {SEQ_DIR} has {len(seqs)}")
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(seqs))[:n_max]
    X = np.stack([one_hot_encode(seqs[i]) for i in pick])
    return to_gpu(X) if device == "cuda" else X


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)))
    ap.add_argument("--kernel", default="estimated", help="kernel type (default -t 2)")
    ap.add_argument("--l", type=int, default=11)
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--threads", type=int, default=16, help="CPU threads")
    ap.add_argument("--check-rows", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=str(RESULTS / "dev" / "gram_index.tsv"))
    args = ap.parse_args()

    import pandas as pd

    from gkmsvm.gram import compute_gram
    from gkmsvm.kernels import lmer_index
    from gkmsvm.svm import KERNEL_BUILDERS, resolve_kernel_type

    threads = set_cpu_threads(args.threads)
    sizes = sorted(int(s) for s in args.sizes.split(","))
    params = {"L": args.l, "k": args.k, "d": args.d}
    kernel = KERNEL_BUILDERS[resolve_kernel_type(args.kernel)](params)
    applies = lmer_index.applies
    if not applies(kernel.base):
        sys.exit("the l-mer index doesn't apply to this kernel")
    X_all = load_onehot(sizes[-1], args.device, args.seed)

    def timed(X, out, index):
        lmer_index.applies = applies if index else (lambda *a, **k: False)
        gpu_sync()
        t0 = time.perf_counter()
        compute_gram(kernel, X, out=out)
        gpu_sync()
        return time.perf_counter() - t0

    # JIT warm-up, untimed
    warm = np.empty((200, 200))
    for index in (False, True):
        timed(X_all[:200], warm, index)

    rng = np.random.default_rng(args.seed)
    rows = []
    for N in sizes:
        X = X_all[:N]
        gram = np.empty((N, N))
        check = np.sort(rng.choice(N, size=min(args.check_rows, N), replace=False))
        t_masked = timed(X, gram, index=False)
        ref = gram[check].copy()
        print(f"N={N} masked: {t_masked:.2f}s", flush=True)
        t_index = timed(X, gram, index=True)
        diff = float(np.abs(gram[check] - ref).max())
        print(
            f"N={N} index: {t_index:.2f}s, masked/index {t_masked / t_index:.2f}x, "
            f"max diff {diff:.1e}",
            flush=True,
        )
        common = {
            "N": N,
            "kernel": resolve_kernel_type(args.kernel),
            "l": args.l,
            "k": args.k,
            "d": args.d,
            "device": args.device,
            "threads": threads if args.device == "cpu" else None,
        }
        rows += [
            {**common, "method": "masked", "gram_s": t_masked},
            {
                **common,
                "method": "index",
                "gram_s": t_index,
                "max_abs_diff": diff,
            },
        ]
        del gram, ref
        write_tsv(
            args.output,
            pd.DataFrame(rows),
            "Self-Gram: l-mer index vs masked join",
            args.device,
        )


if __name__ == "__main__":
    main()
