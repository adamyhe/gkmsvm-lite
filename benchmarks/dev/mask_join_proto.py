"""Prototype: gkm kernel as a signed sum of masked exact-match joins.

For a weight table h(m) that is zero beyond dmax mismatches (-t 0, -t 1, -t 2),
there are coefficients a_t with

    h(m) = sum_t a_t * C(l - m, j_t),   j_t = l - dmax + t,  t = 0..dmax

for every m (both sides vanish for m > dmax). C(l - m, j) counts the size-j
position subsets on which two l-mers with m mismatches agree, so

    K_raw(x, y) = sum_t a_t * sum_{|S| = j_t} <Phi_S(x), Phi_S(y) + Phi_S(rc y)>

where Phi_S(x) is the histogram of x's windows restricted to positions S.
Each inner product is an exact-match join on masked keys, with no verification.

This script checks the identity against the brute-force packed kernel on real
sequences and reports how many join hits it enumerates. It uses scipy.sparse
and is not fast; see docs/design/mask-join-kernel.md.

Usage:
  python benchmarks/dev/mask_join_proto.py --n 200
  python benchmarks/dev/mask_join_proto.py --n 200 -l 10 -k 6
"""

from __future__ import annotations

import argparse
import sys
import time
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import SEQ_DIR  # noqa: E402


def mask_coefficients(table: np.ndarray, l: int, dmax: int) -> tuple[list[int], np.ndarray]:
    """Mask sizes j_t and coefficients a_t with h(m) = sum_t a_t C(l-m, j_t)."""
    js = list(range(l - dmax, l + 1))
    A = np.array([[comb(l - m, j) for j in js] for m in range(dmax + 1)], dtype=float)
    return js, np.linalg.solve(A, table[: dmax + 1])


def masked_keys(packed: np.ndarray, positions: tuple[int, ...]) -> np.ndarray:
    """Gather the 2-bit bases at `positions` of packed windows into a key."""
    out = np.zeros_like(packed)
    for i, p in enumerate(positions):
        out |= ((packed >> (2 * p)) & 3) << (2 * i)
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", type=int, default=200, help="Sequences (half pos, half neg)")
    parser.add_argument("-l", type=int, default=11)
    parser.add_argument("-k", type=int, default=7)
    parser.add_argument("-d", type=int, default=3)
    args = parser.parse_args()

    import scipy.sparse as sp

    from gkmsvm import one_hot_encode, read_fasta
    from gkmsvm.codec import reverse_complement
    from gkmsvm.kernels.direct import _fused_pairwise_packed_numba, _pack_windows_cpu
    from gkmsvm.kernels.esttrunc import EstTruncGkmKernel

    half = args.n // 2
    seqs = ([s for _, s in read_fasta(str(SEQ_DIR / "gm12878_shared.fa"))][:half]
            + [s for _, s in read_fasta(str(SEQ_DIR / "nullseqs_gm12878_shared.1.1.fa"))][:half])
    N = len(seqs)
    X = np.stack([one_hot_encode(s) for s in seqs])
    kern = EstTruncGkmKernel(args.l, args.k, d=args.d)
    h, l, mm = kern._mismatch_table, kern.l, kern._min_matches
    dmax = l - mm

    bx = _pack_windows_cpu(np.ascontiguousarray(kern.base_index_windows(X)))
    brc = _pack_windows_cpu(np.ascontiguousarray(kern.base_index_windows(reverse_complement(X))))
    W = bx.shape[1]

    t0 = time.perf_counter()
    K_ref = (_fused_pairwise_packed_numba(bx, bx, h, l, mm)
             + _fused_pairwise_packed_numba(bx, brc, h, l, mm))
    t_ref = time.perf_counter() - t0

    js, a = mask_coefficients(h, l, dmax)
    seq_x = np.repeat(np.arange(N), W)
    seq_y = np.concatenate([seq_x, seq_x])
    wx = bx.ravel().astype(np.int64)
    wy = np.concatenate([bx.ravel(), brc.ravel()]).astype(np.int64)

    K = np.zeros((N, N))
    hits = 0.0
    t0 = time.perf_counter()
    for t, j in enumerate(js):
        for S in combinations(range(l), j):
            kx, ky = masked_keys(wx, S), masked_keys(wy, S)
            Px = sp.csr_matrix((np.ones(len(kx)), (kx, seq_x)), shape=(4 ** j, N))
            Py = sp.csr_matrix((np.ones(len(ky)), (ky, seq_y)), shape=(4 ** j, N))
            C = (Px.T @ Py).toarray()
            hits += C.sum()
            K += a[t] * C
    t_mask = time.perf_counter() - t0

    rel = np.abs(K - K_ref).max() / np.abs(K_ref).max()
    brute = N * N * W * W * 2
    n_masks = sum(comb(l, j) for j in js)
    print(f"N={N}, {X.shape[2]} bp, l={l} k={args.k} d={args.d}: mask sizes {js}, "
          f"{n_masks} masks, a_t = {np.array2string(a, precision=5)}")
    print(f"max |K_mask - K_brute| / max |K| = {rel:.1e}")
    print(f"brute force {t_ref:.2f}s, scipy prototype {t_mask:.1f}s")
    print(f"window pairs compared, brute force: {brute:.2e}")
    print(f"join hits enumerated:               {hits:.2e} ({brute / hits:.0f}x fewer)")


if __name__ == "__main__":
    main()
