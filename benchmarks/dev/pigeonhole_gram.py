"""Prototype: pigeonhole-filtered CPU pairwise kernel vs the brute-force loop.

The packed CPU kernel (_fused_pairwise_packed_numba) compares every window of
x with every window of y and adds table[m] only when m <= dmax mismatches
(dmax = l - min_matches, = d for -t 2). Split the l positions into dmax + 1
contiguous blocks: a pair with <= dmax mismatches matches exactly on at least
one block. Index each y's windows by block value (counting sort, CSR) and
compare a query window only with windows that share a block. A pair matching
on several blocks is counted at its first matching block only. The skipped
pairs all contribute 0, so the result is the brute-force sum up to
floating-point summation order.

Usage:
  python benchmarks/dev/pigeonhole_gram.py
  python benchmarks/dev/pigeonhole_gram.py --n 500 -l 10 -k 6 -d 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit, prange

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import SEQ_DIR  # noqa: E402

from gkmsvm.kernels.direct import (  # noqa: E402
    _fused_pairwise_packed_numba,
    _pack_windows_cpu,
    _packed_mismatches,
)


def block_layout(l: int, dmax: int) -> tuple[np.ndarray, np.ndarray]:
    """Bit shifts and masks of dmax + 1 contiguous blocks covering l positions."""
    nb = dmax + 1
    sizes = [l // nb + (1 if i < l % nb else 0) for i in range(nb)]
    shifts, masks, pos = [], [], 0
    for s in sizes:
        shifts.append(2 * pos)
        masks.append((1 << (2 * s)) - 1)
        pos += s
    return np.array(shifts, dtype=np.uint32), np.array(masks, dtype=np.uint32)


@njit(parallel=True, cache=True)
def build_block_index(by, shifts, masks):
    """Per sequence and block, window positions sorted by block value (CSR).

    Returns offsets [S, nb, K + 1] and vals [S, nb, W], the packed windows in
    block-value order (K = the largest block's number of values). Storing the
    values, not positions, keeps the candidate scan contiguous.
    """
    S, W = by.shape
    nb = shifts.shape[0]
    K = int(masks.max()) + 1
    offsets = np.zeros((S, nb, K + 1), dtype=np.int32)
    vals = np.empty((S, nb, W), dtype=np.uint32)
    for s in prange(S):
        for blk in range(nb):
            sh, mk = shifts[blk], masks[blk]
            for j in range(W):
                offsets[s, blk, ((by[s, j] >> sh) & mk) + 1] += 1
            for key in range(K):
                offsets[s, blk, key + 1] += offsets[s, blk, key]
            fill = offsets[s, blk, :K].copy()
            for j in range(W):
                key = (by[s, j] >> sh) & mk
                vals[s, blk, fill[key]] = by[s, j]
                fill[key] += 1
    return offsets, vals


@njit(parallel=True, cache=True, fastmath=True)
def pigeonhole_pairwise(bx, offsets, vals, shifts, masks, table, l, min_matches):
    """Same result as _fused_pairwise_packed_numba, visiting only candidate pairs."""
    B, Wx = bx.shape
    S = vals.shape[0]
    nb = shifts.shape[0]
    result = np.empty((B, S), dtype=np.float64)
    for idx in prange(B * S):
        b = idx // S
        s = idx % S
        acc = 0.0
        for i in range(Wx):
            xv = bx[b, i]
            for blk in range(nb):
                key = (xv >> shifts[blk]) & masks[blk]
                for t in range(offsets[s, blk, key], offsets[s, blk, key + 1]):
                    yv = vals[s, blk, t]
                    x = xv ^ yv
                    dup = False
                    for pb in range(blk):
                        if (x >> shifts[pb]) & masks[pb] == 0:
                            dup = True
                            break
                    if dup:
                        continue
                    matches = l - _packed_mismatches(xv, yv)
                    if matches >= min_matches:
                        acc += table[l - matches]
        result[b, s] = acc
    return result


def pair_layout(l: int, dmax: int):
    """dmax + 2 blocks; a pair with <= dmax mismatches matches on >= 2 of them.

    Returns block shifts/masks plus, per pair of blocks (p < q), the shifts,
    masks and low-key width used to form a combined key, and a [nb, nb] table
    mapping (p, q) to its pair index.
    """
    shifts, masks = block_layout(l, dmax + 1)
    nb = len(shifts)
    bits = [int(np.log2(int(m) + 1)) for m in masks]
    pairs = [(p, q) for p in range(nb) for q in range(p + 1, nb)]
    pair_id = -np.ones((nb, nb), dtype=np.int32)
    for i, (p, q) in enumerate(pairs):
        pair_id[p, q] = i
    cs = np.array([[shifts[p], masks[p], shifts[q], masks[q], bits[p]] for p, q in pairs],
                  dtype=np.uint32)
    return shifts, masks, cs, pair_id


@njit(inline="always")
def _pair_key(v, cs, c):
    return ((v >> cs[c, 0]) & cs[c, 1]) | (((v >> cs[c, 2]) & cs[c, 3]) << cs[c, 4])


@njit(parallel=True, cache=True)
def build_pair_index(by, cs):
    """Like build_block_index, keyed on the values of two blocks."""
    S, W = by.shape
    nc = cs.shape[0]
    K = 0
    for c in range(nc):
        K = max(K, int(((cs[c, 3] << cs[c, 4]) | cs[c, 1])) + 1)
    offsets = np.zeros((S, nc, K + 1), dtype=np.int32)
    vals = np.empty((S, nc, W), dtype=np.uint32)
    for s in prange(S):
        for c in range(nc):
            for j in range(W):
                offsets[s, c, _pair_key(by[s, j], cs, c) + 1] += 1
            for key in range(K):
                offsets[s, c, key + 1] += offsets[s, c, key]
            fill = offsets[s, c, :K].copy()
            for j in range(W):
                key = _pair_key(by[s, j], cs, c)
                vals[s, c, fill[key]] = by[s, j]
                fill[key] += 1
    return offsets, vals


@njit(parallel=True, cache=True, fastmath=True)
def pigeonhole_pair_pairwise(bx, offsets, vals, shifts, masks, cs, pair_id,
                             table, l, min_matches):
    """Two-block pigeonhole; each pair is counted at its first two matching blocks."""
    B, Wx = bx.shape
    S = vals.shape[0]
    nb = shifts.shape[0]
    nc = cs.shape[0]
    result = np.empty((B, S), dtype=np.float64)
    for idx in prange(B * S):
        b = idx // S
        s = idx % S
        acc = 0.0
        for i in range(Wx):
            xv = bx[b, i]
            for c in range(nc):
                key = _pair_key(xv, cs, c)
                for t in range(offsets[s, c, key], offsets[s, c, key + 1]):
                    yv = vals[s, c, t]
                    x = xv ^ yv
                    first = -1
                    second = -1
                    for blk in range(nb):
                        if (x >> shifts[blk]) & masks[blk] == 0:
                            if first < 0:
                                first = blk
                            else:
                                second = blk
                                break
                    if pair_id[first, second] != c:
                        continue
                    matches = l - _packed_mismatches(xv, yv)
                    if matches >= min_matches:
                        acc += table[l - matches]
        result[b, s] = acc
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", type=int, default=1000, help="Sequences (half pos, half neg)")
    parser.add_argument("-l", type=int, default=11)
    parser.add_argument("-k", type=int, default=7)
    parser.add_argument("-d", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    import numba

    from gkmsvm import one_hot_encode, read_fasta
    from gkmsvm.codec import reverse_complement
    from gkmsvm.kernels.esttrunc import EstTruncGkmKernel

    half = args.n // 2
    pos = [s for _, s in read_fasta(str(SEQ_DIR / "gm12878_shared.fa"))][:half]
    neg = [s for _, s in read_fasta(str(SEQ_DIR / "nullseqs_gm12878_shared.1.1.fa"))][:half]
    lengths = {len(s) for s in pos + neg}
    if len(lengths) != 1:
        sys.exit(f"prototype needs equal-length sequences, got {sorted(lengths)[:5]}...")
    X = np.stack([one_hot_encode(s) for s in pos + neg])

    kern = EstTruncGkmKernel(l=args.l, k=args.k, d=args.d)
    table, l, mm = kern._mismatch_table, kern.l, kern._min_matches
    dmax = l - mm
    shifts, masks = block_layout(l, dmax)
    bx = _pack_windows_cpu(np.ascontiguousarray(kern.base_index_windows(X)))
    by_rc = _pack_windows_cpu(
        np.ascontiguousarray(kern.base_index_windows(reverse_complement(X)))
    )
    W = bx.shape[1]
    print(f"N={args.n}, {X.shape[2]} bp, W={W} windows, l={l} k={args.k} d={args.d}, "
          f"min_matches={mm}, {len(shifts)} blocks of "
          f"{[int(np.log2(m + 1)) // 2 for m in masks]} bases, "
          f"{numba.get_num_threads()} threads")

    def brute():
        return (_fused_pairwise_packed_numba(bx, bx, table, l, mm)
                + _fused_pairwise_packed_numba(bx, by_rc, table, l, mm))

    def pigeon():
        off, vals = build_block_index(bx, shifts, masks)
        off_rc, vals_rc = build_block_index(by_rc, shifts, masks)
        return (pigeonhole_pairwise(bx, off, vals, shifts, masks, table, l, mm)
                + pigeonhole_pairwise(bx, off_rc, vals_rc, shifts, masks, table, l, mm))

    p_shifts, p_masks, cs, pair_id = pair_layout(l, dmax)

    def pigeon2():
        off, vals = build_pair_index(bx, cs)
        off_rc, vals_rc = build_pair_index(by_rc, cs)
        args2 = (p_shifts, p_masks, cs, pair_id, table, l, mm)
        return (pigeonhole_pair_pairwise(bx, off, vals, *args2)
                + pigeonhole_pair_pairwise(bx, off_rc, vals_rc, *args2))

    # Warm up (JIT) on a small slice
    o2, v2 = build_pair_index(bx[:4], cs)
    pigeonhole_pair_pairwise(bx[:4], o2, v2, p_shifts, p_masks, cs, pair_id, table, l, mm)
    small = slice(0, 4)
    _fused_pairwise_packed_numba(bx[small], bx[small], table, l, mm)
    o, r = build_block_index(bx[small], shifts, masks)
    pigeonhole_pairwise(bx[small], o, r, shifts, masks, table, l, mm)

    def timed(f):
        best, out = np.inf, None
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            out = f()
            best = min(best, time.perf_counter() - t0)
        return out, best

    K_b, t_b = timed(brute)
    K_p, t_p = timed(pigeon)
    K_2, t_2 = timed(pigeon2)
    print(f"brute force:          {t_b:.2f}s")
    for name, K, t in (("pigeonhole, 1 block", K_p, t_p), ("pigeonhole, 2 blocks", K_2, t_2)):
        rel = np.abs(K - K_b).max() / np.abs(K_b).max()
        print(f"{name}: {t:.2f}s  speedup {t_b / t:.1f}x  max |diff|/max |K| {rel:.1e}")

    # Candidate fraction: expected sum over blocks of W / 4^size, per query window
    cand = sum(W / (int(m) + 1) for m in masks)
    print(f"expected candidates per query window: {cand:.1f} of {W} ({cand / W:.1%})")
    cand2 = sum(W / ((int(c[3]) << int(c[4]) | int(c[1])) + 1) for c in cs)
    print(f"  two-block scheme: {cand2:.1f} of {W} ({cand2 / W:.1%}), {len(cs)} pair indexes")


if __name__ == "__main__":
    main()
