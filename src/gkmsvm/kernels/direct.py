"""-t 0 (gkm_cnt): DirectGkmKernel, the base class of the count-table kernels
(esttrunc -t 1/-t 2, weighted -t 4/-t 5), and its numba loops: brute-force
pairs, self-kernels, and the mutant self-kernels ISM needs. The CUDA and Metal
versions are in direct_gpu.py."""

from __future__ import annotations

from math import comb

import numpy as np
from numba import njit, prange

from gkmsvm.kernels.direct_gpu import (
    _fused_cross_diagonal_packed_gpu,
    _fused_pairwise_packed_gpu,
    _fused_pairwise_packed_mlx,
    _mutant_diag_device,
)
from gkmsvm.kernels.packed import PackedGkmKernel, device_of
from gkmsvm.kernels.primitives import _counts, _packed_mismatches, _rc_window, _spans


@njit(parallel=True, cache=True, fastmath=True)
def _fused_pairwise_packed_numba(bx, by, table, l, min_matches):
    """Fused pairwise using packed uint32 comparison with min-matches skip;
    PAD windows (rows of shorter sequences) are skipped."""
    B = bx.shape[0]
    S = by.shape[0]
    sx, sy = _spans(bx), _spans(by)
    result = np.empty((B, S), dtype=np.float64)
    for idx in prange(B * S):
        b = idx // S
        s = idx % S
        acc = 0.0
        for i in range(sx[b, 0], sx[b, 1]):
            bx_val = bx[b, i]
            for j in range(sy[s, 0], sy[s, 1]):
                matches = l - _packed_mismatches(bx_val, by[s, j])
                if matches >= min_matches:
                    acc += table[l - matches]
        result[b, s] = acc
    return result


@njit(inline="always")
def _g(u, v, table):
    """Pair weight table[mismatch count]. Branch-free: zero past the cutoff."""
    return table[_packed_mismatches(u, v)]


@njit(inline="always")
def _block_sum(a, c, symmetric, table):
    """sum over i, j of g(a[i], c[j]); for a symmetric matrix only pairs j >= i are visited.

    Rows go in blocks of 4 so that each pass over c serves four rows (a loop
    pass, not a pair, is the cost on short rows).
    """
    W = a.shape[0]
    diag = 0.0  # j == i, symmetric only
    off = 0.0  # j > i when symmetric, every j otherwise
    i0 = 0
    while i0 < W:
        n = min(4, W - i0)
        if n == 4 and not symmetric:
            u0, u1, u2, u3 = a[i0], a[i0 + 1], a[i0 + 2], a[i0 + 3]
            for j in range(W):
                v = c[j]
                off += (
                    _g(u0, v, table)
                    + _g(u1, v, table)
                    + _g(u2, v, table)
                    + _g(u3, v, table)
                )
        elif n == 4:
            u0, u1, u2, u3 = a[i0], a[i0 + 1], a[i0 + 2], a[i0 + 3]
            diag += (
                _g(u0, c[i0], table)
                + _g(u1, c[i0 + 1], table)
                + _g(u2, c[i0 + 2], table)
                + _g(u3, c[i0 + 3], table)
            )
            off += (
                _g(u0, c[i0 + 1], table)
                + _g(u0, c[i0 + 2], table)
                + _g(u0, c[i0 + 3], table)
                + _g(u1, c[i0 + 2], table)
                + _g(u1, c[i0 + 3], table)
                + _g(u2, c[i0 + 3], table)
            )
            for j in range(i0 + 4, W):
                v = c[j]
                off += (
                    _g(u0, v, table)
                    + _g(u1, v, table)
                    + _g(u2, v, table)
                    + _g(u3, v, table)
                )
        else:
            for i in range(i0, W):
                for j in range(i if symmetric else 0, W):
                    if symmetric and j == i:
                        diag += _g(a[i], c[j], table)
                    else:
                        off += _g(a[i], c[j], table)
        i0 += n
    return diag + 2.0 * off if symmetric else off


@njit(parallel=True, cache=True, fastmath=True)
def _self_diagonal_packed_numba(bx, by, table, l):
    """K_raw(x, x) [B] from packed windows; by is bx, or bx followed by its RC windows.

    Both blocks are symmetric, so each unordered pair is visited once: the
    forward block as (i, j), the RC block as (i, W-1-j), since its column j is
    rc(window W-1-j). A row whose RC block isn't the reverse complement of its
    forward one visits every RC pair. A padded row (a shorter sequence) is
    its first n windows, and its RC block reversed, whose pads come last too.
    """
    B, W = bx.shape
    has_rc = by.shape[1] > W
    counts = _counts(bx)
    result = np.empty(B, dtype=np.float64)
    for b in prange(B):
        n = counts[b]
        fwd = bx[b, :n]
        acc = _block_sum(fwd, fwd, True, table)
        if has_rc:
            # RC windows in forward order: rev[k] = rc(fwd[k]) when paired
            rev = by[b, W:][::-1][:n].copy()
            paired = True
            for k in range(n):
                if rev[k] != _rc_window(fwd[k], l):
                    paired = False
                    break
            acc += _block_sum(fwd, rev, paired, table)
        result[b] = acc
    return result


@njit(inline="always")
def _block_delta(f, nf, a_s, a_e, c, nc, c_s, c_e, table):
    """sum over i, j of g(f'_i, c'_j) - g(f_i, c_j), where f' is f with windows
    a_s..a_e replaced by nf and c' is c with windows c_s..c_e replaced by nc."""
    d = 0.0
    for i in range(a_s, a_e):  # changed rows, every column
        u, u_new = f[i], nf[i - a_s]
        for j in range(c.shape[0]):
            cj = c[j]
            cj_new = nc[j - c_s] if c_s <= j < c_e else cj
            d += _g(u_new, cj_new, table) - _g(u, cj, table)
    for i in range(f.shape[0]):  # unchanged rows, changed columns
        if a_s <= i < a_e:
            continue
        u = f[i]
        for j in range(c_s, c_e):
            d += _g(u, nc[j - c_s], table) - _g(u, c[j], table)
    return d


@njit(parallel=True, cache=True, fastmath=True)
def _mutant_diag_numba(fwd, rc, new, new_rc, diag, table, l):
    """K_raw(mutant, mutant) [4, L, B]: the reference's plus the pairs touching a
    changed window (see PackedGkmKernel.raw_diag_mutants; rc has 0 columns
    without RC). A padded row is its first n windows (its RC row's last n);
    a position past them changes nothing."""
    B, W = fwd.shape
    L = new.shape[1]
    has_rc = rc.shape[1] > 0
    counts = _counts(fwd)
    out = np.empty((4, L, B))
    for item in prange(L * B):
        p, b = item // B, item % B
        n = counts[b]
        a_s, a_e = max(0, p - l + 1), min(p + 1, n)
        if a_s >= a_e:
            for v in range(4):
                out[v, p, b] = diag[b]
            continue
        f = fwd[b, :n]
        # RC windows r_s..r_e change; the row's own are W-n.. (new_rc from r_s)
        q = L - 1 - p
        r_s, r_e = max(0, q - l + 1), min(q + 1, W)
        o = W - n
        c_s = max(r_s, o)
        for v in range(4):
            nf = new[v, p, b]
            d = _block_delta(f, nf, a_s, a_e, f, nf, a_s, a_e, table)
            if has_rc:
                d += _block_delta(
                    f,
                    nf,
                    a_s,
                    a_e,
                    rc[b, o:],
                    new_rc[v, p, b, c_s - r_s :],
                    c_s - o,
                    r_e - o,
                    table,
                )
            out[v, p, b] = diag[b] + d
    return out


# ---------------------------------------------------------------------------
# Kernel class
# ---------------------------------------------------------------------------


class DirectGkmKernel(PackedGkmKernel):
    """Direct gapped k-mer kernel (LS-GKM -t 0 / gkm_cnt).

    For window length l and k informative positions, two windows differing
    in m positions share C(l-m, k) gapped k-mer features. The kernel sums
    these shared features over all pairs of windows from the two sequences.

    d: pairs with more than d mismatches count 0, as in LS-GKM, whose
    search visits only pairs within d mismatches for every kernel type
    (default 3, which drops m = 4 at l = 11, k = 7). None counts every
    pair (the full gapped k-mer count).
    """

    def __init__(
        self,
        l: int,
        k: int,
        *,
        d: int | None = None,
        normalize: bool = True,
        include_rc: bool = True,
    ):
        super().__init__(l, k, normalize=normalize, include_rc=include_rc)
        self.d = d
        self._mismatch_table = np.array(
            [
                comb(l - m, k) if l - m >= k and (d is None or m <= d) else 0
                for m in range(l + 1)
            ],
            dtype=np.float64,
        )
        self._min_matches = self._get_min_matches()

    def _join_plan(self):
        """Mask plan for this kernel's table, or None without a mismatch cutoff."""
        from gkmsvm.kernels.masked import mask_plan

        return mask_plan(tuple(self._mismatch_table), self.l, self._min_matches)

    def _get_min_matches(self) -> int:
        """Minimum match count that yields a non-zero table entry."""
        for m in range(len(self._mismatch_table) - 1, -1, -1):
            if self._mismatch_table[m] != 0.0:
                return self.l - m
        return self.l + 1

    def _brute(self, bx, by, dev):
        args = (self._mismatch_table, self.l, self._min_matches)
        if dev == "cpu":
            return _fused_pairwise_packed_numba(
                np.ascontiguousarray(bx), np.ascontiguousarray(by), *args
            )
        if dev == "cuda":
            return _fused_pairwise_packed_gpu(bx, by, *args)
        return _fused_pairwise_packed_mlx(bx, by, *args)

    def raw_diag_mutants(self, bx, bx_rc, new, new_rc, diag):
        """See PackedGkmKernel.raw_diag_mutants; only pairs touching a changed
        window are visited (about 4 l W per mutant instead of W^2)."""
        w = self.row_weights(bx)
        if device_of(bx) != "cpu":
            return _mutant_diag_device(
                bx, bx_rc, new, new_rc, diag, w, self._mismatch_table, self.l
            )
        B = bx.shape[0]
        rc = (
            np.zeros((B, 0), np.uint32)
            if bx_rc is None
            else np.ascontiguousarray(bx_rc)
        )
        new_rc = np.zeros((4, 1, 1, 1), np.uint32) if new_rc is None else new_rc
        args = (
            np.ascontiguousarray(bx),
            rc,
            np.ascontiguousarray(new),
            np.ascontiguousarray(new_rc),
            np.asarray(diag, dtype=np.float64),
        )
        return self._mutant_diag_cpu(args, w)

    def _mutant_diag_cpu(self, args, w):
        """raw_diag_mutants on CPU: args (bx, bx_rc, new, new_rc, diag); w the
        rows' window weights (None here)."""
        return _mutant_diag_numba(*args, self._mismatch_table, self.l)

    def _brute_cross_diag(self, bx, by, dev):
        args = (self._mismatch_table, self.l, self._min_matches)
        if dev == "cpu":
            return _self_diagonal_packed_numba(
                np.ascontiguousarray(bx), np.ascontiguousarray(by), *args[:2]
            )
        if dev == "cuda":
            import cupy as cp

            return _fused_cross_diagonal_packed_gpu(
                cp.ascontiguousarray(bx), cp.ascontiguousarray(by), *args
            )
        return _fused_pairwise_packed_mlx(bx, by, *args, diag=True)
