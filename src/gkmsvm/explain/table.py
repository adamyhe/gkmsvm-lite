"""The explain table (-t 0/1/2): every l-mer's attribution, precomputed with
Walsh-Hadamard transforms, so a sequence costs one lookup per window."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from gkmsvm.backend import get_array_module
from gkmsvm.explain.common import _batches, _explain_V, _inv_dx

# ---------------------------------------------------------------------------
# Explain table: every window's attribution, precomputed for non-RBF kernels
# ---------------------------------------------------------------------------
# A pair (query window u, SV l-mer v) adds V[v] times a weight that depends
# only on its mismatch count m and, at each position k, on the 2-bit delta
# d = (u ^ v)_k (the SV's base is u_k ^ d): alpha(m) when d == 0, else kappa(m)
# (or the perturbation deltas). So window u's attribution is
#
#     A[u, d, k] = sum_v V[v] G_kd(u ^ v),   G_kd(z) = [z_k == d] t_d(m(z)),
#
# an XOR convolution over the 4^l l-mers. With the Walsh-Hadamard transform H
# (H H = 4^l I), A[:, d, k] = H(H(V) H(G_kd)) / 4^l, and H(G_kd) is H(G_0d)
# with digits 0 and k swapped, so a table costs 1 + 4 + 4l transforms. A query
# window then reads its [4, l] row. Exact up to rounding; nothing per SV and
# no neighbor cutoff, so -t 0 works too. RBF models have no table (their SV
# weights depend on the query).

# 4^12 x 4 x 12 float64 = 6.4 GB
_EXPLAIN_TABLE_MAX_L = 12
# Largest table "auto" builds: [4^11, 4, 11] float64 is 1.5 GB
_EXPLAIN_TABLE_BYTES = 2 * 2**30
# "auto" builds a table once a call explains as many sequences as it takes to
# pay for it: building costs ~2e-8 s per entry of 4^l x l (0.7-1.4 s at l=11
# on an M4), against ~1.5 ms per 300 bp sequence for the CPU l-mer path and
# ~0.5 ms for CUDA's (-t 2; -t 0 and perturbation mode are slower)
_EXPLAIN_TABLE_BUILD_S = 2e-8
_EXPLAIN_SEQ_S = {"cpu": 1.5e-3, "cuda": 0.5e-3}


@njit(parallel=True, cache=True)
def _wht4(a, l):
    """In-place unnormalized Walsh-Hadamard transform of a [4^l], one radix-4
    pass per 2-bit digit."""
    n = a.shape[0]
    h = 1
    for _ in range(l):
        for j in prange(n // 4):
            i = (j // h) * 4 * h + j % h
            a0, a1, a2, a3 = a[i], a[i + h], a[i + 2 * h], a[i + 3 * h]
            s01, d01, s23, d23 = a0 + a1, a0 - a1, a2 + a3, a2 - a3
            a[i] = s01 + s23
            a[i + h] = d01 + d23
            a[i + 2 * h] = s01 - s23
            a[i + 3 * h] = d01 - d23
        h *= 4


@njit(parallel=True, cache=True)
def _wht4_rows(a, l):
    """_wht4 down the rows of a [4^l, C], C columns at once."""
    n, C = a.shape
    h = 1
    for _ in range(l):
        for j in prange(n // 4):
            i = (j // h) * 4 * h + j % h
            for c in range(C):
                a0, a1, a2, a3 = a[i, c], a[i + h, c], a[i + 2 * h, c], a[i + 3 * h, c]
                s01, d01, s23, d23 = a0 + a1, a0 - a1, a2 + a3, a2 - a3
                a[i, c] = s01 + s23
                a[i + h, c] = d01 + d23
                a[i + 2 * h, c] = s01 - s23
                a[i + 3 * h, c] = d01 - d23
        h *= 4


@njit(parallel=True, cache=True)
def _fill_products(fv, fg, l, out):
    """out[z, d, k] = fv[z] * fg[d, z with digits 0 and k swapped] / 4^l."""
    n = fv.shape[0]
    for z in prange(n):
        z0 = z & 3
        f = fv[z] / n
        for k in range(l):
            sk = np.int64(2 * k)
            zk = (z >> sk) & 3
            w = (z & ~(np.int64(3) << sk) & ~np.int64(3)) | (z0 << sk) | zk
            for d in range(4):
                out[z, d, k] = f * fg[d, w]


def _build_explain_table(V, table_a, table_b, l):
    """A [4^l, 4, l] float64 (see above) from V and the tables t_0 = table_a,
    t_1..3 = table_b at mismatch count m."""
    n = 4**l
    fv = V.astype(np.float64)  # copy
    _wht4(fv, l)
    z = np.arange(n, dtype=np.int64)
    m = np.zeros(n, dtype=np.int64)
    for k in range(l):
        m += ((z >> (2 * k)) & 3) != 0
    z0 = z & 3
    fg = np.empty((4, n))
    for d in range(4):
        fg[d] = np.where(z0 == d, (table_a if d == 0 else table_b)[m], 0.0)
        _wht4(fg[d], l)
    del z, m, z0
    # The 4l inverse transforms run together down A's rows
    A = np.empty((n, 4, l))
    _fill_products(fv, fg, l, A)
    del fv, fg
    _wht4_rows(A.reshape(n, 4 * l), l)
    return A


@njit(parallel=True, cache=True, fastmath=True)
def _explain_table_cpu(packed_x, inv_dx, A, l, seqlen, perturbation, wx):
    """[B, 4, seqlen] attributions: each window u adds A[u, d, k] to base u_k ^ d
    at position i + k (perturbation: d == 0 goes to the other three bases),
    times its window weight wx[b, i] (wx [B or 1, W], [1, 0] when unweighted).
    PAD (past a shorter sequence's windows) is no l-mer and adds nothing."""
    B, W = packed_x.shape
    n_lmers = A.shape[0]
    weighted = wx.shape[1] > 0
    out = np.zeros((B, 4, seqlen))
    for b in prange(B):
        acc = out[b]
        wxb = wx[min(np.int64(b), wx.shape[0] - 1)]
        for i in range(W):
            u = packed_x[b, i]
            if np.int64(u) >= n_lmers:
                continue
            a = wxb[i] if weighted else 1.0
            for k in range(l):
                uk = int((u >> np.uint32(2 * k)) & np.uint32(3))
                pos = i + k
                if perturbation:
                    a0 = a * A[u, 0, k]
                    for h in range(4):
                        if h != uk:
                            acc[h, pos] += a0
                else:
                    acc[uk, pos] += a * A[u, 0, k]
                for d in range(1, 4):
                    acc[uk ^ d, pos] += a * A[u, d, k]
        acc *= inv_dx[b]
    return out


def _explain_table(model, n_rows, device, table_a, table_b, perturbation, gamma):
    """The cached explain table for this mode, built if model.use_explain_table
    calls for it, or None."""
    use = model.use_explain_table
    base = model.kernel.base
    l = base.l
    if use is False or gamma is not None or l > _EXPLAIN_TABLE_MAX_L:
        return None
    tables = model._explain_tables
    if perturbation not in tables:
        build_s = _EXPLAIN_TABLE_BUILD_S * 4**l * l
        if use == "auto" and (
            4**l * 4 * l * 8 > _EXPLAIN_TABLE_BYTES
            or n_rows * _EXPLAIN_SEQ_S[device] < build_s
        ):
            return None
        tables[perturbation] = _build_explain_table(
            _explain_V(model), table_a, table_b, l
        )
    return tables[perturbation]


def _explain_with_table(model, x, A, perturbation, batch_size, verbose, lengths):
    """GkmExplain from the explain table, on CPU; results go back to x's device."""
    from gkmsvm.backend import to_cpu

    base = model.kernel.base
    xp = get_array_module(x)
    x_cpu = to_cpu(x)
    seqlen = x_cpu.shape[-1]
    out = np.concatenate(
        [
            _explain_table_cpu(
                packed_x,
                _inv_dx(base, packed_x, packed_x_rc),
                A,
                base.l,
                seqlen,
                perturbation,
                wx,
            )
            for _, packed_x, packed_x_rc, wx in _batches(
                base, x_cpu, lengths, batch_size, verbose
            )
        ]
    )
    return out if xp is np else xp.asarray(out)
