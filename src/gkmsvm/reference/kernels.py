"""NumPy/Numba reference implementation of gkm kernels.

Independent of the PyTorch implementation. Used for correctness
validation and as a CPU performance baseline for GPU optimization.

Requires numba for JIT-compiled kernel computation.
"""

from __future__ import annotations

from math import comb

import numba
import numpy as np


# ── Weight tables ──────────────────────────────────────────────────────


def build_gkm_cnt_table(l: int, k: int) -> np.ndarray:
    """Weight table for -t 0 (gkm_cnt): C(l-m, k) shared gapped k-mers."""
    return np.array(
        [comb(l - m, k) if l - m >= k else 0 for m in range(l + 1)],
        dtype=np.float64,
    )


def build_esttrunc_table(
    l: int, k: int, d: int = 3, *, truncate: bool = True
) -> np.ndarray:
    """Weight table for -t 2 (gkm_esttrunc).

    Reimplemented from calc_gkm_kernel_lmerest_wt in libsvm_gkm.c.
    """
    b = 4

    wLp = [[1.0] * (k + 1) for _ in range(k + 1)]
    wL = [[1.0] * (k + 1) for _ in range(k + 1)]

    for iL in range(1, l + 1):
        for iK in range(1, k + 1):
            wL[iK][0] = wLp[iK][0] + (b - 1) * wLp[iK - 1][0]
            for jM in range(1, iK + 1):
                wL[iK][jM] = (wL[iK - 1][jM - 1] * (iK - iL)) / iK
        wL, wLp = wLp, wL

    nnorm = comb(l, k) * (b**l)
    wm = [wLp[k][i] / nnorm for i in range(k + 1)]

    kernel = [0.0] * (l + 1)
    for m in range(l + 1):
        ub = min(m, k)
        val = 0.0
        for i in range(ub + 1):
            if l - m >= k - i:
                val += wm[i] * comb(l - m, k - i) * comb(m, i)
        kernel[m] = val

    if truncate:
        kernel_tr = [0.0] * (l + 1)
        active = True
        for i in range(l + 1):
            if kernel[i] < 1e-50:
                active = False
            kernel_tr[i] = kernel[i] if active else 0.0
    else:
        kernel_tr = list(kernel)

    res = [0.0] * (l + 1)
    for m in range(l + 1):
        w = 0.0
        for m1 in range(l + 1):
            kt_m1 = kernel_tr[m1]
            if kt_m1 == 0.0:
                continue
            for m2 in range(l + 1):
                kt_m2 = kernel_tr[m2]
                if kt_m2 == 0.0:
                    continue
                for t in range(min(m, l) + 1):
                    r = m1 + m2 - 2 * t - l + m
                    if r < 0:
                        continue
                    diff = m1 - t
                    if diff > l - m:
                        continue
                    if r > diff:
                        continue
                    cc = (
                        comb(m, t)
                        * comb(l - m, diff)
                        * comb(diff, r)
                        * ((b - 1) ** t)
                        * ((b - 2) ** r)
                    )
                    w += cc * kt_m1 * kt_m2
        res[l - m] = w

    table = np.array(res, dtype=np.float64)
    if d < l:
        table[d + 1 :] = 0.0
    return table


# ── Sequence utilities ─────────────────────────────────────────────────

_BASE_MAP = {"A": 0, "a": 0, "C": 1, "c": 1, "G": 2, "g": 2, "T": 3, "t": 3}


def one_hot_encode_np(sequence: str) -> np.ndarray:
    """Encode DNA string to [4, L] one-hot float64 numpy array."""
    L = len(sequence)
    x = np.zeros((4, L), dtype=np.float64)
    for i, base in enumerate(sequence):
        idx = _BASE_MAP.get(base)
        if idx is None:
            raise ValueError(f"Invalid base '{base}' at position {i}")
        x[idx, i] = 1.0
    return x


def reverse_complement_np(x: np.ndarray) -> np.ndarray:
    """Reverse complement of [..., 4, L] one-hot array."""
    return x[..., ::-1, ::-1].copy()


# ── Numba-accelerated kernel ──────────────────────────────────────────


@numba.njit(cache=True)
def _rc(x):
    """Reverse complement of [4, L] array."""
    C, L = x.shape
    out = np.empty((C, L), dtype=x.dtype)
    for c in range(C):
        for p in range(L):
            out[c, p] = x[C - 1 - c, L - 1 - p]
    return out


@numba.njit(cache=True)
def _raw_kernel(x, y, l, table):
    """Raw kernel value between two [4, L] sequences."""
    Lx, Ly = x.shape[1], y.shape[1]
    Wx, Wy = Lx - l + 1, Ly - l + 1

    total = 0.0
    for i in range(Wx):
        for j in range(Wy):
            mm = 0
            for p in range(l):
                match = 0.0
                for c in range(4):
                    match += x[c, i + p] * y[c, j + p]
                if match < 0.5:
                    mm += 1
            total += table[mm]
    return total


@numba.njit(cache=True)
def _pairwise_raw(X, Y, l, table, include_rc):
    """Raw pairwise kernel matrix. X:[B,4,Lx], Y:[S,4,Ly] -> [B,S]."""
    B, S = X.shape[0], Y.shape[0]
    K = np.empty((B, S), dtype=np.float64)
    for b in range(B):
        for s in range(S):
            val = _raw_kernel(X[b], Y[s], l, table)
            if include_rc:
                val += _raw_kernel(X[b], _rc(Y[s]), l, table)
            K[b, s] = val
    return K


@numba.njit(cache=True)
def _diagonal_raw(X, l, table, include_rc):
    """Raw self-kernel values. X:[B,4,L] -> [B]."""
    B = X.shape[0]
    diag = np.empty(B, dtype=np.float64)
    for b in range(B):
        val = _raw_kernel(X[b], X[b], l, table)
        if include_rc:
            val += _raw_kernel(X[b], _rc(X[b]), l, table)
        diag[b] = val
    return diag


# ── Public API ─────────────────────────────────────────────────────────


def pairwise(
    X: np.ndarray,
    Y: np.ndarray,
    l: int,
    table: np.ndarray,
    *,
    include_rc: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Compute pairwise kernel matrix.

    Args:
        X: [B, 4, Lx] query sequences (float64 one-hot)
        Y: [S, 4, Ly] support sequences (float64 one-hot)
        l: window length
        table: [l+1] weight table from build_*_table
        include_rc: include reverse complement contributions
        normalize: normalize by self-similarity

    Returns:
        [B, S] kernel matrix
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    table = np.ascontiguousarray(table, dtype=np.float64)

    K = _pairwise_raw(X, Y, l, table, include_rc)

    if normalize:
        diag_x = _diagonal_raw(X, l, table, include_rc)
        diag_y = _diagonal_raw(Y, l, table, include_rc)
        norm = np.sqrt(np.outer(diag_x, diag_y))
        np.maximum(norm, 1e-10, out=norm)
        K /= norm

    return K


def score(
    X: np.ndarray,
    support_sequences: np.ndarray,
    coefficients: np.ndarray,
    bias: float,
    l: int,
    table: np.ndarray,
    *,
    include_rc: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Compute SVM decision values.

    Args:
        X: [B, 4, Lx] query sequences
        support_sequences: [S, 4, Ls] support vectors
        coefficients: [S] signed dual coefficients
        bias: scalar bias (= -rho)
        l: window length
        table: [l+1] weight table

    Returns:
        [B] decision values
    """
    K = pairwise(
        X, support_sequences, l, table, include_rc=include_rc, normalize=normalize
    )
    return K @ coefficients + bias
