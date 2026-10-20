"""Center-weighted gapped k-mer kernels (LS-GKM -t 4 wgkm, -t 5 wgkmrbf).

The -t 2 kernel with a weight on every window: window i of a sequence with W
windows weighs

    wt_i = min(M, floor(M 2^(-|W/2 - i| / H) + 1))    (W/2 rounded down)

an integer in 1..M, so windows near the sequence center count up to M times
more than those at the ends. A window of the reverse complement carries the
weight of the forward window it came from (RC window j: wt_{W-1-j}). Then

    K_raw(x, y) = sum_{i,j} wt_i(x) wt_j(y) h(m(x_i, y_j))   (+ the RC block)

with h the -t 2 (estimated l-mer, truncated) table. As in LS-GKM
(gkmkernel_new_object in libsvm_gkm.c), weights depend only on a window's
position and the sequence length.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from gkmsvm.backend import to_cpu
from gkmsvm.kernels.direct import _counts, _packed_mismatches, _rc_window, _spans
from gkmsvm.kernels.esttrunc import EstTruncGkmKernel
from gkmsvm.kernels.rbf import RbfKernel


def _window_weights(W: int, M: int, H: float) -> np.ndarray:
    """[W] uint8 window weights, as LS-GKM computes them."""
    center = W // 2
    dist = np.abs(center - np.arange(W))
    wt = np.floor(M * np.exp(-1 * np.log(2) * dist / H) + 1)
    return np.minimum(wt, M).astype(np.uint8)


@njit(parallel=True, cache=True, fastmath=True)
def _brute_weighted_numba(bx, wx, by, wy, table, l, min_matches):
    """[B, S]: sum over window pairs of wx[b, i] wy[s, j] table[m]. wx, wy:
    [B or 1, Wx], [S or 1, Wy] (one row: shared by every row); PAD windows
    are skipped."""
    B = bx.shape[0]
    S = by.shape[0]
    sx, sy = _spans(bx), _spans(by)
    result = np.empty((B, S), dtype=np.float64)
    for idx in prange(B * S):
        b, s = idx // S, idx % S
        wxb = wx[min(np.int64(b), wx.shape[0] - 1)]
        wys = wy[min(np.int64(s), wy.shape[0] - 1)]
        acc = 0.0
        for i in range(sx[b, 0], sx[b, 1]):
            u = bx[b, i]
            row = 0.0
            for j in range(sy[s, 0], sy[s, 1]):
                matches = l - _packed_mismatches(u, by[s, j])
                if matches >= min_matches:
                    row += wys[j] * table[l - matches]
            acc += wxb[i] * row
        result[b, s] = acc
    return result


@njit(inline="always")
def _weighted_block(a, wa, c, wc, symmetric, table):
    """sum over i, j of wa[i] wc[j] table[m(a[i], c[j])]; for a symmetric block
    (wc == wa and g(a[i], c[j]) == g(a[j], c[i])) only pairs j >= i."""
    W = a.shape[0]
    diag = 0.0
    off = 0.0
    for i in range(W):
        u = a[i]
        row = 0.0
        for j in range(i + 1 if symmetric else 0, W):
            row += wc[j] * table[_packed_mismatches(u, c[j])]
        off += wa[i] * row
        if symmetric:
            diag += wa[i] * wc[i] * table[_packed_mismatches(u, c[i])]
    return diag + 2.0 * off if symmetric else off


@njit(parallel=True, cache=True, fastmath=True)
def _self_diagonal_weighted_numba(bx, by, w, table, l):
    """K_raw(x, x) [B]; by is bx, or bx followed by its RC windows.

    In forward order the RC windows (rev[k] = rc(fwd[k])) carry the forward
    weights, and g(fwd[i], rc(fwd[k])) is symmetric in i, k, so both blocks
    visit each unordered pair once. A row whose RC block isn't the reverse
    complement of its forward one visits every RC pair. w: [B or 1, W]
    forward weights (row_weights); a padded row is its first n windows.
    """
    B, W = bx.shape
    has_rc = by.shape[1] > W
    counts = _counts(bx)
    result = np.empty(B, dtype=np.float64)
    for b in prange(B):
        n = counts[b]
        fwd = bx[b, :n]
        wb = w[min(np.int64(b), w.shape[0] - 1), :n]
        acc = _weighted_block(fwd, wb, fwd, wb, True, table)
        if has_rc:
            rev = by[b, W:][::-1][:n].copy()
            paired = True
            for k in range(n):
                if rev[k] != _rc_window(fwd[k], l):
                    paired = False
                    break
            acc += _weighted_block(fwd, wb, rev, wb, paired, table)
        result[b] = acc
    return result


@njit(inline="always")
def _weighted_block_delta(f, nf, a_s, a_e, wf, c, nc, c_s, c_e, wc, table):
    """sum over i, j of wf[i] wc[j] (g(f'_i, c'_j) - g(f_i, c_j)), where f' is f
    with windows a_s..a_e replaced by nf and c' is c with windows c_s..c_e
    replaced by nc."""
    d = 0.0
    for i in range(a_s, a_e):  # changed rows, every column
        u, u_new = f[i], nf[i - a_s]
        row = 0.0
        for j in range(c.shape[0]):
            cj = c[j]
            cj_new = nc[j - c_s] if c_s <= j < c_e else cj
            row += wc[j] * (
                table[_packed_mismatches(u_new, cj_new)]
                - table[_packed_mismatches(u, cj)]
            )
        d += wf[i] * row
    for i in range(f.shape[0]):  # unchanged rows, changed columns
        if a_s <= i < a_e:
            continue
        u = f[i]
        row = 0.0
        for j in range(c_s, c_e):
            row += wc[j] * (
                table[_packed_mismatches(u, nc[j - c_s])]
                - table[_packed_mismatches(u, c[j])]
            )
        d += wf[i] * row
    return d


@njit(parallel=True, cache=True, fastmath=True)
def _mutant_diag_weighted_numba(fwd, rc, new, new_rc, diag, w, table, l):
    """K_raw(mutant, mutant) [4, L, B]: the reference's plus the pairs touching a
    changed window (rc has 0 columns without RC). w: [B or 1, W] forward
    weights (row_weights); a padded row is its first n windows, its RC row's
    last n, whose window j weighs the forward window n-1-j's weight."""
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
        wb = w[min(np.int64(b), w.shape[0] - 1), :n]
        wb_rc = wb[::-1].copy()
        # RC windows r_s..r_e change; the row's own are W-n.. (new_rc from r_s)
        q = L - 1 - p
        r_s, r_e = max(0, q - l + 1), min(q + 1, W)
        o = W - n
        c_s = max(r_s, o)
        for v in range(4):
            nf = new[v, p, b]
            d = _weighted_block_delta(f, nf, a_s, a_e, wb, f, nf, a_s, a_e, wb, table)
            if has_rc:
                d += _weighted_block_delta(
                    f,
                    nf,
                    a_s,
                    a_e,
                    wb,
                    rc[b, o:],
                    new_rc[v, p, b, c_s - r_s :],
                    c_s - o,
                    r_e - o,
                    wb_rc,
                    table,
                )
            out[v, p, b] = diag[b] + d
    return out


def _on_device(out: np.ndarray, like):
    """CPU result out on like's device (float32 on MLX)."""
    from gkmsvm.kernels.packed import device_of

    dev = device_of(like)
    if dev == "mlx":
        import mlx.core as mx

        return mx.array(out.astype(np.float32))
    if dev == "cuda":
        import cupy as cp

        return cp.asarray(out)
    return out


class CenterWeightedGkmKernel(EstTruncGkmKernel):
    """Center-weighted gapped k-mer kernel (LS-GKM -t 4 / wgkm).

    The -t 2 kernel with integer window weights that decay with distance
    from the sequence center: M at the center, halving every H windows,
    at least 1 (see the module docstring).
    """

    def __init__(
        self,
        l: int,
        k: int,
        *,
        d: int = 3,
        M: int = 50,
        H: float = 50.0,
        normalize: bool = True,
        include_rc: bool = True,
    ):
        if not 1 <= M <= 255:
            raise ValueError(
                f"M must be in 1..255 (LS-GKM stores weights as uint8), got {M}"
            )
        if H <= 0:
            raise ValueError(f"H must be positive, got {H}")
        super().__init__(l, k, d=d, normalize=normalize, include_rc=include_rc)
        self.M = int(M)
        self.H = float(H)
        self._weights_cache: dict[int, np.ndarray] = {}

    def window_weights(self, W: int) -> np.ndarray:
        """[W] uint8 weights of a sequence's W windows (forward order)."""
        if W not in self._weights_cache:
            self._weights_cache[W] = _window_weights(W, self.M, self.H)
        return self._weights_cache[W]

    def _brute(self, bx, by, dev, wx=None, wy=None):
        """Weighted pair loop on CPU; the result is on bx's device."""
        out = _brute_weighted_numba(
            np.ascontiguousarray(to_cpu(bx)),
            wx.astype(np.float64),
            np.ascontiguousarray(to_cpu(by)),
            wy.astype(np.float64),
            self._mismatch_table,
            self.l,
            self._min_matches,
        )
        return _on_device(out, bx)

    def raw_diag_packed(self, bx, bx_rc=None):
        """K_raw(x, x), on CPU (_self_diagonal_weighted_numba); the result is on
        bx's device."""
        fwd = np.ascontiguousarray(to_cpu(bx))
        by = fwd if bx_rc is None else np.concatenate([fwd, to_cpu(bx_rc)], axis=1)
        w = self.row_weights(fwd).astype(np.float64)
        return _on_device(
            _self_diagonal_weighted_numba(
                fwd, np.ascontiguousarray(by), w, self._mismatch_table, self.l
            ),
            bx,
        )

    def raw_diag_mutants(self, bx, bx_rc, new, new_rc, diag):
        """See PackedGkmKernel.raw_diag_mutants; on CPU, only pairs touching a
        changed window; the result is on bx's device."""
        fwd = np.ascontiguousarray(to_cpu(bx))
        B = fwd.shape[0]
        rc = (
            np.zeros((B, 0), np.uint32)
            if bx_rc is None
            else np.ascontiguousarray(to_cpu(bx_rc))
        )
        new_rc = np.zeros((4, 1, 1, 1), np.uint32) if new_rc is None else new_rc
        return _on_device(
            _mutant_diag_weighted_numba(
                fwd,
                rc,
                np.ascontiguousarray(new),
                np.ascontiguousarray(new_rc),
                np.asarray(to_cpu(diag), dtype=np.float64),
                self.row_weights(fwd).astype(np.float64),
                self._mismatch_table,
                self.l,
            ),
            bx,
        )


class CenterWeightedRbfGkmKernel(RbfKernel):
    """Center-weighted RBF gapped k-mer kernel (LS-GKM -t 5 / wgkmrbf)."""

    def __init__(
        self,
        l: int,
        k: int,
        *,
        d: int = 3,
        M: int = 50,
        H: float = 50.0,
        gamma: float = 1.0,
        normalize: bool = True,
        include_rc: bool = True,
    ):
        super().__init__(
            CenterWeightedGkmKernel(
                l, k, d=d, M=M, H=H, normalize=True, include_rc=include_rc
            ),
            gamma,
        )
        self.d = d
        self.M = M
        self.H = H
