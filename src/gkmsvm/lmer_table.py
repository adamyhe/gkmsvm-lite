"""Exact l-mer weight table for prediction with non-RBF gkm kernels.

For -t 0, -t 1, -t 2 and -t 4 the decision value is linear in the query's windows
apart from the query's own normalization:

    f(x) = sum_i w(x_i) / sqrt(K_raw(x, x)) + b
    w(u) = sum_sv c_sv * K_raw(u, sv),   c_sv = alpha_sv / sqrt(K_raw(sv, sv))

(c_sv = alpha_sv without normalization), where u is an l-mer and K_raw(u, sv)
counts the SV's reverse complement when the kernel does. So prediction costs
W table lookups plus the query self-kernel, independent of the SV count.

The table is built without touching window pairs. Let V[v] = sum of c_sv over
every occurrence of l-mer v in an SV (and its RC). With the masked-join
identity (kernels/masked.py) writing the pair weight as sum over masks S of
c_S [u|S == v|S],

    w(u) = sum_v V[v] g(u, v) = sum_S c_S M_S(u|S)

where M_S is V summed over the positions outside S, a marginal of V viewed as
a 4 x ... x 4 tensor. The cost is O(#masks * 4^l), independent of the SVs.
"""

from __future__ import annotations

import numpy as np

# 4^12 float64 = 128 MB; larger l falls back to the kernel path
MAX_TABLE_L = 12


def build_lmer_table(kernel, sv_packed, sv_rc_packed, sv_weights,
                     sv_nmask=None, sv_rc_nmask=None) -> np.ndarray | None:
    """Table w over all 4^l packed l-mers, or None if the kernel has no plan.

    sv_packed, sv_rc_packed: [S, W] uint32 packed SV windows (forward and RC;
        sv_rc_packed is None when the kernel has no RC term).
    sv_weights: [S] c_sv as above.
    sv_nmask, sv_rc_nmask: [S, W] N masks of those windows (0b11 at each N
        position), for kernels where N agrees with nothing (-t 4). A window
        with N is left out of V and added for the masks that avoid its Ns.
    """
    l = kernel.l
    if l > MAX_TABLE_L or not hasattr(kernel, "_join_plan"):
        return None
    plan = kernel._join_plan()
    if plan is None:
        return None

    n = 4 ** l
    W = sv_packed.shape[1]
    weights = np.repeat(np.asarray(sv_weights, dtype=np.float64), W)
    blocks = [(sv_packed, sv_nmask)]
    if sv_rc_packed is not None:
        blocks.append((sv_rc_packed, sv_rc_nmask))
    V = np.zeros(n)
    with_n = []  # (windows, N masks, weights) of windows containing N
    for packed, nmask in blocks:
        keys = packed.ravel().astype(np.int64)
        wts = weights
        if nmask is not None:
            has_n = nmask.ravel() != 0
            with_n.append((packed.ravel()[has_n], nmask.ravel()[has_n], weights[has_n]))
            wts = np.where(has_n, 0.0, weights)
        V += np.bincount(keys, weights=wts, minlength=n)

    # Packed index = sum_k base_k 4^k, so in a C-order (4,)*l view axis a holds
    # position l - 1 - a
    V = V.reshape((4,) * l)

    def marginal(S):
        keep = {l - 1 - int(p) for p in S}
        drop = tuple(a for a in range(l) if a not in keep)
        return V.sum(axis=drop, keepdims=True) if drop else V

    # With RC, V is RC-symmetric, so the marginal on the mirrored mask
    # {l-1-p} is the marginal on S read at rc(u). Masks whose mirror has the
    # same coefficient then cost one marginal per pair: w = w_self + w_pair
    # + rc(w_pair).
    coefs = {tuple(int(p) for p in S): coef for _, coef, positions in plan for S in positions}
    w = np.zeros_like(V)
    w_pair = np.zeros_like(V)
    for S, coef in coefs.items():
        mirror = tuple(sorted(l - 1 - p for p in S))
        if sv_rc_packed is None or coefs.get(mirror) != coef or mirror == S:
            w += coef * marginal(S)
        elif S < mirror:
            w_pair += coef * marginal(S)
    # rc(u): reverse the positions (axes) and complement every base (flip)
    w += w_pair + np.flip(w_pair.transpose(tuple(range(l - 1, -1, -1))))
    w = w.ravel()
    for win, nmask, wts in with_n:
        _add_windows(w, win, nmask, wts, coefs, l)
    return w


def _add_windows(w, win, nmask, wts, coefs, l):
    """w[u] += wts * coef_S for each window, mask S without its Ns, and u agreeing on S."""
    for S, coef in coefs.items():
        bits = np.uint32(sum(3 << (2 * p) for p in S))
        ok = (nmask & bits) == 0
        if not ok.any():
            continue
        # Every u agreeing with the window on S: its bases on S, anything elsewhere
        free = [p for p in range(l) if p not in S]
        offsets = np.zeros(1, dtype=np.int64)
        for p in free:
            offsets = (offsets[:, None] + np.arange(4) * 4**p).ravel()
        idx = (win[ok] & bits).astype(np.int64)[:, None] + offsets[None, :]
        np.add.at(w, idx.ravel(), np.repeat(coef * wts[ok], len(offsets)))

