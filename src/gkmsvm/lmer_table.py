"""Exact l-mer weight table for prediction with non-RBF gkm kernels.

For -t 0, -t 1 and -t 2 the decision value is linear in the query's windows
apart from the query's own normalization:

    f(x) = sum_i w(x_i) / sqrt(K_raw(x, x)) + b
    w(u) = sum_sv c_sv * K_raw(u, sv),   c_sv = alpha_sv / sqrt(K_raw(sv, sv))

(c_sv = alpha_sv without normalization), where u is an l-mer and K_raw(u, sv)
counts the SV's reverse complement when the kernel does. So prediction costs
W table lookups plus the query self-kernel, independent of the SV count.

The table is built without touching window pairs. Let V[v] = sum of c_sv over
every occurrence of l-mer v in an SV (and its RC). With the masked-join
identity h(m) = sum_t a_t C(l - m, j_t) (kernels/masked.py),

    w(u) = sum_v V[v] h(m(u, v)) = sum_t a_t sum_{|S| = j_t} M_S(u|S)

where M_S is V summed over the positions outside S, a marginal of V viewed as
a 4 x ... x 4 tensor. The cost is O(#masks * 4^l), independent of the SVs.
"""

from __future__ import annotations

import numpy as np

# 4^12 float64 = 128 MB; larger l falls back to the kernel path
MAX_TABLE_L = 12


def build_lmer_table(kernel, sv_packed, sv_rc_packed, sv_weights) -> np.ndarray | None:
    """Table w over all 4^l packed l-mers, or None if the kernel has no plan.

    sv_packed, sv_rc_packed: [S, W] uint32 packed SV windows (forward and RC;
        sv_rc_packed is None when the kernel has no RC term).
    sv_weights: [S] c_sv as above.
    """
    from gkmsvm.kernels.masked import mask_plan

    l = kernel.l
    if l > MAX_TABLE_L:
        return None
    plan = mask_plan(tuple(kernel._mismatch_table), l, kernel._min_matches)
    if plan is None:
        return None

    n = 4 ** l
    W = sv_packed.shape[1]
    weights = np.repeat(np.asarray(sv_weights, dtype=np.float64), W)
    V = np.bincount(sv_packed.ravel().astype(np.int64), weights=weights, minlength=n)
    if sv_rc_packed is not None:
        V += np.bincount(sv_rc_packed.ravel().astype(np.int64), weights=weights, minlength=n)

    # Packed index = sum_k base_k 4^k, so in a C-order (4,)*l view axis a holds
    # position l - 1 - a
    V = V.reshape((4,) * l)
    w = np.zeros_like(V)
    for _, coef, positions in plan:
        for S in positions:
            keep = {l - 1 - int(p) for p in S}
            drop = tuple(a for a in range(l) if a not in keep)
            M = V.sum(axis=drop, keepdims=True) if drop else V
            w += coef * M
    return w.ravel()

