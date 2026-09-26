from __future__ import annotations

from math import comb

import numpy as np

from gkmsvm.kernels.direct import DirectGkmKernel


class EstTruncGkmKernel(DirectGkmKernel):
    """Estimated l-mer kernel with truncated filter (LS-GKM -t 2 / gkm_esttrunc).

    The default kernel in LS-GKM. Estimates the expected number of matching
    l-mers between two sequences via a probabilistic model, with a truncation
    that zeros out negligible contributions from high-mismatch l-mer pairs.

    Also supports the non-truncated variant (-t 1 / gkm_estfull) via
    truncate=False.
    """

    def __init__(
        self,
        l: int,
        k: int,
        *,
        d: int = 3,
        normalize: bool = True,
        include_rc: bool = True,
        truncate: bool = True,
    ):
        super().__init__(l, k, normalize=normalize, include_rc=include_rc)
        self.d = d
        table = _build_estlmer_table(l, k, truncate)
        if d < l:
            table[d + 1 :] = 0.0
        self._mismatch_table = table
        self._min_matches = self._get_min_matches()


def _build_estlmer_table(l: int, k: int, truncate: bool) -> np.ndarray:
    """Build the weight table for the estimated l-mer kernel.

    Ported from calc_gkm_kernel_lmerest_wt in Dongwon-Lee/lsgkm
    src/libsvm_gkm.c.
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

    return np.array(res, dtype=np.float64)
