"""mutation_impact(): lsgkm gkmexplain -m 5, the GkmExplain variant score."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from gkmsvm.backend import get_array_module
from gkmsvm.explain.common import _rbf_scale
from gkmsvm.svm import (
    GkmSVM,
    _allele_arrays,
    _joint_lengths,
    _nan_rows,
    _progress,
    _rows,
    _short_rows,
)


@njit(parallel=True, cache=True)
def _impact_cpu(k_both, diag_r, diag_sv, norm, coef, gamma):
    """[n] mutation impact from raw K [ref rows; alt rows] against the SVs.

    With norm, both K(ref) and dK are divided by sqrt(K(ref, ref) K(sv, sv)).
    gamma (not NaN) applies the RBF redistribution of mutation_impact().
    """
    n = k_both.shape[0] // 2
    S = k_both.shape[1]
    rbf = not np.isnan(gamma)
    floor = np.exp(-gamma) if rbf else 0.0
    out = np.zeros(n)
    for b in prange(n):
        acc = 0.0
        for s in range(S):
            k = k_both[b, s]
            dk = k_both[n + b, s] - k
            if norm:
                z = max(np.sqrt(diag_r[b] * diag_sv[s]), 1e-10)
                k /= z
                dk /= z
            if rbf:
                if k <= 0:
                    continue
                dk *= (np.exp(gamma * (k - 1.0)) - floor) / k
            acc += coef[s] * dk
        out[b] = acc
    return out


def mutation_impact(
    model: GkmSVM,
    ref,
    alt,
    *,
    ref_lengths=None,
    alt_lengths=None,
    batch_size: int | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """GkmExplain mutation impact score (lsgkm ``gkmexplain -m 5``).

    Shrikumar et al. 2019, as used for the dsQTL evaluation. Per support
    vector j, with base (pre-RBF) kernel values normalized by the *reference*
    sequence's norm::

        K_j  = K_raw(ref, sv_j) / (|ref| |sv_j|)
        dK_j = (K_raw(alt, sv_j) - K_raw(ref, sv_j)) / (|ref| |sv_j|)

    Linear kernels: ``impact = sum_j coef_j * dK_j``. RBF kernels distribute
    the kernel value above its K=0 floor proportionally::

        impact = sum_j coef_j * (exp(g(K_j - 1)) - exp(-g)) * dK_j / K_j

    (contribution is 0 when K_j <= 0). This is a linearization, so it differs
    from ``score(alt) - score(ref)``.

    Args:
        model: Trained GkmSVM.
        ref: [B, 4, L] reference sequences, or a list of [4, L_i] arrays or
            DNA strings of different lengths.
        alt: [B, 4, L'] alternate sequences, as ref.
        ref_lengths, alt_lengths: [B] lengths of the rows of padded ref and
            alt arrays when they differ.
        batch_size: Process inputs in batches of this size, raised as in
            ``GkmSVM.__call__``.
        verbose: Show tqdm progress bar over input batches.

    Returns:
        [B, 1] mutation impact scores; NaN when an allele is shorter than l
        (with a warning).
    """
    from gkmsvm.backend import to_cpu
    from gkmsvm.kernels.packed import device_of

    kernel = model.kernel
    base = kernel.base
    coef = model.coefficients
    ref, alt, ref_lengths, alt_lengths = _allele_arrays(
        ref, alt, ref_lengths, alt_lengths
    )
    L = ref.shape[-1]
    short = None
    if ref_lengths is not None or alt_lengths is not None:
        both_lengths = _joint_lengths(
            L, (ref.shape[0], ref_lengths), (ref.shape[0], alt_lengths)
        )
        short = _short_rows(both_lengths, base.l, "mutation impacts")
        if short is not None:
            short = short.reshape(2, -1).any(axis=0)

    B = ref.shape[0]
    bs = B if batch_size is None else model._min_batch_size(batch_size)
    starts = _progress(range(0, B, bs), verbose, "Mutation impact")

    gamma = kernel.gamma if base is not kernel else np.nan  # RBF
    parts = []
    for s in starts:
        r = model._match_device(ref[s : s + bs])
        a = model._match_device(alt[s : s + bs])
        xp = get_array_module(r)
        n = r.shape[0]
        both = xp.concatenate([r, a])
        # One call for both alleles (the masked join builds its SV index once)
        bx, bx_rc = base.pack_both(
            both,
            _joint_lengths(
                L, (n, _rows(ref_lengths, s, bs)), (n, _rows(alt_lengths, s, bs))
            ),
        )
        diag_r = base.raw_diag_packed(bx[:n], None if bx_rc is None else bx_rc[:n])
        table = model._lmer_table_for(xp)
        if table is not None:
            # Non-RBF: linear in the SVs, so sum_s coef_s dK_raw / sqrt(K(sv, sv))
            # is the l-mer table's alt minus ref (the table holds the SV norms)
            raw = model._table_raw(bx, table)
            dk = raw[n:] - raw[:n]
            if base.normalize:
                dk = dk / xp.sqrt(xp.clip(diag_r.astype(dk.dtype), 1e-10, None))
            parts.append(to_cpu(dk)[:, None])
            continue
        index = model._sv_index()
        if index is not None and (
            xp is np
            or index.impact_on_device(device_of(bx))
            or not index.queries_on_device(device_of(bx))
        ):
            # RBF through the index: one fused pass per variant (on CPU, or in
            # Metal on MLX), the alt row visiting only the windows that differ
            # from ref
            inv_dr, inv_ds = model._inv_norms(
                to_cpu(diag_r) if base.normalize else None, n
            )
            if not index.impact_on_device(device_of(bx)):
                bx = to_cpu(bx)
            out = index.impact(bx, inv_dr, inv_ds, to_cpu(coef), gamma)
            parts.append(to_cpu(out).astype(np.float64)[:, None])
            continue
        k_both = model._sv_raw(bx)
        if xp is np:
            parts.append(
                _impact_cpu(
                    np.ascontiguousarray(k_both, dtype=np.float64),
                    diag_r.astype(np.float64) if base.normalize else np.ones(n),
                    model._sv_diag().astype(np.float64)
                    if base.normalize
                    else np.ones(1),
                    base.normalize,
                    coef.astype(np.float64),
                    float(gamma),
                )[:, None]
            )
            continue
        k_ref = k_both[:n]
        dk = k_both[n:] - k_ref
        if base.normalize:
            norm = xp.clip(
                xp.sqrt(diag_r[:, None] * model._sv_diag()[None, :]), 1e-10, None
            )
            k_ref = k_ref / norm
            dk = dk / norm
        if not np.isnan(gamma):
            dk = dk * _rbf_scale(k_ref, gamma, xp)
        parts.append(to_cpu((dk * coef).sum(axis=1, keepdims=True)))

    return _nan_rows(np.concatenate(parts, axis=0), short)
