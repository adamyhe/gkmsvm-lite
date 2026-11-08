"""gkmexplain(): picks the explain table, the l-mer path or the per-SV kernels for each call."""

from __future__ import annotations

import numpy as np

from gkmsvm.backend import get_array_module, is_mlx
from gkmsvm.codec import codes_to_one_hot
from gkmsvm.explain.common import _batches, _rbf_scale
from gkmsvm.explain.cpu import _explain_cpu
from gkmsvm.explain.cuda import _explain_gpu
from gkmsvm.explain.lmer import _explain_lmer
from gkmsvm.explain.metal import _explain_sv_metal
from gkmsvm.explain.table import _explain_table, _explain_with_table
from gkmsvm.explain.weights import (
    _build_alpha_table,
    _build_kappa_table,
    _build_perturbation_tables,
    _explain_min_matches,
)
from gkmsvm.svm import GkmSVM, _per_position, _short_rows, as_batch

_MODE_ALIASES: dict[int | str, str] = {
    0: "importance",
    1: "hypothetical",
    2: "perturbation",
    "importance": "importance",
    "hypothetical": "hypothetical",
    "perturbation": "perturbation",
}


def gkmexplain(
    model: GkmSVM,
    x,
    *,
    mode: int | str = "importance",
    lengths=None,
    batch_size: int = 50,
    verbose: bool = False,
) -> np.ndarray:
    """Compute GkmExplain attribution scores.

    Args:
        model: A GkmSVM model.
        x: [B, 4, L] one-hot encoded sequences, or a list of [4, L_i] arrays
            or DNA strings of different lengths.
        mode: Attribution mode (string or legacy int).
            ``"importance"`` (0) — hypothetical * one-hot.  Non-reference
                bases are zero; sum over [B, 4, L] equals score - bias
                (completeness axiom).  Use for per-position importance
                visualization.
            ``"hypothetical"`` (1) — hypothetical importance for all four
                bases at every position.  Use for TF-MoDISco (both
                projected and hypothetical contribution inputs).
            ``"perturbation"`` (2) — discrete kernel-value delta for
                each possible single-base mutation (lsgkm C mode 4:
                ``perturbation_eff=1`` over every pair it changes).  Use
                for per-position mutation maps; variant scores
                (``score_variants(method="gkmexplain")``) come from
                ``mutation_impact()``.
        lengths: [B] lengths of x's rows when they differ (x padded to the
            longest).
        batch_size: Process inputs in batches of this size (default 50).
        verbose: Show tqdm progress bar over input batches.

    Returns:
        [B, 4, L] float64 attribution scores on the model's device (NumPy for
        a model on MLX, whose window partials are folded on CPU); 0 past
        each sequence's length, NaN for sequences shorter than l (with a
        warning).
    """
    resolved = _MODE_ALIASES.get(mode)
    if resolved is None:
        raise ValueError(
            f"mode must be 'importance', 'hypothetical', or 'perturbation' "
            f"(or 0/1/2), got {mode!r}"
        )
    x, lengths = as_batch(x, lengths)
    x = model._match_device(x)
    short = _short_rows(lengths, model.kernel.base.l, "attributions")

    if is_mlx(x):
        # The l-mer path and the per-SV kernel run in Metal (window partials
        # folded in float64 on the CPU), the explain table on CPU, from a CPU
        # copy of the model; the result is NumPy float64
        from gkmsvm.backend import to_cpu

        # A CPU copy of the model, kept with the model's caches so its packed
        # SVs, norms and l-mer weights are built once
        cpu_model = model._sv_cache.get("cpu_model")
        if cpu_model is None:
            cpu_model = GkmSVM(
                to_cpu(model.support_sequences),
                to_cpu(model.coefficients),
                model.bias,
                model.kernel_type,
                model._kernel_params,
                sv_lengths=model.sv_lengths,
                sv_chunk_size=model.sv_chunk_size,
            )
            model._sv_cache["cpu_model"] = cpu_model
        # (checked on every call, as on the model itself)
        cpu_model.use_lmer_index = model.use_lmer_index
        cpu_model.use_lmer_table = model.use_lmer_table
        cpu_model.use_explain_table = model.use_explain_table
        out = _gkmexplain(
            cpu_model, to_cpu(x), lengths, resolved, batch_size, verbose, metal=True
        )
        return _per_position(out, lengths, short)
    out = _gkmexplain(model, x, lengths, resolved, batch_size, verbose)
    return _per_position(out, lengths, short)


def _gkmexplain(model, x, lengths, resolved, batch_size, verbose, metal=False):
    """gkmexplain on x on the model's device (not MLX), lengths None or [B].
    metal: x and the model are a CPU copy of an MLX model's, and the l-mer
    path and the per-SV kernel run in Metal."""
    if resolved == "importance":
        hyp = _gkmexplain(model, x, lengths, "hypothetical", batch_size, verbose, metal)
        # N is stored as A, so its importance goes there, as in lsgkm (and the
        # importance then sums to score - bias)
        return hyp * codes_to_one_hot(x, hyp.dtype)

    is_perturbation = resolved == "perturbation"

    kernel = model.kernel
    base = kernel.base
    # RBF (-t 3/-t 5): each SV's contributions scale by _rbf_scale of its kernel value
    gamma = kernel.gamma if base is not kernel else None
    xp = get_array_module(x)
    B, seqlen = x.shape[0], x.shape[-1]
    l = base.l
    d = l if base.d is None else base.d  # -t 0 without d counts every pair
    if is_perturbation:
        table_a, table_b = _build_perturbation_tables(base._mismatch_table, l, d)
    else:
        table_a = _build_alpha_table(base._mismatch_table, l, d)
        table_b = _build_kappa_table(base._mismatch_table, l, d)
    min_matches = _explain_min_matches(table_a, table_b, l)
    A = _explain_table(
        model,
        B,
        "cpu" if xp is np else "cuda",
        table_a,
        table_b,
        is_perturbation,
        gamma,
    )
    if A is not None:
        return _explain_with_table(
            model, x, A, is_perturbation, batch_size, verbose, lengths
        )
    lmer = _explain_lmer(
        model,
        x,
        lengths,
        table_a,
        table_b,
        l - min_matches,
        is_perturbation,
        gamma,
        batch_size,
        verbose,
        metal,
    )
    if lmer is not None:
        return lmer

    S = model.num_support_vectors
    coefs = model.coefficients
    diag_sv = model._sv_diag() if base.normalize else None
    sv_fwd, sv_rc = model._sv_packed()
    packed_sv = sv_fwd if sv_rc is None else xp.concatenate([sv_fwd, sv_rc], axis=1)
    if xp is not np:
        packed_sv = xp.ascontiguousarray(packed_sv.T)  # [Wy, S] for coalesced reads
    cs = model.sv_chunk_size if model.sv_chunk_size is not None else min(S, 2000)
    # [S or 1, Wy] SV window weights, forward then RC (an RC window carries its
    # forward weight); [1, 0] when unweighted
    wy = base.row_weights(sv_fwd)
    if wy is None:
        wy = np.zeros((1, 0))
    else:
        if sv_rc is not None:
            wy_rc = base.row_weights(sv_rc, rc=True)
            n = max(len(wy), len(wy_rc))
            wy = np.concatenate(
                [
                    np.broadcast_to(wy, (n, wy.shape[1])),
                    np.broadcast_to(wy_rc, (n, wy_rc.shape[1])),
                ],
                axis=1,
            )
        wy = np.ascontiguousarray(wy, dtype=np.float64)

    result = xp.zeros((B, 4, seqlen), dtype=np.float64)
    for b0, packed_x, packed_x_rc, wx in _batches(
        base, x, lengths, batch_size, verbose
    ):
        n = len(packed_x)
        diag_x = base.raw_diag_packed(packed_x, packed_x_rc) if base.normalize else None
        if gamma is not None:
            k_norm = model._sv_raw(packed_x) / xp.clip(
                xp.sqrt(diag_x[:, None] * diag_sv[None, :]), 1e-10, None
            )
            rbf = _rbf_scale(k_norm, gamma, xp)
        for s0 in range(0, S, cs):
            s1 = min(s0 + cs, S)
            eff_coef = xp.broadcast_to(coefs[None, s0:s1], (n, s1 - s0))
            if diag_x is not None:
                norm = xp.sqrt(diag_x[:, None] * diag_sv[None, s0:s1])
                eff_coef = eff_coef / xp.clip(norm, 1e-10, None)
            if gamma is not None:
                eff_coef = eff_coef * rbf[:, s0:s1]
            eff_coef = xp.ascontiguousarray(eff_coef, dtype=np.float64)
            wy_s = wy[s0:s1] if len(wy) > 1 else wy
            if xp is np:
                fn = _explain_sv_metal if metal else _explain_cpu
                result[b0 : b0 + n] += fn(
                    packed_x,
                    np.ascontiguousarray(packed_sv[s0:s1]),
                    table_a,
                    table_b,
                    eff_coef,
                    l,
                    min_matches,
                    seqlen,
                    is_perturbation,
                    wx,
                    wy_s,
                )
            else:
                _explain_gpu(
                    packed_x,
                    xp.ascontiguousarray(packed_sv[:, s0:s1]),
                    table_a,
                    table_b,
                    eff_coef,
                    result[b0 : b0 + n],
                    l,
                    min_matches,
                    is_perturbation,
                    wx,
                    wy_s,
                )

    return result
