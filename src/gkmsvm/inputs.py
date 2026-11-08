"""Input handling shared by scoring, ISM, GkmExplain and variant scoring:
batches of base codes (as_batch) with per-row lengths, ref/alt allele pairs,
and the NaN / zero masking of rows and positions past a sequence's length."""

from __future__ import annotations

import numpy as np

from gkmsvm.backend import get_array_module, to_cpu

# LS-GKM's MAX_SEQ_LENGTH (libsvm_gkm.h) is 2048: gkmtrain/gkmpredict read
# at most this many bases of a sequence, with a warning
LSGKM_MAX_LENGTH = 2047


def _uniform_or_lengths(lengths, shape) -> np.ndarray | None:
    """lengths as host int64 [B], or None when every row is shape[-1] long."""
    if lengths is None:
        return None
    lengths = np.asarray(to_cpu(lengths), dtype=np.int64).reshape(-1)
    B, L = shape[0], shape[-1]
    if lengths.shape[0] != B or (lengths < 0).any() or (lengths > L).any():
        raise ValueError(f"lengths must be {B} values in 0..{L}")
    return None if (lengths == L).all() else lengths


def as_batch(x, lengths=None):
    """(x [B, L_max] base codes, lengths or None) from a [B, 4, L] one-hot
    array or [B, L] codes (with lengths of its rows when they differ), or a
    list of [4, L_i] one-hot arrays, [L_i] codes or DNA strings (N allowed).
    Warns about sequences longer than LS-GKM reads."""
    import warnings

    from gkmsvm.codec import encode_codes, to_codes

    if isinstance(x, (list, tuple)):
        if lengths is not None:
            raise ValueError("lengths go with a padded array, not a list")
        if not len(x):
            raise ValueError("no sequences")
        if isinstance(x[0], str):
            x = [encode_codes(s, allow_n=True) for s in x]
        else:
            x = [to_codes(a[None])[0] for a in x]  # [4, L_i] one-hot or [L_i] codes
        lens = np.array([a.shape[-1] for a in x], dtype=np.int64)
        xp = get_array_module(x[0])
        out = xp.zeros((len(x), int(lens.max())), dtype=xp.uint8)
        for i, a in enumerate(x):
            out[i, : a.shape[-1]] = a
        x, lengths = out, lens
    else:
        x = to_codes(x)
    lengths = _uniform_or_lengths(lengths, x.shape)
    longest = int(x.shape[-1] if lengths is None else lengths.max())
    if longest > LSGKM_MAX_LENGTH:
        warnings.warn(
            f"sequences up to {longest} bp: LS-GKM reads only the first "
            f"{LSGKM_MAX_LENGTH} bases of longer ones",
            stacklevel=3,
        )
    return x, lengths


def _allele_arrays(ref, alt, ref_lengths=None, alt_lengths=None):
    """(ref, alt, ref_lengths, alt_lengths): both as batches (as_batch) of
    one length L, padded on the host when they differ; lengths None when
    every row is L long."""
    ref, ref_lengths = as_batch(ref, ref_lengths)
    alt, alt_lengths = as_batch(alt, alt_lengths)
    if ref.shape[0] != alt.shape[0]:
        raise ValueError(
            f"ref and alt hold {ref.shape[0]} and {alt.shape[0]} sequences"
        )
    L = max(ref.shape[-1], alt.shape[-1])

    def widen(x, n):
        if x.shape[-1] == L:
            return x, n
        n = np.full(x.shape[0], x.shape[-1], dtype=np.int64) if n is None else n
        x = to_cpu(x)
        return np.pad(x, ((0, 0), (0, L - x.shape[-1]))), n

    ref, ref_lengths = widen(ref, ref_lengths)
    alt, alt_lengths = widen(alt, alt_lengths)
    return ref, alt, ref_lengths, alt_lengths


def _joint_lengths(L, *parts):
    """Lengths of rows from several batches of one length L (each part a
    (rows, lengths or None) pair), or None when every row is L long."""
    if all(n is None for _, n in parts):
        return None
    return np.concatenate(
        [np.full(rows, L, dtype=np.int64) if n is None else n for rows, n in parts]
    )


def _rows(lengths, start, n):
    """lengths[start:start + n], or None."""
    return None if lengths is None else lengths[start : start + n]


def _progress(chunks, verbose, desc):
    """chunks, under a tqdm progress bar when verbose."""
    if not verbose:
        return chunks
    from tqdm import tqdm

    return tqdm(chunks, desc=desc)


def _short_rows(lengths, l, what="scores"):
    """[B] bool, rows shorter than l (no windows), or None when there are none."""
    if lengths is None:
        return None
    short = lengths < l
    if not short.any():
        return None
    import warnings

    warnings.warn(
        f"{int(short.sum())} sequences are shorter than l = {l}: their {what} are NaN",
        stacklevel=3,
    )
    return short


def _nan_rows(out, short):
    """out [B, ...] with the rows in short set to NaN (on out's device)."""
    if short is None:
        return out
    xp = get_array_module(out)
    mask = xp.asarray(short.reshape(-1, *[1] * (out.ndim - 1)))
    return xp.where(mask, float("nan"), out)


def _per_position(out, lengths, short):
    """Per-position results out [B, 4, L] with positions past each row's length
    0 and rows in short NaN (on out's device)."""
    if lengths is None:
        return out
    xp = get_array_module(out)
    inside = np.arange(out.shape[2])[None, None, :] < lengths[:, None, None]
    out = xp.where(xp.asarray(inside), out, 0.0).astype(out.dtype)
    return _nan_rows(out, short)
