"""One-hot encoding, decoding, reverse complement, and validation.

One-hot encoding adapted from tangermeme (Schreiber 2025,
https://doi.org/10.1101/2025.08.08.669296).
"""

from __future__ import annotations

import numba
import numpy as np

from gkmsvm.backend import get_array_module

BASES = "ACGT"
_BASE_TO_INDEX = {b: i for i, b in enumerate(BASES)}
_BASE_TO_INDEX.update({b.lower(): i for i, b in enumerate(BASES)})

_OHE_MAPPING = np.full(256, -2, dtype=np.int8)
for _i, _b in enumerate(BASES):
    _OHE_MAPPING[ord(_b)] = _i
    _OHE_MAPPING[ord(_b.lower())] = _i
_OHE_MAPPING[ord('N')] = -1
_OHE_MAPPING[ord('n')] = -1


@numba.njit(cache=True)
def _fast_one_hot(out, seq, mapping, allow_n):
    for i in range(len(seq)):
        idx = mapping[seq[i]]
        if idx >= 0:
            out[i, idx] = 1
        elif idx == -1:
            if not allow_n:
                raise ValueError("Invalid base in sequence")
        else:
            raise ValueError("Invalid base in sequence")


def one_hot_encode(
    sequence: str,
    *,
    dtype: np.dtype | type = np.float32,
    allow_n: bool = False,
) -> np.ndarray:
    """Encode a DNA string to a [4, L] one-hot array.

    Channel order: A=0, C=1, G=2, T=3. Case-insensitive.

    Args:
        allow_n: If True, N bases encode as all-zero (no channel set).
    """
    L = len(sequence)
    if L == 0:
        raise ValueError("Sequence must be non-empty")

    seq_bytes = np.frombuffer(bytearray(sequence, "ascii"), dtype=np.uint8)
    ohe = np.zeros((L, 4), dtype=np.int8)
    _fast_one_hot(ohe, seq_bytes, _OHE_MAPPING, allow_n)
    return ohe.T.astype(dtype, copy=False)


def one_hot_decode(arr: np.ndarray) -> str:
    """Decode a [4, L] one-hot array to a DNA string."""
    if arr.ndim != 2 or arr.shape[0] != 4:
        raise ValueError(f"Expected shape [4, L], got {list(arr.shape)}")
    xp = get_array_module(arr)
    indices = xp.argmax(arr, axis=0)
    if xp is not np:
        indices = indices.get()
    return "".join(BASES[i] for i in indices)


def reverse_complement(arr: np.ndarray) -> np.ndarray:
    """Reverse complement a one-hot DNA array.

    Works on [4, L] or [B, 4, L]. Flips channel order (A<->T, C<->G)
    and reverses the sequence dimension.
    """
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array, got {arr.ndim}D")
    if arr.shape[-2] != 4:
        raise ValueError(f"Channel dimension must be 4, got {arr.shape[-2]}")
    xp = get_array_module(arr)
    return xp.ascontiguousarray(xp.flip(xp.flip(arr, axis=-2), axis=-1))


def validate(arr: np.ndarray) -> None:
    """Validate that an array is a proper one-hot DNA encoding.

    Checks shape [..., 4, L], each position sums to 1.0, and each position
    has exactly one nonzero entry. Raises ValueError on failure.
    """
    if arr.shape[-2] != 4:
        raise ValueError(f"Channel dimension must be 4, got {arr.shape[-2]}")
    xp = get_array_module(arr)
    sums = arr.sum(axis=-2)
    if not xp.allclose(sums, xp.ones_like(sums)):
        raise ValueError("Each position must sum to 1.0")
    nonzero_per_pos = (arr != 0).sum(axis=-2)
    if not (nonzero_per_pos == 1).all():
        raise ValueError("Each position must have exactly one nonzero entry")


def encode_batch(
    sequences: list[str],
    *,
    dtype: np.dtype | type = np.float32,
    pad_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode multiple DNA sequences into a padded batch.

    Returns:
        arrays: [B, 4, L_max] one-hot array, padded with pad_value.
        mask: [B, L_max] boolean array, True at valid positions.
    """
    if not sequences:
        raise ValueError("Sequence list must be non-empty")

    encoded = [one_hot_encode(s, dtype=dtype) for s in sequences]
    lengths = [t.shape[1] for t in encoded]
    L_max = max(lengths)
    B = len(sequences)

    batch = np.full((B, 4, L_max), pad_value, dtype=dtype)
    mask = np.zeros((B, L_max), dtype=np.bool_)

    for i, (t, length) in enumerate(zip(encoded, lengths)):
        batch[i, :, :length] = t
        mask[i, :length] = True

    return batch, mask
