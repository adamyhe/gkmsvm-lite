"""One-hot encoding, decoding, reverse complement, and validation.

One-hot encoding adapted from tangermeme (Schreiber 2025,
https://doi.org/10.1101/2025.08.08.669296).
"""

from __future__ import annotations

import numba
import numpy as np

from gkmsvm.backend import get_array_module, is_mlx, to_cpu

BASES = "ACGT"

_OHE_MAPPING = np.full(256, -2, dtype=np.int8)
for _i, _b in enumerate(BASES):
    _OHE_MAPPING[ord(_b)] = _i
    _OHE_MAPPING[ord(_b.lower())] = _i
_OHE_MAPPING[ord("N")] = -1
_OHE_MAPPING[ord("n")] = -1


@numba.njit(cache=True)
def _fast_one_hot(out, seq, mapping, allow_n, allow_other):
    for i in range(len(seq)):
        idx = mapping[seq[i]]
        if idx >= 0:
            out[i, idx] = 1
        elif idx == -1:
            if not allow_n:
                raise ValueError("Invalid base in sequence (pass allow_n=True for N)")
        elif not allow_other:
            raise ValueError(
                "Invalid base in sequence (pass allow_other=True to read "
                "non-ACGTN letters as N)"
            )


def one_hot_encode(
    sequence: str,
    *,
    dtype: np.dtype | type = np.float32,
    allow_n: bool = False,
    allow_other: bool = False,
) -> np.ndarray:
    """Encode a DNA string to a [4, L] one-hot array.

    Channel order: A=0, C=1, G=2, T=3. Case-insensitive.

    Args:
        allow_n: If True, N bases encode as all-zero (no channel set), which
            the kernels score as A on the forward strand, as LS-GKM does.
            Other non-ACGT letters raise unless allow_other.
        allow_other: If True, any other letter (IUPAC codes, gaps, ...) encodes
            as N does, implying allow_n: LS-GKM reads every non-ACGT letter as A.
    """
    L = len(sequence)
    if L == 0:
        raise ValueError("Sequence must be non-empty")

    seq_bytes = np.frombuffer(bytearray(sequence, "ascii"), dtype=np.uint8)
    ohe = np.zeros((L, 4), dtype=np.int8)
    _fast_one_hot(ohe, seq_bytes, _OHE_MAPPING, allow_n or allow_other, allow_other)
    return ohe.T.astype(dtype, copy=False)


# Base codes: A=0, C=1, G=2, T=3; N (and other letters) mark as 4 and 5 here
# and are stored as A, as every kernel reads them (LS-GKM reads non-ACGT as A)
_CODE_MAPPING = np.full(256, 5, dtype=np.uint8)
for _i, _b in enumerate(BASES):
    _CODE_MAPPING[ord(_b)] = _i
    _CODE_MAPPING[ord(_b.lower())] = _i
_CODE_MAPPING[ord("N")] = 4
_CODE_MAPPING[ord("n")] = 4


@numba.njit(cache=True)
def _fast_codes(out, seq, mapping, allow_n, allow_other):
    for i in range(len(seq)):
        c = mapping[seq[i]]
        if c < 4:
            out[i] = c
            continue
        if c == 4:
            if not allow_n:
                raise ValueError("Invalid base in sequence (pass allow_n=True for N)")
        elif not allow_other:
            raise ValueError(
                "Invalid base in sequence (pass allow_other=True to read "
                "non-ACGTN letters as N)"
            )
        out[i] = 0


def encode_codes(
    sequence: str, *, allow_n: bool = False, allow_other: bool = False
) -> np.ndarray:
    """[L] uint8 base codes (A=0, C=1, G=2, T=3) of a DNA string, the form
    every kernel reads. N, and with allow_other any other letter, is stored
    as A: the kernels score it as A on the forward strand, as LS-GKM does.
    allow_n, allow_other as in one_hot_encode."""
    if len(sequence) == 0:
        raise ValueError("Sequence must be non-empty")
    seq_bytes = np.frombuffer(bytearray(sequence, "ascii"), dtype=np.uint8)
    out = np.empty(len(sequence), dtype=np.uint8)
    _fast_codes(out, seq_bytes, _CODE_MAPPING, allow_n or allow_other, allow_other)
    return out


def to_codes(x):
    """[B, L] uint8 base codes of a [B, 4, L] one-hot batch, on x's device.

    N (an all-zero column) becomes A, as the kernels read it. A [B, L]
    integer array is taken as codes already.
    """
    xp = get_array_module(x)
    if x.ndim == 2:
        if str(x.dtype) == "uint8":
            return x
        if not any(t in str(x.dtype) for t in ("int", "bool")):
            raise ValueError(f"[B, L] base codes must be integers, got {x.dtype}")
        return x.astype(xp.uint8)
    if x.ndim != 3 or x.shape[1] != 4:
        raise ValueError(
            f"expected a [B, 4, L] one-hot array or [B, L] base codes, got "
            f"{list(x.shape)}"
        )
    return xp.argmax(x, axis=1).astype(xp.uint8)


def codes_to_one_hot(codes, dtype=np.float32):
    """[B, 4, L] one-hot of [B, L] base codes, on codes' device."""
    xp = get_array_module(codes)
    if is_mlx(codes):  # MLX takes its own dtypes
        dtype = getattr(xp, np.dtype(dtype).name)
    return (codes[:, None, :] == xp.arange(4, dtype=xp.uint8)[None, :, None]).astype(
        dtype
    )


def decode_codes(codes) -> str:
    """DNA string of [L] base codes."""
    return np.frombuffer(b"ACGT", dtype=np.uint8)[to_cpu(codes)].tobytes().decode()


def one_hot_decode(arr: np.ndarray) -> str:
    """Decode a [4, L] one-hot array to a DNA string."""
    if arr.ndim != 2 or arr.shape[0] != 4:
        raise ValueError(f"Expected shape [4, L], got {list(arr.shape)}")
    indices = np.argmax(to_cpu(arr), axis=0)
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
    allow_n: bool = False,
    allow_other: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode multiple DNA sequences into a padded batch.

    allow_n, allow_other: as in one_hot_encode.

    Returns:
        arrays: [B, 4, L_max] one-hot array, padded with pad_value.
        mask: [B, L_max] boolean array, True at valid positions.
    """
    if not sequences:
        raise ValueError("Sequence list must be non-empty")

    encoded = [
        one_hot_encode(s, dtype=dtype, allow_n=allow_n, allow_other=allow_other)
        for s in sequences
    ]
    lengths = [t.shape[1] for t in encoded]
    L_max = max(lengths)
    B = len(sequences)

    batch = np.full((B, 4, L_max), pad_value, dtype=dtype)
    mask = np.zeros((B, L_max), dtype=np.bool_)

    for i, (t, length) in enumerate(zip(encoded, lengths)):
        batch[i, :, :length] = t
        mask[i, :length] = True

    return batch, mask


def center_crop(seqs, length: int, names=None):
    """Center-crop sequences to length bp (the window starts at (L - length) // 2).

    seqs: DNA strings, [4, L] arrays, or a [B, 4, L] array. Returns the same
    kind (a list for list input). Sequences shorter than length can't be
    cropped (there are no flanks to extend from): ValueError naming them
    (names, else their indices); extract windows from the genome instead
    (``fasta.extract_loci``) or keep each length (``lengths=``).
    """
    if not isinstance(seqs, (list, tuple)):
        L = seqs.shape[-1]
        if L < length:
            raise ValueError(f"sequences are {L} bp, shorter than length {length}")
        start = (L - length) // 2
        return seqs[..., start : start + length]
    lens = [len(s) if isinstance(s, str) else s.shape[-1] for s in seqs]
    short = [i for i, n in enumerate(lens) if n < length]
    if short:
        label = [str(names[i]) if names is not None else str(i) for i in short[:5]]
        more = f" and {len(short) - 5} more" if len(short) > 5 else ""
        raise ValueError(
            f"{len(short)} sequences are shorter than {length} bp and can't be "
            f"center-cropped: {', '.join(label)}{more}"
        )
    out = []
    for s, n in zip(seqs, lens):
        start = (n - length) // 2
        out.append(
            s[start : start + length]
            if isinstance(s, str)
            else s[..., start : start + length]
        )
    return out
