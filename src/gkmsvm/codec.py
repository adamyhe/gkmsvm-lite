from __future__ import annotations

import torch

BASES = "ACGT"
_BASE_TO_INDEX = {b: i for i, b in enumerate(BASES)}
_BASE_TO_INDEX.update({b.lower(): i for i, b in enumerate(BASES)})


def one_hot_encode(
    sequence: str, *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Encode a DNA string to a [4, L] one-hot tensor.

    Channel order: A=0, C=1, G=2, T=3. Case-insensitive.
    Raises ValueError on any non-ACGT character.
    """
    L = len(sequence)
    if L == 0:
        raise ValueError("Sequence must be non-empty")

    indices = torch.empty(L, dtype=torch.long)
    for i, base in enumerate(sequence):
        idx = _BASE_TO_INDEX.get(base)
        if idx is None:
            raise ValueError(
                f"Invalid base '{base}' at position {i}. Only A, C, G, T are accepted."
            )
        indices[i] = idx

    tensor = torch.zeros(4, L, dtype=dtype)
    tensor.scatter_(0, indices.unsqueeze(0), 1.0)
    return tensor


def one_hot_decode(tensor: torch.Tensor) -> str:
    """Decode a [4, L] one-hot tensor to a DNA string."""
    if tensor.ndim != 2 or tensor.shape[0] != 4:
        raise ValueError(f"Expected shape [4, L], got {list(tensor.shape)}")
    indices = tensor.argmax(dim=0)
    return "".join(BASES[i] for i in indices.tolist())


def reverse_complement(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse complement a one-hot DNA tensor.

    Works on [4, L] or [B, 4, L]. Flips channel order (A<->T, C<->G)
    and reverses the sequence dimension.
    """
    if tensor.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D tensor, got {tensor.ndim}D")
    if tensor.shape[-2] != 4:
        raise ValueError(f"Channel dimension must be 4, got {tensor.shape[-2]}")
    return tensor.flip(-2, -1)


def validate(tensor: torch.Tensor) -> None:
    """Validate that a tensor is a proper one-hot DNA encoding.

    Checks shape [..., 4, L], each position sums to 1.0, and each position
    has exactly one nonzero entry. Raises ValueError on failure.
    """
    if tensor.shape[-2] != 4:
        raise ValueError(f"Channel dimension must be 4, got {tensor.shape[-2]}")
    sums = tensor.sum(dim=-2)
    if not torch.allclose(sums, torch.ones_like(sums)):
        raise ValueError("Each position must sum to 1.0")
    nonzero_per_pos = (tensor != 0).sum(dim=-2)
    if not (nonzero_per_pos == 1).all():
        raise ValueError("Each position must have exactly one nonzero entry")


def encode_batch(
    sequences: list[str],
    *,
    dtype: torch.dtype = torch.float32,
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode multiple DNA sequences into a padded batch.

    Returns:
        tensors: [B, 4, L_max] one-hot tensor, padded with pad_value.
        mask: [B, L_max] boolean tensor, True at valid positions.
    """
    if not sequences:
        raise ValueError("Sequence list must be non-empty")

    encoded = [one_hot_encode(s, dtype=dtype) for s in sequences]
    lengths = [t.shape[1] for t in encoded]
    L_max = max(lengths)
    B = len(sequences)

    batch = torch.full((B, 4, L_max), pad_value, dtype=dtype)
    mask = torch.zeros(B, L_max, dtype=torch.bool)

    for i, (t, length) in enumerate(zip(encoded, lengths)):
        batch[i, :, :length] = t
        mask[i, :length] = True

    return batch, mask
