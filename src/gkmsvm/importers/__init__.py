"""Model importers: LS-GKM, classic C++ and R gkmSVM, and deltaSVM files.

LS-GKM's ``key value`` header parsing and the one-hot support vectors every
gkm-SVM format holds are here.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np


def _open_auto(path: Path):
    """Open a file, auto-detecting gzip compression."""
    try:
        f = gzip.open(path, "rt")
        f.readline()
        f.seek(0)
        return f
    except gzip.BadGzipFile:
        return open(path, "r")


_INT_KEYS = ("L", "k", "d", "M", "nr_class", "total_sv", "norc")
_FLOAT_KEYS = ("gamma", "H")
_LIST_KEYS = {"rho": float, "label": int, "nr_sv": int, "probA": float, "probB": float}


def parse_header_value(header: dict, key: str, value: str) -> None:
    """Store one header field, typed; unknown keys are ignored.

    ``kernel_type`` keeps names as given and resolves ``-t`` integers to
    names; ``rho`` and the other per-class fields are lists.
    """
    from gkmsvm.svm import KERNEL_ALIASES

    if key == "svm_type":
        header[key] = value
    elif key == "kernel_type":
        header[key] = (
            KERNEL_ALIASES.get(int(value), value) if value.isdigit() else value
        )
    elif key in _INT_KEYS:
        header[key] = int(value)
    elif key in _FLOAT_KEYS:
        header[key] = float(value)
    elif key in _LIST_KEYS:
        header[key] = [_LIST_KEYS[key](x) for x in value.split()]


def build_model(
    header: dict,
    coefficients: list[float],
    sequences: list[str],
    *,
    bias: float,
    n_expected: int | None,
    sv_chunk_size: int | None,
    device: str,
):
    """GkmSVM from parsed SVs, taking L, k, d, gamma, M, H, norc from header."""
    from gkmsvm.codec import encode_codes
    from gkmsvm.svm import GkmSVM

    if not sequences:
        raise ValueError("No support vector sequences found")
    if n_expected is not None and len(sequences) != n_expected:
        raise ValueError(f"Expected {n_expected} support vectors, got {len(sequences)}")
    if len(coefficients) != len(sequences):
        raise ValueError(
            f"Number of coefficients ({len(coefficients)}) does not match "
            f"number of sequences ({len(sequences)})"
        )
    # The model was fit with every non-ACGT letter read as A (as N packs), so
    # SVs holding other letters load as N, with a warning
    other = sum(bool(set(seq.upper()) - set("ACGTN")) for seq in sequences)
    if other:
        import warnings

        warnings.warn(
            f"{other} support vectors contain letters other than ACGTN; "
            "reading them as N (scored as A, as LS-GKM does)",
            stacklevel=3,
        )
    encoded = [encode_codes(seq, allow_other=True) for seq in sequences]
    # SVs of different lengths (LS-GKM trained on peaks as they are) are
    # padded to the longest, with sv_lengths
    from gkmsvm.svm import as_batch

    support_sequences, sv_lengths = as_batch(encoded)
    kernel_type = header.get("kernel_type")
    if kernel_type is None:
        raise ValueError("Model file missing kernel_type field")
    kernel_params = {
        "L": header["L"],
        "k": header["k"],
        "include_rc": header.get("norc", 0) == 0,
    }
    for key in ("d", "gamma", "M", "H"):
        if key in header:
            kernel_params[key] = header[key]
    return GkmSVM(
        support_sequences=support_sequences,
        sv_lengths=sv_lengths,
        coefficients=np.array(coefficients, dtype=np.float64),
        bias=bias,
        kernel_type=kernel_type,
        kernel_params=kernel_params,
        sv_chunk_size=sv_chunk_size,
        device=device,
    )
