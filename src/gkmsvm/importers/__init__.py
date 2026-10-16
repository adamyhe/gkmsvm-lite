"""Model importers: LS-GKM, classic C gkmSVM, R gkmSVM and deltaSVM files.

The gkm-SVM formats share LIBSVM-style ``key value`` headers and one-hot
support vectors, parsed here.
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


_INT_KEYS = ("L", "k", "d", "M", "nr_class", "total_sv", "norc", "nsv", "npos", "nneg")
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
        header[key] = KERNEL_ALIASES.get(int(value), value) if value.isdigit() else value
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
    default_kernel: str | None = None,
    dtype,
    sv_chunk_size: int | None,
    device: str,
):
    """GkmSVM from parsed SVs, taking L, k, d, gamma, M, H, norc from header."""
    from gkmsvm.codec import one_hot_encode
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
    encoded = [one_hot_encode(seq, dtype=dtype, allow_n=True) for seq in sequences]
    lengths = {t.shape[1] for t in encoded}
    if len(lengths) > 1:
        raise ValueError(
            f"All support vector sequences must have the same length, "
            f"got lengths: {sorted(lengths)}"
        )
    kernel_type = header.get("kernel_type", default_kernel)
    if kernel_type is None:
        raise ValueError("Model file missing kernel_type field")
    kernel_params = {"L": header["L"], "k": header["k"],
                     "include_rc": header.get("norc", 0) == 0}
    for key in ("d", "gamma", "M", "H"):
        if key in header:
            kernel_params[key] = header[key]
    return GkmSVM(
        support_sequences=np.stack(encoded),
        coefficients=np.array(coefficients, dtype=np.float64),
        bias=bias,
        kernel_type=kernel_type,
        kernel_params=kernel_params,
        sv_chunk_size=sv_chunk_size,
        device=device,
    )
