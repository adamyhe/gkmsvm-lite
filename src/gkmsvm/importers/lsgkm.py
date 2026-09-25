"""Import LS-GKM model files (.model.txt or .model.txt.gz).

Format: header with key-value pairs (svm_type, kernel_type, L, k, d,
norc, rho, etc.), then SV marker, then lines of
``<signed_coef> <DNA_sequence>``.

Score computation: sum(coef_i * K(x, sv_i)) - rho
Note: rho is the NEGATIVE bias (LIBSVM convention).

ENCODE public gkm-SVM models use this format.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from gkmsvm.codec import one_hot_encode
from gkmsvm.svm import GkmSVM

KERNEL_TYPE_MAP = {
    "gkm_cnt": 0,
    "gkm_estfull": 1,
    "gkm_esttrunc": 2,
    "gkmrbf": 3,
    "wgkm": 4,
    "wgkmrbf": 5,
}

KERNEL_TYPE_REVERSE = {v: k for k, v in KERNEL_TYPE_MAP.items()}


def _open_auto(path: Path):
    """Open a file, auto-detecting gzip compression."""
    try:
        f = gzip.open(path, "rt")
        f.readline()
        f.seek(0)
        return f
    except gzip.BadGzipFile:
        return open(path, "r")


def parse_lsgkm_header(path: str | Path) -> dict:
    """Parse the header of an LS-GKM model file without loading SVs."""
    path = Path(path)
    header = {"norc": 0}

    with _open_auto(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("SV"):
                break

            parts = line.split(None, 1)
            if len(parts) < 2:
                continue

            key, value = parts[0], parts[1]

            if key == "svm_type":
                header["svm_type"] = value
            elif key == "kernel_type":
                header["kernel_type"] = value
                header["kernel_type_id"] = KERNEL_TYPE_MAP.get(value, -1)
            elif key in ("L", "k", "d", "nr_class", "total_sv", "norc"):
                header[key] = int(value)
            elif key == "rho":
                header["rho"] = [float(x) for x in value.split()]
            elif key == "gamma":
                header["gamma"] = float(value)
            elif key == "M":
                header["M"] = int(value)
            elif key == "H":
                header["H"] = float(value)
            elif key == "label":
                header["label"] = [int(x) for x in value.split()]
            elif key == "nr_sv":
                header["nr_sv"] = [int(x) for x in value.split()]
            elif key == "probA":
                header["probA"] = [float(x) for x in value.split()]
            elif key == "probB":
                header["probB"] = [float(x) for x in value.split()]

    return header


def load_lsgkm_model(
    path: str | Path,
    *,
    dtype: np.dtype | type = np.float32,
    sv_chunk_size: int | None = None,
) -> GkmSVM:
    """Load an LS-GKM model file and return a GkmSVM instance.

    Handles both plain text and gzip-compressed (.gz) files.

    Args:
        path: Path to the .model.txt or .model.txt.gz file.
        dtype: Array dtype for model weights.
        sv_chunk_size: Optional chunk size for batched inference over SVs.

    Returns:
        A GkmSVM instance with imported weights.
    """
    path = Path(path)
    header = parse_lsgkm_header(path)

    kernel_type_str = header.get("kernel_type")
    if kernel_type_str is None:
        raise ValueError("Model file missing kernel_type field")

    nr_class = header.get("nr_class", 2)
    total_sv = header.get("total_sv")
    if total_sv is None:
        raise ValueError("Model file missing total_sv field")

    rho_values = header.get("rho")
    if rho_values is None:
        raise ValueError("Model file missing rho field")
    if nr_class != 2:
        raise NotImplementedError(
            f"Only binary classification (nr_class=2) is supported, got {nr_class}"
        )
    rho = rho_values[0]
    n_coefs = nr_class - 1

    coefficients = []
    sequences = []

    with _open_auto(path) as f:
        in_sv_section = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("SV"):
                in_sv_section = True
                continue
            if not in_sv_section:
                continue

            parts = line.split()
            if len(parts) < n_coefs + 1:
                raise ValueError(f"Malformed SV line: {line[:80]}")

            coefs = [float(parts[i]) for i in range(n_coefs)]
            seq = parts[n_coefs]

            coefficients.append(coefs[0] if n_coefs == 1 else coefs)
            sequences.append(seq)

    if len(sequences) != total_sv:
        raise ValueError(f"Expected {total_sv} support vectors, got {len(sequences)}")

    encoded = [one_hot_encode(seq, dtype=dtype, allow_n=True) for seq in sequences]
    lengths = {t.shape[1] for t in encoded}
    if len(lengths) > 1:
        raise ValueError(
            f"All support vector sequences must have the same length, "
            f"got lengths: {sorted(lengths)}"
        )

    support_sequences = np.stack(encoded)
    coef_array = np.array(coefficients, dtype=dtype)

    bias = -rho

    include_rc = header.get("norc", 0) == 0
    kernel_params = {
        "L": header["L"],
        "k": header["k"],
        "include_rc": include_rc,
    }
    if "d" in header:
        kernel_params["d"] = header["d"]
    if "gamma" in header:
        kernel_params["gamma"] = header["gamma"]
    if "M" in header:
        kernel_params["M"] = header["M"]
    if "H" in header:
        kernel_params["H"] = header["H"]

    return GkmSVM(
        support_sequences=support_sequences,
        coefficients=coef_array,
        bias=bias,
        kernel_type=kernel_type_str,
        kernel_params=kernel_params,
        sv_chunk_size=sv_chunk_size,
    )
