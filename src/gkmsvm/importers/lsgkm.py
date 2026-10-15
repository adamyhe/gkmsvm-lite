"""Import LS-GKM model files (.model.txt or .model.txt.gz).

Format: header with key-value pairs (svm_type, kernel_type, L, k, d,
norc, rho, etc.), then SV marker, then lines of
``<signed_coef> <DNA_sequence>``.

Score computation: sum(coef_i * K(x, sv_i)) - rho
Note: rho is the NEGATIVE bias (LIBSVM convention).

ENCODE public gkm-SVM models use this format.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gkmsvm.importers import _open_auto, build_model, parse_header_value
from gkmsvm.svm import KERNEL_ALIASES, GkmSVM

KERNEL_TYPE_MAP = {
    name: idx for idx, name in KERNEL_ALIASES.items() if isinstance(idx, int)
}

KERNEL_TYPE_REVERSE = {v: k for k, v in KERNEL_TYPE_MAP.items()}


def parse_lsgkm_header(path: str | Path) -> dict:
    """Parse the header of an LS-GKM model file without loading SVs."""
    header = {"norc": 0}
    with _open_auto(Path(path)) as f:
        for line in f:
            line = line.strip()
            if line.startswith("SV"):
                break
            parts = line.split(None, 1)
            if len(parts) == 2:
                parse_header_value(header, parts[0], parts[1])
    return header


def load_lsgkm_model(
    path: str | Path,
    *,
    dtype: np.dtype | type = np.float32,
    sv_chunk_size: int | None = None,
    device: str = "cpu",
) -> GkmSVM:
    """Load an LS-GKM model file and return a GkmSVM instance.

    Handles both plain text and gzip-compressed (.gz) files. Support
    vectors may contain N (all-zero one-hot columns).

    Args:
        path: Path to the .model.txt or .model.txt.gz file.
        dtype: Array dtype for model weights.
        sv_chunk_size: ``sv_chunk_size`` of the model (SV chunks in GkmExplain).
        device: ``"cpu"`` (default), ``"cuda"``, ``"mlx"``, or ``"auto"``
            (CuPy if available, else MLX, else CPU).

    Returns:
        A GkmSVM instance with bias = -rho.
    """
    path = Path(path)
    header = parse_lsgkm_header(path)
    if header.get("kernel_type") is None:
        raise ValueError("Model file missing kernel_type field")
    if header.get("total_sv") is None:
        raise ValueError("Model file missing total_sv field")
    if header.get("rho") is None:
        raise ValueError("Model file missing rho field")
    nr_class = header.get("nr_class", 2)
    if nr_class != 2:
        raise NotImplementedError(
            f"Only binary classification (nr_class=2) is supported, got {nr_class}"
        )

    coefficients, sequences = [], []
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
            if len(parts) < 2:
                raise ValueError(f"Malformed SV line: {line[:80]}")
            coefficients.append(float(parts[0]))
            sequences.append(parts[1])

    return build_model(
        header, coefficients, sequences, bias=-header["rho"][0],
        n_expected=header["total_sv"], dtype=dtype, sv_chunk_size=sv_chunk_size,
        device=device,
    )
