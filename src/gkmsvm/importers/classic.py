"""Import classic gkmSVM model files (Ghandi et al. 2014).

The classic format uses two files:
- Model file: header + alpha coefficients (one per line after header)
- Sequence file: FASTA of support vector sequences (same order as alphas)

Also supports single-file format where SVs are embedded (alpha + sequence
per line after the SV marker), identical to LS-GKM layout.

Key difference from LS-GKM: bias = +rho (opposite sign convention).
LS-GKM follows LIBSVM: score = sum(alpha_i * K) - rho, so bias = -rho.
Classic gkmSVM: score = sum(alpha_i * K) + rho, so bias = +rho.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gkmsvm.codec import one_hot_encode
from gkmsvm.fasta import read_fasta
from gkmsvm.importers import _open_auto
from gkmsvm.importers.lsgkm import KERNEL_TYPE_MAP
from gkmsvm.svm import KERNEL_ALIASES, GkmSVM


def _parse_classic_file(f) -> tuple[dict, bool, list[str]]:
    """Parse a classic model file, returning header, SV flag, and trailing lines."""
    header = {"norc": 0}
    found_sv = False
    trailing: list[str] = []

    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith("SV"):
            found_sv = True
            break

        parts = line.split(None, 1)
        if len(parts) < 2:
            trailing.append(line)
            continue

        key, value = parts[0], parts[1]

        if key == "svm_type":
            header["svm_type"] = value
        elif key == "kernel_type":
            if value.isdigit():
                kt_id = int(value)
                header["kernel_type_id"] = kt_id
                header["kernel_type"] = KERNEL_ALIASES.get(kt_id, value)
            else:
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

    if found_sv:
        trailing = [line.strip() for line in f if line.strip()]

    return header, found_sv, trailing


def load_classic_model(
    model_path: str | Path,
    *,
    svseq_path: str | Path | None = None,
    dtype: np.dtype | type = np.float32,
    sv_chunk_size: int | None = None,
    device: str = "cpu",
) -> GkmSVM:
    """Load a classic gkmSVM model.

    Args:
        model_path: Path to the model file.
        svseq_path: Path to FASTA file with support vector sequences.
        dtype: Array dtype for model weights.
        sv_chunk_size: Optional chunk size for batched SV inference.
        device: ``"cpu"`` (default), ``"cuda"``, ``"mlx"``, or ``"auto"``.

    Returns:
        A GkmSVM instance with bias = +rho.
    """
    model_path = Path(model_path)

    coefficients = []
    sequences = []

    with _open_auto(model_path) as f:
        header, has_sv_section, trailing = _parse_classic_file(f)

    nr_class = header.get("nr_class", 2)
    if nr_class != 2:
        raise NotImplementedError(
            f"Only binary classification (nr_class=2) is supported, got {nr_class}"
        )
    n_coefs = nr_class - 1

    if has_sv_section and svseq_path is None:
        for line in trailing:
            parts = line.split()
            if len(parts) < n_coefs + 1:
                raise ValueError(f"Malformed SV line: {line[:80]}")
            coefs = [float(parts[i]) for i in range(n_coefs)]
            seq = parts[n_coefs]
            coefficients.append(coefs[0] if n_coefs == 1 else coefs)
            sequences.append(seq)
    elif has_sv_section and svseq_path is not None:
        for line in trailing:
            parts = line.split()
            coefficients.append(float(parts[0]))
    else:
        for line in trailing:
            if line.startswith("#"):
                continue
            try:
                coefficients.append(float(line))
            except ValueError:
                raise ValueError(
                    f"Expected alpha coefficient, got: {line[:80]}"
                )

    if svseq_path is not None:
        sv_records = read_fasta(svseq_path)
        sequences = [seq for _, seq in sv_records]

    if not sequences:
        raise ValueError("No support vector sequences found")

    total_sv = header.get("total_sv")
    if total_sv is not None and len(sequences) != total_sv:
        raise ValueError(
            f"Expected {total_sv} support vectors, got {len(sequences)}"
        )
    if len(coefficients) != len(sequences):
        raise ValueError(
            f"Number of coefficients ({len(coefficients)}) does not match "
            f"number of sequences ({len(sequences)})"
        )

    encoded = [one_hot_encode(seq, dtype=dtype) for seq in sequences]
    lengths = {t.shape[1] for t in encoded}
    if len(lengths) > 1:
        raise ValueError(
            f"All support vector sequences must have the same length, "
            f"got lengths: {sorted(lengths)}"
        )

    support_sequences = np.stack(encoded)
    coef_array = np.array(coefficients, dtype=dtype)

    rho_values = header.get("rho")
    if rho_values is None:
        raise ValueError("Model file missing rho field")
    rho = rho_values[0]
    bias = rho

    kernel_type = header.get("kernel_type")
    if kernel_type is None:
        raise ValueError("Model file missing kernel_type field")

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
        kernel_type=kernel_type,
        kernel_params=kernel_params,
        sv_chunk_size=sv_chunk_size,
        device=device,
    )
