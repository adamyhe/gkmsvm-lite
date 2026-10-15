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

from gkmsvm.fasta import read_fasta
from gkmsvm.importers import _open_auto, build_model, parse_header_value
from gkmsvm.svm import GkmSVM


def _parse_classic_file(f) -> tuple[dict, bool, list[str]]:
    """Header, whether an SV marker was found, and the remaining data lines."""
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
        else:
            parse_header_value(header, parts[0], parts[1])
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
        sv_chunk_size: ``sv_chunk_size`` of the model (SV chunks in GkmExplain).
        device: ``"cpu"`` (default), ``"cuda"``, ``"mlx"``, or ``"auto"``.

    Returns:
        A GkmSVM instance with bias = +rho.
    """
    with _open_auto(Path(model_path)) as f:
        header, has_sv_section, trailing = _parse_classic_file(f)
    nr_class = header.get("nr_class", 2)
    if nr_class != 2:
        raise NotImplementedError(
            f"Only binary classification (nr_class=2) is supported, got {nr_class}"
        )

    coefficients, sequences = [], []
    for line in trailing:
        if line.startswith("#"):
            continue
        parts = line.split()
        if has_sv_section and svseq_path is None:  # "<coef> <sequence>" lines
            if len(parts) < 2:
                raise ValueError(f"Malformed SV line: {line[:80]}")
            sequences.append(parts[1])
        try:
            coefficients.append(float(parts[0]))
        except ValueError:
            raise ValueError(f"Expected alpha coefficient, got: {line[:80]}") from None
    if svseq_path is not None:
        sequences = [seq for _, seq in read_fasta(svseq_path)]
    if header.get("rho") is None:
        raise ValueError("Model file missing rho field")

    return build_model(
        header, coefficients, sequences, bias=header["rho"][0],
        n_expected=header.get("total_sv"), dtype=dtype, sv_chunk_size=sv_chunk_size,
        device=device,
    )
