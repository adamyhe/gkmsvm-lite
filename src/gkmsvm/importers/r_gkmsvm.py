"""Import R gkmSVM models (Ghandi et al. 2016, R/kernlab package).

Two sub-formats:
  - Unified ``.gkmmodel``: single file with ``#``-prefixed headers
    (``#rho``, ``#nsv``, ``#npos``, ``#nneg``, ``#L``, ``#k``, ``#d``)
    and FASTA entries where the header line is ``>seq_id\\tcoefficient``.
  - Legacy two-file: ``_svalpha.out`` (tab-separated seq_id + coefficient)
    plus ``_svseq.fa`` (FASTA of support vector sequences).

Bias convention: score = ... + rho (kernlab convention, opposite from
LS-GKM's score = ... - rho). The kernel is ``gkm_esttrunc`` unless a
``#kernel_type`` header says otherwise; ``#L``/``#k``/``#d`` default to
11/7/3.

Use ``load_classic_model()`` for the C gkmSVM / LS-GKM-style format
(LIBSVM headers with ``key value`` pairs and ``SV`` marker).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gkmsvm.fasta import read_fasta
from gkmsvm.importers import build_model, parse_header_value
from gkmsvm.svm import GkmSVM


def load_r_gkmsvm_model(
    path: str | Path,
    *,
    svseq_path: str | Path | None = None,
    dtype: np.dtype | type = np.float32,
    sv_chunk_size: int | None = None,
    device: str = "cpu",
) -> GkmSVM:
    """Load an R gkmSVM model.

    Auto-detects format: if *path* is a ``.gkmmodel`` file, parses the
    unified format (``#``-prefixed headers + FASTA SVs). If *path* ends
    with ``_svalpha.out`` (or *svseq_path* is provided), uses the
    legacy two-file format.

    Args:
        path: Path to ``.gkmmodel`` file or ``_svalpha.out`` file.
        svseq_path: Path to ``_svseq.fa`` for the two-file format.
            If *path* ends with ``_svalpha.out`` and this is not given,
            the loader looks for a sibling ``_svseq.fa``.
        dtype: dtype of the one-hot support vectors. Coefficients are float64 (float32 on MLX) whatever it is.
        sv_chunk_size: ``sv_chunk_size`` of the model (SV chunks in GkmExplain).
        device: ``"cpu"`` (default), ``"cuda"``, ``"mlx"``, or ``"auto"``.

    Returns:
        A GkmSVM instance with bias = +rho.
    """
    path = Path(path)

    if svseq_path is not None:
        return _load_twofile(path, Path(svseq_path), dtype, sv_chunk_size, device)

    if path.name.endswith("_svalpha.out"):
        stem = path.name[: -len("_svalpha.out")]
        candidate = path.parent / f"{stem}_svseq.fa"
        if candidate.exists():
            return _load_twofile(path, candidate, dtype, sv_chunk_size, device)
        raise FileNotFoundError(f"Expected companion file {candidate} for {path}")

    return _load_gkmmodel(path, dtype, sv_chunk_size, device)


def _load_gkmmodel(
    path: Path,
    dtype,
    sv_chunk_size,
    device="cpu",
) -> GkmSVM:
    """Parse unified .gkmmodel format with #-prefixed headers."""
    header: dict = {}
    coefficients: list[float] = []
    sequences: list[str] = []

    with open(path) as f:
        current_seq_coef: float | None = None
        seq_lines: list[str] = []

        for line in f:
            line = line.rstrip("\n")

            if line.startswith("#"):
                parts = line[1:].split(None, 1)
                if len(parts) == 2:
                    parse_header_value(header, parts[0], parts[1])
                continue

            if line.startswith(">"):
                if current_seq_coef is not None and seq_lines:
                    coefficients.append(current_seq_coef)
                    sequences.append("".join(seq_lines))

                parts = line[1:].split("\t")
                if len(parts) < 2:
                    raise ValueError(
                        f"Expected >seq_id\\tcoefficient, got: {line[:80]}"
                    )
                current_seq_coef = float(parts[1])
                seq_lines = []
                continue

            if current_seq_coef is not None:
                seq_lines.append(line.strip())

        if current_seq_coef is not None and seq_lines:
            coefficients.append(current_seq_coef)
            sequences.append("".join(seq_lines))

    return _build_model(header, coefficients, sequences, dtype, sv_chunk_size, device)


def _load_twofile(
    alpha_path: Path,
    svseq_path: Path,
    dtype,
    sv_chunk_size,
    device="cpu",
) -> GkmSVM:
    """Parse legacy two-file format: _svalpha.out + _svseq.fa."""
    header: dict = {}
    coefficients: list[float] = []

    with open(alpha_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                parts = line[1:].split(None, 1)
                if len(parts) == 2:
                    parse_header_value(header, parts[0], parts[1])
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                coefficients.append(float(parts[1]))
            else:
                try:
                    coefficients.append(float(parts[0]))
                except ValueError:
                    continue

    sv_records = read_fasta(svseq_path)
    sequences = [seq for _, seq in sv_records]

    return _build_model(header, coefficients, sequences, dtype, sv_chunk_size, device)


def _build_model(
    header, coefficients, sequences, dtype, sv_chunk_size, device
) -> GkmSVM:
    if header.get("rho") is None:
        raise ValueError("Model file missing #rho header")
    header = {"L": 11, "k": 7, "d": 3, **header}
    return build_model(
        header,
        coefficients,
        sequences,
        bias=header["rho"][0],
        n_expected=header.get("nsv"),
        default_kernel="gkm_esttrunc",
        dtype=dtype,
        sv_chunk_size=sv_chunk_size,
        device=device,
    )
