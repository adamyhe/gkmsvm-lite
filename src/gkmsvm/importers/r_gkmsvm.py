"""Import R gkmSVM models (Ghandi et al. 2016, R/kernlab package).

gkmSVM's gkmsvm_train (through 0.83) writes PREFIX_svalpha.out and
PREFIX_svseq.fa, the classic C++ tool's format: no bias (it drops kernlab's
b), and kernel parameters only on gkmsvm_kernel's and gkmsvm_classify's
command lines; see classic.load_svalpha_model, which this calls.

Also reads a single ``.gkmmodel`` file with ``#``-prefixed headers
(``#rho``, ``#nsv``, ``#L``, ``#k``, ``#d``, ...) and FASTA entries whose
header line is ``>seq_id\\tcoefficient``; no released gkmSVM writes it.
There score = ... + rho, the kernel is ``gkm_esttrunc`` unless a
``#kernel_type`` header says otherwise, and ``#L``/``#k``/``#d`` default to
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
    L: int | None = None,
    k: int | None = None,
    d: int | None = None,
    filter_type: int | None = None,
    include_rc: bool | None = None,
    dtype: np.dtype | type = np.float32,
    sv_chunk_size: int | None = None,
    device: str = "cpu",
) -> GkmSVM:
    """Load an R gkmSVM model.

    PREFIX_svalpha.out (with PREFIX_svseq.fa next to it, or *svseq_path*),
    as gkmsvm_train writes, goes to classic.load_svalpha_model; anything
    else is read as a ``.gkmmodel`` file.

    Args:
        path: ``_svalpha.out`` or ``.gkmmodel`` file.
        svseq_path: ``_svseq.fa`` for the two-file format.
        L, k, d, filter_type, include_rc: the kernel, as given to
            gkmsvm_kernel (L, K, maxnmm, useTgkm, addRC); default the
            package's (10, 6, 3, 1, TRUE). Two-file format only.
        dtype: dtype of the one-hot support vectors. Coefficients are float64 (float32 on MLX) whatever it is.
        sv_chunk_size: ``sv_chunk_size`` of the model (SV chunks in GkmExplain).
        device: ``"cpu"`` (default), ``"cuda"``, ``"mlx"``, or ``"auto"``.

    Returns:
        A GkmSVM instance: bias 0 for gkmsvm_train's files (+rho when a
        ``#rho`` line gives one).
    """
    from gkmsvm.importers.classic import load_svalpha_model

    path = Path(path)
    if svseq_path is not None or path.name.endswith("_svalpha.out"):
        return load_svalpha_model(
            path,
            svseq_path,
            L=L,
            k=k,
            d=d,
            filter_type=filter_type,
            include_rc=include_rc,
            dtype=dtype,
            sv_chunk_size=sv_chunk_size,
            device=device,
        )
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
