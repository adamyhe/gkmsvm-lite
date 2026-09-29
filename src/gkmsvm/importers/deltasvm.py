"""Import deltaSVM k-mer weight files.

Format: tab-separated lines of ``<kmer_sequence>\\t<score>``.

deltaSVM is a LINEAR approximation of the full kernel SVM — it precomputes
per-k-mer weights rather than evaluating the full kernel against all support
vectors. score(ALT) - score(REF) from deltaSVM is NOT identical to the
same difference computed via full kernel evaluation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gkmsvm.deltasvm import DeltaSVM, _kmer_to_index
from gkmsvm.importers import _open_auto


def load_deltasvm_model(
    path: str | Path,
    l: int,
    *,
    include_rc: bool | None = None,
    bias: float = 0.0,
    dtype: np.dtype | type = np.float32,
    device: str = "cpu",
) -> DeltaSVM:
    """Load deltaSVM k-mer weights from a tab-separated file.

    The file format is one k-mer per line: ``<sequence>\\t<weight>`` or
    ``<sequence>\\t<revcomp>\\t<weight>`` (RC-collapsed, as in Lee 2015).

    For 3-column RC-collapsed files, both the forward and RC k-mer are
    populated in the weight table and ``include_rc`` defaults to False
    (RC is already represented). For 2-column files, ``include_rc``
    defaults to True.

    Args:
        path: Path to the deltaSVM weight file (plain text or gzip).
        l: Window length (l-mer size). Must match the model's L parameter.
        include_rc: Whether to score both strands. Defaults to False for
            3-column (RC-collapsed) files, True for 2-column files.
        bias: Optional bias term added to scores.
        dtype: Array dtype for weights.
        device: ``"cpu"`` (default), ``"cuda"``, ``"mlx"``, or ``"auto"``.

    Returns:
        A DeltaSVM instance.
    """
    path = Path(path)
    k = None
    kmer_weights: dict[str, float] = {}
    has_rc_column = False
    header_rc_false = False

    with _open_auto(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if "include_rc=false" in line.lower():
                    header_rc_false = True
                continue

            parts = line.split("\t")
            if len(parts) == 1:
                parts = line.split()

            if len(parts) == 3:
                kmer, revcomp, weight_str = parts
                has_rc_column = True
            elif len(parts) == 2:
                kmer, weight_str = parts
                revcomp = None
            else:
                raise ValueError(
                    f"Line {line_num}: expected 2 or 3 columns, got: {line[:80]}"
                )

            kmer = kmer.upper()
            if not all(b in "ACGT" for b in kmer):
                if line_num == 1 and not any(b in "ACGT" for b in kmer):
                    continue
                raise ValueError(f"Line {line_num}: invalid bases in k-mer '{kmer}'")

            if k is None:
                k = len(kmer)
            elif len(kmer) != k:
                raise ValueError(
                    f"Line {line_num}: k-mer length {len(kmer)} != expected {k}"
                )

            kmer_weights[kmer] = float(weight_str)
            if revcomp is not None:
                revcomp = revcomp.upper()
                if all(b in "ACGT" for b in revcomp) and len(revcomp) == k:
                    kmer_weights[revcomp] = float(weight_str)

    if k is None:
        raise ValueError("Empty weight file")

    if l < k:
        raise ValueError(f"l ({l}) must be >= k ({k})")

    if include_rc is None:
        include_rc = not (has_rc_column or header_rc_false)

    weights = np.zeros(4**k, dtype=dtype)
    for kmer, w in kmer_weights.items():
        weights[_kmer_to_index(kmer)] = w

    return DeltaSVM(weights, l, k, include_rc=include_rc, bias=bias, device=device)
