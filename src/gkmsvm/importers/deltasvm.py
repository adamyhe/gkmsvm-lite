"""Import deltaSVM k-mer weight files.

Format: tab-separated lines of ``<kmer_sequence>\\t<score>``.

deltaSVM is a LINEAR approximation of the full kernel SVM — it precomputes
per-k-mer weights rather than evaluating the full kernel against all support
vectors. score(ALT) - score(REF) from deltaSVM is NOT identical to the
same difference computed via full kernel evaluation.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import torch

from gkmsvm.deltasvm import DeltaSVM, _kmer_to_index


def _open_auto(path: Path):
    """Open a file, auto-detecting gzip compression."""
    try:
        f = gzip.open(path, "rt")
        f.readline()
        f.seek(0)
        return f
    except gzip.BadGzipFile:
        return open(path, "r")


def load_deltasvm_weights(
    path: str | Path,
    l: int,
    *,
    include_rc: bool = True,
    bias: float = 0.0,
    dtype: torch.dtype = torch.float32,
) -> DeltaSVM:
    """Load deltaSVM k-mer weights from a tab-separated file.

    The file format is one k-mer per line: ``<sequence>\\t<weight>``.
    The k-mer length k is inferred from the first entry.

    Args:
        path: Path to the deltaSVM weight file (plain text or gzip).
        l: Window length (l-mer size). Must match the model's L parameter.
        include_rc: Whether to score both strands.
        bias: Optional bias term added to scores.
        dtype: Tensor dtype for weights.

    Returns:
        A DeltaSVM instance.
    """
    path = Path(path)
    k = None
    kmer_weights: dict[str, float] = {}

    with _open_auto(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) != 2:
                parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Line {line_num}: expected 'kmer\\tweight', got: {line[:80]}"
                )

            kmer, weight_str = parts
            kmer = kmer.upper()

            if k is None:
                k = len(kmer)
            elif len(kmer) != k:
                raise ValueError(
                    f"Line {line_num}: k-mer length {len(kmer)} != expected {k}"
                )

            if not all(b in "ACGT" for b in kmer):
                raise ValueError(f"Line {line_num}: invalid bases in k-mer '{kmer}'")

            kmer_weights[kmer] = float(weight_str)

    if k is None:
        raise ValueError("Empty weight file")

    if l < k:
        raise ValueError(f"l ({l}) must be >= k ({k})")

    weights = torch.zeros(4**k, dtype=dtype)
    for kmer, w in kmer_weights.items():
        weights[_kmer_to_index(kmer)] = w

    return DeltaSVM(weights, l, k, include_rc=include_rc, bias=bias)
