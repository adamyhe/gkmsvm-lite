"""Import original gkm-SVM models (R gkmSVM package, Ghandi et al. 2014).

Two sub-formats:
  - Legacy two-file: ``_svalpha.out`` (tab-separated seq_id + coefficient)
    plus ``_svseq.fa`` (FASTA of support vector sequences).
  - Unified ``.gkmmodel``: single file with ``#``-prefixed headers
    (``#rho``, ``#nsv``, ``#npos``, ``#nneg``) and FASTA entries where
    the header line contains ``>seq_id\\tcoefficient``.

IMPORTANT: The ``.gkmmodel`` format uses the OPPOSITE sign convention
for bias compared to LS-GKM. LS-GKM: score = ... - rho.
gkmSVM .gkmmodel: score = ... + rho (kernlab convention).
"""

from __future__ import annotations

from pathlib import Path


def load_gkmsvm_model(path: str | Path):
    """Load an original gkm-SVM model.

    Auto-detects format based on file extension / content.

    Args:
        path: Path to .gkmmodel file, or the _svalpha.out file
              (will look for matching _svseq.fa).

    Returns:
        A GkmSVM instance with imported weights.
    """
    raise NotImplementedError
