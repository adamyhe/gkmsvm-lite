"""Import deltaSVM k-mer weight files.

Format: tab-separated lines of ``<kmer_sequence>\\t<score>``.

deltaSVM is a LINEAR approximation of the full kernel SVM — it precomputes
per-k-mer weights rather than evaluating the full kernel against all support
vectors. score(ALT) - score(REF) from deltaSVM is NOT identical to the
same difference computed via full kernel evaluation.
"""

from __future__ import annotations

from pathlib import Path


def load_deltasvm_weights(path: str | Path):
    """Load deltaSVM k-mer weights from a tab-separated file.

    Args:
        path: Path to the deltaSVM weight file.

    Returns:
        A DeltaSvmModel instance.
    """
    raise NotImplementedError
