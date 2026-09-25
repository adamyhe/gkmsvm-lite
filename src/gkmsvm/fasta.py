"""FASTA and genomic sequence I/O.

For simple FASTA read/write, use read_fasta / write_fasta.
For BED + genome → array extraction, use extract_loci (requires pyfaidx).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


_BASES = "ACGT"
_BASE_TO_IDX = {b: i for i, b in enumerate(_BASES)}
_BASE_TO_IDX.update({b.lower(): i for i, b in enumerate(_BASES)})
_IDX_TO_BASE = dict(enumerate(_BASES))


def read_fasta(path: str | Path) -> list[tuple[str, str]]:
    """Read a FASTA file and return (name, sequence) pairs."""
    path = Path(path)
    records: list[tuple[str, str]] = []
    current_header: str | None = None
    current_seq: list[str] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    records.append((current_header, "".join(current_seq)))
                current_header = line[1:].strip()
                current_seq = []
            else:
                current_seq.append(line)

    if current_header is not None:
        records.append((current_header, "".join(current_seq)))

    return records


def write_fasta(
    records: list[tuple[str, str]] | np.ndarray,
    path: str | Path,
    line_width: int = 80,
    headers: list[str] | None = None,
) -> None:
    """Write sequences to a FASTA file.

    Accepts either (name, sequence) string pairs or a [B, 4, L] one-hot array.
    """
    path = Path(path)

    if isinstance(records, np.ndarray):
        _write_onehot_fasta(records, path, headers)
        return

    with open(path, "w") as f:
        for header, sequence in records:
            f.write(f">{header}\n")
            f.writelines(
                sequence[i : i + line_width] + "\n"
                for i in range(0, len(sequence), line_width)
            )


def _write_onehot_fasta(
    arr: np.ndarray, path: Path, headers: list[str] | None
) -> None:
    """Convert [B, 4, L] one-hot array to FASTA."""
    if arr.ndim != 3 or arr.shape[1] != 4:
        raise ValueError(f"Expected [B, 4, L] array, got shape {list(arr.shape)}")

    B = arr.shape[0]
    if headers is None:
        headers = [f"seq_{i}" for i in range(B)]

    indices = arr.argmax(axis=1)  # [B, L]

    with open(path, "w") as f:
        for i in range(B):
            f.write(f">{headers[i]}\n")
            seq = "".join(_IDX_TO_BASE[int(idx)] for idx in indices[i])
            f.write(seq + "\n")


def extract_loci(
    loci,
    sequences,
    *,
    in_window: int = 2114,
    chroms: list[str] | None = None,
    **kwargs,
) -> tuple:
    """Extract one-hot encoded sequences from genomic coordinates.

    Requires pyfaidx. Accepts BED file paths or DataFrames for loci,
    and FASTA path for sequences.

    Args:
        loci: BED file path or DataFrame.
        sequences: Genome FASTA path or pyfaidx.Fasta object.
        in_window: Sequence window size to extract (centered on locus).
        chroms: Restrict to these chromosomes.

    Returns:
        Tuple of (sequences_array, ...).
    """
    try:
        import pandas as pd
        import pyfaidx
    except ImportError as e:
        raise ImportError(
            "extract_loci requires pyfaidx and pandas. "
            "Install with: pip install pyfaidx pandas"
        ) from e

    if isinstance(sequences, (str, Path)):
        sequences = pyfaidx.Fasta(str(sequences))

    if isinstance(loci, (str, Path)):
        loci = pd.read_csv(
            loci, sep="\t", header=None, names=["chrom", "start", "end"],
            usecols=[0, 1, 2],
        )

    if chroms is not None:
        loci = loci[loci["chrom"].isin(chroms)]

    results = []
    for _, row in loci.iterrows():
        chrom = row["chrom"]
        mid = (row["start"] + row["end"]) // 2
        start = mid - in_window // 2
        end = start + in_window

        if chrom not in sequences:
            continue

        seq = str(sequences[chrom][start:end]).upper()
        if len(seq) != in_window:
            continue

        arr = np.zeros((4, in_window), dtype=np.float32)
        for i, base in enumerate(seq):
            idx = _BASE_TO_IDX.get(base)
            if idx is not None:
                arr[idx, i] = 1.0

        results.append(arr)

    if not results:
        return (np.zeros((0, 4, in_window), dtype=np.float32),)

    return (np.stack(results),)
