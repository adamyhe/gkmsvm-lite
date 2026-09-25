"""FASTA and genomic sequence I/O, backed by tangermeme.

For simple FASTA read/write, use read_fasta / write_fasta.
For BED + genome → tensor extraction (the common S2F workflow), use extract_loci.
"""

from __future__ import annotations

from pathlib import Path

import torch
from tangermeme.io import extract_loci as _tangermeme_extract_loci
from tangermeme.io import one_hot_to_fasta


def read_fasta(path: str | Path) -> list[tuple[str, str]]:
    """Read a FASTA file and return (name, sequence) pairs.

    Handles multi-line sequences. Strips whitespace from sequences.

    Args:
        path: Path to the FASTA file.

    Returns:
        List of (header, sequence) tuples. Headers exclude the leading '>'.
    """
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
    records: list[tuple[str, str]] | torch.Tensor,
    path: str | Path,
    line_width: int = 80,
    headers: list[str] | None = None,
) -> None:
    """Write sequences to a FASTA file.

    Accepts either (name, sequence) string pairs or a [B, 4, L] one-hot tensor.

    Args:
        records: List of (header, sequence) tuples, or a [B, 4, L] tensor.
        path: Output file path.
        line_width: Characters per sequence line (used for string records).
        headers: Optional headers when records is a tensor.
    """
    path = Path(path)

    if isinstance(records, torch.Tensor):
        one_hot_to_fasta(records, str(path), headers=headers)
        return

    with open(path, "w") as f:
        for header, sequence in records:
            f.write(f">{header}\n")
            f.writelines(
                sequence[i : i + line_width] + "\n"
                for i in range(0, len(sequence), line_width)
            )


def extract_loci(
    loci,
    sequences,
    *,
    in_window: int = 2114,
    chroms: list[str] | None = None,
    **kwargs,
) -> tuple:
    """Extract one-hot encoded sequences from genomic coordinates.

    Thin wrapper around tangermeme.io.extract_loci. Accepts BED file paths
    or DataFrames for loci, and FASTA path or pyfaidx.Fasta for sequences.

    Args:
        loci: BED file path, DataFrame, or list of either.
        sequences: Genome FASTA path or pyfaidx.Fasta object.
        in_window: Sequence window size to extract (centered on locus).
        chroms: Restrict to these chromosomes.
        **kwargs: Additional arguments passed to tangermeme.io.extract_loci.

    Returns:
        Tuple of (sequences_tensor, ...) as returned by tangermeme.
    """
    return _tangermeme_extract_loci(
        loci,
        sequences,
        in_window=in_window,
        chroms=chroms,
        **kwargs,
    )
