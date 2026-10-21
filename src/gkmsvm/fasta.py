"""FASTA and genomic sequence I/O.

For simple FASTA read/write, use read_fasta / write_fasta.
For BED + genome → array extraction, use extract_loci (requires pyfaidx).

extract_loci adapted from tangermeme (Schreiber 2025,
https://doi.org/10.1101/2025.08.08.669296).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gkmsvm.codec import one_hot_decode, one_hot_encode


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


def _write_onehot_fasta(arr: np.ndarray, path: Path, headers: list[str] | None) -> None:
    """Convert [B, 4, L] one-hot array to FASTA."""
    if arr.ndim != 3 or arr.shape[1] != 4:
        raise ValueError(f"Expected [B, 4, L] array, got shape {list(arr.shape)}")

    B = arr.shape[0]
    if headers is None:
        headers = [f"seq_{i}" for i in range(B)]

    with open(path, "w") as f:
        for i in range(B):
            f.write(f">{headers[i]}\n")
            f.write(one_hot_decode(arr[i]) + "\n")


def read_loci(path) -> dict:
    """BED or narrowPeak file as columns {chrom, start, end, name, summit}
    (NumPy arrays): name is column 4, else chrom:start-end; summit is start +
    column 10 for narrowPeak, when it is not -1, else the midpoint. Lines
    starting with #, track or browser are skipped."""
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            rows.append(line.rstrip("\n").split("\t"))
    out = []
    for c in rows:
        chrom, start, end = c[0], int(c[1]), int(c[2])
        name = (
            c[3] if len(c) > 3 and c[3] not in ("", ".") else f"{chrom}:{start}-{end}"
        )
        peak = int(c[9]) if len(c) > 9 else -1
        out.append(
            (chrom, start, end, name, start + peak if peak >= 0 else (start + end) // 2)
        )
    cols = list(zip(*out)) if out else [()] * 5
    return {
        "chrom": np.array(cols[0], dtype=str),
        "start": np.array(cols[1], dtype=np.int64),
        "end": np.array(cols[2], dtype=np.int64),
        "name": np.array(cols[3], dtype=str),
        "summit": np.array(cols[4], dtype=np.int64),
    }


def extract_loci(
    loci,
    sequences,
    *,
    in_window: int = 2114,
    chroms: list[str] | None = None,
    return_mask: bool = False,
    summit: bool = True,
    **kwargs,
) -> tuple:
    """Extract one-hot encoded sequences from genomic coordinates.

    Requires pyfaidx. Accepts BED/narrowPeak file paths, read_loci's columns
    or a DataFrame for loci, and FASTA path for sequences.

    Each window is ``in_window`` bp starting at ``center - in_window // 2``,
    where center is the narrowPeak summit (``read_loci``; a DataFrame's
    ``summit`` column) when ``summit`` and one is given, else the midpoint.
    Loci on chromosomes outside ``chroms`` or missing from ``sequences``, and
    loci whose window runs off a chromosome end, are dropped. Bases other
    than ACGT (N, IUPAC codes) encode as all-zero, scored as A.

    Args:
        loci: BED/narrowPeak file path, or columns chrom, start, end and
            optionally summit (a dict of arrays, as read_loci gives, or a
            DataFrame).
        sequences: Genome FASTA path or pyfaidx.Fasta object.
        in_window: Sequence window size to extract.
        chroms: Restrict to these chromosomes.
        return_mask: Also return a boolean mask over the input loci, True
            for each locus kept. ``loci[mask]`` lines up with the returned
            sequences.
        summit: Center windows on the summit when the loci give one.

    Returns:
        ``(sequences_array,)``, or ``(sequences_array, mask)`` when
        ``return_mask`` is True.
    """
    try:
        import pyfaidx
    except ImportError as e:
        raise ImportError(
            "extract_loci requires pyfaidx. Install with: pip install pyfaidx"
        ) from e

    if isinstance(sequences, (str, Path)):
        sequences = pyfaidx.Fasta(str(sequences))

    if isinstance(loci, (str, Path)):
        loci = read_loci(loci)
    centers = (
        loci["summit"]
        if summit and "summit" in loci
        else (np.asarray(loci["start"]) + np.asarray(loci["end"])) // 2
    )

    mask = np.zeros(len(loci["chrom"]), dtype=bool)
    results = []
    for i, (chrom, mid) in enumerate(zip(loci["chrom"], centers)):
        if chroms is not None and chrom not in chroms:
            continue
        if chrom not in sequences:
            continue

        start = int(mid) - in_window // 2
        end = start + in_window
        if start < 0 or end > len(sequences[chrom]):
            continue

        # Letters other than ACGT encode as all-zero (N)
        seq = str(sequences[chrom][start:end])
        results.append(one_hot_encode(seq, allow_n=True, allow_other=True))
        mask[i] = True

    if results:
        X = np.ascontiguousarray(np.stack(results))
    else:
        X = np.zeros((0, 4, in_window), dtype=np.float32)

    return (X, mask) if return_mask else (X,)
