"""H1-hESC CTCF training sets for the large-N scaling benchmark (Fig. S3B).

LS-GKM's scaling test (Lee 2016, Bioinformatics 32:2196, Fig. 1 and
Supplementary Methods 2-3) trained on subsamples of H1-hESC CTCF ChIP-seq from
5k to 90k sequences: ENCODE's January 2011 uniformly processed SPP optimal
peaks against an equal number of random genomic regions matched for GC content
and repeat fraction, with chr1 and chr2 held out for testing. This rebuilds
that set on hg19:

  positives  the wgEncodeBroadHistoneH1hescCtcf SPP optimal peaks (55,782; of
             the two H1-hESC CTCF sets the one with enough peaks for 90k), as
             300 bp windows centered on each summit (Lee 2016 used the
             variable-length peaks; here both tools train on these windows).
  negatives  random 300 bp windows without N that overlap no peak, drawn to
             the positives' joint distribution of GC fraction and repeat
             fraction (soft-masked lowercase), in bins of 0.02 x 0.1.

Writes data/ctcf/{train,test}_{pos,neg}.fa: train has every chromosome except
chr1 and chr2 (chr1-22, X), test only chr1 and chr2.

Usage:
  python benchmarks/ctcf_data.py
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np

from _common import DATA

PEAKS_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/ensembl/encode/integration_data_jan2011/"
    "byDataType/peaks/jan2011/spp/optimal/spp.optimal."
    "wgEncodeBroadHistoneH1hescCtcfStdAlnRep0_VS_wgEncodeBroadHistoneH1hescControlStdAlnRep0"
    ".narrowPeak.gz"
)
OUT = DATA / "ctcf"
WIDTH = 300
TEST_CHROMS = ("chr1", "chr2")
CHROMS = tuple(f"chr{c}" for c in [*range(1, 23), "X"])
GC_BIN, REPEAT_BIN = 0.02, 0.1
# Random windows drawn per positive, before matching
CANDIDATES_PER_POSITIVE = 40


def read_peaks(path: Path) -> dict[str, np.ndarray]:
    """{chrom: [n, 3] int64 (start, end, summit)} from a narrowPeak file."""
    by_chrom: dict[str, list] = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            c = line.split("\t")
            start, end, peak = int(c[1]), int(c[2]), int(c[9])
            summit = start + peak if peak >= 0 else (start + end) // 2
            by_chrom.setdefault(c[0], []).append((start, end, summit))
    return {k: np.array(v, dtype=np.int64) for k, v in by_chrom.items()}


def window_stats(seq: np.ndarray, starts: np.ndarray):
    """(GC fraction, repeat fraction, has N) of WIDTH-bp windows of a byte array."""
    w = seq[starts[:, None] + np.arange(WIDTH)]
    upper = w & 0xDF  # ASCII upper case
    gc = ((upper == ord("G")) | (upper == ord("C"))).mean(axis=1)
    repeat = (w >= ord("a")).mean(axis=1)
    has_n = (upper == ord("N")).any(axis=1)
    return gc, repeat, has_n


def overlaps(starts: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Whether [s, s + WIDTH) overlaps any peak interval (peaks sorted by start)."""
    if not len(peaks):
        return np.zeros(len(starts), dtype=bool)
    order = np.argsort(peaks[:, 0])
    p_start, p_end = peaks[order, 0], np.maximum.accumulate(peaks[order, 1])
    # The last peak starting before the window's end; overlap if it ends after the start
    i = np.searchsorted(p_start, starts + WIDTH) - 1
    return (i >= 0) & (p_end[np.maximum(i, 0)] > starts)


def bins(gc: np.ndarray, repeat: np.ndarray) -> np.ndarray:
    n_gc = int(round(1 / GC_BIN)) + 1
    return np.minimum(gc / GC_BIN, n_gc - 1).astype(int) * 100 + np.minimum(
        repeat / REPEAT_BIN, 10
    ).astype(int)


def build(genome_path: Path, peaks_path: Path, out: Path, seed: int = 0) -> None:
    from pyfaidx import Fasta

    rng = np.random.default_rng(seed)
    peaks = read_peaks(peaks_path)
    genome = Fasta(str(genome_path), sequence_always_upper=False, as_raw=True)
    splits = {
        "train": [c for c in CHROMS if c not in TEST_CHROMS],
        "test": list(TEST_CHROMS),
    }
    out.mkdir(parents=True, exist_ok=True)
    for split, chroms in splits.items():
        pos, pos_bins, cand = [], [], []  # cand: (chrom, start, bin)
        n_pos = sum(len(peaks.get(c, ())) for c in chroms)
        total_len = sum(len(genome[c]) for c in chroms)
        for chrom in chroms:
            seq = np.frombuffer(genome[chrom][:].encode("ascii"), dtype=np.uint8)
            p = peaks.get(chrom, np.zeros((0, 3), dtype=np.int64))
            starts = p[:, 2] - WIDTH // 2
            starts = starts[(starts >= 0) & (starts + WIDTH <= len(seq))]
            gc, rep, has_n = window_stats(seq, starts)
            starts, gc, rep = starts[~has_n], gc[~has_n], rep[~has_n]
            pos += [(chrom, int(s)) for s in starts]
            pos_bins.append(bins(gc, rep))
            n_cand = int(CANDIDATES_PER_POSITIVE * n_pos * len(seq) / total_len)
            for s0 in range(0, n_cand, 200_000):
                c_starts = rng.integers(
                    0, len(seq) - WIDTH, size=min(200_000, n_cand - s0)
                )
                gc, rep, has_n = window_stats(seq, c_starts)
                keep = ~has_n & ~overlaps(c_starts, p)
                cand += zip(
                    [chrom] * int(keep.sum()),
                    c_starts[keep].tolist(),
                    bins(gc[keep], rep[keep]).tolist(),
                )
        pos_bins = np.concatenate(pos_bins)
        cand_bins = np.array([b for _, _, b in cand])
        # Each positive bin draws as many candidates without replacement
        chosen = []
        short = 0
        for b, n in zip(*np.unique(pos_bins, return_counts=True)):
            idx = np.flatnonzero(cand_bins == b)
            take = rng.choice(idx, size=min(n, len(idx)), replace=False)
            short += n - len(take)
            chosen.append(take)
        chosen = np.concatenate(chosen)
        if short:  # bins without enough candidates: fill from the rest at random
            rest = np.setdiff1d(np.arange(len(cand)), chosen)
            chosen = np.concatenate(
                [chosen, rng.choice(rest, size=short, replace=False)]
            )
        neg = [cand[i][:2] for i in chosen]
        for name, regions in (("pos", pos), ("neg", neg)):
            path = out / f"{split}_{name}.fa"
            with open(path, "w") as f:
                for chrom, s in regions:
                    f.write(
                        f">{chrom}:{s}-{s + WIDTH}\n{genome[chrom][s : s + WIDTH].upper()}\n"
                    )
        print(
            f"{split}: {len(pos):,} positives, {len(neg):,} matched negatives "
            f"({short} outside their GC/repeat bin), from {len(cand):,} candidates"
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--genome",
        default=None,
        help="Soft-masked hg19 FASTA (default: download UCSC hg19 into data/)",
    )
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    from urllib.request import urlretrieve

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    peaks_path = out / Path(PEAKS_URL).name
    if not peaks_path.exists():
        print("Downloading H1-hESC CTCF SPP optimal peaks...")
        urlretrieve(PEAKS_URL, peaks_path)
    if args.genome:
        genome = Path(args.genome)
    else:
        import dsqtl_auprc as bench

        bench._download_genome(bench.HG19_URL, "hg19")
        genome = bench.DATA_DIR / "hg19.fa"
    build(genome, peaks_path, out)


if __name__ == "__main__":
    main()
