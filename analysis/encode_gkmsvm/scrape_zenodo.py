#!/usr/bin/env python3
"""Scrape metadata for Kipoi/LS-GKM models on Zenodo (record 1466097).

Fetches the file listing from the Zenodo API and parses the ENCODE
wgEncode naming convention to extract source lab, assay type, cell
type, and target factor for each model.

Naming convention:
    wgEncode{Source}{Assay}{CellType}{Factor}{Conditions}_train_gS_11_3_1.model.txt

    Source:  Haib, Sydh, Uw, Broad, OpenChrom
    Assay:   Tfbs (ChIP-seq TF), Histone (ChIP-seq histone), Chip (open chromatin)
    The _gS_11_3_1 suffix encodes kernel params (l=11, d=3, kernel type 1).

Outputs:
    {outdir}/zenodo_manifest.tsv   — one row per model
    {outdir}/zenodo_raw.json       — full API response

Usage:
    python scrape_zenodo.py
    python scrape_zenodo.py --outdir data/encode_gkmsvm
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ZENODO_API = "https://zenodo.org/api/records/1466097"

# ENCODE consortium labs and their abbreviations
SOURCES = {
    "Haib": "HudsonAlpha Institute for Biotechnology",
    "Sydh": "Stanford/Yale/Davis/Harvard",
    "Uw": "University of Washington",
    "Broad": "Broad Institute",
    "OpenChrom": "OpenChromatin Consortium",
    "Uta": "UT Austin",
    "Duke": "Duke University",
}

# Assay type keywords in wgEncode filenames
ASSAY_KEYWORDS = {
    "Tfbs": "ChIP-seq (TF)",
    "Histone": "ChIP-seq (histone)",
    "Chip": "ChIP/OpenChrom",
    "Dnase": "DNase-seq",
    "Faire": "FAIRE-seq",
}

# Known cell type abbreviations
CELL_TYPES = [
    "A549", "Ag04449", "Ag04450", "Ag09309", "Ag09319", "Ag10803",
    "Aoaf", "Be2c", "Bj", "Caco2", "Cmk", "Dnd41",
    "Ecc1", "Gm06990", "Gm10847", "Gm12801", "Gm12864", "Gm12865",
    "Gm12866", "Gm12867", "Gm12868", "Gm12869", "Gm12870", "Gm12871",
    "Gm12872", "Gm12873", "Gm12874", "Gm12875", "Gm12878", "Gm12891",
    "Gm12892", "Gm15510", "Gm18505", "Gm18507", "Gm18526", "Gm18951",
    "Gm19099", "Gm19193", "Gm19238", "Gm19239", "Gm19240",
    "H1hesc", "H7hesc", "H9es", "Hae", "Hbmec", "Hcf", "Hcfaa",
    "Hcm", "Hcpe", "Hct116", "Hee", "Hela", "Helas3",
    "Hepg2", "Hipec", "Hl60", "Hmec", "Hmf", "Hmvecdblad",
    "Hpaf", "Hpdlf", "Hpf", "Hrce", "Hre", "Hrpe",
    "Hsmm", "Hsmmtube", "Htr8", "Huvec",
    "Imr90", "Ishikawa",
    "Jurkat",
    "K562", "Lncap",
    "Mcf7", "Mcf10aer",
    "Nb4", "Nhdfad", "Nhek", "Nhlf",
    "Osteoblast", "Osteobl",
    "Panc1", "Panislets", "Pfsk1", "Progfib",
    "Raji", "Rptec",
    "Saec", "Sknmc", "Sknshra", "Sknshmc", "Sknshsy5y", "Sknshsysy",
    "Sknshramc", "Sknmcmc", "Sknshra",
    "T47d", "Th1", "Th2", "Trexhek293",
    "U2os", "U87", "Urothelia",
    "Werirb1", "Wi38",
]

# Build case-insensitive lookup
_CELL_LOWER = {c.lower(): c for c in CELL_TYPES}


def fetch_zenodo_record() -> dict:
    req = Request(ZENODO_API, headers={"Accept": "application/json"})
    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def parse_filename(filename: str) -> dict:
    """Parse wgEncode filename into structured metadata.

    Example: wgEncodeHaibTfbsK562Atf3V0416101AlnRep0_train_gS_11_3_1.model.txt
    → source=Haib, assay=Tfbs (ChIP-seq TF), cell_type=K562, target=Atf3
    """
    info = {
        "filename": filename,
        "source": "",
        "source_full": "",
        "assay_keyword": "",
        "assay": "",
        "cell_type": "",
        "target": "",
        "conditions": "",
    }

    # Strip suffix
    name = filename
    for suffix in ["_train_gS_11_3_1.model.txt", ".model.txt", ".txt"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    # Strip wgEncode prefix
    if not name.startswith("wgEncode"):
        info["conditions"] = name
        return info
    name = name[len("wgEncode"):]

    # Extract source lab
    for src in sorted(SOURCES.keys(), key=len, reverse=True):
        if name.startswith(src):
            info["source"] = src
            info["source_full"] = SOURCES[src]
            name = name[len(src):]
            break

    # Extract assay keyword
    for kw in sorted(ASSAY_KEYWORDS.keys(), key=len, reverse=True):
        if name.startswith(kw):
            info["assay_keyword"] = kw
            info["assay"] = ASSAY_KEYWORDS[kw]
            name = name[len(kw):]
            break

    # Extract cell type (greedy match against known cell types)
    name_lower = name.lower()
    matched_cell = ""
    for cell_lower, cell_orig in _CELL_LOWER.items():
        if name_lower.startswith(cell_lower) and len(cell_lower) > len(matched_cell):
            matched_cell = cell_lower

    if matched_cell:
        info["cell_type"] = _CELL_LOWER[matched_cell]
        name = name[len(matched_cell):]

    # Remaining text: target factor + conditions
    # Split at first uppercase-to-lowercase boundary after initial caps
    # e.g., "Atf3V0416101AlnRep0" → target="Atf3", conditions="V0416101AlnRep0"
    # Heuristic: target is the first CamelCase word
    m = re.match(r"([A-Z][a-z0-9]*(?:[A-Z][a-z0-9]*)*?)((?:V\d|Sc\d|Pcr|Std|Ucd|Iggab|Iggmus|Ifn|Aln|sc-|ab\d|Rep|Dm).*)?$", name)
    if m:
        info["target"] = m.group(1)
        info["conditions"] = m.group(2) or ""
    else:
        # Fallback: take everything before known suffixes
        for suffix in ["AlnRep", "StdAln", "UcdAln", "PcrAln"]:
            idx = name.find(suffix)
            if idx > 0:
                info["target"] = name[:idx]
                info["conditions"] = name[idx:]
                break
        else:
            info["target"] = name
            info["conditions"] = ""

    return info


def build_manifest(record: dict) -> list[dict]:
    files = record.get("files", [])
    rows = []

    for f in files:
        filename = f.get("key", "")
        if not filename.endswith(".model.txt"):
            continue

        download_url = f.get("links", {}).get("self", "")
        if download_url and not download_url.endswith("/content"):
            download_url += "/content"

        size_bytes = f.get("size", 0)
        size_mb = f"{size_bytes / 1e6:.1f}" if size_bytes else ""
        checksum = f.get("checksum", "")

        parsed = parse_filename(filename)

        row = {
            "filename": filename,
            "source": parsed["source"],
            "source_full": parsed["source_full"],
            "assay": parsed["assay"],
            "cell_type": parsed["cell_type"],
            "target": parsed["target"],
            "conditions": parsed["conditions"],
            "size_mb": size_mb,
            "checksum": checksum,
            "download_url": download_url,
            "kernel_params": "l=11, d=3",
            "genome": "hg19",
        }
        rows.append(row)

    return rows


def write_tsv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    print(f"\nTotal models: {len(rows)}")

    assays: dict[str, int] = {}
    sources: dict[str, int] = {}
    cells: dict[str, int] = {}
    targets: dict[str, int] = {}

    for r in rows:
        a = r["assay"] or "unknown"
        assays[a] = assays.get(a, 0) + 1
        s = r["source"] or "unknown"
        sources[s] = sources.get(s, 0) + 1
        c = r["cell_type"] or "unknown"
        cells[c] = cells.get(c, 0) + 1
        t = r["target"] or "unknown"
        targets[t] = targets.get(t, 0) + 1

    print(f"\nBy assay:")
    for k, v in sorted(assays.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    print(f"\nBy source lab:")
    for k, v in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    print(f"\nTop 15 cell types:")
    for k, v in sorted(cells.items(), key=lambda x: -x[1])[:15]:
        print(f"  {k}: {v}")

    print(f"\nTop 15 targets:")
    for k, v in sorted(targets.items(), key=lambda x: -x[1])[:15]:
        print(f"  {k}: {v}")

    print(f"\nUnique cell types: {len(cells)}")
    print(f"Unique targets: {len(targets)}")


def main():
    parser = argparse.ArgumentParser(
        description="Scrape Zenodo/Kipoi LS-GKM model metadata"
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("data/encode_gkmsvm"),
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Fetching Zenodo record 1466097...")
    record = fetch_zenodo_record()

    # Save raw response
    raw_path = args.outdir / "zenodo_raw.json"
    with open(raw_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Raw JSON: {raw_path}")

    rows = build_manifest(record)

    manifest_path = args.outdir / "zenodo_manifest.tsv"
    write_tsv(rows, manifest_path)
    print(f"Manifest: {manifest_path}")

    print_summary(rows)


if __name__ == "__main__":
    main()
