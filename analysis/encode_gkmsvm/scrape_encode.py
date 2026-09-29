#!/usr/bin/env python3
"""Scrape metadata for all gkm-SVM models on the ENCODE portal.

Queries the ENCODE REST API for all gkm-SVM-model annotations and
extracts structured metadata including assay, biosample, target TF,
organism, and file accessions.

Two-phase approach to avoid ENCODE API response-size limits:
  Phase 1: Fetch annotation-level metadata (lightweight, limit=all)
  Phase 2: Fetch per-annotation file details (heavier, batched)

Outputs:
    {outdir}/encode_manifest.tsv   — one row per annotation
    {outdir}/encode_raw.json       — full API response for debugging

Usage:
    python scrape_encode.py
    python scrape_encode.py --outdir data/encode_gkmsvm
    python scrape_encode.py --organism human
    python scrape_encode.py --skip-files  # annotation metadata only (fast)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ENCODE_BASE = "https://www.encodeproject.org"

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; gkmsvm-lite/0.1)",
}


def fetch_json(url: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            req = Request(url, headers=HEADERS)
            with urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except (HTTPError, OSError) as e:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Retry {attempt + 1} in {wait}s: {e}", file=sys.stderr)
                time.sleep(wait)
            else:
                raise


def fetch_all_annotations() -> list[dict]:
    """Fetch all gkm-SVM annotation metadata (without embedded files)."""
    fields = [
        "accession",
        "biosample_ontology",
        "assay_term_name",
        "targets",
        "organism",
        "derived_from",
        "description",
        "date_released",
    ]
    field_params = "".join(f"&field={f}" for f in fields)
    url = (
        f"{ENCODE_BASE}/search/"
        f"?type=Annotation"
        f"&annotation_type=gkm-SVM-model"
        f"&status=released"
        f"&format=json"
        f"&limit=all"
        f"{field_params}"
    )
    print(f"  Querying annotations...")
    data = fetch_json(url)
    results = data.get("@graph", [])
    print(f"  Got {len(results)} / {data.get('total', '?')} annotations")
    return results


def fetch_annotation_files(accession: str) -> list[dict]:
    """Fetch file details for a single annotation."""
    url = f"{ENCODE_BASE}/annotations/{accession}/?format=json"
    data = fetch_json(url)
    return data.get("files", [])


def _extract_target(annotation: dict) -> str:
    targets = annotation.get("targets", [])
    if not targets:
        return ""
    t = targets[0]
    if isinstance(t, dict):
        return t.get("label", t.get("name", ""))
    if isinstance(t, str):
        parts = t.strip("/").split("/")
        return parts[-1] if parts else t
    return str(t)


def _parse_files(files: list) -> dict:
    """Extract model and kmer file metadata."""
    info = {
        "model_accession": "",
        "model_href": "",
        "model_size_mb": "",
        "model_output_type": "",
        "model_assembly": "",
        "model_derived_from": "",
        "kmer_accession": "",
        "kmer_href": "",
        "kmer_size_mb": "",
    }

    for f in files:
        if isinstance(f, str):
            continue
        if f.get("status") != "released":
            continue

        output_type = f.get("output_type", "").lower()
        accession = f.get("accession", "")
        href = f.get("href", "")
        size = f.get("file_size", 0)
        size_mb = f"{size / 1e6:.1f}" if size else ""
        assembly = f.get("assembly", "")

        derived = f.get("derived_from", [])
        derived_str = ";".join(
            d if isinstance(d, str) else d.get("@id", "")
            for d in (derived or [])
        )

        if "model" in output_type or "prediction" in output_type:
            info["model_accession"] = accession
            info["model_href"] = href
            info["model_size_mb"] = size_mb
            info["model_output_type"] = f.get("output_type", "")
            info["model_assembly"] = assembly
            info["model_derived_from"] = derived_str
        elif "kmer" in output_type or "weight" in output_type:
            info["kmer_accession"] = accession
            info["kmer_href"] = href
            info["kmer_size_mb"] = size_mb

    return info


def build_manifest(
    annotations: list[dict], fetch_files: bool = True,
) -> list[dict]:
    rows = []
    total = len(annotations)

    for i, ann in enumerate(annotations):
        accession = ann.get("accession", "")

        biosample_obj = ann.get("biosample_ontology", {})
        if isinstance(biosample_obj, str):
            biosample_obj = {}

        assay = ann.get("assay_term_name", [])
        if isinstance(assay, list):
            assay = assay[0] if assay else ""

        organism_obj = ann.get("organism", {})
        if isinstance(organism_obj, str):
            organism_obj = {}

        organ_slims = biosample_obj.get("organ_slims", [])

        row = {
            "annotation_accession": accession,
            "assay": assay,
            "biosample": biosample_obj.get("term_name", ""),
            "biosample_type": biosample_obj.get("classification", ""),
            "organ": ";".join(organ_slims) if organ_slims else "",
            "target": _extract_target(ann),
            "organism": organism_obj.get("scientific_name", ""),
            "organism_short": organism_obj.get("name", ""),
            "description": ann.get("description", ""),
            "date_released": ann.get("date_released", ""),
        }

        if fetch_files:
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  Fetching files {i + 1}/{total}...")
            try:
                files = fetch_annotation_files(accession)
                file_info = _parse_files(files)
                row.update(file_info)
            except Exception as e:
                print(f"  Warning: failed to fetch files for {accession}: {e}",
                      file=sys.stderr)
                row.update(_parse_files([]))
            if (i + 1) % 50 == 0:
                time.sleep(0.5)
        else:
            row.update(_parse_files([]))

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
    print(f"\nTotal annotations: {len(rows)}")

    assays: dict[str, int] = {}
    organisms: dict[str, int] = {}
    biosample_types: dict[str, int] = {}
    has_model = 0
    has_kmer = 0

    for r in rows:
        assays[r["assay"]] = assays.get(r["assay"], 0) + 1
        organisms[r["organism_short"]] = organisms.get(r["organism_short"], 0) + 1
        bt = r["biosample_type"] or "unknown"
        biosample_types[bt] = biosample_types.get(bt, 0) + 1
        if r.get("model_accession"):
            has_model += 1
        if r.get("kmer_accession"):
            has_kmer += 1

    print(f"\nBy assay:")
    for k, v in sorted(assays.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    print(f"\nBy organism:")
    for k, v in sorted(organisms.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    print(f"\nBy biosample type:")
    for k, v in sorted(biosample_types.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    if has_model or has_kmer:
        print(f"\nFiles: {has_model} models, {has_kmer} kmer weight files")


def main():
    parser = argparse.ArgumentParser(
        description="Scrape ENCODE gkm-SVM model metadata"
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("data/encode_gkmsvm"),
    )
    parser.add_argument(
        "--organism", type=str, default=None,
        help="Filter by organism (e.g., 'human', 'mouse')",
    )
    parser.add_argument(
        "--skip-files", action="store_true",
        help="Skip per-annotation file fetching (fast, metadata only)",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Fetching ENCODE gkm-SVM annotations...")
    annotations = fetch_all_annotations()

    raw_path = args.outdir / "encode_raw.json"
    with open(raw_path, "w") as f:
        json.dump(annotations, f, indent=2)
    print(f"Raw JSON: {raw_path}")

    if args.organism:
        annotations = [
            a for a in annotations
            if args.organism.lower() in (
                a.get("organism", {}).get("scientific_name", "")
                + a.get("organism", {}).get("name", "")
            ).lower()
        ]
        print(f"Filtered to {len(annotations)} {args.organism} annotations")

    fetch_files = not args.skip_files
    if fetch_files:
        print(f"Fetching file details for {len(annotations)} annotations...")
    rows = build_manifest(annotations, fetch_files=fetch_files)

    manifest_path = args.outdir / "encode_manifest.tsv"
    write_tsv(rows, manifest_path)
    print(f"Manifest: {manifest_path}")

    print_summary(rows)


if __name__ == "__main__":
    main()
