"""Replicate dsQTL variant effect prediction from Lee et al. 2015 (Nature Genetics).

Reproduces the deltaSVM scores and evaluation metrics reported in:
- Lee et al. (2015) "Deltasvm: a new method for predicting effects of regulatory
  noncoding variants." Nat Genet 47(8): 955-961. doi:10.1038/ng.3331
- Pampari et al. (2023) "ChromBPNet" used these as a gkm-SVM baseline:
    Classification AP = 0.19, effect size correlation r = 0.73

Also benchmarks full kernel SVM scoring and GkmExplain on the same variants
using the ENCODE ENCFF579AOX model (72K SVs) for GPU scaling.

Data requirements (auto-downloaded on first run to benchmarks/data/):
    - gm12878_deltasvm_weights.txt  : Beer lab GM12878 deltaSVM 10-mer weights
    - dsqtl_lee2015.tsv             : Lee 2015 Supp Table 1 (28,309 variants)
    - GSE31388_dsQtlTable.txt.gz    : Degner et al. 2012 dsQTL effect sizes
    - hg19.fa                       : hg19 genome FASTA
    - encode_ENCFF579AOX.model.txt.gz : (optional) full ENCODE model for GPU benchmark

Usage:
    python benchmarks/replicate_dsqtl.py                    # deltaSVM replication only
    python benchmarks/replicate_dsqtl.py --full-kernel      # + full kernel SVM scoring
    python benchmarks/replicate_dsqtl.py --full-kernel --device cuda  # GPU benchmark
"""

from __future__ import annotations

import argparse
import csv
import gzip
import time
from pathlib import Path

import numpy as np
import pyfaidx
from sklearn.metrics import average_precision_score

from gkmsvm.backend import HAS_CUPY, HAS_MLX

DATA_DIR = Path(__file__).parent / "data"
FIXTURE_DIR = Path(__file__).parent.parent / "tests" / "fixtures"

FLANK = 9  # bases of flanking context for 10-mer deltaSVM


def _to_device(arr, device):
    if device == "cuda":
        from gkmsvm.backend import to_gpu
        return to_gpu(arr)
    elif device == "mlx":
        from gkmsvm.backend import to_mlx
        return to_mlx(arr)
    return arr


def _to_numpy(arr, device):
    if device == "cuda":
        from gkmsvm.backend import to_cpu
        return to_cpu(arr)
    return np.asarray(arr)


def _sync_device(device):
    if device == "cuda" and HAS_CUPY:
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    elif device == "mlx" and HAS_MLX:
        import mlx.core as mx
        mx.eval()


HG19_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz"
GEO_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE31nnn/GSE31388/suppl/GSE31388_dsQtlTable.txt.gz"
WEIGHTS_URL = "https://beerlab.org/deltasvm/downloads/SupplementaryTable_gm12878weights.txt"
LEE2015_URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.1038%2Fng.3331/MediaObjects/41588_2015_BFng3331_MOESM26_ESM.xlsx"
)


def download_data():
    """Download data files if not present."""
    import shutil
    import urllib.request

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    weights_path = DATA_DIR / "gm12878_deltasvm_weights.txt"
    if not weights_path.exists():
        print("  Downloading Beer lab deltaSVM weights (16 MB)...")
        urllib.request.urlretrieve(WEIGHTS_URL, weights_path)

    tsv_path = DATA_DIR / "dsqtl_lee2015.tsv"
    if not tsv_path.exists():
        import openpyxl
        xlsx_path = DATA_DIR / "lee2015_supp.xlsx"
        if not xlsx_path.exists():
            print("  Downloading Lee 2015 Supplementary Table 1...")
            urllib.request.urlretrieve(LEE2015_URL, xlsx_path)
        print("  Converting XLSX to TSV...")
        wb = openpyxl.load_workbook(xlsx_path, read_only=True)
        ws = wb["SuppTable1"]
        with open(tsv_path, "w") as f:
            for row in ws.iter_rows(values_only=True):
                f.write("\t".join(str(v) if v is not None else "" for v in row) + "\n")
        wb.close()

    geo_path = DATA_DIR / "GSE31388_dsQtlTable.txt.gz"
    if not geo_path.exists():
        print("  Downloading GEO GSE31388 dsQTL effect sizes...")
        urllib.request.urlretrieve(GEO_URL, geo_path)

    fasta_path = DATA_DIR / "hg19.fa"
    if not fasta_path.exists():
        gz_path = DATA_DIR / "hg19.fa.gz"
        if not gz_path.exists():
            print("  Downloading hg19 genome (~900 MB compressed)...")
            urllib.request.urlretrieve(HG19_URL, gz_path)
        print("  Decompressing hg19.fa.gz...")
        with gzip.open(gz_path, "rb") as f_in, open(fasta_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def load_dsqtl_variants(
    tsv_path: Path, fasta_path: Path
) -> dict[str, np.ndarray]:
    """Load dsQTL variants and extract flanking sequences from hg19."""
    genome = pyfaidx.Fasta(str(fasta_path))
    chroms = []
    positions = []
    ref_seqs = []
    alt_seqs = []
    labels = []
    published_scores = []
    snp_names = []

    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            chrom = row["chrom_hg19"]
            pos = int(row["pos_hg19"])
            a1, a2 = row["allele1"], row["allele2"]

            start = pos - 1 - FLANK
            end = pos + FLANK
            if start < 0 or chrom not in genome:
                continue
            seq = str(genome[chrom][start:end]).upper()
            if len(seq) != 2 * FLANK + 1 or "N" in seq:
                continue

            ref_base = seq[FLANK]
            if ref_base not in (a1, a2):
                continue

            a1_seq = list(seq)
            a1_seq[FLANK] = a1
            a1_seq = "".join(a1_seq)
            a2_seq = list(seq)
            a2_seq[FLANK] = a2
            a2_seq = "".join(a2_seq)
            ref_seq = a2_seq
            alt_seq = a1_seq

            chroms.append(chrom)
            positions.append(pos)
            ref_seqs.append(ref_seq)
            alt_seqs.append(alt_seq)
            labels.append(int(row["label"]))
            published_scores.append(float(row["gkm_SVM"]))
            snp_names.append(row["SNPname1"])

    genome.close()
    return {
        "chrom": np.array(chroms),
        "pos": np.array(positions),
        "ref_seq": np.array(ref_seqs),
        "alt_seq": np.array(alt_seqs),
        "label": np.array(labels),
        "published_score": np.array(published_scores),
        "snp_name": np.array(snp_names),
    }


def load_dsqtl_effect_sizes(gz_path: Path) -> dict[str, float]:
    """Load dsQTL effect sizes from Degner et al. 2012 (GSE31388)."""
    effects = {}
    with gzip.open(gz_path, "rt") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            snp_id = f"{row['Chr']}.{row['SNP']}"
            effects[snp_id] = float(row["Estimate"])
    return effects


def run_deltasvm_replication(variants: dict, device: str = "cpu") -> dict:
    """Score all variants with deltaSVM and compute replication metrics."""
    from gkmsvm.codec import one_hot_encode
    from gkmsvm.importers.deltasvm import load_deltasvm_model

    print("Loading deltaSVM weights...")
    model = load_deltasvm_model(
        str(DATA_DIR / "gm12878_deltasvm_weights.txt"),
        l=10,
        include_rc=False,
    )
    if device == "cuda":
        model.cuda()
    elif device == "mlx":
        model.mlx()

    ref_seqs = variants["ref_seq"]
    alt_seqs = variants["alt_seq"]
    n = len(ref_seqs)
    our_scores = np.zeros(n)

    print(f"Scoring {n} variants with deltaSVM...")
    t0 = time.perf_counter()

    batch_size = 512
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        ref_batch = np.stack([one_hot_encode(s) for s in ref_seqs[start:end]])
        alt_batch = np.stack([one_hot_encode(s) for s in alt_seqs[start:end]])
        ref_batch = _to_device(ref_batch, device)
        alt_batch = _to_device(alt_batch, device)
        delta = model.score_variants(ref_batch, alt_batch)
        our_scores[start:end] = _to_numpy(delta.squeeze(1), device)

    elapsed = time.perf_counter() - t0
    print(f"  Scored {n} variants in {elapsed:.2f}s ({n/elapsed:.0f} variants/s)")

    published = variants["published_score"]
    max_diff = np.max(np.abs(our_scores - published))
    corr_with_published = np.corrcoef(our_scores, published)[0, 1]
    print(f"  Max score difference from published: {max_diff:.2e}")
    print(f"  Correlation with published scores: {corr_with_published:.6f}")

    return {
        "scores": our_scores,
        "elapsed": elapsed,
        "max_diff": max_diff,
        "corr_with_published": corr_with_published,
    }


def compute_evaluation_metrics(
    scores: np.ndarray,
    variants: dict,
    effect_sizes: dict[str, float],
    score_name: str = "deltaSVM",
) -> dict:
    """Compute classification AP and effect size correlation."""
    labels = variants["label"]
    binary_labels = (labels == 1).astype(int)

    abs_scores = np.abs(scores)
    ap = average_precision_score(binary_labels, abs_scores)

    snp_names = variants["snp_name"]
    matched_scores = []
    matched_effects = []
    matched_pos_scores = []
    matched_pos_effects = []
    for i, name in enumerate(snp_names):
        if name in effect_sizes:
            matched_scores.append(scores[i])
            matched_effects.append(effect_sizes[name])
            if labels[i] == 1:
                matched_pos_scores.append(scores[i])
                matched_pos_effects.append(effect_sizes[name])

    matched_scores = np.array(matched_scores)
    matched_effects = np.array(matched_effects)
    matched_pos_scores = np.array(matched_pos_scores)
    matched_pos_effects = np.array(matched_pos_effects)
    pearson_r_all = np.corrcoef(matched_scores, matched_effects)[0, 1]
    pearson_r_pos = np.corrcoef(matched_pos_scores, matched_pos_effects)[0, 1]

    print(f"\n--- {score_name} Evaluation ---")
    print(f"  Classification AP:        {ap:.4f}  (published: 0.19)")
    print(f"  Effect size r (pos only): {pearson_r_pos:.4f}  (published: 0.73)")
    print(f"  Effect size r (all):      {pearson_r_all:.4f}")
    print(f"  Matched variants:         {len(matched_scores)} ({len(matched_pos_scores)} pos)")
    print(f"  Total variants:           {len(scores)} ({binary_labels.sum()} pos / {(~binary_labels.astype(bool)).sum()} neg)")

    return {"ap": ap, "pearson_r": pearson_r_pos, "pearson_r_all": pearson_r_all, "n_matched": len(matched_scores)}


def run_full_kernel_benchmark(
    variants: dict,
    effect_sizes: dict[str, float],
    device: str = "cpu",
    sv_chunk_size: int | None = None,
    max_variants: int = 100,
    ism_seqs: int = 5,
) -> dict:
    """Score variants with full kernel SVM (ENCFF579AOX) and benchmark."""
    from gkmsvm.codec import one_hot_encode
    from gkmsvm.importers.lsgkm import load_lsgkm_model

    model_path = FIXTURE_DIR / "encode_ENCFF579AOX.model.txt.gz"
    if not model_path.exists():
        print(f"SKIP: full kernel model not found at {model_path}")
        return {}

    print(f"\nLoading full kernel model (ENCFF579AOX)...")
    t0 = time.perf_counter()
    model = load_lsgkm_model(str(model_path), sv_chunk_size=sv_chunk_size)
    load_time = time.perf_counter() - t0
    print(f"  Loaded in {load_time:.1f}s: {model.num_support_vectors} SVs, L={model.kernel.l}, k={model.kernel.k}")

    if device == "cuda":
        model.cuda()
    elif device == "mlx":
        model.mlx()

    all_ref = variants["ref_seq"]
    all_alt = variants["alt_seq"]
    n_total = len(all_ref)
    n = min(n_total, max_variants)
    ref_seqs = all_ref[:n]
    alt_seqs = all_alt[:n]
    print(f"  Scoring {n}/{n_total} variants (query length: {len(ref_seqs[0])}bp)")

    print(f"\n  Benchmarking forward scoring ({device})...")
    warmup_ref = np.stack([one_hot_encode(s) for s in ref_seqs[:4]])
    warmup_ref = _to_device(warmup_ref, device)
    _ = model(warmup_ref)
    _sync_device(device)

    batch_size = 32 if device != "cpu" else 64
    ref_scores = np.zeros(n)
    alt_scores = np.zeros(n)

    t0 = time.perf_counter()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        ref_batch = np.stack([one_hot_encode(s) for s in ref_seqs[start:end]])
        alt_batch = np.stack([one_hot_encode(s) for s in alt_seqs[start:end]])
        ref_batch = _to_device(ref_batch, device)
        alt_batch = _to_device(alt_batch, device)
        ref_scores[start:end] = _to_numpy(model(ref_batch).squeeze(1), device)
        alt_scores[start:end] = _to_numpy(model(alt_batch).squeeze(1), device)
        if start % (batch_size * 10) == 0 and start > 0:
            elapsed_so_far = time.perf_counter() - t0
            rate = (start * 2) / elapsed_so_far
            eta = (n * 2 - start * 2) / rate
            print(f"    {start}/{n} ({rate:.1f} seqs/s, ETA {eta:.0f}s)")

    _sync_device(device)
    scoring_time = time.perf_counter() - t0

    kernel_deltas = alt_scores - ref_scores
    print(f"  Full kernel scoring: {scoring_time:.1f}s ({n*2/scoring_time:.1f} seqs/s)")
    print(f"  Extrapolated time for all {n_total} variants: {n_total*2/max(1,n*2/scoring_time):.0f}s")

    # ISM benchmark
    ism_results = run_ism_benchmark(model, ref_seqs[:ism_seqs], device) if ism_seqs > 0 else {}

    # GkmExplain benchmark (MLX unsupported, fall back to CPU)
    explain_device = device if device != "mlx" else "cpu"
    if explain_device == "cpu" and device == "mlx":
        from gkmsvm.importers.lsgkm import load_lsgkm_model as _load
        explain_model = _load(str(model_path), sv_chunk_size=sv_chunk_size)
    else:
        explain_model = model
    explain_results = run_explain_benchmark(explain_model, ref_seqs[:ism_seqs], explain_device) if ism_seqs > 0 else {}

    return {
        "scoring_time": scoring_time,
        "seqs_per_sec": n * 2 / scoring_time,
        "load_time": load_time,
        "ism": ism_results,
        "explain": explain_results,
    }


def run_ism_benchmark(model, seqs: np.ndarray, device: str) -> dict:
    """Benchmark ISM on a few sequences."""
    from gkmsvm.codec import one_hot_encode
    from gkmsvm.ism import ism

    n = len(seqs)
    print(f"\n  Benchmarking ISM on {n} sequences ({device})...")
    x = np.stack([one_hot_encode(s) for s in seqs])
    x = _to_device(x, device)

    t0 = time.perf_counter()
    result = ism(model, x)
    _sync_device(device)
    elapsed = time.perf_counter() - t0

    print(f"  ISM: {elapsed:.2f}s for {n} seqs ({n/elapsed:.2f} seqs/s)")
    print(f"  Output shape: {result.shape}")
    return {"elapsed": elapsed, "n_seqs": n, "seqs_per_sec": n / elapsed}


def run_explain_benchmark(model, seqs: np.ndarray, device: str) -> dict:
    """Benchmark GkmExplain on a few sequences."""
    from gkmsvm.codec import one_hot_encode
    from gkmsvm.explain import gkmexplain

    n = len(seqs)
    x_np = np.stack([one_hot_encode(s) for s in seqs])
    x = _to_device(x_np, device)

    print(f"\n  Benchmarking GkmExplain on {n} sequences ({device})...")
    t0 = time.perf_counter()
    hyp = gkmexplain(model, x, mode=1)
    _sync_device(device)
    elapsed = time.perf_counter() - t0
    imp = _to_numpy(hyp, device) * x_np

    print(f"  GkmExplain: {elapsed:.2f}s for {n} seqs ({n/elapsed:.2f} seqs/s)")
    print(f"  Output shape: {_to_numpy(hyp, device).shape}")
    return {"elapsed": elapsed, "n_seqs": n, "seqs_per_sec": n / elapsed}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mlx", "auto"],
                        help="Device for computation (cpu/cuda/mlx/auto)")
    parser.add_argument("--full-kernel", action="store_true", help="Also run full kernel SVM benchmark")
    parser.add_argument("--sv-chunk-size", type=int, default=None, help="SV chunk size for full kernel (default: None = no chunking)")
    parser.add_argument("--full-kernel-variants", type=int, default=100,
                        help="Max variants for full kernel scoring (default: 100)")
    parser.add_argument("--ism-seqs", type=int, default=5,
                        help="Sequences for ISM/GkmExplain benchmark (default: 5)")
    args = parser.parse_args()

    if args.device == "auto":
        if HAS_CUPY:
            args.device = "cuda"
        elif HAS_MLX:
            args.device = "mlx"
        else:
            args.device = "cpu"
        print(f"Auto-selected device: {args.device}")

    if args.device == "cuda":
        if not HAS_CUPY:
            print("CuPy not available, falling back to CPU")
            args.device = "cpu"
        else:
            import cupy as cp
            props = cp.cuda.runtime.getDeviceProperties(0)
            print(f"GPU: {props['name'].decode()}")
            print(f"Memory: {props['totalGlobalMem'] / 1e9:.1f} GB")
    elif args.device == "mlx":
        if not HAS_MLX:
            print("MLX not available, falling back to CPU")
            args.device = "cpu"
        else:
            print("Using MLX (Apple GPU)")

    # --- Download and load data ---
    print("Downloading data (if needed)...")
    download_data()

    print("Loading dsQTL variants and extracting sequences from hg19...")
    variants = load_dsqtl_variants(
        DATA_DIR / "dsqtl_lee2015.tsv",
        DATA_DIR / "hg19.fa",
    )
    print(f"  Loaded {len(variants['label'])} variants "
          f"({(variants['label'] == 1).sum()} pos / {(variants['label'] == -1).sum()} neg)")

    effect_sizes = load_dsqtl_effect_sizes(DATA_DIR / "GSE31388_dsQtlTable.txt.gz")
    print(f"  Loaded {len(effect_sizes)} dsQTL effect sizes")

    # --- DeltaSVM replication ---
    print("\n" + "=" * 60)
    print("DeltaSVM Replication (Lee et al. 2015)")
    print("=" * 60)
    dsvm_result = run_deltasvm_replication(variants, device=args.device)
    dsvm_metrics = compute_evaluation_metrics(
        dsvm_result["scores"], variants, effect_sizes, score_name="deltaSVM"
    )

    # --- Full kernel SVM benchmark ---
    if args.full_kernel:
        print("\n" + "=" * 60)
        print("Full Kernel SVM Benchmark (ENCFF579AOX, 72K SVs)")
        print("=" * 60)
        fk_result = run_full_kernel_benchmark(
            variants, effect_sizes, device=args.device,
            sv_chunk_size=args.sv_chunk_size,
            max_variants=args.full_kernel_variants,
            ism_seqs=args.ism_seqs,
        )

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  deltaSVM score replication:  max_diff={dsvm_result['max_diff']:.2e}")
    print(f"  deltaSVM classification AP:  {dsvm_metrics['ap']:.4f}  (published: 0.19)")
    print(f"  deltaSVM effect size r:      {dsvm_metrics['pearson_r']:.4f}  (published: 0.73)")
    if args.full_kernel and fk_result:
        print(f"  Full kernel scoring:         {fk_result.get('seqs_per_sec', 0):.1f} seqs/s")


if __name__ == "__main__":
    main()
