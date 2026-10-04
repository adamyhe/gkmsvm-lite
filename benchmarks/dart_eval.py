"""DART-Eval benchmarks for gkm-SVM models.

Evaluates gkm-SVM models on DART-Eval Tasks 4 and 5 (Pampari et al. 2024,
NeurIPS):

  Task 4 — Chromatin activity prediction
    Score peak vs nonpeak regions for classification (AUROC/AUPRC).
    Correlate gkm-SVM scores with quantitative log1p(counts) signal.

  Task 5 — Variant effect prediction
    Three VEP scoring methods:
      kernel:     score(alt) - score(ref)  (full kernel, two evals per variant)
      gkmexplain: hyp[alt_base] - hyp[ref_base] at variant position (one
                  gkmexplain call per variant; Shrikumar et al. 2019 §5.2)
      deltasvm:   Σ Δw_kmer  (linear k-mer weight approximation)
    dsQTLs: Yoruban LCL (hg19), caQTLs: African (hg38).

  Sequences are center-cropped to the model's SV length (typically 300 bp)
  before scoring, matching the training context window.

ENCODE ATAC-seq gkm-SVM models (Beer lab, hg38):
  Cell line   Annotation      Enhancer model   Promoter model
  GM12878     ENCSR614LCR     ENCFF953DDE      ENCFF795HXN
  K562        ENCSR735DFK     ENCFF986TYL      ENCFF388NZI
  HepG2       ENCSR847ICH     ENCFF354PGX      ENCFF724LMF
  IMR90       ENCSR571ISC     ENCFF935GNI      ENCFF072NJQ
  H1ESC       (none)          —                —

  Download:
    wget https://www.encodeproject.org/files/ENCFF953DDE/@@download/ENCFF953DDE.txt.gz

Data:
  Download from Synapse (syn59522070). Each task directory includes a
  pre-exported data.h5 with one-hot sequences and labels.

    pip install synapseclient && synapse login
    export DART_WORK_DIR=/path/to/dart-eval
    synapse get -r syn60581044 --downloadLocation $DART_WORK_DIR/refs
    synapse get -r syn60581041 --downloadLocation $DART_WORK_DIR/task_4_chromatin_activity
    synapse get -r syn60581045 --downloadLocation $DART_WORK_DIR/task_5_variant_effect_prediction

  If data.h5 files are absent, the script falls back to BED + genome FASTA
  extraction (requires pyfaidx, pyBigWig).

Usage:
  python benchmarks/dart_eval.py -m ENCFF953DDE.txt.gz --task all \\
      --cell-lines GM12878 --device cuda

  # GkmExplain VEP scoring
  python benchmarks/dart_eval.py -m ENCFF953DDE.txt.gz --task vep \\
      --vep-method gkmexplain --device cuda
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

WORK_DIR = os.environ.get("DART_WORK_DIR", "")

CELL_LINES = ["GM12878", "H1ESC", "HEPG2", "IMR90", "K562"]

CHROMS_TEST = ["chr5", "chr10", "chr14", "chr18", "chr20", "chr22"]

DART_SEQ_LEN = 2114
CROP = 557
BASE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(model_path: str, device: str, sv_chunk_size: int | None):
    from gkmsvm.serialization import load_model
    model = load_model(model_path)
    if sv_chunk_size is not None:
        model.sv_chunk_size = sv_chunk_size
    if device == "cuda":
        model.cuda()
    return model


def _center_crop(seqs: np.ndarray, target_len: int) -> np.ndarray:
    """Center-crop [N, 4, L] sequences to target_len."""
    L = seqs.shape[2]
    if L <= target_len:
        return seqs
    start = (L - target_len) // 2
    return seqs[:, :, start:start + target_len]


def _score_sequences(model, seqs_ohe: np.ndarray, batch_size: int,
                     device: str, verbose: bool = True) -> np.ndarray:
    """Score [N, 4, L] one-hot sequences, return [N] scores."""
    from gkmsvm.backend import to_cpu
    N = seqs_ohe.shape[0]
    scores = np.zeros(N, dtype=np.float64)

    chunks = range(0, N, batch_size)
    if verbose:
        from tqdm import tqdm
        chunks = tqdm(chunks, desc="scoring",
                      total=(N + batch_size - 1) // batch_size)

    for start in chunks:
        end = min(start + batch_size, N)
        xb = model._match_device(seqs_ohe[start:end])
        sb = model(xb, verbose=False).flatten()
        scores[start:end] = to_cpu(sb)

    return scores


def _h5_to_channels_first(seqs_h5: np.ndarray) -> np.ndarray:
    """Convert [N, L, 4] uint8 (DART-Eval H5) → [N, 4, L] float64."""
    return seqs_h5.astype(np.float64).transpose(0, 2, 1)


def _get_sv_len(model) -> int:
    """Get the SV sequence length from the model."""
    return model.support_sequences.shape[2]


# ---------------------------------------------------------------------------
# Task 4: Chromatin Activity
# ---------------------------------------------------------------------------

def _load_task4_h5(cell_line: str) -> dict | None:
    """Load Task 4 test split from pre-exported H5."""
    import h5py

    h5_path = os.path.join(WORK_DIR, "task_4_chromatin_activity/data.h5")
    if not os.path.exists(h5_path):
        return None

    with h5py.File(h5_path, "r") as f:
        if cell_line not in f:
            return None
        test = f[cell_line]["test"]
        result = {}
        for key in ("peaks", "idr_peaks", "nonpeaks"):
            if key in test:
                result[key] = {
                    "seqs": _h5_to_channels_first(test[key]["seqs"][:]),
                    "counts": test[key]["counts"][:],
                }
        return result


def _load_task4_bed(cell_line: str) -> dict | None:
    """Load Task 4 test data from BED files + genome FASTA (fallback)."""
    from gkmsvm.codec import one_hot_encode
    import pyfaidx

    base = os.path.join(WORK_DIR, "task_4_chromatin_activity/processed_data")
    genome_fa = os.path.join(WORK_DIR,
        "refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")

    beds = {
        "peaks": os.path.join(base, f"cell_line_expanded_peaks/{cell_line}_peaks.bed"),
        "idr_peaks": os.path.join(base, f"cell_line_idr_peaks/{cell_line}.bed"),
        "nonpeaks": os.path.join(base, f"cell_line_expanded_peaks/{cell_line}_nonpeaks.bed"),
    }
    bw_path = os.path.join(base, f"bigwigs/{cell_line}_unstranded.bw")

    for f in [*beds.values(), genome_fa]:
        if not os.path.exists(f):
            return None

    fa = pyfaidx.Fasta(genome_fa, one_based_attributes=False)
    result = {}

    for key, bed_path in beds.items():
        df = pd.read_csv(bed_path, sep="\t", header=None,
                         usecols=[0, 1, 2], names=["chrom", "start", "end"])
        df = df[df["chrom"].isin(CHROMS_TEST)].reset_index(drop=True)

        seqs = []
        for _, row in df.iterrows():
            mid = (row["start"] + row["end"]) // 2
            s = mid - DART_SEQ_LEN // 2
            e = s + DART_SEQ_LEN
            seq_str = str(fa[row["chrom"]][max(0, s):e]).upper()
            if len(seq_str) < DART_SEQ_LEN:
                seq_str = seq_str + "N" * (DART_SEQ_LEN - len(seq_str))
            seqs.append(one_hot_encode(seq_str))

        entry = {"seqs": np.stack(seqs)}

        if key == "peaks" and os.path.exists(bw_path):
            import pyBigWig
            bw = pyBigWig.open(bw_path)
            counts = []
            df2 = pd.read_csv(bed_path, sep="\t", header=None,
                              usecols=[0, 1, 2], names=["chrom", "start", "end"])
            df2 = df2[df2["chrom"].isin(CHROMS_TEST)].reset_index(drop=True)
            for _, row in df2.iterrows():
                mid = (row["start"] + row["end"]) // 2
                s2 = mid - DART_SEQ_LEN // 2 + CROP
                e2 = mid + DART_SEQ_LEN // 2 - CROP
                vals = bw.values(row["chrom"], s2, e2)
                vals = np.nan_to_num(vals, nan=0.0)
                counts.append(np.log1p(np.sum(vals)))
            bw.close()
            entry["counts"] = np.array(counts)

        result[key] = entry

    fa.close()
    return result


def run_task4_cell_line(model, cell_line: str, device: str,
                        batch_size: int, verbose: bool) -> dict:
    """Run Task 4 for a single cell line on test chromosomes."""
    sv_len = _get_sv_len(model)
    print(f"\n--- Task 4: {cell_line} ---")

    data = _load_task4_h5(cell_line)
    if data is not None:
        print(f"  Loaded from H5 (pre-exported)")
    else:
        print(f"  H5 not found, falling back to BED+FASTA extraction...")
        data = _load_task4_bed(cell_line)
        if data is None:
            print(f"  SKIP {cell_line}: missing data files", file=sys.stderr)
            return {}

    idr_seqs = _center_crop(data["idr_peaks"]["seqs"], sv_len)
    nonpeak_seqs = _center_crop(data["nonpeaks"]["seqs"], sv_len)
    print(f"  IDR peaks: {len(idr_seqs)}, nonpeaks: {len(nonpeak_seqs)}")
    print(f"  Center-cropped {DART_SEQ_LEN} bp → {sv_len} bp")

    t0 = time.perf_counter()
    print(f"  Scoring IDR peaks...")
    idr_scores = _score_sequences(model, idr_seqs, batch_size, device, verbose)
    print(f"  Scoring nonpeaks...")
    nonpeak_scores = _score_sequences(model, nonpeak_seqs, batch_size, device, verbose)
    elapsed = time.perf_counter() - t0
    total_seqs = len(idr_seqs) + len(nonpeak_seqs)
    print(f"  Scored {total_seqs} sequences in {elapsed:.1f}s "
          f"({total_seqs/elapsed:.1f} seq/s)")

    labels = np.concatenate([np.ones(len(idr_scores)),
                             np.zeros(len(nonpeak_scores))])
    all_scores = np.concatenate([idr_scores, nonpeak_scores])

    auroc = roc_auc_score(labels, all_scores)
    auprc = average_precision_score(labels, all_scores)

    result = {
        "cell_line": cell_line,
        "n_idr_peaks": len(idr_scores),
        "n_nonpeaks": len(nonpeak_scores),
        "auroc": auroc,
        "auprc": auprc,
        "elapsed": elapsed,
    }

    if "peaks" in data and "counts" in data["peaks"]:
        peak_seqs = _center_crop(data["peaks"]["seqs"], sv_len)
        counts = data["peaks"]["counts"]
        print(f"  Scoring all peaks ({len(peak_seqs)}) for quantitative eval...")
        peak_scores = _score_sequences(model, peak_seqs, batch_size, device, verbose)
        r_pearson, _ = pearsonr(peak_scores, counts)
        r_spearman, _ = spearmanr(peak_scores, counts)
        result["pearson_counts"] = r_pearson
        result["spearman_counts"] = r_spearman
        result["n_peaks"] = len(peak_scores)
        print(f"  Quantitative: Pearson={r_pearson:.4f}, "
              f"Spearman={r_spearman:.4f}")

    print(f"  Classification: AUROC={auroc:.4f}, AUPRC={auprc:.4f}")
    return result


# ---------------------------------------------------------------------------
# Task 5: Variant Effect Prediction
# ---------------------------------------------------------------------------

def _load_task5_h5(task_key: str,
                   indices: np.ndarray | None = None) -> dict | None:
    """Load Task 5 allele sequences from pre-exported H5.

    If indices is provided, only loads those rows (avoids OOM on caQTL's
    219K variants × 2114bp × 4 channels × 2 alleles ≈ 30 GB in float64).
    """
    import h5py

    h5_path = os.path.join(WORK_DIR,
                           "task_5_variant_effect_prediction/data.h5")
    if not os.path.exists(h5_path):
        return None

    with h5py.File(h5_path, "r") as f:
        if task_key not in f:
            return None
        grp = f[task_key]
        if indices is not None:
            idx = np.sort(indices)
            return {
                "a1_seqs": _h5_to_channels_first(grp["allele_1_seqs"][idx]),
                "a2_seqs": _h5_to_channels_first(grp["allele_2_seqs"][idx]),
                "is_causal": grp["is_causal"][idx],
                "effect_size": grp["effect_size"][idx],
            }
        return {
            "a1_seqs": _h5_to_channels_first(grp["allele_1_seqs"][:]),
            "a2_seqs": _h5_to_channels_first(grp["allele_2_seqs"][:]),
            "is_causal": grp["is_causal"][:],
            "effect_size": grp["effect_size"][:],
        }


def _extract_variant_seqs(variant_df: pd.DataFrame, genome_fa: str,
                          chrom_col: str, pos_col: str,
                          a1_col: str, a2_col: str,
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Extract allele1 and allele2 sequences from genome FASTA."""
    from gkmsvm.codec import one_hot_encode
    import pyfaidx

    fa = pyfaidx.Fasta(genome_fa, one_based_attributes=False)
    half = DART_SEQ_LEN // 2
    a1_seqs, a2_seqs = [], []

    for _, row in variant_df.iterrows():
        chrom = row[chrom_col]
        pos = int(row[pos_col]) - 1
        allele1 = row[a1_col]
        allele2 = row[a2_col]

        left = str(fa[chrom][max(0, pos - half):pos]).upper()
        right = str(fa[chrom][pos + 1:pos + half]).upper()

        a1_seq = left + allele1 + right
        a2_seq = left + allele2 + right

        for seq_list, seq in [(a1_seqs, a1_seq), (a2_seqs, a2_seq)]:
            if len(seq) < DART_SEQ_LEN:
                seq = seq + "N" * (DART_SEQ_LEN - len(seq))
            elif len(seq) > DART_SEQ_LEN:
                seq = seq[:DART_SEQ_LEN]
            seq_list.append(one_hot_encode(seq))

    fa.close()
    return np.stack(a1_seqs), np.stack(a2_seqs)


def _vep_kernel(model, a1_seqs: np.ndarray, a2_seqs: np.ndarray,
                sv_len: int, batch_size: int, device: str,
                verbose: bool) -> np.ndarray:
    """VEP via full kernel: score(alt) - score(ref)."""
    a1_crop = _center_crop(a1_seqs, sv_len)
    a2_crop = _center_crop(a2_seqs, sv_len)
    print(f"  [kernel] Center-cropped {a1_seqs.shape[2]} bp → {sv_len} bp")
    print(f"  [kernel] Scoring allele1...")
    s1 = _score_sequences(model, a1_crop, batch_size, device, verbose)
    print(f"  [kernel] Scoring allele2...")
    s2 = _score_sequences(model, a2_crop, batch_size, device, verbose)
    return s2 - s1


def _vep_gkmexplain(model, a1_seqs: np.ndarray, a2_seqs: np.ndarray,
                    sv_len: int, batch_size: int, device: str,
                    verbose: bool) -> np.ndarray:
    """VEP via model.score_variants(method='gkmexplain')."""
    from gkmsvm.backend import to_cpu

    a1_crop = _center_crop(a1_seqs, sv_len)
    a2_crop = _center_crop(a2_seqs, sv_len)

    N = a1_crop.shape[0]
    print(f"  [gkmexplain] Center-cropped {a1_seqs.shape[2]} bp → {sv_len} bp")
    print(f"  [gkmexplain] Computing via score_variants(method='gkmexplain')...")

    logfc = np.zeros(N, dtype=np.float64)
    chunks = range(0, N, batch_size)
    if verbose:
        from tqdm import tqdm
        chunks = tqdm(chunks, desc="gkmexplain VEP",
                      total=(N + batch_size - 1) // batch_size)

    for start in chunks:
        end = min(start + batch_size, N)
        ref_b = model._match_device(a1_crop[start:end])
        alt_b = model._match_device(a2_crop[start:end])
        scores = model.score_variants(ref_b, alt_b, method="gkmexplain",
                                      batch_size=end - start,
                                      verbose=False)
        logfc[start:end] = to_cpu(scores).flatten()

    return logfc


def _run_vep_task(model, df: pd.DataFrame, a1_seqs: np.ndarray,
                  a2_seqs: np.ndarray, sv_len: int,
                  label_filter_fn, sig_label, ctrl_label,
                  effect_col: str | None,
                  vep_method: str, batch_size: int, device: str,
                  verbose: bool) -> dict:
    """Shared VEP evaluation logic for dsQTL and caQTL."""
    df = df.copy()

    mask = label_filter_fn(df).index
    n_total = len(df)
    n_used = len(mask)
    if n_used < n_total:
        print(f"  Pre-filtering: {n_used}/{n_total} variants pass filter")
        a1_sub = a1_seqs[mask.values]
        a2_sub = a2_seqs[mask.values]
        del a1_seqs, a2_seqs
        gc.collect()
    else:
        a1_sub = a1_seqs
        a2_sub = a2_seqs

    if vep_method == "kernel":
        logfc_sub = _vep_kernel(model, a1_sub, a2_sub, sv_len,
                                batch_size, device, verbose)
    elif vep_method == "gkmexplain":
        logfc_sub = _vep_gkmexplain(model, a1_sub, a2_sub, sv_len,
                                    batch_size, device, verbose)
    else:
        raise ValueError(f"Unknown VEP method: {vep_method}")

    logfc = np.full(n_total, np.nan)
    logfc[mask.values] = logfc_sub
    df["llm_logfc"] = logfc

    df_used = label_filter_fn(df)
    sig = df_used[df_used["_label"] == sig_label]
    ctrl = df_used[df_used["_label"] == ctrl_label]
    n_sig = len(sig)
    n_ctrl = len(ctrl)
    print(f"  Evaluated variants: {len(df_used)} ({n_sig} sig, {n_ctrl} ctrl)")

    labels = np.concatenate([np.zeros(n_ctrl), np.ones(n_sig)])
    abs_scores = np.concatenate([np.abs(ctrl["llm_logfc"].values),
                                 np.abs(sig["llm_logfc"].values)])

    auroc = roc_auc_score(labels, abs_scores)
    auprc = average_precision_score(labels, abs_scores)

    result = {
        "n_sig": n_sig,
        "n_ctrl": n_ctrl,
        "auroc": auroc,
        "auprc": auprc,
        "method": vep_method,
    }

    if effect_col and effect_col in sig.columns:
        sig_with_effect = sig.dropna(subset=[effect_col])
        if len(sig_with_effect) > 10:
            r_p, _ = pearsonr(sig_with_effect["llm_logfc"],
                              sig_with_effect[effect_col])
            r_s, _ = spearmanr(sig_with_effect["llm_logfc"],
                               sig_with_effect[effect_col])
            result["pearson_effect"] = r_p
            result["spearman_effect"] = r_s
            print(f"  Effect size: Pearson={r_p:.4f}, Spearman={r_s:.4f} "
                  f"(n={len(sig_with_effect)})")

    print(f"  Classification: AUROC={auroc:.4f}, AUPRC={auprc:.4f}")
    return result, df


def run_task5_dsqtl(model, device: str, batch_size: int, verbose: bool,
                    vep_method: str = "kernel",
                    save_scores_dir: str | None = None,
                    max_variants: int | None = None) -> dict:
    """Run Task 5 dsQTL benchmark (Yoruban LCL dsQTLs, hg19)."""
    tsv_path = os.path.join(
        WORK_DIR, "task_5_variant_effect_prediction/input_data/"
        "yoruban.dsqtls.benchmarking.tsv")

    if not os.path.exists(tsv_path):
        print(f"  SKIP dsQTL: missing {tsv_path}", file=sys.stderr)
        return {}

    sv_len = _get_sv_len(model)
    print(f"\n--- Task 5: Yoruban dsQTLs ({vep_method}) ---")
    df = pd.read_csv(tsv_path, sep="\t")

    h5_data = _load_task5_h5("yoruban.dsqtls.benchmarking.tsv")
    if h5_data is not None:
        print(f"  Loaded sequences from H5 ({len(h5_data['a1_seqs'])} variants)")
        a1_seqs = h5_data["a1_seqs"]
        a2_seqs = h5_data["a2_seqs"]
    else:
        genome_fa = os.path.join(WORK_DIR, "refs/male.hg19.fa")
        if not os.path.exists(genome_fa):
            print(f"  SKIP dsQTL: missing {genome_fa}", file=sys.stderr)
            return {}
        print(f"  Extracting sequences from FASTA ({len(df)} variants)...")
        a1_seqs, a2_seqs = _extract_variant_seqs(
            df, genome_fa,
            chrom_col="var.chrom", pos_col="var.pos",
            a1_col="var.allele1", a2_col="var.allele2")

    if max_variants and max_variants < len(df):
        print(f"  Truncating to {max_variants} variants")
        df = df.iloc[:max_variants].reset_index(drop=True)
        a1_seqs = a1_seqs[:max_variants]
        a2_seqs = a2_seqs[:max_variants]

    def label_filter(d):
        d2 = d[d["var.isused"] == True].copy()
        d2["_label"] = d2["var.label"]
        return d2

    t0 = time.perf_counter()
    result, df_scored = _run_vep_task(
        model, df, a1_seqs, a2_seqs, sv_len,
        label_filter_fn=label_filter, sig_label=1, ctrl_label=-1,
        effect_col="obs.estimate",
        vep_method=vep_method, batch_size=batch_size,
        device=device, verbose=verbose)
    result["task"] = "dsQTL"
    result["elapsed"] = time.perf_counter() - t0
    print(f"  Elapsed: {result['elapsed']:.1f}s")

    if save_scores_dir:
        os.makedirs(save_scores_dir, exist_ok=True)
        scores_df = df_scored[["var.chrom", "var.pos", "var.allele1",
                               "var.allele2", "llm_logfc"]].copy()
        scores_df["allele1_scores"] = 0.0
        scores_df["allele2_scores"] = scores_df["llm_logfc"]
        out_path = os.path.join(save_scores_dir, "dsqtl_scores.tsv")
        scores_df.to_csv(out_path, sep="\t", index=False)
        print(f"  Saved scores to {out_path}")

    return result


def run_task5_caqtl(model, device: str, batch_size: int, verbose: bool,
                    vep_method: str = "kernel",
                    save_scores_dir: str | None = None,
                    max_variants: int | None = None) -> dict:
    """Run Task 5 caQTL benchmark (African caQTLs, hg38)."""
    tsv_path = os.path.join(
        WORK_DIR, "task_5_variant_effect_prediction/input_data/"
        "Afr.CaQTLS.tsv")

    if not os.path.exists(tsv_path):
        print(f"  SKIP caQTL: missing {tsv_path}", file=sys.stderr)
        return {}

    sv_len = _get_sv_len(model)
    print(f"\n--- Task 5: African caQTLs ({vep_method}) ---")
    df = pd.read_csv(tsv_path, sep="\t")

    beta_col = "Beta" if "Beta" in df.columns else "beta"

    mask = (df["IsUsed"] == True) & (df["in_peaks"] == True)
    keep_idx = np.where(mask.values)[0]
    print(f"  Pre-filtering: {len(keep_idx)}/{len(df)} variants "
          f"(IsUsed & in_peaks)")

    if max_variants and max_variants < len(keep_idx):
        keep_idx = keep_idx[:max_variants]
        print(f"  Truncating to {max_variants} variants")

    h5_data = _load_task5_h5("Afr.CaQTLS.tsv", indices=keep_idx)
    if h5_data is not None:
        print(f"  Loaded {len(h5_data['a1_seqs'])} filtered sequences from H5")
        a1_seqs = h5_data["a1_seqs"]
        a2_seqs = h5_data["a2_seqs"]
    else:
        genome_fa = os.path.join(
            WORK_DIR, "refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")
        if not os.path.exists(genome_fa):
            print(f"  SKIP caQTL: missing {genome_fa}", file=sys.stderr)
            return {}
        df_sub = df.iloc[keep_idx].reset_index(drop=True)
        print(f"  Extracting sequences from FASTA ({len(df_sub)} variants)...")
        a1_seqs, a2_seqs = _extract_variant_seqs(
            df_sub, genome_fa,
            chrom_col="chr_hg38", pos_col="pos_hg38",
            a1_col="allele1", a2_col="allele2")

    df_filtered = df.iloc[keep_idx].reset_index(drop=True)
    df_filtered["_label"] = df_filtered["label"]

    def label_filter(d):
        d2 = d.copy()
        d2["_label"] = d2["label"]
        return d2

    t0 = time.perf_counter()
    result, df_scored = _run_vep_task(
        model, df_filtered, a1_seqs, a2_seqs, sv_len,
        label_filter_fn=label_filter, sig_label=1, ctrl_label=0,
        effect_col=beta_col,
        vep_method=vep_method, batch_size=batch_size,
        device=device, verbose=verbose)
    result["task"] = "caQTL"
    result["elapsed"] = time.perf_counter() - t0
    print(f"  Elapsed: {result['elapsed']:.1f}s")

    if save_scores_dir:
        os.makedirs(save_scores_dir, exist_ok=True)
        scores_df = df_scored[["chr_hg38", "pos_hg38", "allele1",
                               "allele2", "llm_logfc"]].copy()
        scores_df["allele1_scores"] = 0.0
        scores_df["allele2_scores"] = scores_df["llm_logfc"]
        out_path = os.path.join(save_scores_dir, "caqtl_scores.tsv")
        scores_df.to_csv(out_path, sep="\t", index=False)
        print(f"  Saved scores to {out_path}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-m", "--model", required=True,
                        help="Model file (.npz or .model.txt[.gz]).")
    parser.add_argument("--task", default="all",
                        choices=["all", "activity", "vep"],
                        help="Which DART-Eval tasks to run.")
    parser.add_argument("--cell-lines", nargs="+", default=CELL_LINES,
                        choices=CELL_LINES,
                        help="Cell lines for Task 4 (default: all 5).")
    parser.add_argument("--vep-method", default="kernel",
                        choices=["kernel", "gkmexplain"],
                        help="VEP scoring method (default: kernel).")
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "cuda"],
                        help="Compute device (default: cpu).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32).")
    parser.add_argument("--sv-chunk-size", default=None,
                        type=lambda s: None if s.lower() == 'none' else int(s),
                        help="SV chunk size (default: None = no chunking).")
    parser.add_argument("--work-dir", default=None,
                        help="DART-Eval data directory (overrides DART_WORK_DIR).")
    parser.add_argument("--output", default=None,
                        help="Save results summary to TSV file.")
    parser.add_argument("--save-scores", default=None, metavar="DIR",
                        help="Save VEP allele scores in DART-Eval TSV format.")
    parser.add_argument("--vep-tasks", default="all",
                        choices=["all", "dsqtl", "caqtl"],
                        help="Which VEP datasets to run (default: all).")
    parser.add_argument("--max-variants", type=int, default=None,
                        help="Max variants to score (for quick validation runs).")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show progress bars.")
    args = parser.parse_args()

    global WORK_DIR
    if args.work_dir:
        WORK_DIR = args.work_dir
    if not WORK_DIR:
        WORK_DIR = os.path.join(os.path.dirname(__file__), "data", "dart-eval")

    # Auto-download from Synapse if H5 files are missing
    tasks_needed = []
    if args.task in ("all", "activity"):
        h5 = os.path.join(WORK_DIR, "task_4_chromatin_activity/data.h5")
        if not os.path.exists(h5):
            tasks_needed.append("task_4")
    if args.task in ("all", "vep"):
        h5 = os.path.join(WORK_DIR,
                          "task_5_variant_effect_prediction/data.h5")
        if not os.path.exists(h5):
            tasks_needed.append("task_5")
    if tasks_needed:
        from dart_download import download_dart_data
        download_dart_data(WORK_DIR, tasks=tuple(tasks_needed))

    print(f"DART-Eval benchmark")
    print(f"  Model: {args.model}")
    print(f"  Device: {args.device}")
    print(f"  Data: {WORK_DIR}")

    model = _load_model(args.model, args.device, args.sv_chunk_size)
    sv_len = _get_sv_len(model)
    print(f"  SVs: {model.num_support_vectors}, "
          f"kernel: {model.kernel_type}, "
          f"L={model.kernel.l}, k={model.kernel.k}, "
          f"SV length: {sv_len} bp")

    results = []

    if args.task in ("all", "activity"):
        for cell_line in args.cell_lines:
            r = run_task4_cell_line(
                model, cell_line, args.device, args.batch_size, args.verbose)
            if r:
                results.append(r)

    if args.task in ("all", "activity"):
        del model
        gc.collect()
        model = _load_model(args.model, args.device, args.sv_chunk_size)

    if args.task in ("all", "vep"):
        if args.vep_tasks in ("all", "dsqtl"):
            r = run_task5_dsqtl(model, args.device, args.batch_size,
                                args.verbose,
                                vep_method=args.vep_method,
                                save_scores_dir=args.save_scores,
                                max_variants=args.max_variants)
            if r:
                results.append(r)
            gc.collect()

        if args.vep_tasks in ("all", "caqtl"):
            r = run_task5_caqtl(model, args.device, args.batch_size,
                                args.verbose,
                                vep_method=args.vep_method,
                                save_scores_dir=args.save_scores,
                                max_variants=args.max_variants)
            if r:
                results.append(r)

    # Summary table
    print("\n" + "=" * 70)
    print("DART-Eval Summary")
    print("=" * 70)

    activity_results = [r for r in results if "cell_line" in r]
    if activity_results:
        print(f"\nTask 4 — Chromatin Activity (peak vs nonpeak classification)")
        print(f"  {'Cell line':<12} {'AUROC':>8} {'AUPRC':>8} "
              f"{'Pearson':>8} {'Spearman':>8}")
        print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for r in activity_results:
            pear = f"{r['pearson_counts']:.4f}" if "pearson_counts" in r else "N/A"
            spear = f"{r['spearman_counts']:.4f}" if "spearman_counts" in r else "N/A"
            print(f"  {r['cell_line']:<12} {r['auroc']:>8.4f} "
                  f"{r['auprc']:>8.4f} {pear:>8} {spear:>8}")

    vep_results = [r for r in results if "task" in r]
    if vep_results:
        print(f"\nTask 5 — Variant Effect Prediction ({args.vep_method})")
        print(f"  {'Task':<12} {'AUROC':>8} {'AUPRC':>8} "
              f"{'Pearson':>8} {'n_sig':>8} {'n_ctrl':>8}")
        print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for r in vep_results:
            pear_key = [k for k in r if k.startswith("pearson")]
            pear = f"{r[pear_key[0]]:.4f}" if pear_key else "N/A"
            print(f"  {r['task']:<12} {r['auroc']:>8.4f} "
                  f"{r['auprc']:>8.4f} {pear:>8} "
                  f"{r['n_sig']:>8} {r['n_ctrl']:>8}")

    if args.output and results:
        out_df = pd.DataFrame(results)
        out_df.to_csv(args.output, sep="\t", index=False)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
