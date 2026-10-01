# ENCODE GKM-SVM Analysis Plan

## Context

The ENCODE portal hosts **5,185 gkm-SVM models** (2,879 ChIP-seq, 1,423 DNase-seq, 883 ATAC-seq) trained by the Beer lab using ls-gkm 0.1.1. These are LS-GKM `.txt.gz` models on GRCh38, each paired with a precomputed k-mer weight `.tsv` (deltaSVM table). Each model links to its source IDR-thresholded peak BED files via `derived_from`.

No one has run gkmexplain + TF-MoDISco at scale across these models. This analysis would:
1. Serve as the primary benchmark for the gkmsvm-lite application note
2. Produce a standalone motif compendium resource (potentially publishable)

The pipeline runs on a multi-GPU node and/or Stanford Sherlock (SLURM). Design must scale from a small pilot (~50 models) to the full 5,185.

---

## Pipeline Overview

```
1. Download     ENCODE API → model .txt.gz + peak BED + metadata JSON
2. Extract      BED + hg38 FASTA → one-hot sequences [N, 4, L]
3. GkmExplain   model + sequences → attributions [N, 4, L]
4. TF-MoDISco   attributions + sequences → motif patterns
5. Annotate     motifs vs JASPAR/HOCOMOCO → match rates
6. Aggregate    per-model results → summary tables + figures
```

---

## Phase 1: Data Acquisition

### 1a. Model + metadata download

Query the ENCODE REST API for all `annotation_type=gkm-SVM-model` files:

```
GET /search/?type=Annotation&annotation_type=gkm-SVM-model&format=json&limit=all
```

For each annotation, extract:
- Model file href (`.txt.gz`, `output_type: "TF binding prediction model"`)
- K-mer weights href (`.tsv`, `output_type: "kmer weights"`)
- `derived_from` → source experiment accession → peak BED file href
- Biosample, assay, target TF (for ChIP-seq), organism, assembly

Store metadata in a single `encode_gkmsvm_manifest.tsv` with columns:
`accession | assay | biosample | target | organism | model_href | peaks_href | kmer_href`

### 1b. Reference genome

Download GRCh38 (hg38) and GRCm39 (mm39) FASTA + faidx from ENCODE or UCSC.

### 1c. Directory structure

```
data/encode_gkmsvm/
├── manifest.tsv
├── models/          # {accession}.model.txt.gz
├── peaks/           # {accession}.bed.gz
├── kmers/           # {accession}.kmers.tsv  (optional, for validation)
├── sequences/       # {accession}.seqs.npz   [N, 4, L]
├── attributions/    # {accession}.attr.npz   [N, 4, L]
├── modisco/         # {accession}/           (modisco h5 + reports)
└── summary/         # aggregated results
```

---

## Phase 2: Sequence Extraction

For each model's peak set:

1. Load BED with pandas (chrom, start, end)
2. Center-crop or resize peaks to a fixed window (e.g., 200 bp, matching typical gkm-SVM training)
3. Extract sequences from hg38 using `pyfaidx.Fasta`
4. One-hot encode, save as `{accession}.seqs.npz`

Use `gkmsvm.fasta.extract_loci()` or a vectorized equivalent for speed. Filter out sequences with >10% N bases.

**Window size**: Parse from model header. LS-GKM models typically use l=11, and training sequences were likely 200–600 bp. Extract 200 bp centered on peak summit (or midpoint if no summit column).

**Scale**: Most ENCODE experiments have 10K–100K peaks. At 200 bp, this is 8–80 MB per experiment in one-hot form.

---

## Phase 3: GkmExplain

For each model:

```python
model = load_lsgkm_model(model_path, device="cuda")
x = np.load(seqs_path)["arr_0"]
attr = gkmexplain(model, x, mode=1, batch_size=50, verbose=True)
np.savez(attr_path, arr_0=attr)
np.savez(seqs_out_path, arr_0=x)  # if not already saved
model.cpu()  # free GPU memory
```

**Existing CLI**: `gkmsvm explain -m model.txt.gz -i peaks.fa -o attr.npz -s seqs.npz --device cuda`
This already does the full load → encode → explain → save pipeline for a single model.

**Memory**: The fused reduction kernel accumulates directly into a `[B, 4, L]` result array (~50 × 4 × 200 × 8 = 320 KB) per chunk — no per-SV intermediate. GPU memory is dominated by packed SV windows and the weight tables, well under 1 GB for typical models.

**Runtime estimate**: For a model with ~20K SVs and ~50K peaks at batch_size=50:
- ~1000 batches × ~0.5s per batch ≈ 8 min per model on GPU
- 5,185 models × 8 min ≈ 690 GPU-hours total
- With 4 GPUs on Sherlock: ~7 days wall-clock
- With a pilot of 50 models: ~7 hours

---

## Phase 4: TF-MoDISco

For each model's attributions:

```bash
modisco motifs -s seqs.npz -a attr.npz -n 50000 -o modisco_results.h5
modisco report -i modisco_results.h5 -o report/ -s seqs.npz -a attr.npz
```

Or programmatically via `modiscolite`:

```python
import modiscolite
pos_patterns, neg_patterns = modiscolite.tfmodisco.TFMoDISco(
    hypothetical_contribs=attr, one_hot=seqs, max_seqlets_per_metacluster=50000
)
```

**Runtime**: ~30–90 min per model (CPU-bound, ~8 GB RAM). Embarrassingly parallel across models.

**Output**: H5 file with motif patterns (PPMs, CWMs, seqlet assignments).

---

## Phase 5: Motif Annotation

Use `modisco report` to generate per-model HTML reports with motif logos, seqlet counts, and Tomtom matches against JASPAR/HOCOMOCO:

```bash
modisco report -i modisco/${ACCESSION}/results.h5 \
    -o modisco/${ACCESSION}/report/ \
    -s attributions/${ACCESSION}.seqs.npz \
    -a attributions/${ACCESSION}.attr.npz \
    -m JASPAR2024_CORE_vertebrates_non-redundant_pfms.meme
```

This produces per-pattern logos (PPM + CWM), Tomtom alignment to known motifs, and seqlet statistics — all in a browsable HTML report.

**Validation metric for app note**: "X% of ChIP-seq models recover the target TF motif in the top 3 patterns (Tomtom q < 0.05)"

---

## Phase 6: Motif Compendium

Use **motifcompendium** to aggregate per-model TF-MoDISco results into a unified, queryable compendium:

- Combines all per-model motif patterns into a single database
- Cross-references motifs across models, cell types, and assays
- Identifies shared vs. cell-type-specific motifs
- Generates summary visualizations and searchable tables

### Figures for application note
1. **Scale demonstration**: Runtime vs. number of SVs, number of peaks
2. **Motif recovery**: Fraction of ChIP-seq models recovering target TF (bar chart by cell line)
3. **Example motifs**: Logo plots for well-known TFs (CTCF, GATA1, etc.) from gkmexplain vs. known PWMs
4. **Assay comparison**: Motifs from ATAC-seq/DNase-seq models vs. ChIP-seq for same cell type
5. **Compendium overview**: Heatmap of motif presence across cell types and assays

---

## Execution Strategy

### SLURM job array design

```bash
# One job per model
#SBATCH --array=1-5185
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=2:00:00

ACCESSION=$(sed -n "${SLURM_ARRAY_TASK_ID}p" manifest.txt)
gkmsvm explain -m models/${ACCESSION}.model.txt.gz \
    -i sequences/${ACCESSION}.fa \
    -o attributions/${ACCESSION}.attr.npz \
    -s attributions/${ACCESSION}.seqs.npz \
    --device cuda

modisco motifs -s attributions/${ACCESSION}.seqs.npz \
    -a attributions/${ACCESSION}.attr.npz \
    -n 50000 \
    -o modisco/${ACCESSION}/results.h5

modisco report -i modisco/${ACCESSION}/results.h5 \
    -o modisco/${ACCESSION}/report/ \
    -s attributions/${ACCESSION}.seqs.npz \
    -a attributions/${ACCESSION}.attr.npz
```

### Checkpointing

- Skip models whose output files already exist (simple file-existence check at top of job script)
- Each model is independent — failed jobs can be resubmitted without rerunning completed ones

### Scaling

| Scale | Models | GPU-hours (explain) | CPU-hours (modisco) | Wall-clock (4 GPU) |
|-------|--------|--------------------|--------------------|-------------------|
| Pilot | 50 | ~7 | ~50 | ~2 hours |
| App note | 200 | ~27 | ~200 | ~7 hours |
| Medium | 1000 | ~130 | ~1000 | ~1.5 days |
| Full | 5185 | ~690 | ~5000 | ~7 days |

---

## Deliverables

1. **Pipeline scripts** (in `analysis/encode_gkmsvm/`):
   - `01_download.py` — ENCODE API query + bulk download
   - `02_extract_sequences.py` — BED → one-hot sequences
   - `03_run_gkmexplain.sh` — SLURM array job template
   - `04_run_modisco.sh` — SLURM array job template
   - `05_build_compendium.py` — motifcompendium aggregation
   - `06_figures.py` — Application note figures

2. **Output data**: Per-model modisco H5 + HTML reports, unified motifcompendium

3. **Application note figures**: Runtime benchmarks, motif recovery rates, example logos, compendium heatmap

---

## Verification

- **Pilot test**: Run full pipeline on 5 models (2 ChIP-seq, 2 DNase, 1 ATAC) on single GPU
- **Sanity check**: CTCF ChIP-seq model should recover CTCF motif as top pattern
- **Compare deltaSVM**: For models with precomputed k-mer weights, verify gkmexplain attributions are consistent with deltaSVM variant effect predictions
- **Runtime profiling**: Log per-model wall-clock for explain + modisco, verify scaling estimates
