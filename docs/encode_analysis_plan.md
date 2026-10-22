# ENCODE GKM-SVM Analysis Plan

## Context

The ENCODE portal hosts 5,185 gkm-SVM annotations (`annotation_type=gkm-SVM-model`) from the Beer lab, trained with ls-gkm 0.1.1. 4,767 are released. The other 418 are archived ATAC annotations, each replaced by a re-upload. The released annotations hold **6,655 models**:

| Models | Annotations | Assembly | Training positives |
|---|---|---|---|
| 2,879 TF binding (ChIP-seq) | 2,879 | hg38 | top 10k of the experiment's IDR thresholded peaks |
| 1,888 enhancer + 1,888 promoter (DNase-seq, ATAC-seq) | 1,270 DNase and 374 ATAC on hg38; 153 DNase and 91 ATAC on mm10 | hg38, mm10 | top 10k enhancer or promoter peaks (below) |

Every model is `-t 2` (`gkm_esttrunc`) with l=11, k=7, d=3, trained on 300 bp windows against an equal number of GC- and repeat-matched random genomic sequences. The file names spell this out, e.g. `TF_724_hg38_300_top10k_vs_neg1x_r1_avg.model.txt.gz`. Each model is the concatenation of its five cross-validation folds, which is why it has 6.7k–95k SVs, median about 70k. Every positive SV appears at most 4 times, once per training fold it was in.

DNase and ATAC experiments each have **two separate classifiers**, both trained against background, not enhancer against promoter (Oh & Beer 2024, [doi:10.1038/s41467-024-50708-z](https://doi.org/10.1038/s41467-024-50708-z), Methods):
- **Peaks:** MACS2 run on the combined replicates (p < 1e-9), each peak taken as ±150 bp around the summit.
- **Promoter model:** peaks within 2 kb of an annotated TSS. Positives are the top 10k by MACS2 score.
- **Enhancer model:** the remaining distal peaks, minus any called in more than 30% of ENCODE DHS samples (`hg38ubiq30.bed`/`mm10ubiq30.bed` at http://beerlab.org/gkmalign; the `nc30` in the file names). Positives are the top 10k.

The DNase/ATAC models derive from BAMs, not from an ENCODE peak file. Their training positives aren't on the portal; the Beer lab site has them only for the paper's 45 human/mouse pairs.

Every model is paired with an 11-mer weight `.tsv`, its deltaSVM table (235 GB in all; not needed here).

No one has run GkmExplain + TF-MoDISco + FiNeMo across these models. This analysis would:
1. Serve as the primary benchmark for the gkmsvm-lite application note
2. Produce a standalone motif compendium resource (potentially publishable)

The pipeline runs on Stanford Sherlock (SLURM), scaling from a pilot of about 50 models to all 6,655.

### Existing benchmarks of these models

None per model:
- **ENCODE portal:** every model and k-mer file has empty `quality_metrics`, and the annotations attach no documents.
- **Nature Communications paper** (above; main text, SI, Supplementary Data 1 and source data checked): reports only DNase enhancer-model results:
  - human/mouse cross-species AUROC: mean 0.91 for the 45 matched cell/tissue pairs, 0.65 across mismatched ones;
  - primate enhancer AUROCs (Fig. 1F source data);
  - k-mer weight correlations: 0.72 within the same tissue, 0.31 across tissues;
  - a CRISPRi comparison (Supp. Fig. 18).

  It reports nothing for the TF, ATAC or promoter models. Supplementary Data 1 maps model aliases (`DHS_<n>_hg38`) to ENCODE accessions for the 1,270 human and 153 mouse DNase models.

Our own per-model results so far are the DART-Eval runs (benchmarks panel S5) on 7 ENCODE models. Catalogue-wide accuracy numbers would be new.

---

## Pipeline Overview

```
1. Download     ENCODE API → models + peak files + manifest
2. Extract      peaks + hg38/mm10 FASTA → 300 bp one-hot [N, 4, 300]
3. GkmExplain   model + sequences → hypothetical attributions [N, 4, 300] (mode 1)
4. TF-MoDISco   one-hot + attributions → motif patterns
5. FiNeMo       one-hot + attributions + TF-MoDISco patterns → motif hits
6. Annotate     motifs vs JASPAR/HOCOMOCO → match rates
7. Aggregate    per-model results → summary tables + figures
```

---

## Phase 1: Data Acquisition

### 1a. Models

```
GET /search/?type=Annotation&annotation_type=gkm-SVM-model&status=released&format=json&limit=all&frame=embedded
```

Take each annotation's released files with `output_type` `TF binding prediction model`, `enhancer prediction model` or `promoter prediction model` (6,655 `.txt.gz` files, 41.7 GB). Record:
- the annotation, its assay, biosample, target and organism;
- the assembly: hg38 or mm10, from the submitted file name;
- the experiment, from `experimental_input`.

### 1b. Peak sets: every peak, not the 10k training subset

| Assay | Peak file | Experiments | Peaks |
|---|---|---|---|
| TF ChIP-seq | **IDR ranked peaks** (`idr_ranked_peak`, GRCh38) | 2,871 | 272M (median 89k, max 246k) |
| ATAC-seq | **IDR ranked peaks** (GRCh38 or mm10) | 465 | 79M |
| DNase-seq | **`peaks`** narrowPeak, `preferred_default` (ENCODE's main DNase call set; DNase has no IDR) | 1,420 | 199M |

These peak counts are estimates: file sizes divided by the bytes per peak of about 20 downloaded files.

- **TF ChIP-seq:** match the model's experiment, assembly, analysis and biological replicates. 8 TF experiments have no released IDR ranked file; use the IDR thresholded peaks the model derives from (`derived_from`).
- **Ambiguous matches:** 2,569 TF and 217 ATAC experiments have several IDR ranked files from the same analysis, likely the replicate-pair and pseudo-replicate rankings; the metadata doesn't say which one fed the default thresholded peaks. Choose the pooled true-replicate ranking. The counts above take the largest file, so they are close to an upper bound.
- **DNase:** where an experiment has two preferred-default `peaks` files, take one per experiment (the counts above use the larger). 3 DNase/ATAC experiments have no narrowPeak file on the model's assembly; drop them.

Each DNase/ATAC experiment's peak set is attributed by both of its models. The enhancer model then also sees promoter and ubiquitous peaks it never trained on, which allows comparisons across element classes.

Other files ENCODE has, not used:
- TF ChIP-seq: conservative, optimal and pseudoreplicated IDR thresholded peaks, all thresholded like the default; and `peaks and background as input for IDR` (MACS2 calls, 300k per file, mostly background, 1,766 experiments).
- ATAC: `pseudoreplicated peaks`, the thresholded default.
- DNase: hotspots, FDR cut rate and footprints, which are other kinds of output, not looser peak calls.

### 1c. Reference genomes

GRCh38 and **mm10** FASTA + faidx. The mouse models are mm10, not GRCm39.

### 1d. Manifest and directory structure

One row per model: `model_accession | annotation | model_type (tf/enhancer/promoter) | assay | assembly | experiment | peaks_accession | n_peaks`.

```
data/encode_gkmsvm/
├── manifest.tsv
├── models/          # {model}.model.txt.gz
├── peaks/           # {peaks}.bed.gz
├── sequences/       # {experiment}.seqs.npz   [N, 4, 300], shared by an experiment's models
├── attributions/    # {model}.attr.npz        [N, 4, 300]
├── modisco/         # {model}/                (modisco h5 + report)
├── finemo/          # {model}/                (hits)
└── summary/         # aggregated results
```

---

## Phase 2: Sequence Extraction

For each peak file:

1. Take 300 bp centered on each peak's summit (narrowPeak column 10). Use the midpoint when there is no summit.
2. Collapse peaks whose windows coincide. Relaxed narrowPeak files can list several summits in one region.
3. Extract with `gkmsvm.fasta.extract_loci()` and one-hot encode.
4. Drop windows with >10% N, and windows running off a chromosome end.
5. Save `{experiment}.seqs.npz`.

Windows are 300 bp to match the models' training sequences and SVs.

---

## Phase 3: GkmExplain

```python
model = load_lsgkm_model(model_path, device="cuda")
x = np.load(seqs_path)["arr_0"]
attr = gkmexplain(model, x, mode="hypothetical", batch_size=1000)
np.savez_compressed(attr_path, arr_0=attr.astype(np.float32))
model.cpu()
```

All 6,655 models are `-t 2` with l=11, so GkmExplain takes the l-mer path (`explain._explain_lmer`). Its cost is per query window, not per SV, and the 4^11 V table is cached per model.

Measured: 1,000 300 bp sequences in 3.0 s on an M4 (MLX), for a median-size TF model (ENCFF616SOL, 70.7k SVs). The full run is about 828M (model, sequence) pairs: 272M TF, plus 158M ATAC and 398M DNase, the latter two attributed by both models. At that rate it would take about 690 M4-hours. Time a Sherlock GPU on the pilot before sizing the array job.

---

## Storage

Both arrays are float32, saved with `np.savez_compressed`. Bytes per 300 bp sequence, measured on 1,000 sequences of one ENCODE TF model:

| Array | Raw | Compressed npz |
|---|---|---|
| Hypothetical attributions | 4.8 KB | 4.5 KB |
| One-hot | 4.8 KB | 0.25 KB |

| Peak set | Sequences | Attributions | One-hot | Total |
|---|---|---|---|---|
| TF ChIP-seq IDR ranked | 272M | 1 per model | 1 per model | 1.29 TB |
| ATAC IDR ranked | 79M | 2 per sequence (enhancer, promoter) | 1 per experiment | 0.73 TB |
| DNase `peaks` | 199M | 2 per sequence | 1 per experiment | 1.84 TB |
| **Total** | **550M** | | | **~3.9 TB** (6.6 TB uncompressed) |

- **What dominates:** attributions are nearly all of the total; compression saves 6% on them and 95% on the one-hot arrays. Storing one-hot as int8 would save only about 0.25 TB more, if `modisco` and FiNeMo accept it.
- **Not included:** the models (42 GB), genomes (about 6 GB), and the TF-MoDISco/FiNeMo outputs.
- **FiNeMo's preprocessing:** check whether it writes its own copy of sequences and attributions. If it does, delete that copy after each model.

---

## Phase 4: TF-MoDISco

For each model:

```bash
modisco motifs -s sequences/${EXPERIMENT}.seqs.npz \
    -a attributions/${MODEL}.attr.npz \
    -n 50000 -w 300 \
    -o modisco/${MODEL}/results.h5
```

Programmatically, the `modisco` package (kundajelab/tfmodisco) still imports as `modiscolite`, and `TFMoDISco()` takes `[N, L, 4]`, so transpose our `[N, 4, L]` arrays (the CLI does this itself):

```python
import modiscolite

pos_patterns, neg_patterns = modiscolite.tfmodisco.TFMoDISco(
    hypothetical_contribs=attr.transpose(0, 2, 1),
    one_hot=seqs.transpose(0, 2, 1),
    max_seqlets_per_metacluster=50000,
)
```

**Runtime**: about 30–90 min per model (CPU-bound, about 8 GB RAM). Models run independently, so they parallelize fully.

**Output**: H5 file with motif patterns (PPMs, CWMs, seqlet assignments).

---

## Phase 5: FiNeMo

FiNeMo (kundajelab/finemo_gpu) calls motif instances by fitting the TF-MoDISco patterns to each sequence's attributions. It reads the same one-hot and hypothetical-attribution npz files:

```bash
finemo extract-regions-modisco-fmt -s sequences/${EXPERIMENT}.seqs.npz \
    -a attributions/${MODEL}.attr.npz -w 300 -o finemo/${MODEL}/regions.npz
finemo call-hits -r finemo/${MODEL}/regions.npz -m modisco/${MODEL}/results.h5 \
    -o finemo/${MODEL}/
```

Check these commands against the installed FiNeMo version. It runs on GPU, so schedule it with GkmExplain.

---

## Phase 6: Motif Annotation

Use `modisco report` to generate per-model HTML reports with motif logos, seqlet counts, and Tomtom matches against JASPAR/HOCOMOCO:

```bash
modisco report -i modisco/${MODEL}/results.h5 \
    -o modisco/${MODEL}/report/ \
    -s sequences/${EXPERIMENT}.seqs.npz \
    -a attributions/${MODEL}.attr.npz \
    -m JASPAR2024_CORE_vertebrates_non-redundant_pfms.meme
```

This produces per-pattern logos (PPM + CWM), Tomtom alignment to known motifs, and seqlet statistics — all in a browsable HTML report.

**Validation metric for app note**: "X% of ChIP-seq models recover the target TF motif in the top 3 patterns (Tomtom q < 0.05)"

---

## Phase 7: Motif Compendium

Use **motifcompendium** to aggregate per-model TF-MoDISco results into a unified, queryable compendium:

- Combines all per-model motif patterns into a single database
- Cross-references motifs across models, cell types, assays and element classes (enhancer vs promoter)
- Identifies shared vs. cell-type-specific motifs
- Generates summary visualizations and searchable tables

### Figures for application note
1. **Scale demonstration**: Runtime vs. number of SVs, number of peaks
2. **Motif recovery**: Fraction of ChIP-seq models recovering target TF (bar chart by cell line)
3. **Example motifs**: Logo plots for well-known TFs (CTCF, GATA1, etc.) from gkmexplain vs. known PWMs
4. **Assay comparison**: Motifs from ATAC-seq/DNase-seq models vs. ChIP-seq for same cell type
5. **Element classes**: Enhancer vs. promoter models of the same experiment on the same peaks
6. **Compendium overview**: Heatmap of motif presence across cell types and assays

---

## Execution Strategy

### SLURM job array design

One task per model (6,655). An experiment's sequences are extracted once, before its models run.

```bash
#SBATCH --array=1-6655
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

read MODEL EXPERIMENT < <(sed -n "${SLURM_ARRAY_TASK_ID}p" models.txt)
python -c "
import numpy as np
from gkmsvm import load_lsgkm_model, gkmexplain
m = load_lsgkm_model('models/${MODEL}.model.txt.gz', device='cuda')
x = np.load('sequences/${EXPERIMENT}.seqs.npz')['arr_0']
a = gkmexplain(m, x, mode='hypothetical', batch_size=1000)
np.savez_compressed('attributions/${MODEL}.attr.npz', arr_0=a.astype(np.float32))
"

modisco motifs -s sequences/${EXPERIMENT}.seqs.npz \
    -a attributions/${MODEL}.attr.npz -n 50000 -w 300 \
    -o modisco/${MODEL}/results.h5

finemo extract-regions-modisco-fmt -s sequences/${EXPERIMENT}.seqs.npz \
    -a attributions/${MODEL}.attr.npz -w 300 -o finemo/${MODEL}/regions.npz
finemo call-hits -r finemo/${MODEL}/regions.npz -m modisco/${MODEL}/results.h5 \
    -o finemo/${MODEL}/
```

Set `--time` from the pilot: the largest peak sets (TF up to 246k, DNase several hundred thousand) take several times the median.

### Checkpointing

- Skip models whose output files already exist (simple file-existence check at top of job script)
- Each model is independent — failed jobs can be resubmitted without rerunning completed ones

### Scaling

| Scale | Models | Storage |
|---|---|---|
| Pilot | 50 | about 30 GB |
| App note | 200 | about 120 GB |
| Full | 6,655 | about 3.9 TB |

GPU-hours: from the pilot.

---

## Deliverables

1. **Pipeline scripts** (in `analysis/encode_gkmsvm/`):
   - `01_download.py` — ENCODE API query, peak-file selection (Phase 1b), bulk download, manifest
   - `02_extract_sequences.py` — peaks → 300 bp one-hot, one file per experiment
   - `03_run_gkmexplain.sh` — SLURM array job template
   - `04_run_modisco.sh` — SLURM array job template
   - `05_run_finemo.sh` — SLURM array job template
   - `06_build_compendium.py` — motifcompendium aggregation
   - `07_figures.py` — Application note figures

2. **Output data**: Per-model attributions, modisco H5 + HTML reports, FiNeMo hits, unified motifcompendium

3. **Application note figures**: Runtime benchmarks, motif recovery rates, example logos, compendium heatmap

---

## Verification

- **Pilot test**: Run full pipeline on 5 models (2 ChIP-seq, 2 DNase, 1 ATAC) on single GPU
- **Sanity check**: CTCF ChIP-seq model should recover CTCF motif as top pattern
- **Compare deltaSVM**: For models with precomputed k-mer weights, verify gkmexplain attributions are consistent with deltaSVM variant effect predictions
- **Runtime profiling**: Log per-model wall-clock for explain, modisco and FiNeMo; size the full array from it
