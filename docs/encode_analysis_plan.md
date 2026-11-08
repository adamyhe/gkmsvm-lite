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
5. FiNeMo       one-hot + attributions + TF-MoDISco patterns → motif hits (BED) + HTML report
6. Annotate     motifs vs JASPAR/HOCOMOCO → match rates
7. Aggregate    per-model results → summary tables + figures
8. Tracks       attributions × one-hot → per-model bigWig, viewed as dynseq tracks
```

Every phase runs on every peak: no cap on peaks, and none on seqlets beyond `-n 2000000`.

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

TF has the largest total only because it has the most experiments. Per experiment it has the fewest peaks: about 95k on average, against about 140k for DNase and 170k for ATAC. Within each assay, sizes vary several-fold between experiments. Before sizing jobs, get each file's peak count from the ENCODE API and record it in the manifest's `n_peaks` column. Job limits and the pilot sample then come from the per-experiment distribution, not these totals.

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
├── windows/         # {experiment}.windows.bed.gz   the 300 bp windows, in sequences/ order
├── finemo/          # {model}/                (hits BED + HTML report)
├── tracks/          # {model}.importance.bw   hyp × one-hot, dynseq
└── summary/         # aggregated results
```

---

## Phase 2: Sequence Extraction

For each peak file:

1. Take 300 bp centered on each peak's summit (narrowPeak column 10). Use the midpoint when there is no summit.
2. Collapse peaks whose windows coincide. Relaxed narrowPeak files can list several summits in one region.
3. Extract with `gkmsvm.fasta.extract_loci()` and one-hot encode.
4. Drop windows with >10% N, and windows running off a chromosome end.
5. Save `{experiment}.seqs.npz`, and the kept windows in the same order as `{experiment}.windows.bed.gz`. FiNeMo's genomic coordinates, the bigWig tracks and the published data all use this BED.

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
- **Not included:** the models (42 GB), genomes (about 6 GB), the TF-MoDISco/FiNeMo outputs (size them from the pilot), and the importance bigWigs (at most 1.5 TB, Phase 8).
- **Where:** about 3.9 TB has to outlive the run (until TF-MoDISco, FiNeMo and the tracks finish, and until the attributions are archived), so keep it on Oak, not `$SCRATCH`, which purges after 90 days.
- **FiNeMo's preprocessing:** check whether it writes its own copy of sequences and attributions. If it does, delete that copy after each model.

---

## Phase 4: TF-MoDISco

For each model:

```bash
modisco motifs -s sequences/${EXPERIMENT}.seqs.npz \
    -a attributions/${MODEL}.attr.npz \
    -n 2000000 -w 300 \
    -o modisco/${MODEL}/results.h5
```

`-n` (`max_seqlets_per_metacluster`, default 20,000) is set above any experiment's seqlet count, so it doesn't subsample. `max_passing_windows_frac` (default 0.2) still limits which windows give seqlets. It acts as a significance threshold, so keep it.

Programmatically, the `modisco` package (kundajelab/tfmodisco) still imports as `modiscolite`, and `TFMoDISco()` takes `[N, L, 4]`, so transpose our `[N, 4, L]` arrays (the CLI does this itself):

```python
import modiscolite

pos_patterns, neg_patterns = modiscolite.tfmodisco.TFMoDISco(
    hypothetical_contribs=attr.transpose(0, 2, 1),
    one_hot=seqs.transpose(0, 2, 1),
    max_seqlets_per_metacluster=2_000_000,
)
```

**Runtime**: from the pilot. Without the subsampling cap, clustering is the expensive phase. The coarse affinity compares every pair of seqlets in a metacluster, so time grows about quadratically with seqlets. Memory stays moderate because only the nearest-neighbour graph is kept. Time one large experiment per assay in the pilot. Models run independently, so they parallelize fully.

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

Then `finemo report` writes the HTML quality report (hit statistics, and patterns compared against TF-MoDISco's seqlets). The hits and the report are the outputs to publish:
- **Hits:** one BED per model, `{model}.hits.bed.gz`, in genomic coordinates. Give FiNeMo the windows (`windows/{experiment}.windows.bed.gz`) as its peaks so hits map back to the genome. If it writes only TSV, convert it.
- **Report:** `finemo/{model}/report/`.

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

## Phase 8: Contribution Tracks

These are the main resource to distribute: per-base contribution scores that load into genome browsers.

For each model:
1. Multiply the hypothetical attributions by the one-hot (GkmExplain's `"importance"`, summed over bases) to get one score per position.
2. Write them to `tracks/{model}.importance.bw` with [figwig](https://github.com/jmschrei/figwig) (`pip install figwig`; numpy + numba, multithreaded). Pass the windows from `windows/{experiment}.windows.bed.gz` and the assembly's chrom sizes to `write_bigwig(path, chrom_sizes, chroms, starts, scores[:, None, :])`. For peak sets too large to hold in memory, use `BigWigWriter` in batches.
3. figwig rejects overlapping items, so merge windows whose summits are closer than 300 bp before writing, averaging the scores where they overlap. Write the merged runs as windows of their own widths, or as intervals through `BigWigWriter`. Record the rule in the track description.

figwig's own `.fw` format holds many tracks over one genome, each with JSON attributes. It could carry all of a genome's models in one file for programmatic access. Browsers need bigWig, though (`figwig fw2bw` converts), so the published tracks stay bigWig.

Displayed as dynseq (letters scaled by score): `logo on` in a UCSC track hub's trackDb, or the WashU browser's dynseq tracks. DNase/ATAC experiments get two tracks each, enhancer and promoter model, on the same windows.

**Storage.** Measured with figwig 0.3.0 (default compression, 10 zoom levels) on 2,000 hypothetical attributions from ENCFF579AOX, written as 300 bp windows 1 kb apart:

| Format | Bytes per base | Median TF model (89k peaks, 26.7 Mbp) | All models |
|---|---|---|---|
| bigWig, importance (1 track) | 6.0 | 160 MB | 1.5 TB |
| figwig, importance (1 track) | 3.7 | 100 MB | 0.9 TB |
| bigWig, hypothetical (4 files, one per base) | 23.8 | 640 MB | 5.9 TB |
| figwig, hypothetical (4 tracks, 1 file) | 14.9 | 400 MB | 3.7 TB |
| npz, hypothetical (reference, as in Storage) | 14.9 | | 3.7 TB |

- **Totals:** these are upper bounds. They cover 248 Gbp of model-bases: TF 82 Gbp, plus DNase 60 Gbp and ATAC 24 Gbp, each counted twice for the enhancer and promoter models. Merging overlapping windows lowers that by however much the peaks overlap, which the pilot measures.
- **Zoom levels:** about 1.9 B/base of the bigWig figure (about 30%). Without them the importance bigWig is 4.1 B/base. Browsers need them to show zoomed-out views, so keep them. Raising zlib to level 9 saves nothing.
- **What to publish:** the importance bigWigs (1.5 TB at most) are the browser resource. Hypothetical attributions in figwig take the same space as the npz files and can be read by window, which makes them the better format for the archive. As bigWigs they would be 1.6x larger and four files per model.

---

## Data Release

| Output | Where | Notes |
|---|---|---|
| Models, peaks, genomes | Not republished | The manifest lists ENCODE accessions and md5s |
| Pipeline code | GitHub (`analysis/encode_gkmsvm/`), archived to Zenodo from a release | DOI for the version used |
| Manifest, window BEDs | Zenodo | Windows let anyone rebuild the one-hot arrays from the genome |
| **Contribution bigWigs** (hyp × one-hot) | Host serving HTTP range requests, plus a UCSC/WashU track hub | Main resource. Choose the host before the pilot (S3/GCS bucket, lab web server, or SDR if it serves ranges); don't assume Zenodo works for browser loading |
| **FiNeMo hits BED + HTML report**, per model | Zenodo (split across records if needed) | Size from the pilot |
| TF-MoDISco H5 + reports, motif compendium | Zenodo | |
| Hypothetical attributions (~3.7 TB as figwig) | Stanford Digital Repository or equivalent, archive only | Optional: recomputable from the models, code and windows. figwig (4 tracks per model) is the same size as npz and can be read by window |
| One-hot sequences | Not published | Rebuilt from the windows + genome |

Check current Zenodo record limits (about 50 GB by default) and SDR's terms before the data-availability statement relies on them.

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
    -a attributions/${MODEL}.attr.npz -n 2000000 -w 300 \
    -o modisco/${MODEL}/results.h5

finemo extract-regions-modisco-fmt -s sequences/${EXPERIMENT}.seqs.npz \
    -a attributions/${MODEL}.attr.npz -w 300 -o finemo/${MODEL}/regions.npz
finemo call-hits -r finemo/${MODEL}/regions.npz -m modisco/${MODEL}/results.h5 \
    -p windows/${EXPERIMENT}.windows.bed.gz -o finemo/${MODEL}/
finemo report ...   # HTML quality report; flags per the installed version

python 08_make_tracks.py ${MODEL} ${EXPERIMENT}   # tracks/${MODEL}.importance.bw
```

Set `--time` and `--mem` from the pilot and the per-experiment peak counts. The largest peak sets (TF up to 246k, DNase several hundred thousand) take several times the median for GkmExplain and FiNeMo, and more than that for TF-MoDISco, whose clustering grows about quadratically. TF-MoDISco is CPU-only, so it may be better as a separate CPU array than holding a GPU.

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
   - `08_make_tracks.py` — attributions × one-hot → per-model bigWig; track hub files

2. **Output data** (see Data Release): per-model contribution bigWigs and track hub, FiNeMo hit BEDs + HTML reports, TF-MoDISco H5 + reports, the motif compendium, and an archive of the hypothetical attributions

3. **Application note figures**: Runtime benchmarks, motif recovery rates, example logos, compendium heatmap

---

## Verification

- **Pilot test**: Run full pipeline on 5 models (2 ChIP-seq, 2 DNase, 1 ATAC) on single GPU, then the pilot's 50 models chosen across the per-experiment size distribution, including the largest experiment of each assay
- **Tracks**: a pilot bigWig loads as a dynseq track in UCSC (from the chosen host) and WashU, and its scores match the attribution npz at a few windows
- **Sanity check**: CTCF ChIP-seq model should recover CTCF motif as top pattern
- **Compare deltaSVM**: For models with precomputed k-mer weights, verify gkmexplain attributions are consistent with deltaSVM variant effect predictions
- **Runtime profiling**: Log per-model wall-clock for explain, modisco and FiNeMo; size the full array from it
