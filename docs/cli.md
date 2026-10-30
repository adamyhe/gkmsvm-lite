# Command-line interface

gkmsvm-lite provides the `gkmsvm` command with subcommands for scoring, training, interpretation, and model conversion.

```bash
gkmsvm <command> [options]
```

Global options: `-v` / `--verbose` enables progress bars.

## Inputs

Every command that reads sequences (`predict`, `train`, `trainsvr`, `ism`, `explain`, `deltasvm`, `score-variants`) works on fixed-length windows, as the ENCODE gkm-SVM and DART-Eval pipelines do:

- **Peaks:** `--genome genome.fa` makes the input a BED or narrowPeak file. Each window is `--length` bp around the summit (`start` + column 10 of a narrowPeak), or around the midpoint when there is no summit. Windows past a chromosome end are dropped, and the count is reported. `score-variants` takes FASTA only.
- **FASTA:** records that share a length are used as they are. Records of different lengths, or longer than an explicit `--length`, are center-cropped to `--length`. A record shorter than `--length` is an error that names it, since a FASTA record has no flanks to extend.
- **`--length` defaults:** the model's SV length for `predict`, `ism`, `explain` and `score-variants`; 300 bp for `train` and `trainsvr`; none for `deltasvm`.
- **`--variable-length`** (`predict`, `train`, `trainsvr`, `ism`, `explain`, `score-variants`, `deltasvm`): uses records at their own lengths, as LS-GKM does, for compatibility. `ism` and `explain` write `[B, 4, L_max]` arrays, zero past each record's length, with the lengths in a `lengths` array; `score-variants` then allows indels. Models whose support vectors differ in length (LS-GKM trained on raw peaks) always load. Their queries are cropped to `--length` unless `--variable-length` is given.

Sequences may contain N, which is scored as A on the forward strand, as LS-GKM does. Other letters (IUPAC codes, ...) stop the command with an error unless `--allow-other-bases` is given. That flag reads them as N, matching LS-GKM, which reads every non-ACGT letter as A (with a warning). Model files whose support vectors contain such letters load with a warning.

The tables below leave out these input flags (`--genome`, `--length`, `--variable-length`, `--allow-other-bases`); every command listed above takes them (`score-variants` all but `--genome`).

Python: `fasta.extract_loci(peaks, genome, in_window=300)` (summit-centered) and `codec.center_crop(seqs, 300)`. `model(x, lengths=...)` or `model([...sequences...])` scores sequences of different lengths.

## predict

Score sequences with a trained model.

```bash
gkmsvm predict -m model.npz -i sequences.fa -o scores.tsv --device cuda
```

| Flag | Description |
|---|---|
| `-m`, `--model` | Model file (`.npz` or `.model.txt[.gz]`) |
| `-i`, `--input` | Input FASTA file |
| `-o`, `--output` | Output TSV (default: stdout) |
| `--batch-size` | Batch size (default: 100) |
| `--device` | `auto`, `cpu`, `cuda`, or `mlx` (default: auto) |

Output format: tab-separated `name\tscore` with a header line.

## train

Train a gkm-SVM classifier (C-SVC).

```bash
gkmsvm train -p positive.fa -n negative.fa -o model.npz -t estimated -l 11 -k 7 -d 3 -C 1.0
```

| Flag | Description |
|---|---|
| `-p`, `--positives` | Positive sequences FASTA |
| `-n`, `--negatives` | Negative sequences FASTA |
| `-o`, `--output` | Output model (`.npz` or `.model.txt`) |
| `-t`, `--kernel-type` | Kernel type: 0-6 or name (default: estimated) |
| `-l` | L-mer length (default: 11) |
| `-k` | K informative positions (default: 7) |
| `-d` | Max mismatches (default: 3) |
| `-C` | Regularization parameter (default: 1.0) |
| `-w` | Positives' C is w·C, for unbalanced sets (as LS-GKM `-w`; default: 1.0) |
| `--solver` | `auto`, `libsvm`, `smo`, or `nystrom` (default: auto: libsvm if the Gram fits in RAM, else smo) |
| `--cache-size` | SMO kernel-column cache size (default: sized from available RAM) |
| `--no-rc` | Disable reverse complement |
| `--gamma` | RBF gamma for `-t 3`, `-t 5` (default: 1.0) |
| `-M` | Weight of the center window, 1-255, for `-t 4`, `-t 5` (default 50) |
| `-H` | Half-life of the window weights in bp, for `-t 4`, `-t 5` (default 50) |
| `--device` | Compute device (default: auto) |

Output format is auto-detected from the file extension: `.npz` for native format, anything else for LS-GKM text format.

## trainsvr

Train a gkm-SVR regressor (epsilon-SVR).

```bash
gkmsvm trainsvr -i sequences.fa --labels values.txt -o model.npz -C 1.0 --epsilon 0.1
```

| Flag | Description |
|---|---|
| `-i`, `--input` | Sequences FASTA |
| `--labels` | Labels file (one value per line, same order as FASTA) |
| `-o`, `--output` | Output model |
| `--epsilon` | SVR epsilon / tube width (default: 0.1) |
| `--solver` | `auto`, `libsvm`, `smo`, or `nystrom` (default: auto: libsvm if the Gram fits in RAM, else smo) |
| `--cache-size` | SMO kernel-column cache size (default: sized from available RAM) |

Accepts the same kernel and device options as `train`.

## explain

Compute GkmExplain hypothetical attribution scores (any kernel type). Outputs two `.npz` files directly usable with [TF-MoDISco](https://github.com/kundajelab/tfmodisco) (`uv pip install modisco`; tfmodisco-lite was merged into it).

```bash
gkmsvm explain -m model.npz -i sequences.fa -o attr.npz -s seqs.npz --device cuda
```

| Flag | Description |
|---|---|
| `-m`, `--model` | Model file |
| `-i`, `--input` | Input FASTA |
| `-o`, `--output` | Attributions `.npz` (hypothetical, `[B,4,L]` float32 as `arr_0`) |
| `-s`, `--sequences-output` | Sequences `.npz` (one-hot, `[B,4,L]` as `arr_0`) |
| `--device` | Compute device (default: auto) |

The two output files feed directly into TF-MoDISco:

```bash
modisco motifs -s seqs.npz -a attr.npz -n 2000 -o modisco_results.h5
```

## ism

In-silico mutagenesis: score all single-base mutations.

```bash
gkmsvm ism -m model.npz -i sequences.fa -o ism_results.npz --device cuda
```

| Flag | Description |
|---|---|
| `-m`, `--model` | Model file |
| `-i`, `--input` | Input FASTA |
| `-o`, `--output` | Output `.npz` (contains `scores`, `one_hot`, `names`) |
| `--device` | Compute device (default: auto) |

## score-variants

Score variant effects: `score(alt) - score(ref)`, or the GkmExplain mutation impact.

```bash
gkmsvm score-variants -m model.npz --ref ref.fa --alt alt.fa -o deltas.tsv
```

| Flag | Description |
|---|---|
| `-m`, `--model` | Model file |
| `--ref` | Reference sequences FASTA |
| `--alt` | Alternate sequences FASTA (same order as ref) |
| `-o`, `--output` | Output TSV (default: stdout) |
| `--method` | `kernel` (score delta, default) or `gkmexplain` (mutation impact, lsgkm `gkmexplain -m 5`) |
| `--device` | Compute device (default: auto) |

## deltasvm

Score sequences with pre-computed DeltaSVM k-mer weights.

```bash
gkmsvm deltasvm -w weights.txt -i sequences.fa -l 11 -o scores.tsv
```

| Flag | Description |
|---|---|
| `-w`, `--weights` | DeltaSVM weight file |
| `-i`, `--input` | Input FASTA |
| `-l` | L-mer window length |
| `-o`, `--output` | Output TSV (default: stdout) |
| `--no-rc` | Disable reverse complement |
| `--device` | Compute device (default: auto) |

## to-deltasvm

Convert a trained SVM model to DeltaSVM k-mer weights.

```bash
gkmsvm to-deltasvm -m model.npz -o weights.txt
```

| Flag | Description |
|---|---|
| `-m`, `--model` | Model file |
| `-o`, `--output` | Output DeltaSVM weight file |
| `--query-norm` | Keep each l-mer's normalization: weights = `gkmpredict` on the l-mer minus bias, as lsgkm deltaSVM weights (implied for RBF models) |
| `--device` | Compute device (default: auto) |

`-t 0/1/2/4` models with l <= 12 convert from the exact l-mer table in seconds; RBF models score all 4^l l-mers through the kernel, which takes minutes for a large model.

## import

Import an external model format to native `.npz`.

```bash
gkmsvm import -i model.txt.gz -f lsgkm -o model.npz
```

| Flag | Description |
|---|---|
| `-i`, `--input` | Input model file |
| `-o`, `--output` | Output `.npz` file |
| `-f`, `--format` | Source format: `lsgkm`, `classic`, or `r_gkmsvm` |
| `--svseq` | `PREFIX_svseq.fa` (classic/R; default: next to the input `PREFIX_svalpha.out`) |
| `-l`, `-k`, `-d` | Kernel of `gkmsvm_kernel` (classic/R; defaults 10, 6, 3) |
| `-t`, `--filter-type` | `gkmsvm_kernel -t` (R `useTgkm`): 0 full, 1 truncated (default), 2 gkm counts |
| `--no-rc` | Without reverse complements (`gkmsvm_kernel -R`) |

Classic and R gkmSVM files hold no kernel and no bias, so the model scores `sum_i alpha_i K(x, sv_i)` with the kernel given here (bias 0); see [models](models.md#classic-gkmsvm-c-and-r-gkmsvm).
