# Command-line interface

gkmsvm-lite provides the `gkmsvm` command with subcommands for scoring, training, interpretation, and model conversion.

```bash
gkmsvm <command> [options]
```

Global options: `-v` / `--verbose` enables progress bars.

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
| `-t`, `--kernel-type` | Kernel type: 0-5 or name (default: estimated) |
| `-l` | L-mer length (default: 11) |
| `-k` | K informative positions (default: 7) |
| `-d` | Max mismatches (default: 3) |
| `-C` | Regularization parameter (default: 1.0) |
| `--solver` | `auto`, `smo`, or `libsvm` (default: auto) |
| `--cache-size` | SMO cache columns (default: 256) |
| `--no-rc` | Disable reverse complement |
| `--gamma` | RBF gamma for `-t 3`, `-t 5` (default: 1.0) |
| `-M` | Center-weight window for `-t 4`, `-t 5` |
| `-H` | Center-weight decay for `-t 4`, `-t 5` |
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

Accepts the same kernel and device options as `train`. SVR currently uses the precomputed Gram solver only (`solver="smo"` is not yet supported for SVR), so training dataset size is bounded by available memory.

## explain

Compute GkmExplain attribution scores. Outputs two `.npz` files directly usable with [TF-MoDISco](https://github.com/jmschrei/tfmodisco-lite).

```bash
gkmsvm explain -m model.npz -i sequences.fa -o attr.npz -s seqs.npz --device cuda
```

| Flag | Description |
|---|---|
| `-m`, `--model` | Model file |
| `-i`, `--input` | Input FASTA |
| `-o`, `--output` | Attributions `.npz` (hypothetical, `[B,4,L]` as `arr_0`) |
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

Score variant effects: `score(alt) - score(ref)`.

```bash
gkmsvm score-variants -m model.npz --ref ref.fa --alt alt.fa -o deltas.tsv
```

| Flag | Description |
|---|---|
| `-m`, `--model` | Model file |
| `--ref` | Reference sequences FASTA |
| `--alt` | Alternate sequences FASTA (same order as ref) |
| `-o`, `--output` | Output TSV (default: stdout) |
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
