# Benchmarks

Scripts that produce the application note's figures. Each panel has one script, one result file in `results/` and one `make` target. `figures.ipynb` turns the result files into figures and computes nothing, so it runs on a laptop.

```bash
make -C benchmarks help
make -C benchmarks fig1d DEVICE=cuda     # one panel
make -C benchmarks paper                 # every result file
make -C benchmarks quick                 # small settings, writes results/quick/
make -C benchmarks figures               # results/ -> figures/
```

Make skips a target whose result file is newer than its scripts. Use `make -B <target>` to force a rerun.

Panel 1C needs LS-GKM's `gkmexplain`, which some `ls-gkm` conda builds lack. The other targets use the `lsgkm` conda env (`LSGKM_ENV`). See [Building LS-GKM with gkmexplain](#building-ls-gkm-with-gkmexplain).

Timed runs each execute in their own process, so the reported peak memory is that run's alone. LS-GKM's `-T` accepts only 1, 4 or 16; scripts default to the largest the machine has cores for, run gkmsvm-lite's CPU code on the same number of threads, and record both.

## Panels

| Panel | Shows | Target | Script | Result file |
|---|---|---|---|---|
| 1A | Per-variant agreement with LS-GKM (ISM, GkmExplain, deltaSVM.pl) | `fig1a` | `agreement_lsgkm.py` | `agreement_dsqtl.tsv`, `agreement_checks.tsv` |
| 1B | Training time vs N, against LS-GKM at 1 thread and all cores | `fig1b` | `timing_train.py` | `timing_train.tsv`, `timing_train_T1.tsv` |
| 1C | Time and memory to score the 28,309 dsQTL variants (ISM, GkmExplain, deltaSVM table) | `fig1c` | `timing_variants.py` | `timing_variants.tsv` |
| 1D | dsQTL AUPRC with the paper's models | `fig1d` | `dsqtl_auprc.py` | `dsqtl_summed.tsv`, `dsqtl_per_model.tsv` |
| S1 | Models trained with gkmsvm-lite vs the paper's LS-GKM models | `s1` | `dsqtl_auprc.py` | `dsqtl_retrained_*.tsv` |
| S2 | Prediction throughput, sequence length × batch size | `s2` | `throughput_sweep.py` | `throughput.tsv` |
| S3 | Training time, peak memory and held-out accuracy by solver (precomputed Gram, SMO, Nyström, LS-GKM) | `s3` | `training_memory.py` | `training_memory.tsv` |
| S4 | GkmExplain completeness | `s4` | `completeness.py` | `completeness.tsv` |
| S5 | DART-Eval Tasks 4 and 5 with ENCODE models | `s5` | `dart_eval.py` | `dart_task4_*.tsv`, `dart_task5_dsqtl_*.tsv` |

The dsQTL reproduction behind 1A, 1D and S1 is written up in [`docs/validation.md`](../docs/validation.md).

## Result files

Every result file starts with `# key: value` lines giving the git commit, date, host, command, CPU, GPU, package versions and LS-GKM version. Read them with `pd.read_csv(path, sep="\t", comment="#")`, and read the header with `_common.read_provenance(path)`.

Result files and figures are committed, so the figures can be rebuilt without rerunning anything. Quick runs (`results/quick/`, `figures/quick/`) and logs are not committed.

Besides figures, the notebook writes `figures/text_numbers.tsv` (speedups and AUPRCs quoted in the text) and `figures/table_variant_memory.tsv` (peak host and GPU memory for panel 1C's runs).

## Hardware

Targets run on any machine with the conda envs below; each result file's header records the host, CPU and GPU it ran on. Keep each panel's runs on one host, since LS-GKM and gkmsvm-lite times are compared within a panel.

## Running on a server

`run.sh` sets up a machine and runs targets from its existing checkout. It creates the `lsgkm` and `gkmsvm-bench` conda envs if missing, installs the checkout into `gkmsvm-bench` (`[examples,gpu]` extras), and runs `make` there. It doesn't touch git, so check out the commit you want first; result headers record it.

```bash
bash benchmarks/run.sh fig1b s2
bash benchmarks/run.sh --lsgkm-bin <lsgkm>/bin fig1c
```

## Data

Downloaded on first use into `benchmarks/data/` and `examples/data/`, neither of which is committed:

- Beer lab GM12878 sequence sets and dsQTL variants, hg18 (~940 MB), and the GkmExplain paper's models and outputs: `dsqtl_auprc.py`, `agreement_lsgkm.py`
- ENCODE gkm-SVM models: the Makefile, into `data/models/`
- DART-Eval data: Synapse, see `dart_download.py`, into `data/dart-eval/` (override with `DART_WORK_DIR`)

## Building LS-GKM with gkmexplain

`gkmexplain` is in the kundajelab fork of LS-GKM. Build it with a C compiler and `make`:

```bash
git clone https://github.com/kundajelab/lsgkm.git
cd lsgkm/src
make
make install        # copies gkmtrain, gkmpredict and gkmexplain into ../bin
```

If the Makefile has no `install` target, the binaries are in `src/`. Pass the directory that holds them:

```bash
make -C benchmarks fig1c LSGKM_BIN=<path>/lsgkm/bin
bash benchmarks/run.sh --lsgkm-bin <path>/lsgkm/bin fig1c
```

Running `gkmexplain` with no arguments prints its options. `timing_variants.py` passes `-T` only when that list includes it, otherwise LS-GKM runs single-threaded, and the thread count goes into the result file.

The benchmarks only execute LS-GKM binaries as separate processes and never link against them, so LS-GKM's GPL v3 license doesn't apply to gkmsvm-lite (MIT).

## `dev/`

Experiments that don't feed a figure: earlier solver comparisons (`solver_bench.py`, `bench_smo_vs_gram.py`, `bench_tiled_gram_vs_smo.py`), the Nyström landmark sweep, the Nanog replication, and SVR and caQTL training. Run them from the repo root, e.g. `python benchmarks/dev/solver_bench.py`.
