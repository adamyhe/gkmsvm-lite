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

Per-host settings go in `benchmarks/local.mk`, which make reads if it exists and git ignores:

```make
LSGKM_BIN = /path/to/lsgkm/bin
```

Panel 1C needs LS-GKM's `gkmexplain`, which the Beer lab's LS-GKM and the bioconda `ls-gkm` package don't have; the kundajelab fork does. Every script takes each LS-GKM binary from the `lsgkm` conda env (`LSGKM_ENV`) when the env has it, and from `LSGKM_BIN` otherwise: the kundajelab source build's `gkmpredict` has no `-T`, so it would run single-threaded. 1C times both implementations' CPU code at `THREADS` and at 1 thread (LS-GKM binaries without `-T` at 1 only), so every LS-GKM bar has a gkmsvm-lite CPU bar at its thread count. LS-GKM's `-T` splits each sequence's kernel sum across threads, so it slows down the deltaSVM build (524,800 10-mers): on one 16-core server, 3,773 s at `-T 16` against 684 s at `-T 1`. See [Building LS-GKM with gkmexplain](#building-ls-gkm-with-gkmexplain).

Timed runs each execute in their own process, so the reported peak memory is that run's alone. Scripts that time whole processes (`timing_variants.py`, `training_memory.py`) first run an untimed process that compiles gkmsvm-lite's kernels into the on-disk Numba and CuPy caches (`_common.warm_jit_cache`; about 10 s per device after an install or a kernel edit, 2 s otherwise), so no timed run includes compiling; `--no-warm-jit` skips it for smoke tests. LS-GKM's `-T` accepts only 1, 4 or 16; scripts default to the largest the process has usable cores for (16 on a 16+ core server; override with `THREADS=`), run gkmsvm-lite's CPU code on the same number of threads, and record both.

## Panels

| Panel | Shows | Target | Script | Result file |
|---|---|---|---|---|
| 1A | Per-variant agreement with LS-GKM (ISM, GkmExplain, deltaSVM.pl) | `fig1a` | `agreement_lsgkm.py` | `agreement_dsqtl.tsv`, `agreement_checks.tsv` |
| 1B | Training time vs N with each tool's default solver, both at `THREADS` and at 1 thread | `fig1b` | `timing_train.py` | `timing_train.tsv`, `timing_train_T1.tsv` |
| 1C | Time and memory to score the 28,309 dsQTL variants: ISM and GkmExplain per variant; deltaSVM as a one-time 10-mer table build, then scoring with the table | `fig1c` | `timing_variants.py` | `timing_variants.tsv` |
| 1D | dsQTL AUPRC with the paper's models | `fig1d` | `dsqtl_auprc.py` | `dsqtl_summed.tsv`, `dsqtl_per_model.tsv` |
| S1 | Models trained with gkmsvm-lite vs the paper's LS-GKM models | `s1` | `dsqtl_auprc.py` | `dsqtl_retrained_*.tsv` |
| S2 | Prediction throughput vs number of SVs (`-t 2` l-mer table, `-t 3` l-mer index, masked join, LS-GKM) | `s2` | `throughput_sweep.py` | `throughput.tsv` |
| S3 | Training time, peak memory and held-out accuracy by solver (precomputed Gram, SMO, Nyström, LS-GKM), gkmsvm-lite on GPU and on CPU | `s3` | `training_memory.py` | `training_memory.tsv`, `training_memory_cpu.tsv` |
| S4 | GkmExplain completeness | `s4` | `completeness.py` | `completeness.tsv` |
| S5 | DART-Eval Task 4 (five cell lines) with ENCODE models and models retrained on DART-Eval's training chromosomes, and Task 5 (caQTL, dsQTL) with ENCODE models, against published DART-Eval and Evo 2 results (`dart_reference.tsv`) | `s5` | `dart_eval.py`, `dart_train.py` | `dart_task4_<cell>_<model>.tsv`, `dart_retrained_task4_<cell>.tsv`, `dart_task5_<model>.tsv` |
| check | Retrained models against S5's ENCODE models, Task 4 and Task 5 (GM12878), with training times (`figures/check_dart_retrained.*`; not in the paper) | `s5-retrained` | `dart_train.py`, `dart_eval.py` | `dart_retrained_train_<cell>.tsv`, `dart_retrained_task5_GM12878.tsv` |

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
- DART-Eval reference values: `dart_reference.tsv` (committed), parsed from the DART-Eval v3 LaTeX
  source (Patel et al., arXiv:2412.05430v3) plus Evo 2's zero-shot Task 5 AUROC (Brixi et al. 2026,
  doi:10.1038/s41586-026-10176-5); sources in its header. The gkm-SVM models are the ENCODE models
  trained on the DNase-seq experiment DART-Eval uses per cell line (`DART_TASK4` in the Makefile).
  None holds out a test set: each concatenates the five models of a 5-fold CV over its top 10k
  non-promoter peaks genome-wide, so those of its peaks on DART-Eval's test chromosomes were
  training positives. `s5` also retrains each with gkmsvm-lite on DART-Eval's training split
  (`dart_train.py`: top 10k IDR peaks by reads against 10k background regions, 300 bp, `-t 2`),
  saving models to `data/models/retrained/`.

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

## mLS-GKM baseline

[mLS-GKM](https://github.com/harmstonlab/mLS-GKM) (Howard & Harmston 2026, [arXiv:2608.20576](https://arxiv.org/abs/2608.20576)) extends LS-GKM with multi-class models and, for prediction and `gkmexplain`, worker threads across sequences (`-t`). LS-GKM's `-T` instead splits each sequence's kernel sum, which gains little on short sequences (or loses: 3,773 s at `-T 16` against 684 s at `-T 1` for the deltaSVM build), so mLS-GKM is the multithreaded C baseline. It reads LS-GKM model files. Build it from source (the repository also holds prebuilt binaries; rebuild them):

```bash
git clone https://github.com/harmstonlab/mLS-GKM.git
cd mLS-GKM/src
make clean && make && make install   # binaries in ../bin
```

and pass its `bin` directory:

```bash
make -C benchmarks fig1b fig1c s2 MLSGKM_BIN=<path>/mLS-GKM/bin
bash benchmarks/run.sh --mlsgkm-bin <path>/mLS-GKM/bin
```

With it, 1C adds mLS-GKM rows at every CPU thread count (`gkmpredict -D -t N -T 1`, `gkmexplain -D -m 5 -t N`), S2 adds its `gkmpredict` at `THREADS`, and 1B adds `t_mlsgkm_infer` (prediction with the LS-GKM-trained model). Training stays LS-GKM's `gkmtrain`. 1C's mLS-GKM rows record `max_abs_diff` against LS-GKM's output, a check that both compute the same scores.

## `dev/`

Experiments that don't feed a figure: the CUDA masked-join ablation, the Nanog replication, and SVR and caQTL training. Finished experiments are deleted once their result is recorded (the earlier solver comparisons, the Nyström landmark sweep and the mask-join prototypes are in git history). Run them from the repo root, e.g. `python benchmarks/dev/replicate_nanog.py`.
