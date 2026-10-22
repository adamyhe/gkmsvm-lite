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
| S2 | Prediction throughput vs number of SVs (`-t 2` l-mer table, `-t 3` l-mer index, LS-GKM) | `s2` | `throughput_sweep.py` | `throughput.tsv` |
| S3 | Training time, peak memory and held-out accuracy by solver (precomputed Gram, SMO, Nyström, LS-GKM), gkmsvm-lite on GPU and on CPU | `s3` | `training_memory.py` | `training_memory.tsv`, `training_memory_cpu.tsv` |
| S3B | Large-N scaling on LS-GKM's own design (Lee 2016, Fig. 1): H1-hESC CTCF ChIP-seq vs GC/repeat-matched hg19 windows, LS-GKM's eight sizes (5k–30k, 60k, 90k) plus 45k and 75k, LS-GKM with 100 MB and 8 GB caches. Not part of `paper`: up to a day | `s3b` | `ctcf_data.py`, `training_memory.py --data ctcf` | `training_memory_ctcf.tsv` |
| S4 | GkmExplain completeness | `s4` | `completeness.py` | `completeness.tsv` |
| S5 | DART-Eval Task 4 (five cell lines) with ENCODE models and models retrained on DART-Eval's training chromosomes, and Task 5 (caQTL, dsQTL) with ENCODE models, against published DART-Eval and Evo 2 results (`dart_reference.tsv`) | `s5` | `dart_eval.py`, `dart_train.py` | `dart_task4_<cell>_<model>.tsv`, `dart_retrained_task4_<cell>.tsv`, `dart_task5_<model>.tsv` |
| GPUs | gkmsvm-lite's GPU times in 1B, 1C, S2, S3 and S3B on each host's results (`figures/check_gpus.*`; drawn when `results/<SET>/` exists; not in the paper) | the panels', with `SET=<label>` | as for those panels | `results/<SET>/` |
| check | Retrained models against S5's ENCODE models, Task 4 and Task 5 (GM12878), with training times (`figures/check_dart_retrained.*`; not in the paper) | `s5-retrained` | `dart_train.py`, `dart_eval.py` | `dart_retrained_train_<cell>.tsv`, `dart_retrained_task5_GM12878.tsv` |

The dsQTL reproduction behind 1A, 1D and S1 is written up in [`docs/validation.md`](../docs/validation.md).

## Result files

Every result file starts with `# key: value` lines giving the git commit, date, host, command, CPU, GPU, package versions and LS-GKM version. Read them with `pd.read_csv(path, sep="\t", comment="#")`, and read the header with `_common.read_provenance(path)`.

Result files and figures are committed, so the figures can be rebuilt without rerunning anything. Quick runs (`results/quick/`, `figures/quick/`) and logs are not committed.

Besides figures, the notebook writes `figures/text_numbers.tsv` (speedups and AUPRCs quoted in the text) and `figures/table_variant_memory.tsv` (peak host and GPU memory for panel 1C's runs).

## Hardware

Targets run on any machine with the conda envs below; each result file's header records the host, CPU and GPU it ran on. Keep each panel's runs on one host, since LS-GKM and gkmsvm-lite times are compared within a panel. On Sherlock, each panel's CPU runs share one hardware type and its GPU runs another (see [Running on Sherlock](#running-on-sherlock)).

To add a second GPU, run the timing panels on its host with a label for the result set, which writes `results/<label>/` instead of `results/`:

```bash
make -C benchmarks fig1b fig1c s2 s3 s3b SET=a100   # or: bash benchmarks/run.sh --set a100 ...
make -C benchmarks figures SET=a100                 # figures/a100/, other panels from results/
```

Each set reruns the whole panel, LS-GKM included, since the host's CPU changes too. `make figures` (no `SET`) also draws `check_gpus`, gkmsvm-lite's GPU times from `results/` and every `results/<label>/`, labeled by the GPU each result file records.

## Running on Sherlock

`sherlock/submit.sh` submits the panels as Slurm jobs from this checkout and exits; nothing stays running on the login node. Results go to `results/` in this checkout, and logs to `results/logs/`.

```bash
bash benchmarks/sherlock/submit.sh --quick              # small settings first: results/quick/
bash benchmarks/sherlock/submit.sh                      # fig1b fig1c s2 s3
bash benchmarks/sherlock/submit.sh --no-setup s3b other # S3B, and the panels that time nothing
bash benchmarks/sherlock/submit.sh --gpu a100 --dry-run s2
```

Each timing panel (1B, 1C, S2, S3, S3B) splits into a CPU part (LS-GKM, mLS-GKM and gkmsvm-lite on CPU) and a GPU part (gkmsvm-lite on CUDA). The two parts run on different fixed hardware and are merged afterwards (`make SPLIT=1`):

- **CPU parts** run on a whole AMD EPYC 9384X node in `akundaje` (`CPU_SKU:9384X`, `--exclusive`), one job per part, so no other job shares the cores, caches or memory bandwidth. LS-GKM and gkmsvm-lite get the same 16 threads (`THREADS`).
- **GPU parts** run on one L40S (`--gpu l40s`, the default), with the node's CPU fixed as well (`GPU_SKU:L40S&CPU_SKU:6426Y`) because it runs the GPU part's host-side work. They also get 16 CPUs, and they are chained so that two GPU parts never run at once.
- A **merge** job per panel writes `results/<file>.tsv` once all of the panel's parts have succeeded. The merged file is in the format a single run on both devices would have written, so `figures.ipynb` is unchanged. Its header keeps the CPU part's provenance and adds the GPU part's differing values as `gpu_<key>` (`gpu_host`, ...). `max_abs_diff` (1C) and `r_vs_exact` (S3), which compare rows across the two parts, are filled from score arrays both parts save under `results/parts/` (git-ignored).
- A **setup** job runs first. It creates the `lsgkm` and `gkmsvm-bench` envs in `$GROUP_HOME/$USER/envs` if they are missing, installs this checkout into `gkmsvm-bench` with uv, runs the tests, and downloads every input (`make data`), so concurrent jobs never fetch the same file.
- **`other`** runs 1A, 1D, S1, S4 and S5 in one GPU job. These panels time nothing.

Parts are made only in a job of their kind (the Makefile checks the `BENCH_HW` that `sherlock/job.sh` sets). A merge job that finds a stale or missing part therefore stops instead of running that part on the wrong machine. Every job keeps its temporary files and its Numba and CuPy caches on node-local scratch, so timed runs don't touch the shared filesystems, and each job compiles its kernels in its own untimed warm-up. Result headers record the Slurm job, partition, node, constraint and exclusivity.

Hardware can be overridden from the environment: `PARTITION`, `CPU_CONSTRAINT`, `GPU_CONSTRAINT`, `GPU_CPUS` and `THREADS`. Keep a figure's runs on one hardware type; the result headers show which type each run used.

### Warm-up

No timed region includes compilation or first-use costs:

- gkmsvm-lite scripts first warm up untimed: `warm_jit_cache` compiles into the disk caches, then each worker repeats the timed call on a few inputs or a small model of the same kernel.
- `timing_train.py` warms both prediction paths (l-mer table and index).
- `training_memory.py` workers train once on 100 sequences before the timed fit.
- Each LS-GKM and mLS-GKM binary runs once, untimed, before its first timed run (`warm_binaries`). This reads the executable and its libraries into the page cache.
- 1C copies the model file into the node-local work directory, so both implementations read it from the same disk.
- 1C plots whole-process wall time, which includes start-up and model loading for both implementations but must not include compiling. Before each method's timed gkmsvm-lite runs, an untimed rehearsal process per device makes the timed process's warm-up call, filling the disk caches with exactly the kernels it uses. For the deltaSVM table, the rehearsal runs the whole build: CuPy compiles reductions per launch configuration, which depends on block size, so a smaller warm-up misses some (3 kernels, ~8 s on an L40S). Around every timed process, `jit_cache_files` snapshots the Numba and CuPy caches, which are written only when something compiles. The count of new files is recorded as `jit_compiles` (expected 0, with a warning otherwise).

## Running on a server

`run.sh` sets up a machine and runs targets from its existing checkout. It creates the `lsgkm` and `gkmsvm-bench` conda envs if missing, installs the checkout into `gkmsvm-bench` with uv (`[examples,gpu]` extras), and runs `make` there. It doesn't touch git, so check out the commit you want first; result headers record it.

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

## Large-N scaling (S3B)

LS-GKM's scaling figure (Lee 2016, *Bioinformatics* 32:2196, Fig. 1; Supplementary Methods 2–3) trained on eight subsamples of H1-hESC CTCF ChIP-seq from 5k to 90k sequences, with gkmtrain's default 100 MB and an 8 GB kernel cache. S3B uses those sizes (5k, 10k, 15k, 20k, 25k, 30k, 60k, 90k) plus 45k and 75k, so the range where the 8 GB cache fills (~60k) is sampled too. `ctcf_data.py` rebuilds that set:

- **Positives:** ENCODE's January 2011 SPP optimal peaks for `wgEncodeBroadHistoneH1hescCtcf` (55,782 peaks on hg19; the other H1-hESC CTCF set, from HAIB, has too few for 90k), as 300 bp windows centered on each summit. LS-GKM trained on the variable-length peaks; gkmsvm-lite trains on fixed-length windows.
- **Negatives:** random hg19 windows without N that overlap no peak, matched to the positives' joint GC and repeat-fraction distribution (soft-masked repeats), as in the paper.
- **Split:** chr1 and chr2 held out for testing.

`make -C benchmarks s3b` builds the data (downloading hg19 if `dsqtl_auprc.py` hasn't) and runs `training_memory.py --data ctcf`. The precomputed Gram and Nyström are skipped once N × N float64 exceeds 75% of RAM; SMO runs at every size. In Lee 2016 (Fig. 1B, a 4-core 2016 i7), `gkmtrain` on 4 threads took ~100 min at 90k with the 8 GB cache and ~360 min with 100 MB; single-threaded with 100 MB it took ~200 min at 30k and wasn't run larger. `s3b` is not part of `paper`; the per-run timeout is 48 h.

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

mLS-GKM's `gkmexplain` (commit 2bb06be, 2026-08-30) is wrong for non-RBF models in every mode: on `-t 2` it logs "kmertree for SVs has not been initialized", writes zeros (mode 5) or near-zero attributions, and exits 0. On `-t 3` it equals the kundajelab fork exactly. 1C uses the paper's `-t 3` models; `timing_variants.py` skips mLS-GKM's `gkmexplain` for any other kernel.

## `dev/`

Experiments that don't feed a figure: the CUDA masked-join ablation, the self-Gram through the l-mer index vs the masked join (`gram_index_vs_masked.py`), the Nanog replication, SVR and caQTL training, and an ncu profile of how much each CUDA operation is limited by float64 throughput (`profile_fp64.py`; on an L40S only the short kernels are: ISM, scoring and mutation impact are bound by atomics and memory; on Sherlock, `sbatch -C GPU_SKU:<gpu> benchmarks/sherlock/profile_fp64.sbatch`). Finished experiments are deleted once their result is recorded (the earlier solver comparisons, the Nyström landmark sweep and the mask-join prototypes are in git history). Run them from the repo root, e.g. `python benchmarks/dev/replicate_nanog.py`.
