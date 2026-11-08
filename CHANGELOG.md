# Changelog

All notable changes to gkmsvm-lite are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.1] - 2026-11-08

### Changed

- MLX: RBF ISM and l-mer index queries run in Metal at every index density, 1.4-5x the CPU (10 threads, M4) on the dsQTL `-t 3` model and its SV subsets; on dense indexes they used to fall back to the CPU, where Metal had been slower. ISM's thread blocks and each query row's threadgroup are now 1,024 threads (they were 128 and a thread per item in groups of 256), which keeps their atomics out of DRAM and off one another. End to end on MLX: RBF prediction 1.4-4.3x the CPU (was 0.86-1.14x), ISM 2.6-5.4x (was 1.0-3.2x).
- MLX: index scoring (deltaSVM tables, prediction, variant and ISM scoring through the SV index) runs a fused Metal query + reduce: persistent threadgroups scatter each row's hits into a per-group scratch of length S and reduce coef · f(K) on the GPU, so the [B, S] block and the host combine are gone. A 65,536 l-mer deltaSVM chunk on the dsQTL `-t 3` model: 21.4 s to 0.68 s (CPU, 10 threads: 2.46 s).
- MLX: query launches with fewer than 40 rows split each row across several threadgroups, so single SMO columns use the whole GPU: 6.3 to 1.2 ms per column; SMO at N = 10k with a 2,048-column cache 121 to 34 s.
- MLX: mutation impact (GkmExplain `-m 5`, `score_variants(method="gkmexplain")`) runs a fused Metal pass at every index density: 28,309 dsQTL SNPs in 21 s on an M5 (10 CPU threads: 52 s), about 2e-5 relative to the CPU.
- MLX: RBF GkmExplain (hypothetical, perturbation) overlaps each batch's Metal kernel with the next batch's CPU query of its SV weights: dsQTL `-t 3` model, 500 × 300 bp on an M5, hypothetical 8.8 to 5.6 s (10 CPU threads: 9.1 s), perturbation 1.6× the CPU.
- MLX: base codes of a one-hot input are taken on the host (unified memory, no copy) rather than by an argmax on the GPU that packing then waited on. Small `-t 2` calls now match the CPU: predict 2,000 × 300 bp 48.6 to 35.8 ms (CPU 32.0), 2,000 variants 8.7 to 4.6 ms (CPU 4.6).
- `solver="auto"` takes the precomputed Gram only when 1.1 × the Gram plus 2 GiB fits in 90% of available RAM (was: the Gram in 75%). Runs measure the Gram plus 0.9-2.0 GB, so on a 16 GB Mac the old rule let through runs that would swap.

### Fixed

- MLX precomputed Gram: MLX's buffer cache is cleared after each column panel. The panels grow, so the cache kept one buffer of every size, O(N²) beside the Gram: peak memory at N = 20k (M5) 11.1 to 5.0 GB.
- macOS: available memory is read from the VM statistics (free + inactive pages). macOS has no `SC_AVPHYS_PAGES`, so every Mac used to read as 8 GB, which set `solver="auto"`'s precomputed-Gram limit (N of about 28k whatever the RAM) and the SMO cache size.

### Added

- Benchmarks: Fig. S4B–F, GkmExplain (hypothetical, perturbation) and full ISM throughput on 300 bp sequences against LS-GKM, with float64-reference and summation-to-delta checks per run (`benchmarks/explain_throughput.py`, `make s4-explain`).
- Benchmarks: `--device mlx` for the timing panels and S4, and `make apple` (CPU, MLX and LS-GKM on one Mac, into `results/apple/`); `dev/gram_index_vs_masked.py --device mlx`.
- Benchmarks: Fig. S7, gkmsvm-lite on Apple Silicon (`make apple`, `results/apple/`), and `dev/mlx_index_density.py`, the Metal paths against the CPU by index density.

## [0.2.0.post1] - 2026-11-01

### Changed

- `_csmo.c` no longer carries a BSD-2-Clause label; like the rest of the package it is MIT (`LICENSE`). No code changes.

## [0.2.0] - 2026-10-30

### Breaking changes

Scores and outputs for the same inputs:

- `-t 3` (gkmrbf) is now LS-GKM's definition, `exp(gamma * (K_norm - 1))` on the normalized base kernel; 0.1.0 applied the RBF to the unnormalized kernel distance, so `-t 3` scores and trained models change.
- `-t 4` / `-t 5` (wgkm / wgkmrbf) are now LS-GKM's window-weighted kernel (the `-t 2` kernel with an integer weight per l-mer window by its distance from the sequence center); 0.1.0 weighted positions inside each l-mer, a different kernel that was 0.16-0.24 off `gkmpredict`. They now also take `d`.
- `d` now applies to `-t 0` (gkm_cnt), as in LS-GKM: pairs with more than `d` mismatches are not counted. Imported LS-GKM `-t 0` models were scored with the wrong kernel (up to 44% of score - bias); `-t 0` models saved by 0.1.0 (no `d` in their parameters) keep counting every pair.
- N is read as A on the forward strand and T on the reverse strand, as in LS-GKM, for every kernel type and in deltaSVM scoring; this changes scores of sequences (and SVs) containing N.
- GkmExplain importance mode now assigns N positions' importance to A (as lsgkm does) instead of zeroing them, so attributions sum to score - bias for sequences with N.
- Scores, ISM, attributions, Gram matrices, coefficients and deltaSVM weights are float64 on CPU and CUDA whatever the input dtype (float32 on MLX). In 0.1.0 precision followed the input or the SVs, so default-loaded models and CUDA scoring computed in float32.
- `score_variants(method="gkmexplain")` now returns the GkmExplain mutation impact score (lsgkm `gkmexplain -m 5`) instead of reading hypothetical attributions at the variant positions.
- `gkmexplain()` on an MLX model returns a NumPy float64 array, not an MLX array.
- `GkmSVM.support_sequences` is now `[S, L]` uint8 base codes (A=0/C=1/G=2/T=3), not `[S, 4, L]` one-hot. One-hot SVs passed to the constructor (e.g. from 0.1.0 npz files) are converted; npz files saved by 0.2.0 store codes and cannot be read by 0.1.0.
- `extract_loci` returns int8 one-hot arrays (was float32) and centers windows on the narrowPeak/DataFrame `summit` when present (`summit=False` restores the midpoint).

Removed or renamed API:

- The importers' `dtype` argument is removed (`load_lsgkm_model`, `load_classic_model`, `load_r_gkmsvm_model`).
- The classic and R gkmSVM importers read only the files `gkmsvm_train` actually writes (`PREFIX_svalpha.out` + `PREFIX_svseq.fa`); the LIBSVM-style-header and single `.gkmmodel` formats are dropped. The kernel is now given as arguments (`load_classic_model(alpha_path, svseq_path, L=10, k=6, d=3, filter_type=1)`, R names `L, K, maxnmm, useTgkm, addRC` for `load_r_gkmsvm_model`) and the bias is 0.
- `sv_chunk_size` is removed from `ism()` and `gkmexplain()`; use `batch_size` (inputs) and `model.sv_chunk_size`.
- The pure-Python SMO fallback is removed: without the compiled `_csmo` extension, `solver="smo"` raises.
- `train_gkmsvr` rejects unknown solver names (choices: `auto`, `libsvm`, `smo`, `nystrom`).
- `-t 4` / `-t 5` require `1 <= M <= 255` (LS-GKM stores weights as uint8).

### Added

- Masked-join gkm kernel (CPU, CUDA, Metal) for every kernel type, used for Gram matrices, SMO columns and scoring.
- Exact 4^l l-mer weight table for prediction with `-t 0/1/2/4` (l <= 12), controlled by `model.use_lmer_table`.
- Cached l-mer index for scoring, Gram matrices and SMO columns when the mismatch neighborhood is small (`model.use_lmer_index`), on CPU, CUDA and Metal.
- `mutation_impact(model, ref, alt)`: lsgkm's `gkmexplain -m 5` variant score (exported from `gkmsvm`).
- GkmExplain for every kernel type (`-t 0` to `-t 6`), a `"perturbation"` mode (lsgkm mode 4), and string mode names (`"importance"`, `"hypothetical"`, `"perturbation"`; integers still accepted).
- GkmExplain fast paths: SV l-mer weights, a precomputed explain table for `-t 0/1/2` (`model.use_explain_table`), and CUDA and Metal kernels.
- ISM works for every kernel type, with l-mer table and fused l-mer index paths on CPU, CUDA and Metal and bounded memory.
- `-t 6` (gkm_cnt_rbf / direct_rbf): RBF on the gapped k-mer count kernel, from the kundajelab lsgkm fork.
- Column-cached SMO for epsilon-SVR: `train_gkmsvr(solver="smo")` and `gkmsvm trainsvr --solver smo --cache-size`; `train_gkmsvr` also takes `nystrom`, `cache_size`, `tol`, `max_iter` and `max_gram_gb`.
- `pos_weight` in `train_gkmsvm` and `gkmsvm train -w` (LS-GKM's `-w`).
- Sequences of different lengths: scoring, training, ISM, GkmExplain, mutation impact and DeltaSVM accept mixed-length lists or `lengths=`; models keep `sv_lengths` through importers, npz and LS-GKM files.
- Inputs can be `[B, 4, L]` one-hot arrays, DNA strings, or `[B, L]` uint8 base codes.
- N accepted in string inputs (CLI and training); `allow_other` / `allow_other_bases` / `--allow-other-bases` read other non-ACGT letters as N, as LS-GKM does.
- CLI: `--genome` reads BED/narrowPeak summit windows from a genome FASTA, `--length` center-crops FASTA records, `--variable-length` keeps records at their own lengths; `train --solver nystrom`, `to-deltasvm --query-norm --device`, `score-variants --method`; `-t` accepts `0`-`6`.
- `extract_loci(return_mask=True)` returns a mask of the kept loci; `read_loci` reads BED/narrowPeak files without pandas.
- `to_deltasvm(query_norm=True)` builds weights equal to `gkmpredict` on each l-mer, as lsgkm deltaSVM pipelines do.
- `save_deltasvm_npz` / `load_deltasvm_npz` (and `load_model` detects DeltaSVM npz files).
- `batch_size` on `model()` and `ism()` (and `None` default on `score_variants()`); `GkmSVM.clear_caches()`.
- Oracle tests against LS-GKM (`gkmpredict`, `gkmtrain`, `gkmexplain` modes 0/1/4/5), lsgkm-svr, and the classic and R gkmSVM tools.
- Application-note benchmark pipeline (`benchmarks/`, one make target per figure panel), including a Stanford Sherlock submission pipeline for fixed CPU and GPU hardware.

### Changed

- Training is much faster: N=10k `-t 2` l=11 k=7 d=3 on a 16-thread Xeon, CPU 1711 s -> 55 s (LS-GKM 517 s), GPU 33 s -> 9 s; `-t 2` SMO with a 2048-column cache at N=5000, 283 s -> 5.7 s.
- Prediction against 8.5k SVs: 128 s -> 0.10 s on CPU and 0.02 s on GPU (LS-GKM 64 s).
- ISM: 1.5 s -> 4 ms per 300 bp sequence on a 72k-SV model (M4); no longer holds every mutant's per-SV values in memory.
- GkmExplain: 1.7 s -> 3.5 ms per 300 bp sequence on a 72k-SV model through the l-mer path, ~0.035 ms with the explain table.
- Mutation impact through the l-mer table: 500 dsQTL variants 7.0 s -> 0.01 s on CPU.
- RBF `to_deltasvm` on a 32,715-SV `-t 3` model: 318 s -> 88 s at 1 thread.
- Kernel scoring raises small batch sizes so each masked-join call amortizes its SV index.
- SMO `cache_size` defaults to sizing from available RAM (was 2048 in the API, 256 in the CLI); `solver="auto"` and the SMO cache respect a Slurm job's cgroup memory limit.
- `-t 4` / `-t 5` `M` and `H` default to LS-GKM's 50 and 50 (were required).
- `device="auto"` uses CUDA only when CuPy finds a GPU.
- `save_lsgkm` writes `norc` only when RC is off, so kundajelab lsgkm builds can read the files.
- Development, CI and benchmark installs use uv (`uv pip install -e . --group dev`).

### Fixed

- `ism()` crashed for `-t 3/4/5`.
- `gkmsvm trainsvr` crashed by passing `cache_size` to `train_gkmsvr`.
- `-t 4` SMO columns allocated tens of GB at once.
- `to_deltasvm(query_norm=False)` gave wrong weights for RBF models, and `to_deltasvm` no longer mutates the model or exhausts GPU memory.
- Classic and R gkmSVM importers failed on real `gkmsvm_train` output.
- The CLI's `-t 0`-`-t 5` strings were not resolved as kernel types.
- SV-chunked scoring of MLX models tried to import CuPy.
- MLX GkmExplain cache grew on every call; `one_hot_decode` failed on MLX arrays.
- The C SMO extension failed to load on newer macOS (duplicate `-rpath`) and training silently fell back.
- `device="auto"` failed with a CUDA driver error where CuPy is installed but no GPU is available.
- Python 3.14 compatibility in the dsQTL example's archive extraction.

## [0.1.0] - 2026-10-03

First release.

[0.2.1]: https://github.com/adamyhe/gkmsvm-lite/compare/v0.2.0.post1...v0.2.1
[0.2.0.post1]: https://github.com/adamyhe/gkmsvm-lite/compare/v0.2.0...v0.2.0.post1
[0.2.0]: https://github.com/adamyhe/gkmsvm-lite/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/adamyhe/gkmsvm-lite/releases/tag/v0.1.0
