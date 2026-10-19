# Design decisions

These decisions are load-bearing — do not deviate without discussion.

## Inference-first

Ship an LS-GKM model importer and predictor before building training. Match `gkmpredict` output within floating-point tolerance. Oracle scores are validated against Dongwon-Lee/lsgkm built from source.

## Kernel modes are distinct

`-t 0` (direct gapped k-mer / `gkm_cnt`) and `-t 2` (truncated estimated l-mer / `gkm_esttrunc`, the LS-GKM default) use different math. The binomial coefficient table `C(l-m, k)` for `-t 0` does NOT apply to `-t 2`. The `-t 2` weight table is ported from `calc_gkm_kernel_lmerest_wt` in `libsvm_gkm.c`.

`-t 1` (`gkm_estfull`) is supported via `EstTruncGkmKernel(truncate=False)`.

`-t 3` (`gkmrbf`) wraps the normalized `-t 2` kernel with an RBF transform: `K_rbf(x,y) = exp(gamma * (K_norm(x,y) - 1))`, matching lsgkm. This equals `exp(-gamma/2 * ||phi(x) - phi(y)||^2)` on unit-norm features. Self-similarity is always 1.

`-t 4` (`wgkm`) is the `-t 2` kernel with a weight on each l-mer window, as in LS-GKM's `gkmkernel_new_object`: window i of a sequence with W windows weighs `min(M, floor(M 2^(-|W/2 - i|/H) + 1))` (W/2 rounded down), an integer from 1 to M, and a window pair contributes the product of its windows' weights times the `-t 2` table entry. A reverse-complement window carries the weight of the forward window it came from, so with an even W the kernel isn't exactly RC-invariant (neither is LS-GKM's). Weights depend only on a window's position and the sequence length, so packed kernels take them as one `[W]` vector per side (`window_weights(W)`): the brute-force loop multiplies them in, the masked join and the l-mer index store each column window's weight next to its id (uint8; the CUDA join counts in uint64, exact), and the l-mer table weights each SV window's contribution and each query lookup.

`-t 5` (`wgkmrbf`) applies the same RBF transform to the normalized center-weighted kernel.

## Score formula

`score(x) = Σ dual_coef_i × K(x, support_i) + bias`

- Coefficients are signed (positive and negative class)
- `bias = -rho` (LIBSVM sign convention)
- LS-GKM's `rho` is the NEGATIVE bias

## Kernel normalization

Normalize by self-similarity: `K_norm(x,y) = K(x,y) / sqrt(K(x,x) * K(y,y))`. Unequal-length sequences have different norms. Normalization is on by default.

## Reverse-complement equivalence

On by default (`norc=0` in LS-GKM). Scores must be invariant under RC. When enabled, kernel computes K(x,y) + K(x, RC(y)).

## Kernel naming

Each kernel mode has three identifiers: LS-GKM `-t N` integer, an internal name (used in model files), and a descriptive alias. `resolve_kernel_type()` accepts any of these and returns the canonical internal name. `GkmSVM` resolves on construction — `model.kernel_type` is always the internal name.

## Kernel computation

Every kernel computes on packed windows: each l-mer is a uint32 (2 bits per base), and a window pair's mismatch count is an XOR + popcount (see "Packed uint32 comparison" below). The interface is `PackedGkmKernel` (`kernels/packed.py`): `pack(x)`, `raw_packed(bx, by, by_rc)` with the reverse-complement block in the same call, and `raw_diag_packed(bx, bx_rc)`. RBF kernels expose their base kernel as `kernel.base` and apply `kernel.finish(k_norm) = exp(γ(k_norm - 1))` last, so Gram tiles, SMO columns, scoring, ISM and mutation impact share one code path for all six kernel types. `raw_packed` dispatches to:

- **Masked join** (`kernels/masked.py`) when both sides have 16+ sequences: h(m) as a signed sum of C(l-m, j) terms, each an exact-match join on masked l-mer keys. See [design/mask-join-kernel.md](design/mask-join-kernel.md).
- **Brute force** below that: every window pair, skipping pairs below `min_matches`. CPU: Numba `@njit(parallel=True, fastmath=True)`. NVIDIA: CuPy RawKernel with coalesced SV reads and shared-memory caching of the table and query windows. Apple: Metal kernels via `mx.fast.metal_kernel` (pairwise, and cross-diagonal for self-kernels against [x, rc(x)]).

Scoring against a model's SVs adds two faster paths for fixed SVs: the exact l-mer weight table (`lmer_table.py`, `-t 0/1/2/4`, l <= 12) and the cached l-mer index (`kernels/lmer_index.py`, `-t 1/2/3` with a small d-neighborhood).

**GkmExplain** (`-t 0/1/2`): the same packed comparison selects the ~0.1% of window pairs within the mismatch cutoff, and only those are decomposed per position, with base identities read from the packed words (`(packed >> 2k) & 3`). Importance is `hypothetical * one_hot_input`; only hypothetical and perturbation have kernels (one kernel with a mode flag per device). The inner loop multiplies by the normalized coefficient and reduces over SVs, accumulating into the `[B, 4, L]` result: CPU Numba with `prange` over the batch; CUDA one thread per (sequence, SV) pair with float64 `atomicAdd` (CC >= 6.0). MLX falls back to CPU. Usually that per-SV loop is skipped: the attribution is linear in the SVs, so summing them into a 4^l table of l-mer weights first lets each query window visit just the l-mers within the cutoff (CPU Numba, or a CUDA kernel with 16 lanes per window), at no cost per SV. For `-t 0/1/2`, since a pair's weight at each position depends only on its mismatch count and the base difference there, the attribution of every possible window is an XOR convolution over the 4^l l-mers: Walsh-Hadamard transforms precompute it as a [4^l, 4, l] table (1.4 GiB, ~1 s at l=11), and explaining a sequence becomes W row lookups. It is built automatically once a call explains enough sequences to pay for it.

## Memory in scoring

Kernel scoring holds [rows, S] kernel values, so `GkmSVM.__call__` processes inputs in row blocks of at most `_MAX_SCORE_BYTES` (512 MB of float64), whatever `batch_size` says; for the 72,145-SV ENCODE model ENCFF579AOX that is ~930 rows per block. The model also caches packed SV windows (4 bytes per window and strand) and, for d-cutoff kernels, the l-mer index (8·4^l bytes of offsets plus 4 bytes per SV window and strand: ~110 MB at l=11 for 32k SVs). `GkmSVM(sv_chunk_size=N)` only chunks SVs inside GkmExplain.

## Training

Two solver backends, selected automatically based on available memory:

**Precomputed Gram + sklearn** (`solver="libsvm"`, default for small N): Computes the full N×N kernel matrix via tiled Gram computation, then calls scikit-learn's LIBSVM-backed `SVC(kernel='precomputed')` / `SVR(kernel='precomputed')`. Supports both C-SVC (`train_gkmsvm`, `-s 0`) and epsilon-SVR (`train_gkmsvr`, `-s 3`). Fast — LIBSVM's SMO is highly optimized — but requires O(N²) memory.

**Column-cached SMO** (`solver="smo"`, large N): WSS3 working set selection (Fan et al. 2005) with shrinking and cached kernel columns. Memory is O(cache_size × N) instead of O(N²). Currently supports C-SVC only. A C extension (`_csmo`) handles the optimization loop and the column cache; kernel columns come from `KernelColumnCache` on the training device (Numba CPU, CuPy GPU, or MLX): one query row of an l-mer index over the training set when it applies (`-t 1/2/3`), else batches through the masked join; misses prefetch likely next working-set columns into free cache slots. Installs still succeed if the extension fails to compile, but `solver="smo"` then raises.

**Nyström approximation** (`solver="nystrom"`): kernel values against m landmarks (O(N m) evaluations), a low-rank N×N approximation, then sklearn. Saves kernel evaluations, not memory; the solution is approximate.

`solver="auto"` uses the precomputed Gram when N²×8 bytes fits in 75% of available RAM (and under `max_gram_gb`); the Gram is always held in host RAM, whatever the training device. Otherwise it falls back to SMO.

The scoring formula `Σ coef_i × K(x, sv_i) + bias` is identical for SVC and SVR — only the dual coefficients differ (α_i × y_i for SVC, α_i* − α_i for SVR). The `GkmSVM` model class is shared.

## No dense Gram matrix at scale

LS-GKM exists because the full N×N kernel matrix doesn't fit in memory at scale (50k examples ≈ 19 GB, 90k ≈ 60 GB of float64). Training then uses column-cached SMO, and scoring computes [rows, S] blocks capped at `_MAX_SCORE_BYTES`.

## The `d` parameter

`d` limits mismatch depth in LS-GKM's tree-based kernel evaluation. In our brute-force window comparison, this is implemented by zeroing weight table entries for m > d. Without this cutoff, scores diverge ~1.5% from `gkmpredict`.

## LS-GKM model format

Header key-value pairs until `SV` marker, then `<signed_coef> <DNA_sequence>` per line. Key fields: `svm_type`, `kernel_type`, `L`, `k`, `d`, `norc`, `rho`, `nr_class`, `total_sv`. Auto-detects gzip. Binary classification only (nr_class=2). `norc` is optional (missing = RC on); some builds (kundajelab lsgkm) reject the key, so `save_lsgkm` writes it only for RC-off models.

## Classic gkmSVM format (C)

Original gkmSVM C implementation uses OPPOSITE sign convention: `bias = +rho`. LIBSVM-style `key value` headers followed by `SV` marker. Supports both embedded SVs (single file: `coef sequence` per line) and two-file format (model + FASTA). Integer kernel types (0-5) are mapped to string names. Load via `load_classic_model(model_path, svseq_path=...)`.

## R gkmSVM format

R gkmSVM package (Ghandi et al. 2016, kernlab-based) uses `#`-prefixed headers (`#rho`, `#nsv`, `#npos`, `#nneg`, `#L`, `#k`, `#d`). Two sub-formats:

- **Unified `.gkmmodel`**: headers then FASTA entries where the header line is `>seq_id\tcoefficient`.
- **Legacy two-file**: `_svalpha.out` (tab-separated `seq_id\tcoef`) plus `_svseq.fa`.

Same bias convention as C gkmSVM: `bias = +rho`. Load via `load_r_gkmsvm_model(path, svseq_path=...)`. Auto-detects `_svseq.fa` companion file when loading `_svalpha.out`.

## SV cache

The model caches, per device, the packed SV windows, their base-kernel self-kernels and the l-mer index (`GkmSVM._sv_cache`, built on first use and cleared by `cuda()`/`mlx()`/`cpu()`), plus the l-mer weight table.

## NumPy + CuPy + MLX (no PyTorch)

gkm-SVMs are not differentiable — autograd provides no value. The array operations (matmul, einsum, fancy indexing) are identical in NumPy and CuPy, so a single codebase handles both CPU and NVIDIA GPU via `gkmsvm.backend.get_array_module()`.

CPU: NumPy arrays + Numba `@njit(parallel=True)` fused pairwise kernels.
NVIDIA GPU: CuPy arrays (optional `[gpu]` extra) + CuPy RawKernel CUDA code. `model.cuda()` moves data to GPU.
Apple GPU: MLX arrays (optional `[mlx]` extra) + custom Metal shaders via `mx.fast.metal_kernel`. `model.mlx()` moves data to Apple Silicon GPU. MLX arrays lack `.strides`, `.copy()`, and NumPy-style fancy indexing — an `_MlxShim` wrapper and `get_strides()` helper provide compatibility. Packing runs on the CPU and the packed windows move to the device; the Metal kernels do the pairwise work. Metal has no float64, so the masked join and the l-mer index count matches in int32 and combine them in float64 on the CPU.

tangermeme interop is vendored — only `extract_loci` (pyfaidx) and FASTA I/O are needed. ledidi requires differentiable models and does not work with gkm-SVMs.

## Packed uint32 comparison

Each l-mer window is packed into a uint32 (2 bits per base, 22 bits for l=11). Match counting uses XOR + popcount on 2-bit fields instead of l individual byte comparisons. This replaces 11 strided memory reads with a single contiguous uint32 read per window pair. On GPU, the SV array goes from `[l, Wy, S]` int8 (229MB for 72K SVs at 300bp) to `[Wy, S]` uint32 (84MB) — 2.8x memory reduction plus drastically better cache locality. On CPU, ~3x faster than the int8 loop.

## Mismatch table sparsity

The mismatch table has zero entries for high-mismatch counts. For esttrunc l=11 k=7 d=3, only m=0..3 are non-zero (min_matches=8). For random DNA, P(match)=0.25, so P(>=8 matches out of 11) ≈ 0.12% — 99.88% of window pairs contribute nothing. Both CPU and GPU paths skip the table lookup for pairs below min_matches. On GPU, shared memory caches the query sequence's packed windows so all threads in a block share a single load from global memory.

## Epsilon-SVR

`train_gkmsvr()` trains an epsilon-SVR for continuous-valued prediction (e.g. quantitative chromatin accessibility from lsgkm-svr). The epsilon parameter controls the tube width — errors within ±epsilon are not penalized. Uses LIBSVM's `-s 3` solver with precomputed Gram matrix. SMO SVR (2N dual variables, different box constraints) is deferred — the Gram path handles typical regression dataset sizes.

## References

- Ghandi M, Lee D, Mohammad-Noori M, Beer MA. Enhanced regulatory sequence prediction using gapped k-mer features. *PLoS Comput Biol* 10(7):e1003711 (2014). doi:10.1371/journal.pcbi.1003711
- Ghandi M, Mohammad-Noori M, Ghareghani N, Lee D, Garraway L, Beer MA. gkmSVM: an R package for gapped-kmer SVM. *Bioinformatics* 32(14):2205–2207 (2016). doi:10.1093/bioinformatics/btw203
- Lee D. LS-GKM: a new gkm-SVM for large-scale datasets. *Bioinformatics* 32(14):2196–2198 (2016). doi:10.1093/bioinformatics/btw142
- Shrikumar A, Prakash E, Kundaje A. GkmExplain: fast and accurate interpretation of nonlinear gapped k-mer SVMs. *Bioinformatics* 35(14):i173–i182 (2019). doi:10.1093/bioinformatics/btz322
- Lee D, Gorkin DU, Baker M, Strober BJ, Asoni AL, McCallion AS, Beer MA. A method to predict the impact of regulatory variants from DNA sequence. *Nat Genet* 47(8):955–961 (2015). doi:10.1038/ng.3331
- Chang CC, Lin CJ. LIBSVM: a library for support vector machines. *ACM Trans Intell Syst Technol* 2(3):1–27 (2011). doi:10.1145/1961189.1961199
- Dongwon-Lee/lsgkm: https://github.com/Dongwon-Lee/lsgkm
- kundajelab/lsgkm (with GkmExplain): https://github.com/kundajelab/lsgkm
