# GkmExplain on MLX (Metal)

Status: done for the l-mer and index paths (`explain._explain_lmer_metal`), as planned in steps 1 and 3 below plus a compensated sum for the RBF neighbor weights (step 2 was not needed). The explain table and the per-SV kernel still run on a CPU copy of the model; the result is NumPy float64 on every path.

Measured (100 random 300 bp sequences, M4; error relative to the CPU's largest attribution):

| Model | Mode | Error | CPU ms/seq | Metal ms/seq |
|---|---|---|---|---|
| ENCFF579AOX (`-t 2`) | hypothetical | 3.9e-7 | 3.5 | 0.9 |
| ENCFF579AOX | perturbation | 6.0e-7 | 27.6 | 5.4 |
| dsQTL `-t 2` | hypothetical | 4.1e-7 | 1.5 | 0.8 |
| dsQTL `-t 2` | perturbation | 5.7e-7 | 14.4 | 3.5 |
| dsQTL `-t 3` | hypothetical | 3.3e-7 | 20.3 | 16.6 |
| dsQTL `-t 3` | perturbation | 5.1e-7 | 71.9 | 44.8 |

Plain float32 per-window sums held to 4-6e-7 for `-t 2`. For `-t 3` the neighbor sums over the index's CSR entries (signed SV weights, ~18 entries per l-mer on 32,715 SVs) gave 6e-6 in float32 and 3e-7 with a Neumaier sum. The `-t 3` time is mostly the CPU index query for the RBF weights (the Metal index query is slower on that SV set: contended atomics). Importance sums to score − bias to 1.4e-7 of the largest score.

The plan as written before the implementation follows.

## Why it runs on CPU today

Metal has no float64. GkmExplain's importance sums to score − bias (the completeness axiom), and the paths accumulate in float64 to keep that, and to match the kundajelab fork's gkmexplain to its 6 printed digits.

## Where precision matters, by path

- **Explain table** (`-t 0/1/2/4`, l <= 12, large batches): each output entry is a sum of at most l table entries A[u, d, k], which already hold the SV sums (built in float64). Float32 lookups and sums would be fine, but the CPU path costs 0.035 ms per 300 bp sequence (M4), so a GPU gains little.
- **l-mer path** (`_explain_lmer_cpu`) and **index path** (`_explain_index_cpu`, always taken by RBF models `-t 3/5`): each query window visits its dmax-neighborhood (up to `MAX_NEIGHBORS` = 8,192 l-mers) and adds V[v] (or the per-query sum of SV weights over the index entries) times the alpha/kappa weight. A window's partial is a sum of thousands of signed terms, so cancellation is where float32 could fail. This is the path a GPU would speed up: 3.5 ms per sequence for ENCFF579AOX and 29 ms for the dsQTL `-t 3` model on an M4 CPU; the CUDA version (`_explain_lmer_gpu`) is 5-14x the CPU on a Titan Xp.
- **Per-SV kernel** (`_explain_cpu`, large neighborhoods without the l-mer path): one (query, SV) pair per thread, accumulating into the result.

## Plan

1. **No atomics, float64 across windows.** Each Metal thread group takes one query window and writes its [4, l] partial (four sums by base delta per position, as the CUDA kernel keeps in registers) to a [B, W, 4, l] buffer: about 50 KB per 300 bp sequence in float32. The CPU folds the overlapping windows into [B, 4, L] in float64, applying 1/sqrt(K(x, x)) there. Every sum across windows (and, for the per-SV kernel, across SVs, by writing [B, S-chunk, ...] partials or reducing per chunk) then happens in float64.
2. **Float-float within a window, if needed.** Keep each lane's running sum as an unevaluated pair of float32s (TwoSum/FastTwoSum with `fma`), about 48 bits of mantissa, and store V and the alpha/kappa tables as hi/lo pairs. A few extra flops per term, which the GPU has to spare.
3. **RBF weights.** The index path needs w[b, s] = coef_s / sqrt(K(sv, sv)) · f(K_norm(x, sv)). The Metal l-mer index already gives exact K_raw(x, sv) (int32 counts per mismatch level, combined on CPU), so w is computed in float64 on CPU and uploaded per batch.

Not quantization: rounding V or the tables to int8/int16 adds ~1e-3 relative error (fails the gkmexplain oracles). Exact fixed point (a value split over several int32 limbs, each accumulated separately and recombined on CPU) would make sums exact and order-independent, like the Metal index's level counts, but needs a per-model scale and more bookkeeping than float-float.

## First step: measure

Replay `_explain_lmer_cpu` and `_explain_index_cpu` in NumPy with float32 per-window sums on ENCFF579AOX (`-t 2`) and the dsQTL `-t 3` model, against float64. If plain float32 holds to ~1e-6 of the largest attribution, steps 1 and 3 suffice (about a day's work: one Metal kernel shaped like `_EXPLAIN_LMER_CUDA`, the fold, and tests against the CPU on every kernel type); otherwise add step 2.

## Validation

The CPU explain paths are the oracle (they match gkmexplain): `tests/test_mlx.py` on every kernel type and mode, with N, odd/even window counts, mixed-length queries and SVs, at a tolerance set from the measurement above; completeness (importance sums to score − bias) to the same tolerance.
