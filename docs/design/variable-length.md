# Variable-length sequences

Status: accepted (2026-10-19); in progress on the `variable-length` branch.

## Problem

LS-GKM trains on and scores sequences of any length (up to `MAX_SEQ_LENGTH` = 2048 bp in `libsvm_gkm.h`). Each sequence has its own `seqlen - L + 1` windows. For `-t 4/5`, its window weights come from its own window count (`gkmkernel_new_object`). gkmsvm-lite assumes one length per array everywhere:

- **Importing** an LS-GKM model whose SVs differ in length raises (`importers.build_model`). Many real models are like this, since LS-GKM is usually trained on peaks.
- **Training and the CLI** stack per-sequence one-hot arrays, so mixed lengths raise. Benchmark S3B works around it: LS-GKM trains on the variable-length peaks and gkmsvm-lite on fixed 300 bp windows (`benchmarks/ctcf_data.py`).
- **`model()`, ISM, GkmExplain and mutation impact** take `[B, 4, L]` with a single L.

## What LS-GKM computes

Nothing about the kernel changes:

    K_raw(x, y) = sum over x's windows i and y's windows j of wt_i(x) wt_j(y) h(m(x_i, y_j))   (+ RC block)

Each sequence contributes only its own windows. wt is 1 except for `-t 4/5`, where it is computed from that sequence's own W. Normalization uses each sequence's own K(x, x). So a variable-length batch only needs to represent **which windows exist** and **each existing window's weight**.

## What the C tools do at the edges

Checked by running the Beer-lab LS-GKM, the kundajelab fork (both `MAX_SEQ_LENGTH` 2048) and mLS-GKM (`MAX_SEQ_LENGTH` 4096, commit 2bb06be) on the same model and inputs:

| | LS-GKM (both builds) | mLS-GKM |
|---|---|---|
| Mixed lengths in one file | each sequence has its own windows, `-t 4/5` weights and norm | same; scores identical up to 2,047 bp |
| Longer than the maximum | first 2,047 bases, with a warning | first 4,095 bases; `gkmpredict` gives no warning |
| Shorter than l, scoring | score `nan`, no warning (gkmexplain: `nan` attributions) | score `nan`, attributions 0 |
| Shorter than l, training | one such sequence gives a model with 0 SVs and rho `nan`, exit 0, no warning | same |
| gkmexplain output | one line per sequence at its own length | same |
| gkmexplain `-m 5`, even length | assertion abort (exit 134) | same |

## Decisions

1. **API:** both forms. A `[B, 4, L_max]` array plus `lengths`, or a list of `[4, L_i]` arrays or DNA strings (converted with `encode_batch`).
2. **ISM and GkmExplain output:** padded `[B, 4, L_max]`, zero past each length, for both input forms. It holds the same information as gkmexplain's ragged lines and stays on the device.
3. **CLI `explain` with mixed lengths:** write padded arrays and warn that TF-MoDISco needs one length.
4. **Long sequences:** full length. The C tools disagree (2,047 vs 4,095, with and without a warning), so no truncation would reproduce "LS-GKM". A warning names sequences over 2,047 bp, which LS-GKM would cut. An optional `max_length` (e.g. 2047) truncates the same way LS-GKM does. Oracle fixtures stay at 2,047 bp or less.
5. **Sequences shorter than l:** training raises an error that names them (LS-GKM silently fits a useless model). Scoring returns `nan` for them with a warning, as LS-GKM does, so one bad sequence doesn't stop a batch. ISM and GkmExplain give `nan` rows.
6. **Mutation impact (`-m 5` semantics) on even lengths:** a clear error, not an abort.

## Representation: padded arrays with explicit lengths

- **One-hot:** a batch is `[B, 4, L_max]` plus `lengths [B]` (int). Positions past a sequence's length are all-zero. N is also all-zero, so the lengths must be explicit; they can't be inferred from the array.
- **Packed:** `pack(x, lengths)` gives `[B, W_max]` uint32, with `PAD_WINDOW` (0xFFFFFFFF) for every window past `lengths[b] - l + 1`. `_rc_windows` maps PAD to PAD and reverses rows as it does now. The RC row of a padded sequence then holds the pads first and the real RC windows after, which is exactly the reverse of the forward row. The self-kernels' "is this the RC of that" check keeps working.
- **Window weights (`-t 4/5`):** per sequence and window, `[B, W_max]` uint8: `window_weights(W_b)` for the real windows, 0 for pads. An RC row's weights are the forward row's weights reversed, as now.
- **Uniform length** (`lengths=None`, every current caller): unchanged. No PAD, `[W]` weights, the same code paths. Variable length is a superset, so the fixed-length fast paths are kept.

**PAD must be skipped explicitly, never through mismatch counts.** Today's code relies on PAD cancelling in new − old (ISM). That isn't enough here:
- PAD against PAD has 0 mismatches, so it would get the full weight h(0).
- Masked to an l-mer's bits, PAD equals the all-T l-mer, so a masked join would pair it with real TTT… windows.
- Brute-force kernels index `table[popcount(...)]` with up to 16 mismatches, past the end of the `l + 1` table.

So every kernel that reads packed windows gets a `w == PAD_WINDOW → skip` test on both sides. PAD is only distinct from every real window for l ≤ 15, so variable-length input requires l ≤ 15 (l ≤ 12 in practice).

## Changes by path

| Path | Change | Devices |
|---|---|---|
| `pack` / `pack_both` | Take `lengths`; write PAD past each length; RC keeps PAD | CPU, CUDA, MLX (MLX packs on CPU) |
| Brute force (`_fused_pairwise_packed_*`, `_brute_weighted_numba`) | Skip PAD rows and columns; weighted: per-row weights `[B, W]` | CPU, CUDA, Metal |
| Self-kernels (`_self_diagonal_packed_numba`, `_self_diagonal_weighted_numba`, CUDA/Metal cross-diagonal), `raw_diag_mutants` | Skip PAD; weights per row | CPU, CUDA, Metal |
| Masked join (`_build_column_index`, `_join_rows*`, CUDA `masked_hist`/`masked_scatter`/`masked_join*`, Metal index/join) | Skip PAD when building the column index and when joining rows. Column weights are already per window (`wby [R, W]`); row weights become `[B, W]` | CPU, CUDA, Metal |
| l-mer index (`LmerIndex`) | Leave PAD out of the CSR keys (otherwise `bincount` would grow to 2^32); per-entry `vals` from per-row weights. Queries already skip `v >= 4^l`; weighted queries scale by per-row weights | CPU, CUDA, Metal |
| l-mer table (`build_lmer_table`, `_score_lmer_table`) | Build: drop PAD before `bincount`, weights per row. Score: PAD reads a zero entry (append one to the table and map PAD to it), times per-row weights | CPU, CUDA, MLX |
| Gram (`compute_gram`), SMO (`KernelColumnCache`), Nyström | Carry `lengths` with X; tiles and panels are subsets of rows, so their lengths and weights are slices | all |
| GkmExplain (table, l-mer, index, per-SV; CUDA `explain_lmer`, `fused_explain`) | Skip PAD query windows (output past a sequence's length stays 0) and PAD SV windows; per-row weights | CPU, CUDA |
| ISM (`_packed_mutations`, table / index / kernel paths) | Positions past a sequence's length give delta 0. Per-row slot weights (`_slot_weights` per length) | CPU, CUDA, MLX |
| Mutation impact | Packs ref/alt with their lengths; ref and alt have the same length per variant (SNVs) | all |
| `to_deltasvm`, `DeltaSVM` scoring | Weights are per l-mer, so unchanged. `DeltaSVM` scoring of a padded batch skips PAD windows | all |

## API

- **Inputs:** `model(x, lengths=None)`, `ism`, `gkmexplain`, `mutation_impact`, `score_variants`, `compute_gram`, `train_gkmsvm`/`train_gkmsvr` and `DeltaSVM.__call__` accept either:
  - a `[B, 4, L_max]` array plus `lengths`, or
  - a list of `[4, L_i]` arrays or DNA strings, converted with `codec.encode_batch` (which already returns padded arrays plus a mask; it would return lengths).
- **Outputs:** scores are `[B, 1]` as now. ISM and GkmExplain return `[B, 4, L_max]`, zero past each length, for either input form.
- **Models:** `GkmSVM.support_sequences` stays `[S, 4, L_max]`, plus `sv_lengths [S]` (None when uniform). npz stores `sv_lengths` (older npz files have none, so they are uniform). `save_lsgkm` writes each SV trimmed to its length. `DeviceModel` moves `sv_lengths` with the arrays.
- **CLI:** FASTAs with mixed lengths work in every subcommand. `explain` writes padded `[B, 4, L_max]` attribution and sequence arrays and warns when lengths differ (TF-MoDISco needs one length).

## Validation (LS-GKM oracles)

Using the Beer-lab and kundajelab builds, as for the existing fixtures (`benchmarks/dev/lsgkm_wgkm_fixtures.py`):

1. **Scores:** models trained by gkmtrain on mixed-length sequences (for example 60-140 bp, with N), `-t 0` to `-t 5`; gkmpredict on mixed-length test sets.
2. **Attributions:** gkmexplain modes 0/1/4 on mixed-length queries; mode 5 on odd lengths (it explains the central base, so each sequence must be odd).
3. **Training:** `train_gkmsvm` on mixed-length sets against gkmtrain's models (SV count, held-out scores, as in the fixed-length check).
4. **Every path and device:** each kernel path on padded input against the same sequences one at a time (fixed length, already checked), on CPU, CUDA and MLX.

## Performance and memory

- **Memory and work:** padding costs memory, and work for paths that loop over all windows (brute force, the column index build), in proportion to `L_max / mean L`. The table and index paths are built from SV windows, so only their packed SV arrays pay. Peak sets usually span about 2x. For a very skewed set (for example 50 bp to 2 kb), the Gram and SMO code could sort sequences by length so that each tile pads only to its own maximum. This is optional and only worth doing if measured.
- **Uniform length:** nothing changes (`lengths=None` keeps today's paths). Benchmarks S2 and 1C should show the same times.

## Stages

1. **Representation and the CPU kernels:** pack with lengths and PAD; skip PAD in brute force, the self-kernels, the masked join, the index and the table; per-row weights. Scoring padded input on CPU matches one-at-a-time scoring and gkmpredict.
2. **Models:** importers (lift the error), `sv_lengths`, npz, `save_lsgkm`, device moves; CLI `predict` on mixed FASTAs.
3. **Training:** Gram, SMO and Nyström with lengths; `train_gkmsvm`/`train_gkmsvr` and the CLI; gkmtrain fixtures.
4. **Interpretation:** ISM, GkmExplain (all paths), mutation impact; gkmexplain fixtures.
5. **CUDA and Metal:** PAD skips and per-row weights in the RawKernels and Metal kernels; `test_gpu.py` / `test_mlx.py` variable-length cases.
6. **Benchmarks:** S3B trains both tools on the same variable-length peaks.

Stages 1 to 3 make imported models and training work on CPU. Each stage is one commit on one branch.
