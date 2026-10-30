# Validation

gkmsvm-lite is checked against LS-GKM in two ways: oracle tests against `gkmpredict`, `gkmexplain` and `gkmtrain` output on small fixtures, and a full reproduction of the GkmExplain paper's dsQTL benchmark using the paper's own models.

## Oracle tests

| What | Reference | Test |
|---|---|---|
| Forward pass, `-t 2` | `gkmpredict` on the Nanog reference model and an ENCODE model | `tests/test_replication.py`, `tests/test_importers/test_encode_integration.py`, `tests/test_reference/test_reference.py` |
| Forward pass, `-t 0` to `-t 6` | `gkmpredict` on fixture models (`-t 6`: the kundajelab fork's), with N and odd/even window counts | `tests/test_explain.py::test_fixture_models_match_gkmpredict`, `tests/test_kernels/test_weighted.py::test_matches_gkmpredict`, `tests/test_explain.py::TestMutationImpactOracle` |
| GkmExplain importance, hypothetical, perturbation | kundajelab fork's `gkmexplain -m 0/1/4` (rtol 2e-5) on every kernel type, every CPU explain path | `tests/test_explain.py::test_matches_lsgkm_gkmexplain` |
| Mutation impact score | lsgkm `gkmexplain -m 5` on every kernel type, through the l-mer table, index, masked join and brute force | `tests/test_explain.py::TestMutationImpactOracle`, `test_matches_lsgkm_mode5_fixture_models` |
| Training, C-SVC | `gkmtrain` for `-t 0` to `-t 6` and `-w`: same SVs, coefficients to the solvers' 1e-3 tolerance, `gkmpredict`'s scores to 5e-4 (LIBSVM and SMO); gkmtrain's solutions meet our KKT conditions | `tests/test_train_oracle.py` |
| Training, epsilon-SVR | kundajelab/lsgkm-svr for `-t 2/3/4` (LIBSVM and SMO) | `tests/test_train_oracle.py::test_svr_matches_lsgkm_svr` |
| Sequences of different lengths | `gkmpredict`, `gkmtrain` and `gkmexplain` (modes 0/1/4/5) on mixed-length models and queries (`lsgkm_ragged_*`) | `tests/test_variable_length.py`, `tests/test_explain.py` |
| Classic and R gkmSVM models | each tool's `gkmsvm_kernel` (rtol 1e-5) | `tests/test_importers/test_gkmsvm_tools.py` |

Fixtures were generated with lsgkm built from source, with all parameters passed explicitly (`tests/fixtures/lsgkm_*`; `benchmarks/dev/lsgkm_wgkm_fixtures.py`, `benchmarks/dev/gkmsvm_fixtures.py`).

## dsQTL variant effect prediction

Shrikumar et al. (2019), Figure 8, reproduced with the paper's published lsgkm models ([kundajelab/gkmexplain](https://github.com/kundajelab/gkmexplain/tree/master/dsQTL)).

**Data.** 28,309 GM12878 SNPs from Lee et al. (2015): 574 dsQTLs and 27,735 controls. Variants are ranked by |score| and evaluated by AUPRC. A random ranking gives the fraction of dsQTLs, 0.0203 (the AUPRC base rate), for every table below.

**Models.** Five lsgkm models per kernel, one per Beer lab negative set, `l=10 k=6 d=3`:
- `-t 2` (gkm_esttrunc)
- `-t 3` (gkmrbf, `c=10 g=2`)

**Methods.**

| Method | Input | Score |
|---|---|---|
| deltaSVM | 19bp Beer lab sequences | Σ w(alt l-mers) − Σ w(ref l-mers), w = `gkmpredict` on each 10-mer (`to_deltasvm(query_norm=True)`) |
| ISM | 51bp hg18 context | score(alt) − score(ref) |
| GkmExplain | 51bp hg18 context | mutation impact score, lsgkm `gkmexplain -m 5` (`score_variants(method="gkmexplain")`) |

### Per-model AUPRC, `-t 3`

ISM and GkmExplain match the paper's per-variant outputs to within 1e-5 on every model.

| Negset | ISM | paper | GkmExplain | paper | deltaSVM.pl | paper deltaSVM* | deltaSVM |
|---|---|---|---|---|---|---|---|
| 1 | 0.18808 | 0.18808 | 0.18905 | 0.18905 | 0.1829 | 0.18287 | 0.16423 |
| 2 | 0.18968 | 0.18968 | 0.19101 | 0.19101 | 0.1865 | 0.18649 | 0.18202 |
| 3 | 0.18433 | 0.18433 | 0.18523 | 0.18523 | 0.1800 | 0.18001 | 0.17567 |
| 4 | 0.18569 | 0.18569 | 0.18698 | 0.18698 | 0.1803 | 0.18028 | 0.17825 |
| 5 | 0.19432 | 0.19432 | 0.19477 | 0.19477 | 0.1873 | 0.18733 | 0.17515 |

### Per-model AUPRC, `-t 2`

The paper reports deltaSVM only for the `-t 2` models; ISM and GkmExplain are ours.

| Negset | deltaSVM.pl | paper deltaSVM* | deltaSVM | ISM | GkmExplain |
|---|---|---|---|---|---|
| 1 | 0.17918 | 0.17918 | 0.18216 | 0.1858 | 0.1860 |
| 2 | 0.18566 | 0.18565 | 0.18795 | 0.1910 | 0.1926 |
| 3 | 0.17697 | 0.17697 | 0.17952 | 0.1825 | 0.1833 |
| 4 | 0.17943 | 0.17943 | 0.18247 | 0.1867 | 0.1873 |
| 5 | 0.18483 | 0.18483 | 0.18890 | 0.1918 | 0.1925 |

\* The paper's deltaSVM scores come from a scoring-script quirk (below). `deltaSVM.pl` reproduces it; its scores match the paper's per variant to within 5.4e-5 (6-decimal output, float32 weight table).

### Scores summed over the five models

| Method | `-t 3` | paper | `-t 2` | paper |
|---|---|---|---|---|
| deltaSVM | 0.1821 | — | 0.1931 | — |
| deltaSVM.pl | 0.1902 | 0.19021 | 0.1900 | 0.18997 |
| ISM | 0.1955 | 0.19550 | 0.1974 | — |
| GkmExplain | 0.1965 | 0.19647 | 0.1982 | — |

The paper evaluated ISM and GkmExplain on the `-t 3` models only. The paper column is the AUPRC of the summed per-negset output files in the gkmexplain repo. The repo README's deltaSVM sums (0.18943 and 0.18795) cover only three of the five models (below).

### Differences from the paper's scripts

Two bugs in the paper's scripts affect its deltaSVM numbers. Neither affects ISM or GkmExplain.

1. **Dropped poly-A/T l-mers.** The Beer lab `deltasvm.pl` reads the first line of the weight file as a bias header and discards it. The paper scored the 10-mers from lsgkm `nrkmers.py` with `gkmpredict`, so that first line is `AAAAAAAAAA`. Its weight, and the weight of its reverse complement `TTTTTTTTTT`, therefore count as 0. This changes the scores of the 63 variants whose 19bp windows contain either 10-mer. On the `-t 3` models it raises deltaSVM AUPRC on every model, and on the `-t 2` models it lowers it on every model. `deltaSVM.pl` in the benchmark reproduces it. Because `to_deltasvm` weights have the bias subtracted, a dropped l-mer is `-bias` there.
2. **Three-model sums.** `deltasvm_out/combine_folds.sh` pastes five 2-column files and keeps columns 1, 2, 6, 10, 14 and 18. Those positions fit 4-column files, so the sum includes negsets 1, 3 and 5 only. Columns 14 and 18 are empty and count as 0. The ISM script keeps the correct columns.

Plain deltaSVM (every l-mer scored) is the method as described. With both bugs removed, deltaSVM still ranks below ISM and GkmExplain for both kernels (summed: 0.1821 vs 0.1955 and 0.1965 for `-t 3`; 0.1931 vs 0.1974 and 0.1982 for `-t 2`).

GkmExplain's mutation impact score is a linearization of the kernel change, not ISM. The two differ per variant, and GkmExplain scores slightly higher than ISM on all ten models.

### Reproducing

```bash
# AUPRC tables (downloads hg18, ~940 MB, and the paper's models)
make -C benchmarks fig1d     # or: python benchmarks/dsqtl_auprc.py --device cuda --params paper_t2,paper_t3

# Per-variant comparison against the paper's output files
make -C benchmarks fig1a     # or: python benchmarks/agreement_lsgkm.py --device cuda
```

`dsqtl_auprc.py` also trains and evaluates models with gkmsvm-lite (`--params l10k6`, `l11k7`, `l10k6_rbf`, `l11k7_rbf`; `make -C benchmarks s1`). On an NVIDIA L40S, each 51bp `-t 3` model takes about 6 s for ISM and 5 s for GkmExplain over all 28,309 variants, and 6 s to build its deltaSVM weight table (40 s, 28 s and 20 s on 16 CPU threads; `benchmarks/results/timing_variants.tsv`, see [performance](performance.md#measurements)).

The tables above were last regenerated on 2026-10-29 on an L40S, with all five negsets and both kernels (`make -C benchmarks fig1a fig1d`, `results/agreement_checks.tsv`): per variant, ISM matches the paper's outputs to 1.0e-5 and GkmExplain to 5.0e-6 on every `-t 3` model, and `deltaSVM.pl` to 5.4e-5 on all ten. These runs check the CUDA paths against lsgkm's outputs directly.

## References

- Shrikumar A, Prakash E, Kundaje A. GkmExplain: fast and accurate interpretation of nonlinear gapped k-mer SVMs. *Bioinformatics* 35(14):i173–i182 (2019).
- Lee D, Gorkin DU, Baker M, Strober BJ, Asoni AL, McCallion AS, Beer MA. A method to predict the impact of regulatory variants from DNA sequence. *Nat Genet* 47(8):955–961 (2015).
