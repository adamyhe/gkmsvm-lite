# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

gkmsvm-lite is a PyTorch implementation of gapped k-mer SVMs (gkm-SVMs) for DNA sequence classification. It loads existing LS-GKM models and integrates into sequence-to-function (S2F) workflows. Scores are validated against Dongwon-Lee/lsgkm `gkmpredict` to floating-point precision.

Design decisions: `docs/design.md`. Attribution methods: `docs/attribution.md`. Roadmap: `docs/roadmap.md`.

## Build and test

```bash
pip install -e ".[dev]"
pytest tests/ -v
pytest tests/test_codec.py          # single file
pytest tests/ -k "test_rc"          # pattern match
```

## Conventions

- `src/` layout, source under `src/gkmsvm/`
- Python >=3.10, PyTorch >=2.0, tangermeme >=1.4
- Tensor format: `[batch, 4, length]` one-hot DNA, channel order A=0/C=1/G=2/T=3
- Output: `[batch, 1]` floating-point margin scores
- Kernel normalization on by default, RC equivalence on by default
- Score = `Σ coef_i × K(x, sv_i) + bias` where `bias = -rho` (LS-GKM) or `+rho` (classic gkmSVM)
- Kernel modes: `-t 0` (direct), `-t 1` (est full), `-t 2` (est trunc, default), `-t 3` (RBF), `-t 4` (center-weighted), `-t 5` (center-weighted RBF)
- ISM via `ism(model, x)` returns `[B, 4, L]` score deltas using window-delta optimization
- GkmExplain via `gkmexplain(model, x, mode=0|1)` returns `[B, 4, L]` attribution scores, 20-30x faster than ISM
- Gradient-based methods (DeepLIFT, SHAP, captum) are incompatible — use GkmExplain or ISM
- ledidi requires differentiable models and does not work with gkm-SVMs
- GPU: `model.cuda().compile(dtype=torch.bfloat16)` fuses kernel + uses bf16 tensor cores (5.3x faster than LS-GKM C)
- GPU uses float32 (or bf16 with compile), CPU uses float64. Score diff is ~2e-6, negligible for scoring

## Gotchas

- LS-GKM defaults are `-t 2 -l 11 -k 7 -d 3`. Always pass parameters explicitly when generating oracle fixtures.
- The `d` parameter zeros weight table entries for m > d. Without this, scores diverge ~1.5% from `gkmpredict`.
- `-t 0` and `-t 2` use completely different weight table math. Do not mix.
- `tangermeme.kmers.gapped_kmers` is NOT usable for kernel computation (returns CSR, caps at 10 k-mers).
- `score(ALT) - score(REF)` differs from deltaSVM's k-mer-weight linear approximation.
- Original gkmSVM (`.gkmmodel`) uses opposite sign convention for bias vs LS-GKM. Use `load_classic_model()` not `load_lsgkm_model()`.
- `-t 4`/`-t 5` (wgkm/wgkmrbf) require M and H parameters; these use per-position DP, not the matmul+table path.
