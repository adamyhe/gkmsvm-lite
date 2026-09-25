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
- Score = `Σ coef_i × K(x, sv_i) + bias` where `bias = -rho`
- ISM via `ism(model, x)` returns `[B, 4, L]` score deltas using window-delta optimization
- GkmExplain via `gkmexplain(model, x, mode=0|1)` returns `[B, 4, L]` attribution scores, 20-30x faster than ISM
- Gradient-based attribution (DeepLIFT, SHAP, captum, ledidi) is incompatible — use GkmExplain or ISM

## Gotchas

- LS-GKM defaults are `-t 2 -l 11 -k 7 -d 3`. Always pass parameters explicitly when generating oracle fixtures.
- The `d` parameter zeros weight table entries for m > d. Without this, scores diverge ~1.5% from `gkmpredict`.
- `-t 0` and `-t 2` use completely different weight table math. Do not mix.
- `tangermeme.kmers.gapped_kmers` is NOT usable for kernel computation (returns CSR, caps at 10 k-mers).
- `score(ALT) - score(REF)` differs from deltaSVM's k-mer-weight linear approximation.
- Original gkmSVM (`.gkmmodel`) uses opposite sign convention for bias vs LS-GKM.
