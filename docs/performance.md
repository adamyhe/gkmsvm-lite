# GPU performance

## Compute precision

Kernel computation uses **float32 on GPU** (CUDA and MPS) and **float64 on CPU**.

Consumer NVIDIA GPUs (GeForce/RTX) have severely degraded float64 throughput
— 1:64 FP64:FP32 ratio on Ampere (RTX 3080), meaning float64 arithmetic is
**64x slower** than float32. Even data-center cards vary (A100 is 1:2, but
L4/T4 are 1:32). float32 is the right default for GPU inference.

Measured on RTX 3080, bmm at the kernel's actual shape (1000× 290×44 @ 44×290):
- float32: 1.1 ms/iter
- float64: 26.3 ms/iter (24x slower)

### Why float32 is safe for gkm kernels

The match computation (`einsum("bif,sjf->bsij", wx, wy)`) produces dot
products of one-hot window vectors — exact integers ≤ l (typically 11). These
are representable exactly in float32 (23-bit mantissa) and even in TF32
(19-bit mantissa, integers up to 524,288).

The mismatch table lookup (`_apply_table_histogram`) multiplies integer counts
by table weights and sums at most d+1 = 4 terms. float32 rounding here is
negligible for scoring.

Normalization (`K_raw / sqrt(diag_x * diag_y)`) accumulates table weights
over all window pairs. The diagonal values are sums of O(W²) terms where
W ≈ 290, so ~84,100 terms. float32 relative error is ~1e-7 per operation,
so accumulated error is ~1e-4 — well within tolerance for SVM scoring.

### TF32 (Tensor Float 32)

On Ampere and later GPUs, PyTorch enables TF32 by default for float32
matmuls (`torch.backends.cuda.matmul.allow_tf32 = True` since PyTorch 1.12).
TF32 uses 19-bit mantissa (vs float32's 23-bit) but provides up to 8x
throughput on tensor cores. Since gkm kernel match counts are small integers,
TF32 is exact for the matmul step and provides a free speedup.

If exact float32 results are needed (e.g., for oracle validation), disable
TF32 explicitly: `torch.backends.cuda.matmul.allow_tf32 = False`.

## Pre-cached SV diagonal

The SVM's `forward()` caches the support-vector self-kernel diagonal
(`_raw_diagonal(sv)`) on first call. This avoids recomputing the diagonal
(a chunked loop over all SVs with window extraction, bmm, and table lookup)
on every inference call. For ENCODE-scale models (72K SVs), this saves
hundreds of GPU kernel launches per forward pass.

The cache auto-invalidates on device change (e.g., after `.cuda()` or `.cpu()`).

## Bottleneck analysis: gkm kernel vs. RBF kernel

The gapped k-mer kernel is inherently more expensive than a standard RBF
kernel on pre-extracted features:

| | RBF on features (e.g., pydreg) | gkm string kernel |
|---|---|---|
| Kernel eval per (query, SV) | D multiply-adds (one dot product) | W² × l values (all window-pair matches) |
| Typical dimensions | D = 360 | W = 290, l = 11 → 84,100 × 11 |
| GPU op pattern | Single GEMM (ideal for cuBLAS) | Batched small matmuls + histogram |
| FLOPs per SV | ~720 | ~5,000,000 |

This is inherent to the kernel function, not an implementation gap. The gkm
kernel evaluates sequence similarity directly from DNA, without a separate
feature extraction step — that generality costs compute.

## torch.compile fusion

`model.compile()` uses `torch.compile` to fuse the einsum + histogram
table lookup into a single Triton GPU kernel. Instead of 4 separate passes
over the match tensor (one per nonzero table entry), the compiled kernel
reads it once and accumulates all bins — ~15x less memory traffic.

The first forward call after `.compile()` triggers compilation (~5s).
Subsequent calls run the fused kernel.

## bf16 (bfloat16) compute

`model.compile(dtype=torch.bfloat16)` converts flat windows to bf16 before
the compiled pairwise kernel. On Ampere+ GPUs, bf16 matmuls run on tensor
cores at ~60% higher throughput than float32.

bf16 is safe for gkm kernels because the einsum produces integer match
counts ≤ l (typically 11). bf16 has a 7-bit mantissa (integers exact up
to 127), so match counts are exact. The histogram accumulation stays in
float32 (the table dtype on GPU). Measured score diff vs float32: ~2e-6.

## Throughput comparison (ENCODE ENCFF579AOX, 72K SVs, 300bp, RTX 3080)

| Implementation | seq/s | vs LS-GKM C |
|---|---|---|
| gkmsvm-lite GPU compiled bf16 | **6.6** | **5.3x faster** |
| gkmsvm-lite GPU compiled f32 | 4.1 | 3.3x faster |
| LS-GKM C (`gkmpredict`) | ~1.25 | — |
| gkmsvm-lite GPU uncompiled | ~0.5 | 2.5x slower |
| gkmsvm-lite CPU | ~0.017 | 75x slower |

Usage:
```python
model = load_lsgkm_model("model.txt", sv_chunk_size=5000)
model = model.cuda().compile(dtype=torch.bfloat16)
scores = model(sequences)  # first call compiles (~5s), then 6+ seq/s
```

Score accuracy: GPU bf16 vs CPU float64 differ by ~2e-6 (negligible).
Both differ from LS-GKM C by ~3% on the ENCODE model due to N-base
handling (4 of 72K SVs contain N characters; gkmsvm-lite encodes N as
all-zero, LS-GKM C handles N differently).
