"""Throughput sweep: batch size × sequence length × device.

Measures steady-state inference throughput for the ENCODE ENCFF579AOX model
(72K SVs, esttrunc l=11 k=7 d=3). Separates model loading, warmup/JIT,
and scoring time.

Usage:
    python benchmarks/throughput_sweep.py              # CPU only
    python benchmarks/throughput_sweep.py --device cuda # GPU
    python benchmarks/throughput_sweep.py --device both # side-by-side
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

FIXTURE_DIR = Path(__file__).parent.parent / "tests" / "fixtures"
MODEL_PATH = FIXTURE_DIR / "encode_ENCFF579AOX.model.txt.gz"

SEQ_LENGTHS = [19, 50, 100, 200, 300]
BATCH_SIZES = [1, 4, 16, 64, 256]
MIN_SEQS = 64
MIN_TIME = 2.0


def _sync_gpu():
    try:
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    except ImportError:
        pass


def random_onehot(batch: int, length: int, xp=np) -> np.ndarray:
    idx = xp.random.randint(0, 4, size=(batch, length))
    x = xp.zeros((batch, 4, length), dtype=xp.float64)
    for b in range(batch):
        for p in range(length):
            x[b, idx[b, p], p] = 1.0
    return x


def bench_one(model, seq_len: int, batch_size: int, device: str) -> dict:
    if device == "cuda":
        import cupy as cp
        xp = cp
    else:
        xp = np

    x = random_onehot(batch_size, seq_len, xp)

    # Warmup (JIT + cache population)
    _ = model(x)
    _sync_gpu()

    # Score enough sequences to get a stable measurement
    n_iters = max(1, MIN_SEQS // batch_size)
    while True:
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = model(x)
        _sync_gpu()
        elapsed = time.perf_counter() - t0
        if elapsed >= MIN_TIME or n_iters >= 1024:
            break
        n_iters = min(n_iters * 2, 1024)

    total_seqs = n_iters * batch_size
    seqs_per_sec = total_seqs / elapsed
    ms_per_seq = elapsed / total_seqs * 1000

    return {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "device": device,
        "seqs_per_sec": seqs_per_sec,
        "ms_per_seq": ms_per_seq,
        "total_seqs": total_seqs,
        "elapsed": elapsed,
    }


def run_sweep(device: str):
    from gkmsvm.importers.lsgkm import load_lsgkm_model

    if not MODEL_PATH.exists():
        print(f"ERROR: model not found at {MODEL_PATH}")
        return []

    print(f"\n{'='*70}")
    print(f"  Throughput sweep — {device.upper()}")
    print(f"  Model: ENCFF579AOX ({MODEL_PATH.name})")
    print(f"{'='*70}")

    print(f"\nLoading model...", end=" ", flush=True)
    t0 = time.perf_counter()
    model = load_lsgkm_model(str(MODEL_PATH))
    load_time = time.perf_counter() - t0
    print(f"{load_time:.1f}s ({model.num_support_vectors} SVs)")

    if device == "cuda":
        print("Moving to GPU...", end=" ", flush=True)
        t0 = time.perf_counter()
        model.cuda()
        cuda_time = time.perf_counter() - t0
        print(f"{cuda_time:.1f}s")

    # JIT warmup
    print("JIT warmup...", end=" ", flush=True)
    if device == "cuda":
        import cupy as cp
        xp = cp
    else:
        xp = np
    warmup = random_onehot(4, 50, xp)
    t0 = time.perf_counter()
    _ = model(warmup)
    _sync_gpu()
    jit_time = time.perf_counter() - t0
    print(f"{jit_time:.1f}s")

    # Header
    print(f"\n{'len':>5} {'batch':>6}  {'seq/s':>10} {'ms/seq':>10} {'total':>6} {'time':>7}")
    print("-" * 55)

    results = []
    for seq_len in SEQ_LENGTHS:
        for batch_size in BATCH_SIZES:
            r = bench_one(model, seq_len, batch_size, device)
            results.append(r)
            print(
                f"{r['seq_len']:>5} {r['batch_size']:>6}"
                f"  {r['seqs_per_sec']:>10.1f} {r['ms_per_seq']:>10.2f}"
                f" {r['total_seqs']:>6} {r['elapsed']:>7.2f}s"
            )
        print()

    # Summary: best batch size per sequence length
    print(f"\nBest batch size per sequence length ({device}):")
    print(f"{'len':>5} {'batch':>6} {'seq/s':>10}")
    print("-" * 25)
    for seq_len in SEQ_LENGTHS:
        group = [r for r in results if r["seq_len"] == seq_len]
        best = max(group, key=lambda r: r["seqs_per_sec"])
        print(f"{seq_len:>5} {best['batch_size']:>6} {best['seqs_per_sec']:>10.1f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "both"],
    )
    args = parser.parse_args()

    all_results = {}
    devices = ["cpu", "cuda"] if args.device == "both" else [args.device]

    for device in devices:
        if device == "cuda":
            try:
                import cupy  # noqa: F401
            except ImportError:
                print("SKIP: CuPy not available")
                continue
        all_results[device] = run_sweep(device)

    if len(all_results) == 2 and "cpu" in all_results and "cuda" in all_results:
        print(f"\n{'='*70}")
        print(f"  GPU / CPU speedup")
        print(f"{'='*70}")
        print(f"{'len':>5} {'batch':>6} {'CPU seq/s':>10} {'GPU seq/s':>10} {'speedup':>8}")
        print("-" * 45)
        cpu_by_key = {(r["seq_len"], r["batch_size"]): r for r in all_results["cpu"]}
        for r_gpu in all_results["cuda"]:
            key = (r_gpu["seq_len"], r_gpu["batch_size"])
            r_cpu = cpu_by_key.get(key)
            if r_cpu:
                speedup = r_gpu["seqs_per_sec"] / max(r_cpu["seqs_per_sec"], 0.001)
                print(
                    f"{key[0]:>5} {key[1]:>6}"
                    f" {r_cpu['seqs_per_sec']:>10.1f} {r_gpu['seqs_per_sec']:>10.1f}"
                    f" {speedup:>7.1f}x"
                )


if __name__ == "__main__":
    main()
