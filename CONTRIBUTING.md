# Contributing

## Install from source

```bash
git clone https://github.com/adamyhe/gkmsvm-lite.git
cd gkmsvm-lite

uv pip install -e "." --group dev              # CPU + test dependencies
uv pip install -e ".[gpu]" --group dev         # + CuPy (NVIDIA GPU)
uv pip install -e ".[mlx]" --group dev         # + MLX (Apple Silicon GPU)
uv pip install -e ".[examples]" --group dev    # + notebook and benchmark dependencies
```

## Run tests

```bash
pytest tests/ -v
pytest tests/test_codec.py       # single file
pytest tests/ -k "test_rc"       # pattern match
```

## Run benchmarks

```bash
make -C benchmarks help      # one target per paper figure panel
make -C benchmarks quick     # small end-to-end run, writes benchmarks/results/quick/
```

See [benchmarks/README.md](benchmarks/README.md) for what each target produces.

## Before a release: MLX (Apple Silicon)

CI runs on Linux, where every MLX test is skipped, so the Metal kernels are
only tested on a Mac. On Apple Silicon with MLX installed:

1. Run the MLX tests and confirm none were skipped (a missing or broken MLX
   install skips the whole file):

   ```bash
   pytest tests/test_mlx.py -rs
   ```

2. Check MLX against the CPU on l-mer indexes from sparse to dense, which
   covers the fused Metal query + reduce, mutation impact, ISM and the RBF
   GkmExplain path. Every operation should agree with the CPU to float32
   precision (max relative difference about 1e-4 or less) and run faster than
   it; the script's docstring has the last release's ratios:

   ```bash
   python benchmarks/dev/mlx_index_density.py
   ```

3. If the change touches training on MLX, compare peak memory and time with
   `results/apple/` (`make -C benchmarks results/apple/training_memory.tsv
   DEVICE=mlx SET=apple THREADS=4`; see benchmarks/README.md, Apple Silicon).

## Project structure

See [CLAUDE.md](CLAUDE.md) for source layout, conventions, and implementation details.
