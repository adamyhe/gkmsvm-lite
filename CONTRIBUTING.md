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

## Project structure

See [CLAUDE.md](CLAUDE.md) for source layout, conventions, and implementation details.
