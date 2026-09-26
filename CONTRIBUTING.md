# Contributing

## Install from source

```bash
git clone https://github.com/adamyhe/gkmsvm-lite.git
cd gkmsvm-lite

uv pip install -e ".[dev]"          # CPU + test dependencies
uv pip install -e ".[dev,gpu]"      # + CuPy GPU support
uv pip install -e ".[dev,bench]"    # + benchmark dependencies (scikit-learn, py2bit)
```

## Run tests

```bash
pytest tests/ -v
pytest tests/test_codec.py       # single file
pytest tests/ -k "test_rc"       # pattern match
```

## Run benchmarks

```bash
python benchmarks/solver_bench.py
python benchmarks/throughput_sweep.py
python benchmarks/replicate_dsqtl.py
```

## Project structure

See [CLAUDE.md](CLAUDE.md) for source layout, conventions, and implementation details.
