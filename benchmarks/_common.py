"""Shared paths, measurement helpers and result-file provenance for the paper benchmarks.

Timed runs execute in their own child process (``run_measured``) so that the
peak RSS reported for a run belongs to that run alone, whether it is an LS-GKM
binary or a gkmsvm-lite worker.

Every result TSV starts with ``# key: value`` lines recording the git commit,
hardware and software versions it was produced with, followed by a normal
tab-separated table. Read them with ``pd.read_csv(path, sep="\\t", comment="#")``
and the header with ``read_provenance(path)``.
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent
RESULTS = BENCH / "results"
FIGURES = BENCH / "figures"
DATA = BENCH / "data"
MODELS = BENCH / "models"
SEQ_DIR = REPO / "examples" / "data" / "gm12878_sequence_sets"


# LS-GKM's gkmtrain/gkmpredict accept only these -T values.
LSGKM_THREADS = (1, 4, 16)


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30,
                             check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _git_commit() -> str:
    commit = _run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO) or "unknown"
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO)
    return commit + ("-dirty" if dirty else "")


def _cpu_model() -> str:
    if sys.platform == "darwin":
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if name:
            return name
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _gpu() -> str | None:
    try:
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        name = name.decode() if isinstance(name, bytes) else name
        cc = f"{props['major']}.{props['minor']}"
        return f"{name} ({props['totalGlobalMem'] / 1024**3:.0f} GB, CC {cc})"
    except (ImportError, RuntimeError):
        return None


def _versions() -> str:
    from importlib.metadata import PackageNotFoundError, version

    import numba
    import numpy as np

    try:
        lite = version("gkmsvm-lite")
    except PackageNotFoundError:
        lite = "?"
    parts = [
        f"gkmsvm-lite {lite}",
        f"python {platform.python_version()}",
        f"numpy {np.__version__}",
        f"numba {numba.__version__} ({numba.config.NUMBA_NUM_THREADS} threads)",
    ]
    try:
        import cupy as cp

        parts.append(f"cupy {cp.__version__}")
    except ImportError:
        pass
    return ", ".join(parts)


def lsgkm_default_threads() -> int:
    """Largest LS-GKM -T value the machine has cores for."""
    return max(t for t in LSGKM_THREADS if t <= (os.cpu_count() or 1))


def lsgkm_binary(name: str, conda_env: str = "lsgkm", bin_dir: str | None = None) -> str | None:
    """Path to an LS-GKM binary: ``bin_dir/name`` if given, else from ``conda_env``."""
    if bin_dir:
        path = Path(bin_dir) / name
        return str(path) if path.exists() else None
    return _run(["conda", "run", "-n", conda_env, "which", name])


def lsgkm_has_option(binary: str, option: str) -> bool:
    """Whether an LS-GKM binary's usage text lists ``option`` (e.g. ``-T``)."""
    try:
        out = subprocess.run([binary], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return f" {option} " in out.stdout + out.stderr


def lsgkm_version(conda_env: str = "lsgkm") -> str | None:
    """Version of the bioconda ``ls-gkm`` package in ``conda_env``, if any."""
    out = _run(["conda", "list", "-n", conda_env, "ls-gkm", "--json"])
    if not out:
        return None
    pkgs = json.loads(out)
    return f"ls-gkm {pkgs[0]['version']} (bioconda)" if pkgs else None


def provenance(device: str | None = None, **extra) -> dict[str, str]:
    """Key/value header describing how a result was produced."""
    info = {
        "commit": _git_commit(),
        "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "host": socket.gethostname(),
        "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "cpu": f"{_cpu_model()}, {os.cpu_count()} logical cores",
        "versions": _versions(),
    }
    if device is not None:
        info["device"] = device
    if device in ("cuda", "both"):
        info["gpu"] = _gpu() or "unavailable"
    info.update({k: str(v) for k, v in extra.items() if v is not None})
    return info


def write_header(f, title: str, device: str | None = None, **extra) -> None:
    """Write the provenance header to an open text file."""
    f.write(f"# {title}\n")
    for k, v in provenance(device, **extra).items():
        f.write(f"# {k}: {v}\n")


def write_tsv(path, df, title: str, device: str | None = None, **extra) -> Path:
    """Write a DataFrame as TSV preceded by the provenance header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        write_header(f, title, device, **extra)
        df.to_csv(f, sep="\t", index=False)
    print(f"Saved {path}")
    return path


def read_provenance(path) -> dict[str, str]:
    """Parse the ``# key: value`` header of a result file."""
    info = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            key, sep, value = line[1:].strip().partition(": ")
            if sep:
                info[key] = value
    return info


# ── Measured child processes ───────────────────────────────────────


@dataclass
class Measured:
    """Outcome of ``run_measured``."""

    status: str  # "ok", "failed" or "timeout"
    wall_s: float
    peak_rss_mb: float | None
    stdout: str
    stderr: str


def run_measured(cmd: list[str], timeout: float | None = None) -> Measured:
    """Run ``cmd`` and report its wall time and its own peak RSS.

    Uses ``os.wait4`` for the child's resource usage, so it works on Linux and
    macOS without GNU time. Output goes through temp files, so large outputs
    can't fill a pipe and stall the child.
    """
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, stdout=out, stderr=err, text=True)
        status = "ok"
        while True:
            pid, code, usage = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                break
            if timeout is not None and time.perf_counter() - t0 > timeout:
                proc.kill()
                pid, code, usage = os.wait4(proc.pid, 0)
                status = "timeout"
                break
            time.sleep(0.05)
        wall = time.perf_counter() - t0
        proc.returncode = os.waitstatus_to_exitcode(code)
        if status == "ok" and proc.returncode != 0:
            status = "failed"
        out.seek(0)
        err.seek(0)
        stdout, stderr = out.read(), err.read()
    # ru_maxrss is KiB on Linux, bytes on macOS
    rss = usage.ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024)
    return Measured(status, wall, rss, stdout, stderr)


def worker_result(stdout: str) -> dict | None:
    """Parse the ``RESULT {json}`` line a benchmark worker prints last."""
    for line in reversed(stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    return None


def emit_result(**values) -> None:
    """Print a worker's result for ``worker_result`` to pick up."""
    print("RESULT " + json.dumps(values), flush=True)


def gpu_sync() -> None:
    try:
        import cupy as cp

        cp.cuda.Stream.null.synchronize()
    except ImportError:
        pass


def gpu_pool_mb() -> float | None:
    """MB held by CuPy's default memory pool (blocks are cached, so this tracks the peak)."""
    try:
        import cupy as cp
    except ImportError:
        return None
    return cp.get_default_memory_pool().total_bytes() / 1024**2
