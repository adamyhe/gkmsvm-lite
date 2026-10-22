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
import warnings
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
        out = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _git_commit() -> str:
    commit = _run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO) or "unknown"
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO)
    return commit + ("-dirty" if dirty else "")


def usable_cores() -> int:
    """CPUs this process may run on: its affinity mask, capped by a cgroup quota.

    os.cpu_count() counts every core on the machine, which overstates what a
    job gets under taskset, a batch scheduler or a container CPU limit.
    """
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:  # macOS
        n = os.cpu_count() or 1
    quota = None
    try:  # cgroup v2
        q, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if q != "max":
            quota = int(q) / int(period)
    except (OSError, ValueError):
        try:  # cgroup v1
            q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if q > 0:
                quota = q / period
        except (OSError, ValueError):
            pass
    if quota is not None:
        n = min(n, max(1, int(quota)))
    return n


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
        driver = cp.cuda.runtime.driverGetVersion()
        return (
            f"{name} ({props['totalGlobalMem'] / 1024**3:.0f} GB, CC {cc}, "
            f"CUDA driver {driver // 1000}.{driver % 1000 // 10})"
        )
    except (ImportError, RuntimeError):
        return None


def _slurm() -> str | None:
    """The Slurm job this process runs in: id, partition, node, constraint."""
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return None
    parts = [f"job {job}"]
    for label, var in (
        ("partition", "SLURM_JOB_PARTITION"),
        ("node", "SLURMD_NODENAME"),
        # Set by benchmarks/sherlock/job.sh: Slurm exports no constraint variable
        ("constraint", "BENCH_CONSTRAINT"),
        ("exclusive", "BENCH_EXCLUSIVE"),
    ):
        if os.environ.get(var):
            parts.append(f"{label} {os.environ[var]}")
    return ", ".join(parts)


def _versions() -> str:
    from importlib.metadata import PackageNotFoundError, version

    import gkmsvm  # noqa: F401  pins Numba's threading layer before first use
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
        (
            f"numba {numba.__version__} ({numba.get_num_threads()} threads, "
            f"{numba.threading_layer()} layer)"
        ),
    ]
    try:
        import cupy as cp

        parts.append(f"cupy {cp.__version__}")
    except ImportError:
        pass
    return ", ".join(parts)


# Thread variables for processes started by run_measured(); see set_cpu_threads
_CHILD_THREAD_ENV: dict[str, str] = {}


def set_cpu_threads(n: int) -> int:
    """Run gkmsvm-lite's CPU code on ``n`` threads, here and in child processes.

    Matches gkmsvm-lite to LS-GKM's -T so CPU timings compare like for like.
    Sets Numba's thread count in this process, and the thread variables that
    run_measured() passes to worker processes. This process's environment is
    left alone: Numba rereads NUMBA_NUM_THREADS whenever it compiles and
    raises if it differs from the pool it already started. Returns the count
    used (capped at the cores Numba sees).
    """
    import gkmsvm  # noqa: F401  pins Numba's threading layer before first use
    import numba

    usable = usable_cores()
    if n > usable:
        warnings.warn(
            f"{n} threads requested but this process can use {usable} CPU(s) "
            "(affinity or cgroup limit); CPU timings will not reflect that thread "
            "count. Request more cores from the scheduler.",
            RuntimeWarning,
            stacklevel=2,
        )
    n = max(1, min(n, numba.config.NUMBA_NUM_THREADS))
    numba.set_num_threads(n)
    for var in (
        "NUMBA_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        _CHILD_THREAD_ENV[var] = str(n)
    return n


def lsgkm_default_threads() -> int:
    """Largest LS-GKM -T value this process has usable cores for."""
    return max(t for t in LSGKM_THREADS if t <= usable_cores())


def lsgkm_binary(
    name: str, conda_env: str = "lsgkm", bin_dir: str | None = None
) -> str | None:
    """Path to an LS-GKM binary: from ``conda_env`` if it has one, else ``bin_dir/name``.

    The conda build comes first because its gkmpredict takes -T; ``bin_dir``
    supplies binaries the env lacks (gkmexplain, which the kundajelab fork
    has; its gkmpredict is single-threaded).
    """
    path = _run(["conda", "run", "-n", conda_env, "which", name])
    if path:
        return path
    if bin_dir and (Path(bin_dir) / name).exists():
        return str(Path(bin_dir) / name)
    return None


def warm_binaries(*binaries: str | None) -> None:
    """Run each C binary once, untimed, before timed runs.

    Without arguments LS-GKM's binaries print their usage and exit, which
    reads the executable and its shared libraries (from a conda env or a
    source build, on a network filesystem on clusters) into the page cache,
    so the first timed run doesn't pay for that and later ones don't skip it.
    """
    for b in dict.fromkeys(b for b in binaries if b):
        for _ in range(2):
            try:
                subprocess.run([b], capture_output=True, timeout=60, check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass


def lsgkm_has_option(binary: str, option: str) -> bool:
    """Whether an LS-GKM binary's usage text lists ``option`` (e.g. ``-T``)."""
    try:
        out = subprocess.run(
            [binary], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    import re

    # Usage lines may indent options with tabs: match on any whitespace
    return (
        re.search(
            rf"(^|\s){re.escape(option)}(\s|$)", out.stdout + out.stderr, re.MULTILINE
        )
        is not None
    )


# mLS-GKM (Howard & Harmston 2026, github.com/harmstonlab/mLS-GKM): LS-GKM with
# multi-class models and worker threads across sequences (-t). Built from
# source; it reads LS-GKM model files and, for binary models with -D, writes
# gkmpredict/gkmexplain output in LS-GKM's format
MLSGKM = "mLS-GKM"


def mlsgkm_binary(name: str, bin_dir: str | None) -> str | None:
    """Path to an mLS-GKM binary in bin_dir, or None."""
    path = Path(bin_dir) / name if bin_dir else None
    return str(path) if path is not None and path.exists() else None


def c_threads_opts(impl: str, n_threads: int) -> list[str]:
    """Options running LS-GKM or mLS-GKM gkmpredict/gkmexplain on n_threads CPU threads.

    LS-GKM's -T splits each sequence's kernel sum (callers pass it only to
    binaries that list -T); mLS-GKM's -t runs that many worker threads over
    sequences with -T 1, as its README advises for many sequences, and -D
    selects binary decision values (it requires an output mode).
    """
    if impl == MLSGKM:
        return ["-D", "-t", str(n_threads), "-T", "1"]
    return ["-T", str(n_threads)] if n_threads > 1 else []


def mlsgkm_version(bin_dir: str | None) -> str | None:
    """git commit of an mLS-GKM checkout (bin_dir is <checkout>/bin), if any."""
    if not bin_dir:
        return None
    commit = _run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=Path(bin_dir).resolve().parent
    )
    return f"mLS-GKM {commit}" if commit else "mLS-GKM"


def lsgkm_version(conda_env: str = "lsgkm") -> str | None:
    """Version of the bioconda ``ls-gkm`` package in ``conda_env``, if any."""
    out = _run(["conda", "list", "-n", conda_env, "ls-gkm", "--json"])
    if not out:
        return None
    pkgs = json.loads(out)
    return f"ls-gkm {pkgs[0]['version']} (bioconda)" if pkgs else None


def lsgkm_sources(binaries: dict[str, str | None], conda_env: str = "lsgkm") -> str:
    """Where each LS-GKM binary came from, for result headers.

    ``lsgkm_binary`` takes a binary from the conda env when it has it (bioconda's
    ls-gkm, the official Dongwon-Lee/lsgkm) and from ``bin_dir`` otherwise (a
    source build, e.g. the kundajelab fork for gkmexplain), so one run can mix both.
    """
    parts = []
    for name, path in binaries.items():
        if not path:
            continue
        prefix = Path(path).resolve().parent.parent
        conda = (prefix / "conda-meta").is_dir()
        origin = (lsgkm_version(conda_env) or "conda") if conda else "source build"
        parts.append(f"{name}: {origin} ({path})")
    return "; ".join(parts)


def provenance(device: str | None = None, **extra) -> dict[str, str]:
    """Key/value header describing how a result was produced."""
    info = {
        "commit": _git_commit(),
        "date": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"
        ),
        "host": socket.gethostname(),
        "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "cpu": f"{_cpu_model()}, {os.cpu_count()} logical cores, {usable_cores()} usable",
        "versions": _versions(),
    }
    if _slurm():
        info["slurm"] = _slurm()
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


# ── CPU and GPU parts of one result file ───────────────────────────


def _read_part(path: Path):
    """(title, provenance, {section or None: DataFrame of strings})."""
    import io

    import pandas as pd

    lines = path.read_text().splitlines()
    title = lines[0][2:] if lines and lines[0].startswith("# ") else ""
    sections, current = {None: []}, None
    for line in lines:
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
        elif line and not line.startswith("#"):
            sections[current].append(line)
    frames = {
        k: pd.read_csv(
            io.StringIO("\n".join(v)), sep="\t", dtype=str, keep_default_na=False
        ).replace({"NA": None, "": None})
        for k, v in sections.items()
        if v
    }
    return title, read_provenance(path), frames


def merge_parts(parts: dict[str, Path], out: Path, on: dict | None = None) -> Path:
    """Combine result files that ran the CPU and GPU halves of one benchmark.

    ``parts`` maps a label ("cpu", "gpu") to a part file written by
    ``write_tsv``/``write_header``, the first part taking precedence. Tables
    are concatenated, or with ``on`` ({section: key columns}) joined on those
    columns, each cell taken from the first part that has a value (for
    timing_train.py's wide, sectioned file). Header keys that agree across
    parts are written once; otherwise the first part's value keeps the key and
    the others are written as ``<label>_<key>`` (``gpu_host`` etc.). Values
    are copied as text, so numbers keep the parts' formatting.
    """
    import pandas as pd

    read = {label: _read_part(Path(p)) for label, p in parts.items()}
    labels = list(read)
    title = read[labels[0]][0]
    provs = {label: r[1] for label, r in read.items()}
    commits = {p.get("commit") for p in provs.values()}
    if len(commits) > 1:
        warnings.warn(f"{out}: parts ran different commits {commits}", stacklevel=2)
    header = {}
    for label in labels:
        for key, value in provs[label].items():
            if key not in header:
                header[key] = value
            elif header[key] != value:
                header[f"{label}_{key}"] = value
    devices = {p.get("device") for p in provs.values()}
    if "cuda" in devices:
        header["device"] = "cuda"  # what a single run on CPU and GPU recorded
        for label in labels:
            header.pop(f"{label}_device", None)
    header["parts"] = ", ".join(f"{label} {Path(parts[label]).name}" for label in labels)

    sections = list(dict.fromkeys(k for r in read.values() for k in r[2]))
    merged = {}
    for sec in sections:
        frames = [r[2][sec] for r in read.values() if sec in r[2]]
        keys = (on or {}).get(sec)
        if keys:
            df = frames[0].set_index(keys)
            for f in frames[1:]:
                df = df.combine_first(f.set_index(keys))
            # combine_first sorts the index and the columns; keep the first part's order
            cols = list(dict.fromkeys(c for f in frames for c in f.columns if c not in keys))
            df = df.reset_index()[keys + cols]
            df = df.iloc[pd.to_numeric(df[keys[0]], errors="coerce").argsort(kind="stable")]
        else:
            df = pd.concat(frames, ignore_index=True)
        merged[sec] = df

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"# {title}\n")
        for key, value in header.items():
            f.write(f"# {key}: {value}\n")
        for sec, df in merged.items():
            if sec is not None:
                f.write(f"\n## {sec}\n")
            df.to_csv(f, sep="\t", index=False, na_rep="NA" if on else "")
    print(f"Merged {', '.join(str(p) for p in parts.values())} -> {out}")
    return out


def rewrite_table(path, df) -> None:
    """Replace the table of a single-table result file, keeping its header."""
    path = Path(path)
    header = [line for line in path.read_text().splitlines() if line.startswith("#")]
    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        df.to_csv(f, sep="\t", index=False)


# ── Measured child processes ───────────────────────────────────────


def write_seqs(path, seqs: list[str], prefix: str) -> None:
    """FASTA with records named prefix0, prefix1, ..."""
    with open(path, "w") as f:
        f.writelines(f">{prefix}{i}\n{s}\n" for i, s in enumerate(seqs))


def read_seqs(path) -> list[str]:
    """Sequences of a FASTA file (gzipped if the name ends in .gz)."""
    import gzip

    path = Path(path)
    with gzip.open(path, "rt") if path.suffix == ".gz" else open(path) as f:
        records = f.read().split(">")[1:]
    return ["".join(r.splitlines()[1:]) for r in records]


@dataclass
class Measured:
    """Outcome of ``run_measured``."""

    status: str  # "ok", "failed" or "timeout"
    wall_s: float
    peak_rss_mb: float | None
    stdout: str
    stderr: str


def _vm_hwm_mb(pid: int) -> float | None:
    """Peak RSS of ``pid``'s current address space (Linux ``VmHWM``), in MB."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError):
        pass
    return None


def run_measured(cmd: list[str], timeout: float | None = None) -> Measured:
    """Run ``cmd`` and report its wall time and its own peak RSS.

    On Linux the peak is the child's ``VmHWM``, polled while it runs: the
    ``ru_maxrss`` that ``wait4`` returns there also counts the address space
    the child had before exec, which (fork or vfork) is this process's, so
    every child would measure at least this process's RSS. ``VmHWM`` only
    grows, so polling misses at most the last poll interval. Without /proc
    (macOS, where ``ru_maxrss`` doesn't inherit) it falls back to ``wait4``.
    Output goes through temp files, so large outputs can't fill a pipe and
    stall the child.
    """
    hwm = None
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        t0 = time.perf_counter()
        # Popen returns once the child has exec'd, so every poll sees its own memory
        proc = subprocess.Popen(
            cmd,
            stdout=out,
            stderr=err,
            text=True,
            env={**os.environ, **_CHILD_THREAD_ENV},
        )
        status = "ok"
        while True:
            h = _vm_hwm_mb(proc.pid)
            if h is not None:
                hwm = max(hwm or 0.0, h)
            pid, code, usage = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                break
            if timeout is not None and time.perf_counter() - t0 > timeout:
                proc.kill()
                pid, code, usage = os.wait4(proc.pid, 0)
                status = "timeout"
                break
            time.sleep(0.01)
        wall = time.perf_counter() - t0
        proc.returncode = os.waitstatus_to_exitcode(code)
        if status == "ok" and proc.returncode != 0:
            status = "failed"
        out.seek(0)
        err.seek(0)
        stdout, stderr = out.read(), err.read()
    # ru_maxrss is KiB on Linux, bytes on macOS
    rss = (
        hwm
        if hwm is not None
        else usage.ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024)
    )
    return Measured(status, wall, rss, stdout, stderr)


def worker_result(stdout: str) -> dict | None:
    """Parse the ``RESULT {json}`` line a benchmark worker prints last."""
    for line in reversed(stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    return None


def emit_result(**values) -> None:
    """Print a worker's result for ``worker_result`` to pick up."""
    print("RESULT " + json.dumps(values), flush=True)


def gpu_sync() -> None:
    """Wait for queued GPU work; a no-op without a GPU (CuPy also imports on a
    cluster's CPU nodes, where synchronizing raises)."""
    from gkmsvm.backend import cuda_available

    if cuda_available():
        import cupy as cp

        cp.cuda.Stream.null.synchronize()


def gpu_pool_mb() -> float | None:
    """MB held by CuPy's default memory pool (blocks are cached, so this tracks the peak)."""
    from gkmsvm.backend import cuda_available

    if not cuda_available():
        return None
    import cupy as cp

    return cp.get_default_memory_pool().total_bytes() / 1024**2


def _jit_workload(device: str) -> None:
    """Small calls through the training, scoring and attribution kernels on device."""
    import numpy as np
    from gkmsvm import gkmexplain, ism, one_hot_encode, train_gkmsvm
    from gkmsvm.backend import to_cpu

    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGT"), 60)) for _ in range(80)]
    X = np.stack([one_hot_encode(s) for s in seqs])
    alt = X.copy()
    alt[:, :, 30] = np.roll(X[:, :, 30], 1, axis=1)
    for kernel in ("estimated", "rbf"):
        for solver in ("libsvm", "smo", "nystrom"):
            model = train_gkmsvm(
                seqs[:40],
                seqs[40:],
                kernel_type=kernel,
                l=6,
                k=4,
                d=2,
                solver=solver,
                device=device,
                verbose=False,
            )
        if device == "cuda":
            model.cuda()
        for table, index in ((True, True), (False, True), (False, False)):
            model.use_lmer_table, model.use_lmer_index = table, index
            to_cpu(model(X))
        for method in ("kernel", "gkmexplain"):
            to_cpu(model.score_variants(X, alt, method=method))
        to_cpu(ism(model, X[:4]))
        if kernel == "estimated":
            to_cpu(gkmexplain(model, X[:4]))
        dsvm = model.to_deltasvm(query_norm=True)
        if device == "cuda":
            dsvm.cuda()
        to_cpu(dsvm.score_variants(X, alt))
    gpu_sync()


def jit_cache_files() -> set[str]:
    """Files in the on-disk Numba and CuPy kernel caches.

    Both write a cache file only when they compile (loading writes nothing),
    so files that appear while a timed process runs mean it compiled
    something: compare snapshots taken before and after it.
    """
    from importlib.util import find_spec

    files = set()
    numba_dir = os.environ.get("NUMBA_CACHE_DIR")
    if numba_dir:
        roots = [Path(numba_dir)]
    else:  # next to the sources, in __pycache__
        spec = find_spec("gkmsvm")
        roots = [Path(spec.origin).parent] if spec and spec.origin else []
    for root in roots:
        if root.is_dir():
            files |= {str(p) for ext in ("*.nbi", "*.nbc") for p in root.rglob(ext)}
    cupy_dir = Path(
        os.environ.get("CUPY_CACHE_DIR") or Path.home() / ".cupy" / "kernel_cache"
    )
    if cupy_dir.is_dir():
        files |= {str(p) for p in cupy_dir.rglob("*") if p.is_file()}
    return files


def warm_jit_cache(device: str) -> None:
    """Fill the on-disk kernel caches before timed runs, untimed.

    Numba (cache=True) and CuPy keep compiled kernels on disk. After an
    install or a kernel edit the first process to use a kernel compiles it,
    which would land in whichever timed run comes first. device "cuda" warms
    CPU and CUDA kernels.
    """
    t0 = time.perf_counter()
    for dev in ["cpu", "cuda"] if device == "cuda" else ["cpu"]:
        m = run_measured(
            [sys.executable, str(Path(__file__).resolve()), "--warm-jit", dev]
        )
        if m.status != "ok":
            print(f"JIT warm-up on {dev} {m.status}: {m.stderr[-500:]}")
    print(f"JIT cache warm-up: {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__" and sys.argv[1:2] == ["--warm-jit"]:
    _jit_workload(sys.argv[2])
