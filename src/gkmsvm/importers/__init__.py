from __future__ import annotations

import gzip
from pathlib import Path


def _open_auto(path: Path):
    """Open a file, auto-detecting gzip compression."""
    try:
        f = gzip.open(path, "rt")
        f.readline()
        f.seek(0)
        return f
    except gzip.BadGzipFile:
        return open(path, "r")
