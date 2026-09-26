"""Model serialization: npz (native) and LS-GKM text (interop) formats."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from gkmsvm.codec import one_hot_decode
from gkmsvm.importers.lsgkm import KERNEL_TYPE_REVERSE


def save_npz(model, path: str | Path) -> None:
    """Save a GkmSVM model in native npz format."""
    from gkmsvm.backend import to_cpu

    metadata = {
        "bias": float(model.bias),
        "kernel_type": model.kernel_type,
        "kernel_params": model._kernel_params,
    }
    if model.sv_chunk_size is not None:
        metadata["sv_chunk_size"] = model.sv_chunk_size

    np.savez(
        path,
        support_sequences=to_cpu(model.support_sequences),
        coefficients=to_cpu(model.coefficients),
        metadata=np.void(json.dumps(metadata).encode("utf-8")),
    )


def load_npz(path: str | Path):
    """Load a GkmSVM model from native npz format."""
    from gkmsvm.svm import GkmSVM

    data = np.load(path, allow_pickle=False)
    metadata = json.loads(bytes(data["metadata"]))

    return GkmSVM(
        support_sequences=data["support_sequences"],
        coefficients=data["coefficients"],
        bias=metadata["bias"],
        kernel_type=metadata["kernel_type"],
        kernel_params=metadata["kernel_params"],
        sv_chunk_size=metadata.get("sv_chunk_size"),
    )


def save_lsgkm(model, path: str | Path) -> None:
    """Save a GkmSVM model in LS-GKM text format."""
    from gkmsvm.backend import to_cpu

    path = Path(path)
    params = model._kernel_params
    coefs = to_cpu(model.coefficients)
    svs = to_cpu(model.support_sequences)

    n_pos = int((coefs > 0).sum())
    n_neg = int((coefs <= 0).sum())

    lines = []
    lines.append(f"svm_type c_svc")
    lines.append(f"kernel_type {model.kernel_type}")
    lines.append(f"L {params['L']}")
    lines.append(f"k {params['k']}")
    if "d" in params:
        lines.append(f"d {params['d']}")
    if "gamma" in params:
        lines.append(f"gamma {params['gamma']}")
    if "M" in params:
        lines.append(f"M {params['M']}")
    if "H" in params:
        lines.append(f"H {params['H']}")

    include_rc = params.get("include_rc", True)
    lines.append(f"norc {0 if include_rc else 1}")
    lines.append(f"nr_class 2")
    lines.append(f"total_sv {len(coefs)}")
    lines.append(f"rho {-model.bias}")
    lines.append(f"label 1 -1")
    lines.append(f"nr_sv {n_pos} {n_neg}")
    lines.append("SV")

    for i in range(len(coefs)):
        seq_str = one_hot_decode(svs[i])
        lines.append(f"{coefs[i]} {seq_str}")

    text = "\n".join(lines) + "\n"

    if str(path).endswith(".gz"):
        with gzip.open(path, "wt") as f:
            f.write(text)
    else:
        path.write_text(text)


def load_model(path: str | Path):
    """Load a GkmSVM model, auto-detecting format from extension.

    Supports:
        - ``.npz`` — native format
        - ``.txt``, ``.txt.gz``, ``.model`` — LS-GKM text format
    """
    path = Path(path)
    name = path.name.lower()

    if name.endswith(".npz"):
        return load_npz(path)

    from gkmsvm.importers.lsgkm import load_lsgkm_model
    return load_lsgkm_model(str(path))
