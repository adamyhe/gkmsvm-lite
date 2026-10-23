"""Import classic gkmSVM models (Ghandi et al. 2014).

gkmsvm_train (the C++ tool, and the R gkmSVM package's, which shares its
code) writes a model as two files: PREFIX_svalpha.out, one ``SV id <TAB>
signed alpha`` line per SV, and PREFIX_svseq.fa. They hold no bias and no
kernel parameters: gkmsvm_classify scores sum_i alpha_i K(x, sv_i) with
normalized K, and takes -l/-k/-d/-t (R: L, K, maxnmm, useTgkm) on its command
line, so they are arguments here. Both tools read FASTA by dropping every
letter other than ACGT, so SVs are loaded that way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gkmsvm.fasta import read_fasta
from gkmsvm.importers import _open_auto, build_model
from gkmsvm.svm import GkmSVM

# gkmsvm_kernel/gkmsvm_classify -t (R useTgkm) as kernel types: full filter,
# truncated filter (the tools' default) and gkm counts h[m]; each truncated
# at d mismatches. 3 (wildcard) and 4 (mismatch kernel) have no equivalent.
GKMSVM_FILTERS = {0: "gkm_estfull", 1: "gkm_esttrunc", 2: "gkm_cnt"}


def load_classic_model(
    alpha_path: str | Path,
    svseq_path: str | Path | None = None,
    *,
    L: int = 10,
    k: int = 6,
    d: int = 3,
    filter_type: int = 1,
    include_rc: bool = True,
    sv_chunk_size: int | None = None,
    device: str = "cpu",
) -> GkmSVM:
    """Load a model gkmsvm_train wrote (classic C++ or R gkmSVM).

    Scores equal sum_i alpha_i K(x, sv_i) with the K of the tools'
    gkmsvm_kernel, the kernel the alphas were fit with. gkmsvm_classify's
    own output differs from that by up to ~2% (its tree and simple
    algorithms normalize differently; tests/test_importers/test_gkmsvm_tools.py).

    Args:
        alpha_path: PREFIX_svalpha.out (``id <TAB> alpha`` lines).
        svseq_path: PREFIX_svseq.fa; default: next to alpha_path. SVs are
            matched to alphas by id, and their non-ACGT letters dropped, as
            the tools read them.
        L, k, d: word length, informative positions and maximum mismatches,
            as given to gkmsvm_kernel (-l/-k/-d, R L/K/maxnmm); default
            10, 6, 3 (the tools' defaults).
        filter_type: gkmsvm_kernel -t (R useTgkm): 0 full filter, 1
            truncated (default), 2 gkm counts.
        include_rc: Reverse complements (default True; False for -R, R
            addRC=FALSE).
        sv_chunk_size, device: as in load_lsgkm_model.

    Returns:
        A GkmSVM with bias 0.
    """
    import re
    import warnings

    alpha_path = Path(alpha_path)
    if svseq_path is None:
        name = alpha_path.name
        if not name.endswith("svalpha.out"):
            raise ValueError(f"give svseq_path for {alpha_path}")
        svseq_path = alpha_path.with_name(name[: -len("svalpha.out")] + "svseq.fa")
    ids, coefficients = [], []
    with _open_auto(alpha_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Expected 'id <TAB> alpha', got: {line[:80]}")
            ids.append(parts[0])
            coefficients.append(float(parts[1]))
    by_id = {name.split()[0]: seq for name, seq in read_fasta(str(svseq_path))}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(
            f"{len(missing)} SV ids of {alpha_path} are not in {svseq_path}, "
            f"e.g. {missing[0]}"
        )
    sequences = [re.sub("[^ACGTacgt]", "", by_id[i]).upper() for i in ids]
    dropped = sum(len(by_id[i]) != len(s) for i, s in zip(ids, sequences))
    if dropped:
        warnings.warn(
            f"{dropped} support vectors hold letters other than ACGT, dropped "
            "as gkmSVM reads them (queries with N score as in gkmsvm_classify "
            "only with those letters dropped too)",
            stacklevel=2,
        )
    if filter_type not in GKMSVM_FILTERS:
        raise NotImplementedError(
            f"gkmSVM filter type {filter_type} (wildcard/mismatch kernels) is "
            f"not supported; use {sorted(GKMSVM_FILTERS)}"
        )
    params = {
        "L": L,
        "k": k,
        "d": d,
        "norc": 0 if include_rc else 1,
        "kernel_type": GKMSVM_FILTERS[filter_type],
    }
    return build_model(
        params,
        coefficients,
        sequences,
        bias=0.0,
        n_expected=None,
        sv_chunk_size=sv_chunk_size,
        device=device,
    )
