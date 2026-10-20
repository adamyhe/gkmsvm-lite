"""Import R gkmSVM models (Ghandi et al. 2016, R/kernlab package).

The package's gkmsvm_train (through 0.83) writes the classic C++ tool's
files, PREFIX_svalpha.out and PREFIX_svseq.fa, with no bias (it drops
kernlab's b) and no kernel parameters: see classic.load_classic_model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gkmsvm.svm import GkmSVM


def load_r_gkmsvm_model(
    path: str | Path,
    svseq_path: str | Path | None = None,
    *,
    L: int = 10,
    K: int = 6,
    maxnmm: int = 3,
    useTgkm: int = 1,
    addRC: bool = True,
    dtype: np.dtype | type = np.float32,
    sv_chunk_size: int | None = None,
    device: str = "cpu",
) -> GkmSVM:
    """Load a model R gkmSVM's gkmsvm_train wrote.

    Args:
        path: PREFIX_svalpha.out.
        svseq_path: PREFIX_svseq.fa; default: next to path.
        L, K, maxnmm, useTgkm, addRC: the kernel, as given to gkmsvm_kernel
            (the package's defaults).
        dtype, sv_chunk_size, device: as in load_lsgkm_model.

    Returns:
        A GkmSVM with bias 0 (classic.load_classic_model).
    """
    from gkmsvm.importers.classic import load_classic_model

    return load_classic_model(
        path,
        svseq_path,
        L=L,
        k=K,
        d=maxnmm,
        filter_type=useTgkm,
        include_rc=addRC,
        dtype=dtype,
        sv_chunk_size=sv_chunk_size,
        device=device,
    )
