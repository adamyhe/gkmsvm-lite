"""Import classic gkmSVM models (Ghandi et al. 2014).

gkmsvm_train (the C++ tool, and the R gkmSVM package's, which shares its
code) writes a model as two files: PREFIX_svalpha.out, one ``SV id <TAB>
signed alpha`` line per SV, and PREFIX_svseq.fa. They hold no bias and no
kernel parameters: gkmsvm_classify scores sum_i alpha_i K(x, sv_i) with
normalized K, and takes -l/-k/-d/-t (R: L, K, maxnmm, useTgkm) on its command
line, so they are arguments here (load_svalpha_model). Both tools read FASTA
by dropping every letter other than ACGT, so SVs are loaded that way.

load_classic_model also reads LIBSVM-style files with ``key value`` headers
and ``rho`` (bias = +rho), with alphas after the header and the SVs in a
FASTA, or ``alpha sequence`` lines after an ``SV`` marker.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gkmsvm.fasta import read_fasta
from gkmsvm.importers import _open_auto, build_model, parse_header_value
from gkmsvm.svm import GkmSVM

# gkmsvm_kernel/gkmsvm_classify -t (R useTgkm) as kernel types: full filter,
# truncated filter (the tools' default) and gkm counts h[m]; each truncated
# at d mismatches. 3 (wildcard) and 4 (mismatch kernel) have no equivalent.
GKMSVM_FILTERS = {0: "gkm_estfull", 1: "gkm_esttrunc", 2: "gkm_cnt"}


def load_svalpha_model(
    alpha_path: str | Path,
    svseq_path: str | Path | None = None,
    *,
    L: int | None = None,
    k: int | None = None,
    d: int | None = None,
    filter_type: int | None = None,
    include_rc: bool | None = None,
    dtype: np.dtype | type = np.float32,
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
            10, 6, 3 (the tools' defaults), or ``#L``/``#k``/``#d`` lines in
            alpha_path.
        filter_type: gkmsvm_kernel -t (R useTgkm): 0 full filter, 1
            truncated (default), 2 gkm counts.
        include_rc: Reverse complements (default True; False for -R, R
            addRC=FALSE).
        dtype, sv_chunk_size, device: as in load_lsgkm_model.

    Returns:
        A GkmSVM with bias 0 (or ``#rho``, when alpha_path has one).
    """
    import re
    import warnings

    alpha_path = Path(alpha_path)
    if svseq_path is None:
        name = alpha_path.name
        if not name.endswith("svalpha.out"):
            raise ValueError(f"give svseq_path for {alpha_path}")
        svseq_path = alpha_path.with_name(name[: -len("svalpha.out")] + "svseq.fa")
    header: dict = {}
    ids, coefficients = [], []
    with _open_auto(alpha_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                parts = line[1:].split(None, 1)
                if len(parts) == 2:
                    parse_header_value(header, parts[0], parts[1])
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
    if filter_type is None:
        filter_type = 1
    if filter_type not in GKMSVM_FILTERS:
        raise NotImplementedError(
            f"gkmSVM filter type {filter_type} (wildcard/mismatch kernels) is "
            f"not supported; use {sorted(GKMSVM_FILTERS)}"
        )
    params = {
        "L": L if L is not None else header.get("L", 10),
        "k": k if k is not None else header.get("k", 6),
        "d": d if d is not None else header.get("d", 3),
        "norc": 0 if (include_rc is None or include_rc) else 1,
        "kernel_type": GKMSVM_FILTERS[filter_type],
    }
    return build_model(
        params,
        coefficients,
        sequences,
        bias=header["rho"][0] if header.get("rho") else 0.0,
        n_expected=header.get("nsv"),
        dtype=dtype,
        sv_chunk_size=sv_chunk_size,
        device=device,
    )


def _is_svalpha(path: Path) -> bool:
    """Whether path holds gkmsvm_train's ``id <TAB> alpha`` lines."""
    if path.name.endswith("svalpha.out"):
        return True
    with _open_auto(path) as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            if len(parts) != 2 or "\t" not in line:
                return False
            try:
                float(parts[1])
            except ValueError:
                return False
            return True
    return False


def _parse_classic_file(f) -> tuple[dict, bool, list[str]]:
    """Header, whether an SV marker was found, and the remaining data lines."""
    header = {"norc": 0}
    found_sv = False
    trailing: list[str] = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith("SV"):
            found_sv = True
            break
        parts = line.split(None, 1)
        if len(parts) < 2:
            trailing.append(line)
        else:
            parse_header_value(header, parts[0], parts[1])
    if found_sv:
        trailing = [line.strip() for line in f if line.strip()]
    return header, found_sv, trailing


def load_classic_model(
    model_path: str | Path,
    *,
    svseq_path: str | Path | None = None,
    dtype: np.dtype | type = np.float32,
    sv_chunk_size: int | None = None,
    device: str = "cpu",
) -> GkmSVM:
    """Load a classic gkmSVM model.

    gkmsvm_train's PREFIX_svalpha.out (with PREFIX_svseq.fa) goes to
    load_svalpha_model with the tools' default kernel (l=10, k=6, d=3,
    truncated filter); call that for other parameters.

    Args:
        model_path: Path to the model file.
        svseq_path: Path to FASTA file with support vector sequences.
        dtype: dtype of the one-hot support vectors. Coefficients are float64 (float32 on MLX) whatever it is.
        sv_chunk_size: ``sv_chunk_size`` of the model (SV chunks in GkmExplain).
        device: ``"cpu"`` (default), ``"cuda"``, ``"mlx"``, or ``"auto"``.

    Returns:
        A GkmSVM instance with bias = +rho.
    """
    if _is_svalpha(Path(model_path)):
        return load_svalpha_model(
            model_path,
            svseq_path,
            dtype=dtype,
            sv_chunk_size=sv_chunk_size,
            device=device,
        )
    with _open_auto(Path(model_path)) as f:
        header, has_sv_section, trailing = _parse_classic_file(f)
    nr_class = header.get("nr_class", 2)
    if nr_class != 2:
        raise NotImplementedError(
            f"Only binary classification (nr_class=2) is supported, got {nr_class}"
        )

    coefficients, sequences = [], []
    for line in trailing:
        if line.startswith("#"):
            continue
        parts = line.split()
        if has_sv_section and svseq_path is None:  # "<coef> <sequence>" lines
            if len(parts) < 2:
                raise ValueError(f"Malformed SV line: {line[:80]}")
            sequences.append(parts[1])
        try:
            coefficients.append(float(parts[0]))
        except ValueError:
            raise ValueError(f"Expected alpha coefficient, got: {line[:80]}") from None
    if svseq_path is not None:
        sequences = [seq for _, seq in read_fasta(svseq_path)]
    if header.get("rho") is None:
        raise ValueError("Model file missing rho field")

    return build_model(
        header,
        coefficients,
        sequences,
        bias=header["rho"][0],
        n_expected=header.get("total_sv"),
        dtype=dtype,
        sv_chunk_size=sv_chunk_size,
        device=device,
    )
