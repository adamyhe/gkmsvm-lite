"""Training: fit a C-SVM with gapped k-mer kernel."""

from __future__ import annotations

import numpy as np

from gkmsvm.backend import get_array_module
from gkmsvm.codec import one_hot_encode
from gkmsvm.svm import KERNEL_BUILDERS, GkmSVM, resolve_kernel_type


def train_gkmsvm(
    pos_seqs,
    neg_seqs,
    *,
    kernel_type: str | int = "estimated",
    l: int = 11,
    k: int = 7,
    d: int = 3,
    C: float = 1.0,
    gamma: float = 1.0,
    M: int | None = None,
    H: float | None = None,
    include_rc: bool = True,
    solver: str = "auto",
    cache_size: int = 256,
    tol: float = 1e-3,
    max_iter: int = 10_000_000,
    sv_chunk_size: int | None = None,
    gram_chunk_size: int = 1000,
    verbose: bool = False,
) -> GkmSVM:
    """Train a gapped k-mer SVM on positive and negative sequences.

    Args:
        pos_seqs: Positive sequences — list of DNA strings or [N+, 4, L] array.
        neg_seqs: Negative sequences — list of DNA strings or [N-, 4, L] array.
        kernel_type: Kernel mode — alias, internal name, or -t integer.
        l: L-mer window length.
        k: Number of informative positions.
        d: Maximum mismatch depth.
        C: SVM regularization parameter.
        gamma: RBF gamma (for -t 3, -t 5).
        M: Center-weight window size (for -t 4, -t 5).
        H: Center-weight decay (for -t 4, -t 5).
        include_rc: Include reverse complement in kernel.
        solver: ``"auto"`` (precomputed Gram if N²×8 < 50% available RAM/VRAM,
            else SMO), ``"smo"`` (column-cached SMO), or ``"libsvm"``
            (precomputed Gram + LIBSVM C solver).
        cache_size: Number of kernel columns to cache (SMO only).
        tol: KKT violation tolerance (SMO only).
        max_iter: Maximum SMO iterations.
        sv_chunk_size: Chunk size for inference on the returned model.
        gram_chunk_size: Tile size for Gram matrix computation (libsvm only).
        verbose: Show progress.

    Returns:
        A trained GkmSVM model.
    """
    X_pos = _to_onehot(pos_seqs)
    X_neg = _to_onehot(neg_seqs)
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.array([1] * X_pos.shape[0] + [-1] * X_neg.shape[0])

    kernel_type = resolve_kernel_type(kernel_type)
    kernel_params = {"L": l, "k": k, "include_rc": include_rc}
    if kernel_type in ("gkm_esttrunc", "gkm_estfull", "gkmrbf"):
        kernel_params["d"] = d
    if kernel_type in ("gkmrbf", "wgkmrbf"):
        kernel_params["gamma"] = gamma
    if kernel_type in ("wgkm", "wgkmrbf"):
        if M is None or H is None:
            raise ValueError("M and H are required for weighted kernels (-t 4/-t 5)")
        kernel_params["M"] = M
        kernel_params["H"] = H

    kernel = KERNEL_BUILDERS[kernel_type](kernel_params)

    if solver == "auto":
        use_smo = not _gram_fits_in_memory(X.shape[0], get_array_module(X))
        if verbose and use_smo:
            gram_gb = X.shape[0] ** 2 * 8 / 1024**3
            print(f"Gram matrix would be {gram_gb:.1f} GB — using SMO solver")
    elif solver == "smo":
        use_smo = True
    elif solver == "libsvm":
        use_smo = False
    else:
        raise ValueError(
            f"Unknown solver {solver!r}. Use 'auto', 'smo', or 'libsvm'."
        )

    if use_smo:
        return _train_smo(
            kernel, X, y, C, kernel_type, kernel_params,
            cache_size=cache_size, tol=tol, max_iter=max_iter,
            sv_chunk_size=sv_chunk_size, verbose=verbose,
        )
    return _train_libsvm(
        kernel, X, y, C, kernel_type, kernel_params,
        gram_chunk_size=gram_chunk_size,
        sv_chunk_size=sv_chunk_size, verbose=verbose,
    )


def _train_libsvm(
    kernel, X, y, C, kernel_type, kernel_params, *,
    gram_chunk_size, sv_chunk_size, verbose,
) -> GkmSVM:
    from libsvm.svmutil import svm_train

    from gkmsvm.gram import compute_gram

    N = X.shape[0]
    if verbose:
        print(f"Computing {N}x{N} Gram matrix ({N * N:,} kernel evaluations)...")
    gram = compute_gram(kernel, X, chunk_size=gram_chunk_size, verbose=verbose)

    ids = np.arange(1, N + 1, dtype=np.float64).reshape(-1, 1)
    x_train = np.hstack([ids, gram])
    del gram

    quiet = "" if verbose else " -q"
    if verbose:
        print("Fitting SVM...")
    model = svm_train(y.tolist(), x_train, f"-s 0 -t 4 -c {C}{quiet}")

    n_sv = model.l
    sv_indices = np.array(
        [model.sv_indices[i] - 1 for i in range(n_sv)]
    )
    coefficients = np.array(
        [model.sv_coef[0][i] for i in range(n_sv)],
        dtype=X.dtype,
    )
    bias = float(-model.rho[0])

    if verbose:
        n_pos = int((coefficients > 0).sum())
        n_neg = int((coefficients <= 0).sum())
        print(
            f"Training complete: {n_sv} support vectors "
            f"({n_pos} pos, {n_neg} neg)"
        )

    return GkmSVM(
        support_sequences=X[sv_indices],
        coefficients=coefficients,
        bias=bias,
        kernel_type=kernel_type,
        kernel_params=kernel_params,
        sv_chunk_size=sv_chunk_size,
    )


def _train_smo(
    kernel, X, y, C, kernel_type, kernel_params, *,
    cache_size, tol, max_iter, sv_chunk_size, verbose,
) -> GkmSVM:
    from gkmsvm.solver import smo_solve

    if verbose:
        print(f"Training with SMO solver (N={X.shape[0]}, cache={cache_size})...")

    coefficients, bias = smo_solve(
        kernel, X, y, C=C, tol=tol, max_iter=max_iter,
        cache_size=cache_size, verbose=verbose,
    )

    sv_mask = np.abs(coefficients) > 1e-10
    support_sequences = X[sv_mask]
    sv_coefficients = coefficients[sv_mask].astype(support_sequences.dtype)

    if verbose:
        n_pos = int((sv_coefficients > 0).sum())
        n_neg = int((sv_coefficients <= 0).sum())
        print(
            f"Training complete: {sv_mask.sum()} support vectors "
            f"({n_pos} pos, {n_neg} neg)"
        )

    return GkmSVM(
        support_sequences=support_sequences,
        coefficients=sv_coefficients,
        bias=bias,
        kernel_type=kernel_type,
        kernel_params=kernel_params,
        sv_chunk_size=sv_chunk_size,
    )


def _available_memory(xp) -> int:
    """Available memory in bytes on the device backing *xp*."""
    if xp is not np:
        try:
            import cupy as cp
            free, _ = cp.cuda.Device().mem_info
            return free
        except Exception:
            return 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, OSError):
        pass
    try:
        import os
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return pages * page_size
    except (AttributeError, ValueError, OSError):
        pass
    return 8 * 1024**3


def _gram_fits_in_memory(N: int, xp) -> bool:
    """Check if an N x N float64 Gram matrix fits in available memory."""
    gram_bytes = N * N * 8
    available = _available_memory(xp)
    return gram_bytes < available * 0.5


def _to_onehot(seqs) -> np.ndarray:
    """Convert sequences to [N, 4, L] one-hot array."""
    if isinstance(seqs, np.ndarray) and seqs.ndim == 3:
        return seqs
    if isinstance(seqs, (list, tuple)):
        if len(seqs) == 0:
            raise ValueError("Sequence list must be non-empty")
        if isinstance(seqs[0], str):
            return np.stack([one_hot_encode(s) for s in seqs])
        if isinstance(seqs[0], np.ndarray):
            return np.stack(seqs)
    raise TypeError(
        f"Expected list of DNA strings, list of arrays, or [N, 4, L] array, "
        f"got {type(seqs)}"
    )
