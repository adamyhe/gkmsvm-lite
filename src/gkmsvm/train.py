"""Training: fit gkm-SVM (classification) and gkm-SVR (regression)."""

from __future__ import annotations

import numpy as np

from gkmsvm.backend import HAS_CUPY, HAS_MLX, get_array_module, is_mlx, to_cpu, to_gpu, to_mlx
from gkmsvm.codec import one_hot_encode
from gkmsvm.svm import KERNEL_BUILDERS, GkmSVM, resolve_kernel_type


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
    cache_size: int = 2048,
    tol: float = 1e-3,
    max_iter: int = 10_000_000,
    max_gram_gb: float | None = None,
    sv_chunk_size: int | None = None,
    gram_chunk_size: int = 1000,
    device: str = "auto",
    verbose: bool = False,
) -> GkmSVM:
    """Train a gapped k-mer SVM classifier on positive and negative sequences.

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
        solver: ``"auto"`` (precomputed Gram if it fits in device memory;
            on GPU, falls back to GPU-compute + CPU-solve if it fits in
            system RAM; else column-cached SMO), ``"smo"`` (column-cached
            SMO), or ``"libsvm"`` (precomputed Gram + sklearn solver).
        cache_size: Number of kernel columns to cache (SMO only).
        tol: KKT violation tolerance (SMO only).
        max_iter: Maximum SMO iterations.
        max_gram_gb: Hard cap on Gram matrix size in GB. On shared compute,
            set this to your allocation limit to avoid OOM. ``None`` uses
            automatic memory detection.
        sv_chunk_size: Chunk size for inference on the returned model.
        gram_chunk_size: Tile size for Gram matrix computation (libsvm only).
        device: ``"auto"`` (MLX if available), ``"mlx"``, or ``"cpu"``.
        verbose: Show progress.

    Returns:
        A trained GkmSVM model.
    """
    X_pos = _to_onehot(pos_seqs)
    X_neg = _to_onehot(neg_seqs)
    X = np.concatenate([X_pos, X_neg], axis=0)
    X = _move_to_device(X, device, verbose)
    y = np.array([1] * X_pos.shape[0] + [-1] * X_neg.shape[0])

    kernel, kernel_type, kernel_params = _build_kernel(
        kernel_type, l, k, d, gamma, M, H, include_rc,
    )

    xp = get_array_module(X)
    N = X.shape[0]
    gram_gb = N * N * 8 / 1024**3

    if max_gram_gb is not None and gram_gb > max_gram_gb:
        gram_capped = True
    else:
        gram_capped = False

    if solver == "auto":
        if not gram_capped and _gram_fits_in_memory(N, xp):
            use_smo = False
            gpu_gram_cpu_solve = False
        elif not gram_capped and xp is not np and _gram_fits_on_cpu(N):
            use_smo = False
            gpu_gram_cpu_solve = True
            if verbose:
                print(
                    f"Gram matrix ({gram_gb:.1f} GB) exceeds GPU memory "
                    f"— computing on GPU, solving on CPU"
                )
        else:
            use_smo = True
            gpu_gram_cpu_solve = False
            if verbose:
                print(f"Gram matrix would be {gram_gb:.1f} GB — using SMO solver")
    elif solver == "smo":
        use_smo = True
        gpu_gram_cpu_solve = False
    elif solver == "libsvm":
        use_smo = False
        gpu_gram_cpu_solve = False
    else:
        raise ValueError(
            f"Unknown solver {solver!r}. Use 'auto', 'smo', or 'libsvm'."
        )

    if use_smo:
        return _train_smo(
            kernel, X, y, C, kernel_type, kernel_params,
            cache_size=cache_size, tol=tol, max_iter=max_iter,
            sv_chunk_size=sv_chunk_size, device=device, verbose=verbose,
        )
    if gpu_gram_cpu_solve:
        return _fit_libsvm_gpu_gram(
            kernel, X, y, C,
            kernel_type, kernel_params,
            gram_chunk_size=gram_chunk_size,
            sv_chunk_size=sv_chunk_size, verbose=verbose,
        )
    return _fit_libsvm(
        kernel, X, y, C,
        kernel_type, kernel_params,
        gram_chunk_size=gram_chunk_size,
        sv_chunk_size=sv_chunk_size, device=device, verbose=verbose,
    )


def train_gkmsvr(
    sequences,
    labels,
    *,
    kernel_type: str | int = "estimated",
    l: int = 11,
    k: int = 7,
    d: int = 3,
    C: float = 1.0,
    epsilon: float = 0.1,
    gamma: float = 1.0,
    M: int | None = None,
    H: float | None = None,
    include_rc: bool = True,
    sv_chunk_size: int | None = None,
    gram_chunk_size: int = 1000,
    device: str = "auto",
    verbose: bool = False,
) -> GkmSVM:
    """Train a gapped k-mer SVR (epsilon-SVR) for regression.

    Predicts continuous values from DNA sequences using the gapped
    k-mer kernel with sklearn's epsilon-SVR solver.

    Args:
        sequences: DNA sequences — list of strings or [N, 4, L] array.
        labels: [N] continuous target values (list or array).
        kernel_type: Kernel mode — alias, internal name, or -t integer.
        l: L-mer window length.
        k: Number of informative positions.
        d: Maximum mismatch depth.
        C: Regularization parameter.
        epsilon: Epsilon-tube width — errors within ±epsilon are ignored.
        gamma: RBF gamma (for -t 3, -t 5).
        M: Center-weight window size (for -t 4, -t 5).
        H: Center-weight decay (for -t 4, -t 5).
        include_rc: Include reverse complement in kernel.
        sv_chunk_size: Chunk size for inference on the returned model.
        gram_chunk_size: Tile size for Gram matrix computation.
        device: ``"auto"`` (MLX if available), ``"mlx"``, or ``"cpu"``.
        verbose: Show progress.

    Returns:
        A trained GkmSVM model. Scores are predicted continuous values.
    """
    X = _to_onehot(sequences)
    X = _move_to_device(X, device, verbose)
    y = np.asarray(labels, dtype=np.float64)
    if y.shape[0] != X.shape[0]:
        raise ValueError(
            f"Number of labels ({y.shape[0]}) must match "
            f"number of sequences ({X.shape[0]})"
        )

    kernel, kernel_type, kernel_params = _build_kernel(
        kernel_type, l, k, d, gamma, M, H, include_rc,
    )

    xp = get_array_module(X)
    N = X.shape[0]
    if xp is not np and not _gram_fits_in_memory(N, xp) and _gram_fits_on_cpu(N):
        if verbose:
            gram_gb = N * N * 8 / 1024**3
            print(
                f"Gram matrix ({gram_gb:.1f} GB) exceeds GPU memory "
                f"— computing on GPU, solving on CPU"
            )
        return _fit_libsvm_gpu_gram(
            kernel, X, y, C,
            kernel_type, kernel_params,
            epsilon=epsilon, gram_chunk_size=gram_chunk_size,
            sv_chunk_size=sv_chunk_size, verbose=verbose,
        )
    return _fit_libsvm(
        kernel, X, y, C,
        kernel_type, kernel_params,
        epsilon=epsilon, gram_chunk_size=gram_chunk_size,
        sv_chunk_size=sv_chunk_size, device=device, verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _move_to_device(X: np.ndarray, device: str, verbose: bool) -> np.ndarray:
    """Move training data to the requested accelerator.

    For MLX, conservative memory policy: auto-detection only engages when
    training data fits within 25% of available system RAM, to avoid pressure
    on unified memory (macOS users are typically running other applications).
    """
    if device == "cpu":
        return X
    if device in ("cuda", "gpu"):
        if not HAS_CUPY:
            raise RuntimeError(
                "CuPy is not installed. Install with: pip install cupy-cuda12x[ctk]"
            )
        if verbose:
            print("Moving training data to CUDA GPU")
        return to_gpu(X)
    if device == "auto":
        if HAS_CUPY:
            if verbose:
                print("Moving training data to CUDA GPU")
            return to_gpu(X)
        if HAS_MLX:
            data_bytes = X.nbytes
            available = _available_memory(np)
            if data_bytes > available * 0.25:
                if verbose:
                    data_mb = data_bytes / 1024**2
                    avail_mb = available / 1024**2
                    print(
                        f"Training data ({data_mb:.0f} MB) exceeds 25% of "
                        f"available memory ({avail_mb:.0f} MB) — staying on CPU"
                    )
                return X
            if verbose:
                print("Moving training data to MLX")
            return to_mlx(X)
        return X
    if device == "mlx":
        if not HAS_MLX:
            raise RuntimeError("MLX is not installed. Install with: pip install mlx")
        if verbose:
            print("Moving training data to MLX")
        return to_mlx(X)
    raise ValueError(
        f"Unknown device {device!r}. Use 'auto', 'cpu', 'cuda', or 'mlx'."
    )


def _build_kernel(kernel_type, l, k, d, gamma, M, H, include_rc):
    """Resolve kernel type, build params dict, instantiate kernel."""
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
    return kernel, kernel_type, kernel_params


def _fit_libsvm_gpu_gram(
    kernel, X, y, C, kernel_type, kernel_params, *,
    epsilon=None, gram_chunk_size, sv_chunk_size, verbose=False,
) -> GkmSVM:
    """Compute Gram on GPU in tiles, accumulate on CPU, solve with sklearn."""
    from gkmsvm.gram import compute_gram

    N = X.shape[0]
    gram = np.empty((N, N), dtype=np.float64)
    if verbose:
        print(f"Computing {N}x{N} Gram matrix on GPU, storing on CPU...")
    compute_gram(kernel, X, chunk_size=gram_chunk_size, verbose=verbose,
                 out=gram)

    y_cpu = to_cpu(y) if not isinstance(y, np.ndarray) else y
    clf = _fit_sklearn(gram, y_cpu, C, epsilon=epsilon, verbose=verbose)

    sv_indices = clf.support_
    sv_seqs = to_cpu(X)[sv_indices]
    coefficients = clf.dual_coef_[0].astype(sv_seqs.dtype)
    bias = float(clf.intercept_[0])

    if verbose:
        print(f"Training complete: {len(sv_indices)} support vectors")

    return GkmSVM(
        support_sequences=sv_seqs,
        coefficients=coefficients,
        bias=bias,
        kernel_type=kernel_type,
        kernel_params=kernel_params,
        sv_chunk_size=sv_chunk_size,
    )


def _fit_libsvm(
    kernel, X, y, C, kernel_type, kernel_params, *,
    epsilon=None, gram_chunk_size, sv_chunk_size, device="cpu", verbose=False,
) -> GkmSVM:
    """Compute Gram matrix and fit with sklearn's LIBSVM-backed solver."""
    from gkmsvm.gram import compute_gram

    N = X.shape[0]
    gram = np.empty((N, N), dtype=np.float64)
    if verbose:
        print(f"Computing {N}x{N} Gram matrix ({N * N:,} kernel evaluations)...")
    compute_gram(kernel, X, chunk_size=gram_chunk_size, verbose=verbose,
                 out=gram)

    y_fit = to_cpu(y) if not isinstance(y, np.ndarray) else y
    clf = _fit_sklearn(gram, y_fit, C, epsilon=epsilon, verbose=verbose)

    sv_indices = clf.support_
    if is_mlx(X):
        sv_seqs = to_cpu(X)[sv_indices]
    else:
        sv_seqs = to_cpu(X[sv_indices])
    coefficients = clf.dual_coef_[0].astype(sv_seqs.dtype)
    bias = float(clf.intercept_[0])

    if verbose:
        print(f"Training complete: {len(sv_indices)} support vectors")

    return GkmSVM(
        support_sequences=sv_seqs,
        coefficients=coefficients,
        bias=bias,
        kernel_type=kernel_type,
        kernel_params=kernel_params,
        sv_chunk_size=sv_chunk_size,
    )


def _fit_sklearn(gram, y, C, *, epsilon=None, verbose=False):
    """Fit SVC or SVR with a precomputed kernel matrix."""
    from sklearn.svm import SVC, SVR

    if epsilon is not None:
        clf = SVR(C=C, epsilon=epsilon, kernel="precomputed")
        if verbose:
            print("Fitting SVR with sklearn...")
    else:
        clf = SVC(C=C, kernel="precomputed")
        if verbose:
            print("Fitting SVC with sklearn...")
    clf.fit(gram, y)
    return clf


def _train_smo(
    kernel, X, y, C, kernel_type, kernel_params, *,
    cache_size, tol, max_iter, sv_chunk_size, device="cpu", verbose=False,
) -> GkmSVM:
    from gkmsvm.solver import smo_solve

    if verbose:
        print(f"Training with SMO solver (N={X.shape[0]}, cache={cache_size})...")

    coefficients, bias = smo_solve(
        kernel, X, y, C=C, tol=tol, max_iter=max_iter,
        cache_size=cache_size, verbose=verbose,
    )

    sv_mask = np.abs(to_cpu(coefficients)) > 1e-10
    if is_mlx(X):
        X_cpu = to_cpu(X)
        support_sequences = X_cpu[sv_mask]
    else:
        support_sequences = to_cpu(X[sv_mask])
    sv_coefficients = to_cpu(coefficients)[sv_mask].astype(support_sequences.dtype)

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
            pass
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
    """Check if Gram matrix fits in available device memory."""
    gram_bytes = N * N * 8
    available = _available_memory(xp)
    return gram_bytes < available * 0.75


def _gram_fits_on_cpu(N: int) -> bool:
    """Check if Gram matrix fits in CPU RAM."""
    gram_bytes = N * N * 8
    available = _available_memory(np)
    return gram_bytes < available * 0.75


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
