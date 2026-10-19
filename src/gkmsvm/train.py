"""Training: fit gkm-SVM (classification) and gkm-SVR (regression)."""

from __future__ import annotations

import numpy as np

from gkmsvm.backend import (
    HAS_CUPY,
    HAS_MLX,
    get_array_module,
    is_mlx,
    to_cpu,
    to_gpu,
    to_mlx,
)
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
    n_components: int | None = None,
    cache_size: int | None = None,
    tol: float = 1e-3,
    max_iter: int = 10_000_000,
    max_gram_gb: float | None = None,
    sv_chunk_size: int | None = None,
    gram_chunk_size: int = 1000,
    device: str = "auto",
    allow_other_bases: bool = False,
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
        M: Weight of the center window, 1-255 (-t 4, -t 5; default 50, as LS-GKM).
        H: Half-life of the window weights in bp (-t 4, -t 5; default 50).
        include_rc: Include reverse complement in kernel.
        solver: ``"auto"`` (``"libsvm"`` when the N x N Gram fits in 75%
            of available RAM and under ``max_gram_gb``, else ``"smo"``),
            ``"libsvm"`` (precomputed Gram in host RAM, tiles computed on
            ``device``, solved by sklearn's LIBSVM), ``"smo"`` (column-cached
            SMO, O(cache_size x N) memory), or ``"nystrom"`` (Nyström
            low-rank approximation: fewer kernel evaluations, approximate).
        n_components: Number of landmark points for Nyström approximation.
            Defaults to ``min(N, max(1000, int(sqrt(N) * 10)))``.
        cache_size: Number of kernel columns to cache (SMO only); None sizes
            it from available RAM.
        tol: KKT violation tolerance (SMO only).
        max_iter: Maximum SMO iterations.
        max_gram_gb: Largest Gram matrix (GB) ``solver="auto"`` may allocate.
            On shared compute, set this to your allocation limit. ``None``
            uses automatic memory detection.
        sv_chunk_size: ``sv_chunk_size`` of the returned model (GkmExplain).
        gram_chunk_size: Tile size for Gram matrix computation (libsvm, nystrom).
        device: Where to train: ``"auto"`` (CUDA, else MLX, if available),
            ``"cuda"``, ``"mlx"``, or ``"cpu"``. The returned model is on
            CPU; move it with ``model.cuda()`` or ``model.mlx()``.
        allow_other_bases: Read letters other than ACGTN in DNA strings as N
            (scored as A, as LS-GKM reads them) instead of raising.
        verbose: Show progress.

    Returns:
        A trained GkmSVM model.
    """
    X_pos = _to_onehot(pos_seqs, allow_other_bases)
    X_neg = _to_onehot(neg_seqs, allow_other_bases)
    X = _move_to_device(np.concatenate([X_pos, X_neg], axis=0), device, verbose)
    y = np.array([1] * X_pos.shape[0] + [-1] * X_neg.shape[0])
    if solver not in ("auto", "libsvm", "nystrom", "smo"):
        raise ValueError(
            f"Unknown solver {solver!r}. Use 'auto', 'libsvm', 'nystrom', or 'smo'."
        )
    if solver == "auto":
        N = X.shape[0]
        gram_gb = N * N * 8 / 1024**3
        capped = max_gram_gb is not None and gram_gb > max_gram_gb
        solver = "libsvm" if not capped and _gram_fits_on_cpu(N) else "smo"
        if solver == "smo":
            import warnings

            warnings.warn(
                f"Gram matrix ({gram_gb:.1f} GB) does not fit in memory — falling back to "
                f"column-cached SMO, which is slower than a precomputed Gram. Consider "
                f"downsampling or solver='nystrom' for faster approximate training.",
                stacklevel=2,
            )
    return _fit(
        X,
        y,
        solver,
        C=C,
        kernel_type=kernel_type,
        l=l,
        k=k,
        d=d,
        gamma=gamma,
        M=M,
        H=H,
        include_rc=include_rc,
        n_components=n_components,
        gram_chunk_size=gram_chunk_size,
        sv_chunk_size=sv_chunk_size,
        verbose=verbose,
        smo_args={"cache_size": cache_size, "tol": tol, "max_iter": max_iter},
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
    solver: str = "auto",
    n_components: int | None = None,
    sv_chunk_size: int | None = None,
    gram_chunk_size: int = 1000,
    device: str = "auto",
    allow_other_bases: bool = False,
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
        M: Weight of the center window, 1-255 (-t 4, -t 5; default 50, as LS-GKM).
        H: Half-life of the window weights in bp (-t 4, -t 5; default 50).
        include_rc: Include reverse complement in kernel.
        solver: ``"auto"`` (precomputed Gram if it fits), ``"nystrom"``
            (Nyström low-rank approximation — faster but approximate),
            or ``"libsvm"`` (precomputed Gram + sklearn solver).
        n_components: Number of landmark points for Nyström approximation.
        sv_chunk_size: ``sv_chunk_size`` of the returned model (GkmExplain).
        gram_chunk_size: Tile size for Gram matrix computation.
        device: Where to train: ``"auto"`` (CUDA, else MLX, if available),
            ``"cuda"``, ``"mlx"``, or ``"cpu"``. The returned model is on
            CPU; move it with ``model.cuda()`` or ``model.mlx()``.
        allow_other_bases: Read letters other than ACGTN in DNA strings as N
            (scored as A, as LS-GKM reads them) instead of raising.
        verbose: Show progress.

    Returns:
        A trained GkmSVM model. Scores are predicted continuous values.
    """
    if solver not in ("auto", "libsvm", "nystrom"):
        raise ValueError(
            f"solver must be 'auto', 'libsvm' or 'nystrom' for SVR, got {solver!r}"
        )
    X = _move_to_device(_to_onehot(sequences, allow_other_bases), device, verbose)
    y = np.asarray(labels, dtype=np.float64)
    if y.shape[0] != X.shape[0]:
        raise ValueError(
            f"Number of labels ({y.shape[0]}) must match "
            f"number of sequences ({X.shape[0]})"
        )
    return _fit(
        X,
        y,
        "nystrom" if solver == "nystrom" else "libsvm",
        C=C,
        epsilon=epsilon,
        kernel_type=kernel_type,
        l=l,
        k=k,
        d=d,
        gamma=gamma,
        M=M,
        H=H,
        include_rc=include_rc,
        n_components=n_components,
        gram_chunk_size=gram_chunk_size,
        sv_chunk_size=sv_chunk_size,
        verbose=verbose,
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
    raise ValueError(f"Unknown device {device!r}. Use 'auto', 'cpu', 'cuda', or 'mlx'.")


def _build_kernel(kernel_type, l, k, d, gamma, M, H, include_rc):
    """Resolve kernel type, build params dict, instantiate kernel."""
    kernel_type = resolve_kernel_type(kernel_type)
    kernel_params = {"L": l, "k": k, "d": d, "include_rc": include_rc}
    if kernel_type in ("gkmrbf", "wgkmrbf"):
        kernel_params["gamma"] = gamma
    if kernel_type in ("wgkm", "wgkmrbf"):
        kernel_params["M"] = 50 if M is None else M  # LS-GKM defaults
        kernel_params["H"] = 50.0 if H is None else H
    kernel = KERNEL_BUILDERS[kernel_type](kernel_params)
    return kernel, kernel_type, kernel_params


def _fit(
    X,
    y,
    solver,
    *,
    C,
    kernel_type,
    l,
    k,
    d,
    gamma,
    M,
    H,
    include_rc,
    n_components,
    gram_chunk_size,
    sv_chunk_size,
    verbose,
    epsilon=None,
    smo_args=None,
) -> GkmSVM:
    """Train with `solver` ("libsvm", "nystrom" or "smo") and build the model."""
    kernel, kernel_type, kernel_params = _build_kernel(
        kernel_type,
        l,
        k,
        d,
        gamma,
        M,
        H,
        include_rc,
    )
    if solver == "smo":
        from gkmsvm.solver import smo_solve

        if verbose:
            print(
                f"Training with SMO solver (N={X.shape[0]}, "
                f"cache={smo_args['cache_size']})..."
            )
        coefs, bias = smo_solve(kernel, X, y, C=C, verbose=verbose, **smo_args)
        sv_idx = np.flatnonzero(np.abs(coefs) > 1e-10)
        coefs = coefs[sv_idx]
    else:
        if solver == "nystrom":
            gram = _nystrom_gram(kernel, X, n_components, gram_chunk_size, verbose)
        else:
            from gkmsvm.gram import compute_gram

            N = X.shape[0]
            gram = np.empty((N, N), dtype=np.float64)  # host RAM; tiles on X's device
            if verbose:
                print(
                    f"Computing {N}x{N} Gram matrix ({N * N:,} kernel evaluations)..."
                )
            compute_gram(
                kernel, X, chunk_size=gram_chunk_size, verbose=verbose, out=gram
            )
        clf = _fit_sklearn(gram, y, C, epsilon=epsilon, verbose=verbose)
        del gram
        sv_idx, coefs, bias = clf.support_, clf.dual_coef_[0], float(clf.intercept_[0])

    sv_seqs = to_cpu(X)[sv_idx]
    coefs = np.asarray(coefs, dtype=np.float64)
    if verbose:
        print(
            f"Training complete: {len(sv_idx)} support vectors "
            f"({int((coefs > 0).sum())} positive coefficients)"
        )
    return GkmSVM(
        support_sequences=sv_seqs,
        coefficients=coefs,
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
    else:
        clf = SVC(C=C, kernel="precomputed")
    if verbose:
        print(f"Fitting {type(clf).__name__} with sklearn...")
    clf.fit(gram, y)
    return clf


def _nystrom_gram(kernel, X, n_components, chunk_size, verbose) -> np.ndarray:
    """N x N Nystrom approximation of the Gram matrix from m landmarks.

    Computes K_mm (m x m) and K_nm (N x m) exactly, then K ~ K_nm K_mm^+ K_mn,
    so kernel evaluations cost O(N m) instead of O(N^2). The solver still
    sees a dense N x N matrix. The trained model scores with the exact
    kernel; the approximation only affects training.
    """
    from gkmsvm.gram import compute_gram

    N = X.shape[0]
    if n_components is None:
        n_components = min(N, max(1000, int(np.sqrt(N) * 10)))
    m = min(n_components, N)
    if verbose:
        print(f"Nyström approximation: {m} landmarks from {N} samples")

    landmark_idx = np.sort(np.random.RandomState(42).choice(N, size=m, replace=False))
    X_cpu = to_cpu(X)
    if is_mlx(X):
        X_landmarks = to_mlx(X_cpu[landmark_idx])
    elif get_array_module(X) is not np:
        X_landmarks = to_gpu(X_cpu[landmark_idx])
    else:
        X_landmarks = X_cpu[landmark_idx]

    K_mm = np.empty((m, m), dtype=np.float64)
    compute_gram(kernel, X_landmarks, chunk_size=chunk_size, verbose=verbose, out=K_mm)
    K_nm = np.empty((N, m), dtype=np.float64)
    compute_gram(
        kernel, X, X_landmarks, chunk_size=chunk_size, verbose=verbose, out=K_nm
    )

    # Features phi = K_nm U L^-1/2 over the well-conditioned eigenpairs of K_mm
    eigenvalues, U = np.linalg.eigh(K_mm)
    valid = eigenvalues > max(1e-10, eigenvalues.max() * 1e-8)
    if verbose:
        print(f"Effective rank: {int(valid.sum())}/{m}")
    features = (K_nm @ U[:, valid]) / np.sqrt(eigenvalues[valid])[None, :]
    del K_nm, K_mm, U

    gram = np.empty((N, N), dtype=np.float64)
    for i in range(0, N, chunk_size):
        gram[i : i + chunk_size] = features[i : i + chunk_size] @ features.T
    return gram


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


def _gram_fits_on_cpu(N: int) -> bool:
    """Whether an N x N float64 Gram fits in 75% of available RAM (it is
    always held in host memory, whatever the training device)."""
    gram_bytes = N * N * 8
    available = _available_memory(np)
    return gram_bytes < available * 0.75


def _to_onehot(seqs, allow_other: bool = False) -> np.ndarray:
    """Convert sequences to [N, 4, L] one-hot array (DNA strings may hold N,
    and with allow_other any letter, read as N)."""
    if isinstance(seqs, np.ndarray) and seqs.ndim == 3:
        return seqs
    if isinstance(seqs, (list, tuple)):
        if len(seqs) == 0:
            raise ValueError("Sequence list must be non-empty")
        if isinstance(seqs[0], str):
            # N encodes as all-zero (scored as A, as in LS-GKM)
            return np.stack(
                [one_hot_encode(s, allow_n=True, allow_other=allow_other) for s in seqs]
            )
        if isinstance(seqs[0], np.ndarray):
            return np.stack(seqs)
    raise TypeError(
        f"Expected list of DNA strings, list of arrays, or [N, 4, L] array, "
        f"got {type(seqs)}"
    )
