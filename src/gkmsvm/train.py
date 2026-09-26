"""Training: fit a C-SVM with gapped k-mer kernel via sklearn."""

from __future__ import annotations

import numpy as np

from gkmsvm.codec import one_hot_encode
from gkmsvm.gram import compute_gram
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
    sv_chunk_size: int | None = None,
    gram_chunk_size: int = 1000,
    verbose: bool = False,
) -> GkmSVM:
    """Train a gapped k-mer SVM on positive and negative sequences.

    Uses sklearn's SVC with a precomputed gkm kernel matrix.

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
        sv_chunk_size: Chunk size for inference on the returned model.
        gram_chunk_size: Tile size for Gram matrix computation.
        verbose: Show progress bars.

    Returns:
        A trained GkmSVM model.

    Raises:
        ImportError: If scikit-learn is not installed.
    """
    try:
        from sklearn.svm import SVC
    except ImportError:
        raise ImportError(
            "Training requires scikit-learn. "
            "Install with: pip install gkmsvm-lite[train]"
        )

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

    if verbose:
        n = X.shape[0]
        print(f"Computing {n}x{n} Gram matrix ({n * n:,} kernel evaluations)...")
    gram = compute_gram(kernel, X, chunk_size=gram_chunk_size, verbose=verbose)

    if verbose:
        print("Fitting SVM...")
    clf = SVC(kernel="precomputed", C=C)
    clf.fit(gram, y)

    sv_indices = clf.support_
    support_sequences = X[sv_indices]
    coefficients = clf.dual_coef_[0].astype(support_sequences.dtype)
    bias = float(clf.intercept_[0])

    if verbose:
        print(
            f"Training complete: {len(sv_indices)} support vectors "
            f"({(clf.dual_coef_[0] > 0).sum()} pos, "
            f"{(clf.dual_coef_[0] <= 0).sum()} neg)"
        )

    return GkmSVM(
        support_sequences=support_sequences,
        coefficients=coefficients,
        bias=bias,
        kernel_type=kernel_type,
        kernel_params=kernel_params,
        sv_chunk_size=sv_chunk_size,
    )


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
