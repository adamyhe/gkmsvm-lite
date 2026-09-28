"""Gram matrix computation with tiled chunking and symmetry exploitation."""

from __future__ import annotations

import numpy as np

from gkmsvm.backend import get_array_module, to_cpu
from gkmsvm.codec import reverse_complement
from gkmsvm.kernels.base import GkmKernel


def compute_gram(
    kernel: GkmKernel,
    X: np.ndarray,
    Y: np.ndarray | None = None,
    *,
    chunk_size: int = 1000,
    verbose: bool = False,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Compute kernel matrix K(X, Y) with tiled chunking.

    When Y is None, computes K(X, X) with symmetry exploitation —
    only upper-triangle tiles are evaluated, then mirrored.

    Uses the packed uint32 path when available (DirectGkmKernel,
    EstTruncGkmKernel) for ~3x faster computation on CPU.

    Args:
        kernel: A GkmKernel instance.
        X: [N, 4, L] one-hot encoded sequences.
        Y: [M, 4, L] one-hot encoded sequences, or None for self-Gram.
        chunk_size: Tile size for row and column chunking.
        verbose: Show tqdm progress bar over tiles.
        out: Optional [N, M] output array to write into (avoids allocation).

    Returns:
        [N, M] or [N, N] kernel matrix.
    """
    xp = get_array_module(X)
    symmetric = Y is None
    if symmetric:
        Y = X
    N = X.shape[0]
    M = Y.shape[0]

    use_fast = hasattr(kernel, "pairwise_from_indices")

    gram = np.zeros((N, M), dtype=np.float64) if out is None else out

    row_ranges = list(range(0, N, chunk_size))
    col_ranges = list(range(0, M, chunk_size))

    if symmetric:
        tiles = [
            (r, c)
            for r in row_ranges
            for c in col_ranges
            if r <= c
        ]
    else:
        tiles = [(r, c) for r in row_ranges for c in col_ranges]

    if verbose:
        from tqdm import tqdm
        tiles = tqdm(tiles, desc="Gram matrix")

    if use_fast:
        diag_x = kernel._raw_diagonal(X)
        if symmetric:
            diag_y = diag_x
        else:
            diag_y = kernel._raw_diagonal(Y)

    for r_start, c_start in tiles:
        r_end = min(r_start + chunk_size, N)
        c_end = min(c_start + chunk_size, M)

        if use_fast:
            tile = _fast_pairwise_tile(
                kernel, X[r_start:r_end], Y[c_start:c_end]
            )
            if kernel.normalize:
                norm = xp.sqrt(
                    diag_x[r_start:r_end, None] * diag_y[c_start:c_end][None, :]
                )
                tile = tile / xp.clip(norm, 1e-10, None)
        else:
            tile = kernel.pairwise(X[r_start:r_end], Y[c_start:c_end])

        tile_np = to_cpu(tile)
        gram[r_start:r_end, c_start:c_end] = tile_np

        if symmetric and r_start != c_start:
            gram[c_start:c_end, r_start:r_end] = tile_np.T

    return gram


def _fast_pairwise_tile(kernel, x, y):
    """Compute raw pairwise kernel using the packed uint32 path."""
    bx = kernel.base_index_windows(x)
    by = kernel.base_index_windows(y)
    result = kernel.pairwise_from_indices(bx, by)

    if kernel.include_rc:
        by_rc = kernel.base_index_windows(reverse_complement(y))
        result = result + kernel.pairwise_from_indices(bx, by_rc)

    return result.astype(x.dtype)
