"""Gram matrix computation with tiled chunking and symmetry exploitation."""

from __future__ import annotations

import numpy as np

from gkmsvm.backend import get_array_module
from gkmsvm.kernels.base import GkmKernel


def compute_gram(
    kernel: GkmKernel,
    X: np.ndarray,
    Y: np.ndarray | None = None,
    *,
    chunk_size: int = 1000,
    verbose: bool = False,
) -> np.ndarray:
    """Compute kernel matrix K(X, Y) with tiled chunking.

    When Y is None, computes K(X, X) with symmetry exploitation —
    only upper-triangle tiles are evaluated, then mirrored.

    Args:
        kernel: A GkmKernel instance.
        X: [N, 4, L] one-hot encoded sequences.
        Y: [M, 4, L] one-hot encoded sequences, or None for self-Gram.
        chunk_size: Tile size for row and column chunking.
        verbose: Show tqdm progress bar over tiles.

    Returns:
        [N, M] or [N, N] kernel matrix.
    """
    xp = get_array_module(X)
    symmetric = Y is None
    if symmetric:
        Y = X
    N = X.shape[0]
    M = Y.shape[0]

    gram = xp.zeros((N, M), dtype=np.float64)

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

    for r_start, c_start in tiles:
        r_end = min(r_start + chunk_size, N)
        c_end = min(c_start + chunk_size, M)

        tile = kernel.pairwise(X[r_start:r_end], Y[c_start:c_end])
        gram[r_start:r_end, c_start:c_end] = tile

        if symmetric and r_start != c_start:
            gram[c_start:c_end, r_start:r_end] = tile.T

    return gram
