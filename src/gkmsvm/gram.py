"""Gram matrix computation with tiled chunking and symmetry exploitation."""

from __future__ import annotations

import numpy as np

from gkmsvm.backend import float_dtype, get_array_module, is_mlx, to_cpu
from gkmsvm.kernels.base import GkmKernel
from gkmsvm.svm import _progress


def compute_gram(
    kernel: GkmKernel,
    X: np.ndarray,
    Y: np.ndarray | None = None,
    *,
    chunk_size: int = 1000,
    verbose: bool = False,
    out: np.ndarray | None = None,
    lengths=None,
    y_lengths=None,
) -> np.ndarray:
    """Compute kernel matrix K(X, Y) with tiled chunking.

    When Y is None, computes K(X, X) with symmetry exploitation —
    only upper-triangle tiles are evaluated, then mirrored.

    Uses the base kernel's packed path when it applies (RBF kernels via
    their base kernel). When the kernel uses the masked join, whole column
    panels are computed per call so each column tile's index is built once;
    otherwise square tiles are used.

    Args:
        kernel: A GkmKernel instance.
        X: [N, 4, L] one-hot encoded sequences.
        Y: [M, 4, L] one-hot encoded sequences, or None for self-Gram.
        chunk_size: Tile size for row and column chunking.
        verbose: Show tqdm progress bar over tiles.
        out: Optional [N, M] output array to write into (avoids allocation).
        lengths, y_lengths: [N], [M] sequence lengths of X and Y when they
            differ (the arrays padded to their longest).

    Returns:
        [N, M] or [N, N] kernel matrix.
    """
    xp = get_array_module(X)
    symmetric = Y is None
    if symmetric:
        Y = X
    N = X.shape[0]
    M = Y.shape[0]
    gram = np.zeros((N, M), dtype=np.float64) if out is None else out

    # Tiles come from the base kernel's packed path (the base of -t 3/-t 5,
    # then finish() applies the RBF)
    base = kernel.base

    # The masked join builds an index of the column sequences on every call,
    # so compute whole column panels (all rows, or all rows up to the
    # diagonal) to build each column tile's index once
    panels = base._masked_plan() is not None
    row_ranges = range(0, N, chunk_size)
    col_ranges = range(0, M, chunk_size)
    if panels:
        tiles = [(0, c) for c in col_ranges]
    elif symmetric:
        tiles = [(r, c) for r in row_ranges for c in col_ranges if r <= c]
    else:
        tiles = [(r, c) for r in row_ranges for c in col_ranges]
    tiles = _progress(tiles, verbose, "Gram matrix")

    bx, bx_rc = base.pack_both(X, lengths)
    by, by_rc = (bx, bx_rc) if symmetric else base.pack_both(Y, y_lengths)
    # MLX tiles are normalized on the host in float64 (Metal has no float64)
    txp = np if xp is not np and is_mlx(X) else xp

    def tile_float(a):
        return to_cpu(a).astype(np.float64) if txp is np else a.astype(float_dtype(a))

    if base.normalize:
        diag_x = tile_float(base.raw_diag_packed(bx, bx_rc))
        diag_y = diag_x if symmetric else tile_float(base.raw_diag_packed(by, by_rc))

    for r_start, c_start in tiles:
        c_end = min(c_start + chunk_size, M)
        r_end = (c_end if symmetric else N) if panels else min(r_start + chunk_size, N)

        diag_offset = (
            c_start if symmetric and panels and get_array_module(bx) is np else None
        )
        tile = tile_float(
            base.raw_packed(
                bx[r_start:r_end],
                by[c_start:c_end],
                None if by_rc is None else by_rc[c_start:c_end],
                diag_offset=diag_offset,
            )
        )
        if base.normalize:
            norm = txp.sqrt(
                diag_x[r_start:r_end, None] * diag_y[c_start:c_end][None, :]
            )
            tile = tile / txp.clip(norm, 1e-10, None)
        tile = kernel.finish(tile)

        tile_np = to_cpu(tile)
        gram[r_start:r_end, c_start:c_end] = tile_np

        if symmetric and panels:
            # Rows above the diagonal block mirror into the lower triangle;
            # the diagonal block takes its upper triangle so K stays exactly
            # symmetric (the two triangles differ in summation order)
            gram[c_start:c_end, :c_start] = tile_np[:c_start].T
            blk = tile_np[c_start:c_end]
            gram[c_start:c_end, c_start:c_end] = np.triu(blk) + np.triu(blk, 1).T
        elif symmetric and r_start != c_start:
            gram[c_start:c_end, r_start:r_end] = tile_np.T

    return gram
