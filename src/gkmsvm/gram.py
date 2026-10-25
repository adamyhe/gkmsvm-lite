"""Gram matrix computation with tiled chunking and symmetry exploitation."""

from __future__ import annotations

import numpy as np

from gkmsvm.backend import float_dtype, get_array_module, is_mlx, to_cpu
from gkmsvm.kernels.base import GkmKernel
from gkmsvm.svm import GkmSVM, _progress


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

    Uses the base kernel's packed path (RBF kernels via their base kernel).
    When the l-mer index applies (kernels/lmer_index.py, the same rule as
    SMO columns) and X isn't on MLX, Y is indexed once and X is queried in
    row blocks, each self-Gram row only from the diagonal on. Otherwise, when
    the kernel uses the masked join, whole column panels are computed per
    call so each column tile's index is built once; else square tiles are
    used.

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

    bx, bx_rc = base.pack_both(X, lengths)
    by, by_rc = (bx, bx_rc) if symmetric else base.pack_both(Y, y_lengths)
    # MLX tiles are normalized on the host in float64 (Metal has no float64)
    txp = np if xp is not np and is_mlx(X) else xp

    def tile_float(a):
        return to_cpu(a).astype(np.float64) if txp is np else a.astype(float_dtype(a))

    if base.normalize:
        diag_x = tile_float(base.raw_diag_packed(bx, bx_rc))
        diag_y = diag_x if symmetric else tile_float(base.raw_diag_packed(by, by_rc))

    from gkmsvm.kernels import lmer_index

    # MLX keeps the masked join: the Metal index query loses to it at every N
    # (0.2-0.37x at 1k-20k on an M4; its neighbor lookups alone take longer)
    if lmer_index.applies(base) and not is_mlx(X):
        index = lmer_index.LmerIndex(base, by, by_rc)
        rows = max(1, GkmSVM._MAX_SCORE_BYTES // (8 * M))
        for r in _progress(range(0, N, rows), verbose, "Gram matrix"):
            e = min(r + rows, N)
            c = r if symmetric else 0
            tile = index.pairwise(bx[r:e], r if symmetric else None)[:, c:]
            tile = tile_float(tile)
            if base.normalize:
                norm = txp.sqrt(diag_x[r:e, None] * diag_y[c:][None, :])
                tile = tile / txp.clip(norm, 1e-10, None)
            tile = to_cpu(kernel.finish(tile))
            if not symmetric:
                gram[r:e] = tile
                continue
            # The diagonal block holds its upper triangle (the rest was skipped)
            blk = tile[:, : e - r]
            gram[r:e, r:e] = np.triu(blk) + np.triu(blk, 1).T
            gram[r:e, e:] = tile[:, e - r :]
            gram[e:, r:e] = tile[:, e - r :].T
        return gram

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
