"""Gram matrix computation with tiled chunking and symmetry exploitation."""

from __future__ import annotations

import numpy as np

from gkmsvm.backend import get_array_module, to_cpu
from gkmsvm.codec import reverse_complement
from gkmsvm.kernels.base import GkmKernel
from gkmsvm.kernels.rbf import RbfGkmKernel


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
    EstTruncGkmKernel, and RbfGkmKernel via its base kernel). When the
    kernel uses the masked join, whole column panels are computed per call so
    each column tile's index is built once; otherwise square tiles are used,
    and on CPU diagonal tiles compute only their upper triangle.

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

    # Kernel whose packed path computes the tiles: the kernel itself, or the
    # base kernel of -t 3 gkmrbf (exp is applied after normalizing)
    if hasattr(kernel, "pairwise_from_indices"):
        fast = kernel
    elif isinstance(kernel, RbfGkmKernel):
        fast = kernel._base
    else:
        fast = None
    use_fast = fast is not None

    gram = np.zeros((N, M), dtype=np.float64) if out is None else out

    row_ranges = list(range(0, N, chunk_size))
    col_ranges = list(range(0, M, chunk_size))

    # The masked-join kernel builds an index of the column sequences on every
    # call, so compute whole column panels (all rows, or all rows up to the
    # diagonal) to build each column tile's index once
    plan_fn = getattr(fast, "_masked_plan", None) if use_fast else None
    panels = plan_fn is not None and plan_fn() is not None

    if panels:
        tiles = [(0, c) for c in col_ranges]
    elif symmetric:
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
        diag_x = fast._raw_diagonal(X)
        if symmetric:
            diag_y = diag_x
        else:
            diag_y = fast._raw_diagonal(Y)

    for r_start, c_start in tiles:
        c_end = min(c_start + chunk_size, M)
        if panels:
            r_end = c_end if symmetric else N
        else:
            r_end = min(r_start + chunk_size, N)

        if use_fast:
            tile = _fast_pairwise_tile(
                fast, X[r_start:r_end], Y[c_start:c_end],
                symmetric=symmetric and not panels and r_start == c_start,
                diag_offset=c_start if symmetric and panels else None,
            )
            if fast.normalize:
                norm = xp.sqrt(
                    diag_x[r_start:r_end, None] * diag_y[c_start:c_end][None, :]
                )
                tile = tile / xp.clip(norm, 1e-10, None)
            if fast is not kernel:
                tile = xp.exp(kernel.gamma * (tile - 1))
        else:
            tile = kernel.pairwise(X[r_start:r_end], Y[c_start:c_end])

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


def _fast_pairwise_tile(kernel, x, y, *, symmetric=False, diag_offset=None):
    """Compute raw pairwise kernel using the packed uint32 path.

    symmetric: x and y are the same sequences (a diagonal Gram tile).
    diag_offset: x is sequences 0..B-1 and y sequences diag_offset.. of one
        set; entries below the diagonal may be left 0 (a column panel).
    """
    bx = kernel.base_index_windows(x)
    by = bx if symmetric else kernel.base_index_windows(y)
    if diag_offset is not None and get_array_module(bx) is np:
        plan = kernel._masked_plan() if hasattr(kernel, "_masked_plan") else None
        if plan is not None:
            # One masked call for the forward and RC blocks (one shared index)
            from gkmsvm.kernels.direct import _pack_windows_cpu
            from gkmsvm.kernels.masked import masked_pairwise

            def pack(a):
                return _pack_windows_cpu(np.ascontiguousarray(a))

            by_rc = (pack(kernel.base_index_windows(reverse_complement(y)))
                     if kernel.include_rc else None)
            return masked_pairwise(pack(bx), pack(by), plan, by_rc,
                                   diag_offset=diag_offset).astype(x.dtype)
    result = kernel.pairwise_from_indices(bx, by, symmetric=symmetric)

    if kernel.include_rc:
        by_rc = kernel.base_index_windows(reverse_complement(y))
        result = result + kernel.pairwise_from_indices(bx, by_rc, symmetric=symmetric)

    return result.astype(x.dtype)
