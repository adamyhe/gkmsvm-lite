"""Kernels whose raw value sums a pair weight over packed l-mer windows.

K_raw(x, y) = sum_{i,j} g(x_i, y_j), with x_i, y_j the length-l windows of x
and y packed 2 bits per base into uint32 (position k in bits 2k, 2k+1). Every
device holds packed windows as [B, W] arrays on that device (MLX arrays are
packed on the CPU). Reverse complements enter as a second column block.

Subclasses provide the brute-force pair loop (`_brute`, `_brute_cross_diag`
or their own `raw_diag_packed`) and the mask plan for the masked join (`_join_plan`, kernels/masked.py).

Sequences of different lengths share one array: pack(x, lengths) writes
PAD_WINDOW past each row's last window (RC rows, being reversed, hold their
pads first). Every kernel skips PAD explicitly; a row's windows and, for
-t 4/5, their weights follow from its count of non-PAD windows
(docs/design/variable-length.md).
"""

from __future__ import annotations

import numpy as np

from gkmsvm.backend import float_dtype, get_array_module, is_mlx, to_cpu
from gkmsvm.kernels.base import GkmKernel

# Below this many query or column sequences, the masked join's per-call index
# build costs more than it saves (e.g. single SMO columns)
_MASKED_MIN_ROWS = 16

# Window value standing for "no window": past a shorter sequence's last
# window, and in ISM's window sets. It has bits set above every l-mer (l <=
# 15), so the l-mer index never finds it; every kernel skips it explicitly
# (against itself it would score as a full match, and masked to an l-mer's
# bits it equals the all-T l-mer).
PAD_WINDOW = np.uint32(0xFFFFFFFF)
# Longest l: a uint32 packs 16 bases, and at 16 the all-T l-mer is PAD_WINDOW
# (LS-GKM allows l <= 12)
MAX_L = 15


def _rc_windows(packed, l):
    """Packed windows of the reverse complement: window j is window W-1-j with
    its bases complemented (b ^ 3: A<->T, C<->G) and in reverse order. PAD
    stays PAD, so a padded row's RC holds its pads first."""
    xp = get_array_module(packed)
    rev = packed[:, ::-1]
    u = rev ^ xp.uint32((1 << (2 * l)) - 1)
    out = xp.zeros_like(u)
    for k in range(l):
        out |= ((u >> xp.uint32(2 * k)) & xp.uint32(3)) << xp.uint32(2 * (l - 1 - k))
    pad = rev == PAD_WINDOW
    if bool(pad.any()):
        out = xp.where(pad, PAD_WINDOW, out)
    return xp.ascontiguousarray(out)


def is_padded(packed) -> bool:
    """Whether any row of packed windows holds PAD_WINDOW."""
    if packed is None:
        return False
    if is_mlx(packed):
        packed = np.asarray(packed)
    return bool((packed == PAD_WINDOW).any())


def window_counts(packed) -> np.ndarray:
    """[R] int64 number of windows (non-PAD) in each row, on CPU (counted on
    packed's device, so only the counts move)."""
    if is_mlx(packed):
        packed = np.asarray(packed)
    n = (packed != PAD_WINDOW).sum(axis=1)
    return np.asarray(to_cpu(n)).astype(np.int64)


def device_of(x) -> str:
    """'cpu', 'cuda' or 'mlx'."""
    if is_mlx(x):
        return "mlx"
    return "cpu" if get_array_module(x) is np else "cuda"


class PackedGkmKernel(GkmKernel):
    """Base for -t 0/1/2 (DirectGkmKernel) and -t 4 (CenterWeightedGkmKernel)."""

    # Use the masked-join kernel when a plan exists. Set False to force the
    # brute-force loop (e.g. as a test oracle).
    use_masked = True

    def __init__(self, l: int, k: int, **kwargs):
        if l > MAX_L:
            raise ValueError(f"l must be <= {MAX_L}, got {l}")
        super().__init__(l, k, **kwargs)

    def _join_plan(self):
        """Mask plan for kernels/masked.py, or None."""

    def _masked_plan(self):
        """_join_plan() for the pairwise kernel, unless use_masked is off."""
        return self._join_plan() if self.use_masked else None

    def window_weights(self, W: int):
        """[W] uint8 weights of a sequence's W windows (forward order; a reverse
        complement's windows carry them reversed), or None when every window
        counts once. Only -t 4/-t 5 weight windows (kernels/weighted.py)."""

    def row_weights(self, packed, rc: bool = False):
        """Window weights of packed rows: [1, W] uint8 when no row is padded,
        else [R, W] (each row's from its own window count, 0 on PAD); None
        when every window counts once. rc: the rows are RC windows (weights
        reversed, as their windows are)."""
        if packed is None:
            return None
        W = packed.shape[1]
        if self.window_weights(W) is None:
            return None
        counts = window_counts(packed)
        if (counts == W).all():
            w = self.window_weights(W)[None]
        else:
            w = np.zeros((len(counts), W), dtype=np.uint8)
            for n in np.unique(counts[counts > 0]):
                w[counts == n, :n] = self.window_weights(int(n))
        return np.ascontiguousarray(w[:, ::-1]) if rc else w

    def _check_length(self, L: int) -> None:
        if L < self.l:
            raise ValueError(
                f"Sequence length {L} is shorter than window length {self.l}"
            )

    def pack(self, x, lengths=None):
        """[B, W] uint32 packed windows of one-hot x, on x's device.

        N (all-zero) positions pack as A. lengths [B]: each row's sequence
        length (x is padded to its longest); windows past a row's last are
        PAD_WINDOW, and a row shorter than l has none.
        """
        B, _, L = x.shape
        self._check_length(L)
        W = L - self.l + 1
        dev = device_of(x)
        xp = np if dev == "mlx" else get_array_module(x)
        base = (xp.argmax(to_cpu(x) if dev == "mlx" else x, axis=1)).astype(xp.uint32)
        packed = xp.zeros((B, W), dtype=xp.uint32)
        for k in range(self.l):
            packed |= base[:, k : k + W] << xp.uint32(2 * k)
        if lengths is not None:
            lengths = np.asarray(to_cpu(lengths), dtype=np.int64)
            if lengths.shape != (B,) or (lengths > L).any() or (lengths < 0).any():
                raise ValueError(f"lengths must be [{B}] values in 0..{L}")
            if (lengths < L).any():
                n = xp.asarray(np.clip(lengths - self.l + 1, 0, None))
                past = xp.arange(W)[None, :] >= n[:, None]
                packed = xp.where(past, PAD_WINDOW, packed).astype(xp.uint32)
        if dev == "mlx":
            import mlx.core as mx

            return mx.array(packed)
        return packed

    def pack_both(self, x, lengths=None):
        """(packed x, packed reverse complement of x or None); lengths as in pack.

        The RC windows come from the packed forward ones, so N packs as A on
        the forward strand and T on the reverse one, as in lsgkm (which reads
        N as A before taking the reverse complement). Packing the one-hot RC
        would give A on both strands, and an RC block that isn't the reverse
        complement of the forward one.
        """
        fwd = self.pack(x, lengths)
        if not self.include_rc:
            return fwd, None
        if device_of(fwd) == "mlx":
            import mlx.core as mx

            return fwd, mx.array(_rc_windows(np.asarray(fwd), self.l))
        return fwd, _rc_windows(fwd, self.l)

    # -- raw kernel from packed windows ------------------------------------

    def raw_packed(self, bx, by, by_rc=None, *, diag_offset=None):
        """K_raw [B, S] between packed rows bx [B, Wx] and columns by [S, Wy].

        by_rc: the columns' reverse complements, added in the same call (one
            masked-join index serves both blocks). When the columns hold more
            windows than the rows, the masked join instead indexes only the
            forward columns and probes with both strands of the rows, since
            K(x, rc(y)) = K(rc(x), y): the index build, which dominates
            scoring against many SVs, halves.
        diag_offset: rows are sequences 0..B-1 and columns sequences
            diag_offset.. of one set (a self-Gram column panel); entries below
            the diagonal may be left 0 (CPU masked join only).

        Rows and columns are whole sequences' windows, in order, PAD past the
        end of shorter ones: window weights (row_weights) follow their
        positions.

        CPU and CUDA return float64; MLX returns float32 on the device.
        """
        dev = device_of(bx)
        plan = self._masked_plan()
        small = min(bx.shape[0], by.shape[0]) < _MASKED_MIN_ROWS
        wx, wy = self.row_weights(bx), self.row_weights(by)
        wy_rc = self.row_weights(by_rc, rc=True)
        if plan is not None and not small:
            if by_rc is not None and by.size > bx.size:
                bx, wx = _with_rc_rows(bx, wx, self.l)
                by_rc = wy_rc = None
            return _masked(bx, by, plan, by_rc, diag_offset, dev, wx, wy, wy_rc)
        if wx is None:
            raw = self._brute(bx, by, dev)
            if by_rc is not None:
                raw = raw + self._brute(bx, by_rc, dev)
            return raw
        raw = self._brute(bx, by, dev, wx, wy)
        if by_rc is not None:
            raw = raw + self._brute(bx, by_rc, dev, wx, wy_rc)
        return raw

    def _brute(self, bx, by, dev):
        """Brute-force pair loop: [B, S] over all window pairs."""
        raise NotImplementedError

    def raw_diag_packed(self, bx, bx_rc=None):
        """K_raw(x, x) [B]: each row's windows against themselves (and its RC)."""
        return self._brute_cross_diag(
            bx, bx if bx_rc is None else _concat_windows(bx, bx_rc), device_of(bx)
        )

    def _brute_cross_diag(self, bx, by, dev):
        """[B]: sum of g over row b of bx against row b of by."""
        raise NotImplementedError

    def raw_diag_mutants(self, bx, bx_rc, new, new_rc, diag):
        """K_raw(mutant, mutant) [4, L, B] for every single-base substitution of bx.

        new [4, L, B, l] (CPU): the forward windows covering position p after
        substituting base v, padded with PAD_WINDOW (ism._packed_mutations).
        new_rc: the same for the RC windows covering L-1-p, indexed by the
        forward base v (None without RC). diag: K_raw(x, x) [B] of the
        reference. The result is on bx's device.
        """
        raise NotImplementedError

    # -- GkmKernel interface --------------------------------------------------

    def _raw_pairwise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        by, by_rc = self.pack_both(y)
        return self.raw_packed(self.pack(x), by, by_rc).astype(float_dtype(x))

    def _raw_diagonal(
        self, x: np.ndarray, *, chunk_size: int | None = None
    ) -> np.ndarray:
        xp = get_array_module(x)
        B = x.shape[0]
        if chunk_size is not None and chunk_size < B:
            return xp.concatenate(
                [
                    self._raw_diagonal(x[s : s + chunk_size])
                    for s in range(0, B, chunk_size)
                ]
            )
        return self.raw_diag_packed(*self.pack_both(x)).astype(float_dtype(x))


def _masked(bx, by, plan, by_rc, diag_offset, dev, wx=None, wy=None, wy_rc=None):
    """wx, wy, wy_rc: row_weights of the rows, columns and RC columns."""
    from gkmsvm.kernels import masked

    if dev == "cpu":
        return masked.masked_pairwise(
            bx, by, plan, by_rc, diag_offset=diag_offset, wx=wx, wy=wy, wy_rc=wy_rc
        )
    if dev == "cuda":
        return masked.masked_pairwise_gpu(
            bx, by, plan, by_rc, wx=wx, wy=wy, wy_rc=wy_rc
        )
    import mlx.core as mx

    # Exact integer counts combined in float64; Metal holds float32
    out = masked.masked_pairwise_mlx(bx, by, plan, by_rc, wx=wx, wy=wy, wy_rc=wy_rc)
    return mx.array(out.astype(np.float32))


def _with_rc_rows(bx, wx, l):
    """Rows holding their RC windows and then their own ([B, 2W]), and the
    windows' weights (wx: row_weights of bx, or None; an RC window carries
    its forward window's weight).

    The masked join on CPU visits one run of non-PAD windows per row, so
    padded rows move their non-PAD windows to the front (their order doesn't
    change the sum). MLX rows come back as NumPy, which masked_pairwise_mlx
    takes.
    """
    if is_mlx(bx):
        bx = np.asarray(bx)
    xp = get_array_module(bx)
    both = _concat_windows(_rc_windows(bx, l), bx)
    w = None if wx is None else np.concatenate([wx[:, ::-1], wx], axis=1)
    if is_padded(both):
        order = xp.argsort(both == PAD_WINDOW, axis=1)
        both = xp.ascontiguousarray(xp.take_along_axis(both, order, axis=1))
        if w is not None:
            w = np.broadcast_to(w, both.shape)
            w = np.take_along_axis(w, to_cpu(order), axis=1)
    if w is not None:
        w = np.ascontiguousarray(w)
    return both, w


def _concat_windows(a, b):
    """Concatenate packed window arrays along the window axis."""
    return get_array_module(a).concatenate([a, b], axis=1)
