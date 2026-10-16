"""Kernels whose raw value sums a pair weight over packed l-mer windows.

K_raw(x, y) = sum_{i,j} g(x_i, y_j), with x_i, y_j the length-l windows of x
and y packed 2 bits per base into uint32 (position k in bits 2k, 2k+1). Every
device holds packed windows as [B, W] arrays on that device (MLX arrays are
packed on the CPU). Reverse complements enter as a second column block.

Subclasses provide the brute-force pair loop (`_brute`, `_brute_cross_diag`)
and the mask plan for the masked join (`_join_plan`, kernels/masked.py).
"""

from __future__ import annotations

import numpy as np

from gkmsvm.backend import float_dtype, get_array_module, is_mlx, to_cpu, to_gpu, to_mlx
from gkmsvm.codec import reverse_complement
from gkmsvm.kernels.base import GkmKernel

# Below this many query or column sequences, the masked join's per-call index
# build costs more than it saves (e.g. single SMO columns)
_MASKED_MIN_ROWS = 16

# Window value standing for "no window" in padded window sets (ISM). It has
# bits set above every l-mer, so the l-mer index skips it; the brute and
# masked kernels count it identically wherever it appears, so it cancels in
# differences between window sets padded the same way.
PAD_WINDOW = np.uint32(0xFFFFFFFF)


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

    def _join_plan(self):
        """Mask plan for kernels/masked.py, or None."""

    def _masked_plan(self):
        """_join_plan() for the pairwise kernel, unless use_masked is off."""
        return self._join_plan() if self.use_masked else None

    def packs(self, x) -> bool:
        """Whether the packed path is exact for x (always, unless overridden)."""
        return True

    def _check_length(self, L: int) -> None:
        if L < self.l:
            raise ValueError(f"Sequence length {L} is shorter than window length {self.l}")

    def pack(self, x):
        """[B, W] uint32 packed windows of one-hot x, on x's device.

        N (all-zero) positions pack as A.
        """
        B, _, L = x.shape
        self._check_length(L)
        W = L - self.l + 1
        dev = device_of(x)
        xp = np if dev == "mlx" else get_array_module(x)
        base = (xp.argmax(to_cpu(x) if dev == "mlx" else x, axis=1)).astype(xp.uint32)
        packed = xp.zeros((B, W), dtype=xp.uint32)
        for k in range(self.l):
            packed |= base[:, k:k + W] << xp.uint32(2 * k)
        if dev == "mlx":
            import mlx.core as mx

            return mx.array(packed)
        return packed

    def pack_both(self, x):
        """(packed x, packed reverse complement of x or None)."""
        return self.pack(x), (self.pack(reverse_complement(x)) if self.include_rc else None)

    # -- raw kernel from packed windows ------------------------------------

    def raw_packed(self, bx, by, by_rc=None, *, diag_offset=None):
        """K_raw [B, S] between packed rows bx [B, Wx] and columns by [S, Wy].

        by_rc: the columns' reverse complements, added in the same call (one
            masked-join index serves both blocks).
        diag_offset: rows are sequences 0..B-1 and columns sequences
            diag_offset.. of one set (a self-Gram column panel); entries below
            the diagonal may be left 0 (CPU masked join only).

        CPU and CUDA return float64; MLX returns float32 on the device.
        """
        dev = device_of(bx)
        plan = self._masked_plan()
        small = min(bx.shape[0], by.shape[0]) < _MASKED_MIN_ROWS
        if plan is not None and not (small and self._HAS_BRUTE):
            return _masked(bx, by, plan, by_rc, diag_offset, dev)
        if not self._HAS_BRUTE:
            raise ValueError(f"{type(self).__name__} needs the masked join (use_masked=True)")
        raw = self._brute(bx, by, dev)
        if by_rc is not None:
            raw = raw + self._brute(bx, by_rc, dev)
        return raw

    # Whether _brute exists; without it the masked join serves every size
    _HAS_BRUTE = False

    def _brute(self, bx, by, dev):
        """Brute-force pair loop: [B, S] over all window pairs."""
        raise NotImplementedError

    def raw_diag_packed(self, bx, bx_rc=None):
        """K_raw(x, x) [B]: each row's windows against themselves (and its RC)."""
        return self._brute_cross_diag(
            bx, bx if bx_rc is None else _concat_windows(bx, bx_rc), device_of(bx))

    def _brute_cross_diag(self, bx, by, dev):
        """[B]: sum of g over row b of bx against row b of by."""
        raise NotImplementedError

    def raw_diag_mutants(self, bx, bx_rc, new, new_rc, diag):
        """K_raw(mutant, mutant) [4, L, B] for every single-base substitution of bx.

        new [4, L, B, l] (CPU): the forward windows covering position p after
        substituting base v, padded with PAD_WINDOW (ism._packed_mutations).
        new_rc: the same for the RC windows covering L-1-p, indexed by the
        forward base v (None without RC). diag: K_raw(x, x) [B] of the
        reference. This default builds every mutant's windows and calls
        raw_diag_packed; the result is on bx's device.
        """
        fwd = to_cpu(bx)
        B, W = fwd.shape
        L = new.shape[1]
        l = self.l

        def mutate(ref, windows, flip):
            mut = np.broadcast_to(ref, (4, L, B, W)).copy()
            for p in range(L):
                q = L - 1 - p if flip else p
                a_s, a_e = max(0, q - l + 1), min(q + 1, W)
                mut[:, p, :, a_s:a_e] = windows[:, p, :, : a_e - a_s]
            return mut.reshape(4 * L * B, W)

        dev = {"cpu": np.asarray, "cuda": to_gpu, "mlx": to_mlx}[device_of(bx)]
        mut_rc = None if new_rc is None else dev(mutate(to_cpu(bx_rc), new_rc, True))
        return self.raw_diag_packed(dev(mutate(fwd, new, False)), mut_rc).reshape(4, L, B)

    # -- GkmKernel interface --------------------------------------------------

    def _raw_pairwise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.packs(x) and self.packs(y):
            by, by_rc = self.pack_both(y)
            return self.raw_packed(self.pack(x), by, by_rc).astype(float_dtype(x))
        return self._raw_pairwise_onehot(x, y)

    def _raw_diagonal(self, x: np.ndarray, *, chunk_size: int | None = None) -> np.ndarray:
        xp = get_array_module(x)
        B = x.shape[0]
        if chunk_size is not None and chunk_size < B:
            return xp.concatenate([self._raw_diagonal(x[s:s + chunk_size])
                                   for s in range(0, B, chunk_size)])
        if self.packs(x):
            return self.raw_diag_packed(*self.pack_both(x)).astype(float_dtype(x))
        return self._raw_diagonal_onehot(x)

    def _raw_pairwise_onehot(self, x, y):
        raise NotImplementedError

    def _raw_diagonal_onehot(self, x):
        raise NotImplementedError


def _masked(bx, by, plan, by_rc, diag_offset, dev):
    from gkmsvm.kernels import masked

    if dev == "cpu":
        return masked.masked_pairwise(bx, by, plan, by_rc, diag_offset=diag_offset)
    if dev == "cuda":
        return masked.masked_pairwise_gpu(bx, by, plan, by_rc)
    import mlx.core as mx

    # Exact integer counts combined in float64; Metal holds float32
    return mx.array(masked.masked_pairwise_mlx(bx, by, plan, by_rc).astype(np.float32))


def _concat_windows(a, b):
    """Concatenate packed window arrays along the window axis."""
    return get_array_module(a).concatenate([a, b], axis=1)
