from __future__ import annotations

import numpy as np

from gkmsvm.backend import get_array_module, is_gpu, to_cpu, to_gpu
from gkmsvm.codec import reverse_complement
from gkmsvm.kernels.base import GkmKernel
from gkmsvm.kernels.direct import DirectGkmKernel
from gkmsvm.kernels.esttrunc import EstTruncGkmKernel
from gkmsvm.kernels.rbf import RbfGkmKernel
from gkmsvm.kernels.weighted import (
    CenterWeightedGkmKernel,
    CenterWeightedRbfGkmKernel,
)

# Kernel modes:
#   -t 0  gkm_cnt       / direct          Exact gapped k-mer count
#   -t 1  gkm_estfull   / estimated_full  Estimated l-mer kernel, full
#   -t 2  gkm_esttrunc  / estimated       Estimated l-mer kernel, truncated (default)
#   -t 3  gkmrbf        / rbf             RBF on estimated kernel
#   -t 4  wgkm          / weighted        Center-weighted gapped k-mer
#   -t 5  wgkmrbf       / weighted_rbf    Center-weighted RBF

KERNEL_BUILDERS = {
    "gkm_cnt": lambda p: DirectGkmKernel(
        l=p["L"], k=p["k"], normalize=True, include_rc=p.get("include_rc", True)
    ),
    "gkm_esttrunc": lambda p: EstTruncGkmKernel(
        l=p["L"],
        k=p["k"],
        d=p.get("d", 3),
        normalize=True,
        include_rc=p.get("include_rc", True),
        truncate=True,
    ),
    "gkm_estfull": lambda p: EstTruncGkmKernel(
        l=p["L"],
        k=p["k"],
        d=p.get("d", 3),
        normalize=True,
        include_rc=p.get("include_rc", True),
        truncate=False,
    ),
    "gkmrbf": lambda p: RbfGkmKernel(
        l=p["L"],
        k=p["k"],
        d=p.get("d", 3),
        gamma=p.get("gamma", 1.0),
        normalize=True,
        include_rc=p.get("include_rc", True),
    ),
    "wgkm": lambda p: CenterWeightedGkmKernel(
        l=p["L"],
        k=p["k"],
        M=p["M"],
        H=p["H"],
        normalize=True,
        include_rc=p.get("include_rc", True),
    ),
    "wgkmrbf": lambda p: CenterWeightedRbfGkmKernel(
        l=p["L"],
        k=p["k"],
        M=p["M"],
        H=p["H"],
        gamma=p.get("gamma", 1.0),
        normalize=True,
        include_rc=p.get("include_rc", True),
    ),
}

KERNEL_ALIASES: dict[str | int, str] = {
    "direct":         "gkm_cnt",
    "estimated":      "gkm_esttrunc",
    "estimated_full": "gkm_estfull",
    "rbf":            "gkmrbf",
    "weighted":       "wgkm",
    "weighted_rbf":   "wgkmrbf",
    0: "gkm_cnt",
    1: "gkm_estfull",
    2: "gkm_esttrunc",
    3: "gkmrbf",
    4: "wgkm",
    5: "wgkmrbf",
}


def resolve_kernel_type(kernel_type: str | int) -> str:
    """Resolve a kernel type alias or ``-t N`` integer to the internal name.

    Accepts: internal names (``gkm_cnt``), aliases (``direct``),
    or LS-GKM ``-t`` integers (``0``).
    """
    if isinstance(kernel_type, int):
        canonical = KERNEL_ALIASES.get(kernel_type)
        if canonical is None:
            raise ValueError(
                f"Unknown kernel type integer {kernel_type}. "
                f"Valid: 0-5 (LS-GKM -t flag)"
            )
        return canonical
    if kernel_type in KERNEL_BUILDERS:
        return kernel_type
    canonical = KERNEL_ALIASES.get(kernel_type)
    if canonical is not None:
        return canonical
    supported = ", ".join(
        f"{a!r}" for a in KERNEL_ALIASES if isinstance(a, str)
    )
    raise NotImplementedError(
        f"Unknown kernel type {kernel_type!r}. "
        f"Aliases: {supported}. "
        f"Internal names: {', '.join(KERNEL_BUILDERS)}"
    )


class GkmSVM:
    """Gapped k-mer SVM model for sequence classification.

    Stores support sequences, signed dual coefficients, bias, and kernel
    configuration. Produces uncalibrated decision values.

    Score = sum_i(coef_i * K(x, sv_i)) + bias
    """

    def __init__(
        self,
        support_sequences: np.ndarray,
        coefficients: np.ndarray,
        bias: float,
        kernel_type: str | int,
        kernel_params: dict,
        *,
        sv_chunk_size: int | None = None,
    ):
        if support_sequences.ndim != 3 or support_sequences.shape[1] != 4:
            raise ValueError(
                f"support_sequences must be [S, 4, L], got {list(support_sequences.shape)}"
            )
        if coefficients.ndim != 1:
            raise ValueError(
                f"coefficients must be [S], got {list(coefficients.shape)}"
            )
        if support_sequences.shape[0] != coefficients.shape[0]:
            raise ValueError(
                f"Number of support sequences ({support_sequences.shape[0]}) "
                f"must match number of coefficients ({coefficients.shape[0]})"
            )

        self.support_sequences = support_sequences
        self.coefficients = coefficients
        self.bias = bias
        self.kernel_type = resolve_kernel_type(kernel_type)
        self._kernel_params = kernel_params
        self.kernel = KERNEL_BUILDERS[self.kernel_type](kernel_params)
        self.sv_chunk_size = sv_chunk_size
        self._cached_sv_diag: np.ndarray | None = None
        self._sv_idx_windows: np.ndarray | None = None
        self._sv_rc_idx_windows: np.ndarray | None = None
        self._sv_packed_t: np.ndarray | None = None
        self._sv_rc_packed_t: np.ndarray | None = None
        self._on_gpu = False

    @property
    def num_support_vectors(self) -> int:
        return self.support_sequences.shape[0]

    @property
    def kernel_params(self) -> dict:
        return dict(self._kernel_params)

    def _invalidate_caches(self):
        self._cached_sv_diag = None
        self._sv_idx_windows = None
        self._sv_rc_idx_windows = None
        self._sv_packed_t = None
        self._sv_rc_packed_t = None

    def cuda(self) -> GkmSVM:
        """Move model arrays to GPU (CuPy)."""
        self.support_sequences = to_gpu(self.support_sequences)
        self.coefficients = to_gpu(self.coefficients)
        self._invalidate_caches()
        self._on_gpu = True
        return self

    def cpu(self) -> GkmSVM:
        """Move model arrays to CPU (NumPy)."""
        self.support_sequences = to_cpu(self.support_sequences)
        self.coefficients = to_cpu(self.coefficients)
        self._invalidate_caches()
        self._on_gpu = False
        return self

    def save(self, path: str, *, format: str | None = None) -> None:
        """Save model to disk.

        Format is auto-detected from extension unless overridden:
            - ``.npz`` — fast native format
            - ``.txt``, ``.txt.gz`` — LS-GKM text format (interop)

        Args:
            path: Output file path.
            format: ``"npz"`` or ``"lsgkm"``. Auto-detected if None.
        """
        from gkmsvm.serialization import save_lsgkm, save_npz

        if format is None:
            name = str(path).lower()
            if name.endswith(".npz"):
                format = "npz"
            else:
                format = "lsgkm"

        if format == "npz":
            save_npz(self, path)
        elif format == "lsgkm":
            save_lsgkm(self, path)
        else:
            raise ValueError(f"Unknown format {format!r}. Use 'npz' or 'lsgkm'.")

    def _get_sv_diag(self) -> np.ndarray:
        sv = self.support_sequences
        if self._cached_sv_diag is not None:
            if is_gpu(self._cached_sv_diag) == is_gpu(sv):
                return self._cached_sv_diag
        self._cached_sv_diag = self.kernel._raw_diagonal(sv, chunk_size=1000)
        return self._cached_sv_diag

    def _get_sv_index_windows(self):
        """Lazily compute and cache int8 base-index windows for SVs (fwd + RC).

        On GPU, also caches packed uint32 windows for the CUDA kernel.
        """
        if self._sv_idx_windows is None:
            kernel = self.kernel
            self._sv_idx_windows = kernel.base_index_windows(
                self.support_sequences
            )
            if kernel.include_rc:
                self._sv_rc_idx_windows = kernel.base_index_windows(
                    reverse_complement(self.support_sequences)
                )
            if hasattr(kernel, '_min_matches'):
                from gkmsvm.kernels.direct import (
                    _pack_windows_cpu, _pack_windows_uint32,
                )
                if self._on_gpu:
                    xp = get_array_module(self._sv_idx_windows)
                    packed = _pack_windows_uint32(self._sv_idx_windows, xp)
                    self._sv_packed_t = xp.ascontiguousarray(packed.T)
                    if self._sv_rc_idx_windows is not None:
                        packed_rc = _pack_windows_uint32(
                            self._sv_rc_idx_windows, xp
                        )
                        self._sv_rc_packed_t = xp.ascontiguousarray(
                            packed_rc.T
                        )
                else:
                    self._sv_packed_t = _pack_windows_cpu(
                        np.ascontiguousarray(self._sv_idx_windows)
                    )
                    if self._sv_rc_idx_windows is not None:
                        self._sv_rc_packed_t = _pack_windows_cpu(
                            np.ascontiguousarray(self._sv_rc_idx_windows)
                        )
        return self._sv_idx_windows, self._sv_rc_idx_windows

    def __call__(self, x: np.ndarray, *, verbose: bool = False) -> np.ndarray:
        """Compute SVM decision values.

        Args:
            x: [B, 4, L] one-hot encoded DNA sequences.
            verbose: Show tqdm progress bar over SV chunks.

        Returns:
            [B, 1] uncalibrated decision values.
        """
        xp = get_array_module(x)
        S = self.num_support_vectors
        chunk = self.sv_chunk_size
        kernel = self.kernel
        do_norm = kernel.normalize

        if do_norm:
            diag_x = kernel._raw_diagonal(x)
            diag_sv = self._get_sv_diag()

        if hasattr(kernel, "pairwise_from_indices") and (chunk is None or chunk >= S):
            sv_idx, sv_rc_idx = self._get_sv_index_windows()
            bx = kernel.base_index_windows(x)
            raw = kernel.pairwise_from_indices(
                bx, sv_idx, by_packed_t=self._sv_packed_t
            )
            if kernel.include_rc:
                raw = raw + kernel.pairwise_from_indices(
                    bx, sv_rc_idx, by_packed_t=self._sv_rc_packed_t
                )
            raw = raw.astype(x.dtype)
            if do_norm:
                norm = xp.sqrt(diag_x[:, None] * diag_sv[None, :])
                raw = raw / xp.clip(norm, 1e-10, None)
            scores = (raw * self.coefficients).sum(axis=1, keepdims=True)
        elif chunk is not None and chunk < S:
            scores = xp.zeros((x.shape[0], 1), dtype=x.dtype)
            sv_iter = range(0, S, chunk)
            if verbose:
                from tqdm import tqdm
                sv_iter = tqdm(sv_iter, desc="SV chunks", total=(S + chunk - 1) // chunk)
            for start in sv_iter:
                end = min(start + chunk, S)
                raw = kernel._raw_pairwise(
                    x, self.support_sequences[start:end]
                )
                if do_norm:
                    norm = xp.sqrt(
                        diag_x[:, None] * diag_sv[start:end][None, :]
                    )
                    raw = raw / xp.clip(norm, 1e-10, None)
                scores += (raw * self.coefficients[start:end]).sum(
                    axis=1, keepdims=True
                )
        else:
            raw = kernel._raw_pairwise(x, self.support_sequences)
            if do_norm:
                norm = xp.sqrt(diag_x[:, None] * diag_sv[None, :])
                raw = raw / xp.clip(norm, 1e-10, None)
            scores = (raw * self.coefficients).sum(axis=1, keepdims=True)

        return scores + self.bias

    def score_variants(
        self, ref: np.ndarray, alt: np.ndarray, *, verbose: bool = False
    ) -> np.ndarray:
        """Variant effect scores: score(alt) - score(ref).

        Args:
            ref: [B, 4, L] reference sequences.
            alt: [B, 4, L] alternate sequences.
            verbose: Show tqdm progress bar.

        Returns:
            [B, 1] score differences.
        """
        return self(alt, verbose=verbose) - self(ref, verbose=verbose)
