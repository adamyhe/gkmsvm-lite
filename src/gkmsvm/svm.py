from __future__ import annotations

import numpy as np

from gkmsvm.backend import get_array_module, is_gpu, to_cpu, to_gpu
from gkmsvm.kernels.base import GkmKernel
from gkmsvm.kernels.direct import DirectGkmKernel
from gkmsvm.kernels.esttrunc import EstTruncGkmKernel
from gkmsvm.kernels.rbf import RbfGkmKernel
from gkmsvm.kernels.weighted import (
    CenterWeightedGkmKernel,
    CenterWeightedRbfGkmKernel,
)

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


def _build_kernel(kernel_type: str, kernel_params: dict) -> GkmKernel:
    builder = KERNEL_BUILDERS.get(kernel_type)
    if builder is None:
        supported = ", ".join(KERNEL_BUILDERS)
        raise NotImplementedError(
            f"Kernel type '{kernel_type}' is not yet implemented. "
            f"Supported: {supported}"
        )
    return builder(kernel_params)


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
        kernel_type: str,
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
        self.kernel_type = kernel_type
        self._kernel_params = kernel_params
        self.kernel = _build_kernel(kernel_type, kernel_params)
        self.sv_chunk_size = sv_chunk_size
        self._cached_sv_diag: np.ndarray | None = None
        self._on_gpu = False

    @property
    def num_support_vectors(self) -> int:
        return self.support_sequences.shape[0]

    @property
    def kernel_params(self) -> dict:
        return dict(self._kernel_params)

    def cuda(self) -> GkmSVM:
        """Move model arrays to GPU (CuPy)."""
        self.support_sequences = to_gpu(self.support_sequences)
        self.coefficients = to_gpu(self.coefficients)
        self._cached_sv_diag = None
        self._on_gpu = True
        return self

    def cpu(self) -> GkmSVM:
        """Move model arrays to CPU (NumPy)."""
        self.support_sequences = to_cpu(self.support_sequences)
        self.coefficients = to_cpu(self.coefficients)
        self._cached_sv_diag = None
        self._on_gpu = False
        return self

    def _get_sv_diag(self) -> np.ndarray:
        sv = self.support_sequences
        if self._cached_sv_diag is not None:
            if is_gpu(self._cached_sv_diag) == is_gpu(sv):
                return self._cached_sv_diag
        self._cached_sv_diag = self.kernel._raw_diagonal(sv, chunk_size=1000)
        return self._cached_sv_diag

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Compute SVM decision values.

        Args:
            x: [B, 4, L] one-hot encoded DNA sequences.

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

        if chunk is None or chunk >= S:
            raw = kernel._raw_pairwise(x, self.support_sequences)
            if do_norm:
                norm = xp.sqrt(diag_x[:, None] * diag_sv[None, :])
                raw = raw / xp.clip(norm, 1e-10, None)
            scores = (raw * self.coefficients).sum(axis=1, keepdims=True)
        else:
            scores = xp.zeros((x.shape[0], 1), dtype=x.dtype)
            for start in range(0, S, chunk):
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

        return scores + self.bias

    def score_variants(self, ref: np.ndarray, alt: np.ndarray) -> np.ndarray:
        """Compute variant effect scores as score(alt) - score(ref).

        Args:
            ref: [B, 4, L] reference sequences.
            alt: [B, 4, L] alternate sequences.

        Returns:
            [B, 1] score differences.
        """
        return self(alt) - self(ref)
