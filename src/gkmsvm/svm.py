from __future__ import annotations

import torch
from torch import nn

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


class GkmSVM(nn.Module):
    """Gapped k-mer SVM model for sequence classification.

    Stores support sequences, signed dual coefficients, bias, and kernel
    configuration as buffers. Produces uncalibrated decision values.

    Score = sum_i(coef_i * K(x, sv_i)) + bias
    """

    def __init__(
        self,
        support_sequences: torch.Tensor,
        coefficients: torch.Tensor,
        bias: float,
        kernel_type: str,
        kernel_params: dict,
        *,
        sv_chunk_size: int | None = None,
    ):
        super().__init__()

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

        self.register_buffer("support_sequences", support_sequences)
        self.register_buffer("coefficients", coefficients)
        self.register_buffer("_bias", torch.tensor(bias, dtype=support_sequences.dtype))

        self.kernel_type = kernel_type
        self._kernel_params = kernel_params
        self.kernel = _build_kernel(kernel_type, kernel_params)
        self.sv_chunk_size = sv_chunk_size

    @property
    def bias(self) -> float:
        return self._bias.item()

    @property
    def num_support_vectors(self) -> int:
        return self.support_sequences.shape[0]

    @property
    def kernel_params(self) -> dict:
        return dict(self._kernel_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute SVM decision values.

        Args:
            x: [B, 4, L] one-hot encoded DNA sequences.

        Returns:
            [B, 1] uncalibrated decision values.
        """
        if self.sv_chunk_size is None or self.sv_chunk_size >= self.num_support_vectors:
            K = self.kernel.pairwise(x, self.support_sequences)
            scores = (K * self.coefficients).sum(dim=1, keepdim=True)
        else:
            scores = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
            S = self.num_support_vectors
            for start in range(0, S, self.sv_chunk_size):
                end = min(start + self.sv_chunk_size, S)
                sv_chunk = self.support_sequences[start:end]
                coef_chunk = self.coefficients[start:end]
                K_chunk = self.kernel.pairwise(x, sv_chunk)
                scores += (K_chunk * coef_chunk).sum(dim=1, keepdim=True)

        return scores + self._bias

    def score_variants(self, ref: torch.Tensor, alt: torch.Tensor) -> torch.Tensor:
        """Compute variant effect scores as score(alt) - score(ref).

        Args:
            ref: [B, 4, L] reference sequences.
            alt: [B, 4, L] alternate sequences.

        Returns:
            [B, 1] score differences.
        """
        return self.forward(alt) - self.forward(ref)
