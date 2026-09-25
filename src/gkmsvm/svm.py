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
        self._cached_sv_diag: torch.Tensor | None = None
        self._compiled_pairwise: object | None = None
        self._compile_dtype: torch.dtype | None = None

    @property
    def bias(self) -> float:
        return self._bias.item()

    @property
    def num_support_vectors(self) -> int:
        return self.support_sequences.shape[0]

    @property
    def kernel_params(self) -> dict:
        return dict(self._kernel_params)

    def compile(self, *, dtype: torch.dtype | None = None) -> GkmSVM:
        """Enable torch.compile for the kernel computation.

        Fuses the einsum + histogram table lookup into a single GPU kernel,
        giving ~15x speedup on the pairwise computation. Call once before
        inference; the first forward call will trigger compilation.

        Args:
            dtype: Compute dtype for the compiled pairwise kernel. Use
                ``torch.bfloat16`` on Ampere+ GPUs for ~60% additional
                speedup via tensor cores (5x faster than LS-GKM C).
                Match counts are exact in bf16 (integers ≤ l ≤ 15).

        Returns self for chaining: ``model.cuda().compile(dtype=torch.bfloat16)``.
        """
        from gkmsvm.kernels.direct import DirectGkmKernel

        kernel = self.kernel
        if not isinstance(kernel, DirectGkmKernel):
            return self
        self._compiled_pairwise = torch.compile(kernel.pairwise_from_windows)
        self._compile_dtype = dtype
        return self

    def _pairwise_from_windows(
        self, wx: torch.Tensor, wy: torch.Tensor
    ) -> torch.Tensor:
        if self._compiled_pairwise is not None:
            return self._compiled_pairwise(wx, wy)
        return self.kernel.pairwise_from_windows(wx, wy)

    def _get_sv_diag(self) -> torch.Tensor:
        sv = self.support_sequences
        if self._cached_sv_diag is not None and self._cached_sv_diag.device == sv.device:
            return self._cached_sv_diag
        self._cached_sv_diag = self.kernel._raw_diagonal(sv, chunk_size=1000)
        return self._cached_sv_diag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute SVM decision values.

        Args:
            x: [B, 4, L] one-hot encoded DNA sequences.

        Returns:
            [B, 1] uncalibrated decision values.
        """
        S = self.num_support_vectors
        chunk = self.sv_chunk_size
        kernel = self.kernel
        do_norm = kernel.normalize
        has_flat = hasattr(kernel, "flat_windows")

        if do_norm:
            diag_x = kernel._raw_diagonal(x)
            diag_sv = self._get_sv_diag()

        if chunk is None or chunk >= S:
            raw = self._compute_raw_pairwise(x, self.support_sequences, has_flat)
            if do_norm:
                norm = torch.sqrt(diag_x.unsqueeze(1) * diag_sv.unsqueeze(0))
                raw = raw / torch.clamp(norm, min=1e-10)
            scores = (raw * self.coefficients).sum(dim=1, keepdim=True)
        else:
            scores = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
            for start in range(0, S, chunk):
                end = min(start + chunk, S)
                raw = self._compute_raw_pairwise(
                    x, self.support_sequences[start:end], has_flat
                )
                if do_norm:
                    norm = torch.sqrt(
                        diag_x.unsqueeze(1) * diag_sv[start:end].unsqueeze(0)
                    )
                    raw = raw / torch.clamp(norm, min=1e-10)
                scores += (raw * self.coefficients[start:end]).sum(dim=1, keepdim=True)

        return scores + self._bias

    def _compute_raw_pairwise(
        self, x: torch.Tensor, y: torch.Tensor, has_flat: bool
    ) -> torch.Tensor:
        kernel = self.kernel
        if not has_flat or self._compiled_pairwise is None:
            return kernel._raw_pairwise(x, y)
        wx = kernel.flat_windows(x)
        wy = kernel.flat_windows(y)
        if self._compile_dtype is not None:
            wx = wx.to(self._compile_dtype)
            wy = wy.to(self._compile_dtype)
        result = self._pairwise_from_windows(wx, wy)
        if kernel.include_rc:
            from gkmsvm.codec import reverse_complement

            wy_rc = kernel.flat_windows(reverse_complement(y))
            if self._compile_dtype is not None:
                wy_rc = wy_rc.to(self._compile_dtype)
            result = result + self._pairwise_from_windows(wx, wy_rc)
        return result.to(x.dtype)

    def score_variants(self, ref: torch.Tensor, alt: torch.Tensor) -> torch.Tensor:
        """Compute variant effect scores as score(alt) - score(ref).

        Args:
            ref: [B, 4, L] reference sequences.
            alt: [B, 4, L] alternate sequences.

        Returns:
            [B, 1] score differences.
        """
        return self.forward(alt) - self.forward(ref)
