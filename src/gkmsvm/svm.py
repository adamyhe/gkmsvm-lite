from __future__ import annotations

import numpy as np

from gkmsvm.backend import get_array_module, is_gpu, is_mlx, to_cpu, to_gpu, to_mlx
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
        device: str = "cpu",
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
        self._on_mlx = False

        if device != "cpu":
            self._to_device(device)

    def _to_device(self, device: str) -> None:
        """Move model to the specified device."""
        from gkmsvm.backend import HAS_CUPY, HAS_MLX
        if device in ("cuda", "gpu"):
            self.cuda()
        elif device == "mlx":
            self.mlx()
        elif device == "auto":
            if HAS_CUPY:
                self.cuda()
            elif HAS_MLX:
                self.mlx()
        elif device != "cpu":
            raise ValueError(
                f"Unknown device {device!r}. Use 'auto', 'cpu', 'cuda', or 'mlx'."
            )

    @property
    def num_support_vectors(self) -> int:
        return self.support_sequences.shape[0]

    @property
    def kernel_params(self) -> dict:
        return dict(self._kernel_params)

    def _invalidate_caches(self):
        had_gpu = is_gpu(self._sv_idx_windows) if self._sv_idx_windows is not None else self._on_gpu
        had_mlx = is_mlx(self._sv_idx_windows) if self._sv_idx_windows is not None else self._on_mlx
        self._cached_sv_diag = None
        self._sv_idx_windows = None
        self._sv_rc_idx_windows = None
        self._sv_packed_t = None
        self._sv_rc_packed_t = None
        if had_mlx:
            try:
                import mlx.core as mx
                mx.clear_cache()
            except (ImportError, AttributeError):
                pass
        if had_gpu:
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except (ImportError, AttributeError):
                pass

    def cuda(self) -> GkmSVM:
        """Move model arrays to GPU (CuPy)."""
        self.support_sequences = to_gpu(self.support_sequences)
        self.coefficients = to_gpu(self.coefficients)
        self._invalidate_caches()
        self._on_gpu = True
        self._on_mlx = False
        return self

    def mlx(self) -> GkmSVM:
        """Move model arrays to Apple GPU (MLX)."""
        self.support_sequences = to_mlx(to_cpu(self.support_sequences))
        self.coefficients = to_mlx(to_cpu(self.coefficients))
        self._invalidate_caches()
        self._on_gpu = False
        self._on_mlx = True
        return self

    def cpu(self) -> GkmSVM:
        """Move model arrays to CPU (NumPy)."""
        self.support_sequences = to_cpu(self.support_sequences)
        self.coefficients = to_cpu(self.coefficients)
        self._invalidate_caches()
        self._on_gpu = False
        self._on_mlx = False
        return self

    def _match_device(self, x: np.ndarray) -> np.ndarray:
        """Convert input array to match the model's device."""
        if self._on_gpu and not is_gpu(x):
            return to_gpu(np.asarray(x) if is_mlx(x) else x)
        if self._on_mlx and not is_mlx(x):
            return to_mlx(to_cpu(x) if is_gpu(x) else x)
        if not self._on_gpu and not self._on_mlx and (is_gpu(x) or is_mlx(x)):
            return to_cpu(x)
        return x

    def save(self, path: str, *, fmt: str | None = None) -> None:
        """Save model to disk.

        Format is auto-detected from extension unless overridden:
            - ``.npz`` — fast native format
            - ``.txt``, ``.txt.gz`` — LS-GKM text format (interop)

        Args:
            path: Output file path.
            fmt: ``"npz"`` or ``"lsgkm"``. Auto-detected if None.
        """
        from gkmsvm.serialization import save_lsgkm, save_npz

        if fmt is None:
            name = str(path).lower()
            if name.endswith(".npz"):
                fmt = "npz"
            else:
                fmt = "lsgkm"

        if fmt == "npz":
            save_npz(self, path)
        elif fmt == "lsgkm":
            save_lsgkm(self, path)
        else:
            raise ValueError(f"Unknown format {fmt!r}. Use 'npz' or 'lsgkm'.")

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
                if is_mlx(self._sv_idx_windows):
                    xp = get_array_module(self._sv_idx_windows)
                    self._sv_packed_t = xp.asarray(_pack_windows_cpu(
                        np.ascontiguousarray(to_cpu(self._sv_idx_windows))
                    ))
                    if self._sv_rc_idx_windows is not None:
                        self._sv_rc_packed_t = xp.asarray(_pack_windows_cpu(
                            np.ascontiguousarray(
                                to_cpu(self._sv_rc_idx_windows)
                            )
                        ))
                elif self._on_gpu:
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
        x = self._match_device(x)
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
        elif chunk is not None and chunk < S and hasattr(kernel, "pairwise_from_indices"):
            self._get_sv_index_windows()
            bx = kernel.base_index_windows(x)
            on_gpu = is_gpu(bx)
            if on_gpu:
                import cupy as cp
                from gkmsvm.kernels.direct import _pack_windows_uint32
                bx_p = cp.ascontiguousarray(
                    _pack_windows_uint32(cp.asarray(bx), cp)
                )
            else:
                bx_p = None

            def _slice_packed(packed, s, e):
                if on_gpu:
                    return xp.ascontiguousarray(packed[:, s:e])
                return packed[s:e]

            scores = xp.zeros((x.shape[0], 1), dtype=x.dtype)
            sv_iter = range(0, S, chunk)
            if verbose:
                from tqdm import tqdm
                sv_iter = tqdm(sv_iter, desc="SV chunks", total=(S + chunk - 1) // chunk)
            for start in sv_iter:
                end = min(start + chunk, S)
                by_t = _slice_packed(self._sv_packed_t, start, end)
                raw = kernel.pairwise_from_indices(
                    bx, None, by_packed_t=by_t, bx_packed=bx_p
                )
                if kernel.include_rc:
                    by_rc_t = _slice_packed(self._sv_rc_packed_t, start, end)
                    raw = raw + kernel.pairwise_from_indices(
                        bx, None, by_packed_t=by_rc_t, bx_packed=bx_p
                    )
                raw = raw.astype(x.dtype)
                if do_norm:
                    norm = xp.sqrt(
                        diag_x[:, None] * diag_sv[start:end][None, :]
                    )
                    raw = raw / xp.clip(norm, 1e-10, None)
                scores += (raw * self.coefficients[start:end]).sum(
                    axis=1, keepdims=True
                )
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
        self, ref: np.ndarray, alt: np.ndarray, *,
        method: str = "kernel",
        batch_size: int = 50,
        verbose: bool = False,
    ) -> np.ndarray:
        """Variant effect scores.

        Args:
            ref: [B, 4, L] reference sequences.
            alt: [B, 4, L] alternate sequences.
            method: Scoring strategy.
                "kernel" — score(alt) - score(ref) via full kernel (default).
                "gkmexplain" — GkmExplain hypothetical importance at the
                    variant position (Shrikumar et al. 2019 §5.2).  Computes
                    mode-1 attributions on ref, then reads off the predicted
                    effect at each position where ref and alt differ.  Faster
                    than kernel when the model has many SVs, because only one
                    attribution pass is needed per sequence.
            batch_size: Batch size for gkmexplain (ignored for kernel).
            verbose: Show tqdm progress bar.

        Returns:
            [B, 1] score differences.
        """
        if method == "kernel":
            ref = self._match_device(ref)
            alt = self._match_device(alt)
            return self(alt, verbose=verbose) - self(ref, verbose=verbose)

        if method == "gkmexplain":
            from gkmsvm.explain import gkmexplain
            from gkmsvm.backend import get_array_module, to_cpu

            ref = self._match_device(ref)
            alt = self._match_device(alt)
            hyp = gkmexplain(self, ref, mode=1, batch_size=batch_size,
                             verbose=verbose)
            diff_mask = ref != alt
            ref_contrib = (hyp * ref * diff_mask).sum(axis=(1, 2))
            alt_contrib = (hyp * alt * diff_mask).sum(axis=(1, 2))
            xp = get_array_module(hyp)
            return xp.reshape(alt_contrib - ref_contrib, (-1, 1))

        raise ValueError(
            f"method must be 'kernel' or 'gkmexplain', got {method!r}"
        )

    _MAX_LMER_TABLE_L = 14  # 4^14 ≈ 268M entries, ~1 GB

    def to_deltasvm(
        self, *, device: str = "cpu", verbose: bool = False,
    ) -> "DeltaSVM":
        """Convert to a DeltaSVM linear scoring model.

        Scores all 4^l l-mers through the full model to build a weight
        lookup table. The resulting DeltaSVM uses one table lookup per
        window position (no gapped k-mer decomposition).

        The approximation is in query normalization: the full SVM divides
        by sqrt(K(x,x)), which varies per query. The DeltaSVM omits this
        and gives unnormalized scores. For variant scoring, K(ref,ref)
        and K(alt,alt) are nearly equal (single-SNP changes affect few
        windows), so the normalization cancels and variant effects
        correlate near-perfectly (r > 0.999 for l >= 10).

        Works for any kernel type. Requires l <= 14 (4^l table entries).

        Args:
            device: Device for the returned DeltaSVM model.
            verbose: Show progress bar during weight computation.

        Returns:
            DeltaSVM with k=l (one weight per l-mer).
        """
        from gkmsvm.deltasvm import DeltaSVM

        l = self.kernel.l
        if l > self._MAX_LMER_TABLE_L:
            raise ValueError(
                f"to_deltasvm() requires l <= {self._MAX_LMER_TABLE_L} "
                f"(4^{l} = {4**l:,} table entries would use "
                f"{4**l * 4 / 1e9:.1f} GB). Got l={l}."
            )

        was_on_gpu = self._on_gpu
        was_on_mlx = self._on_mlx

        n_lmers = 4**l
        weights = np.zeros(n_lmers, dtype=np.float32)

        S = self.num_support_vectors
        # [B, S] intermediate × 2 (RC) × 2 (norm + accumulation)
        bytes_per_seq = S * 8 * 4
        if self._on_gpu:
            import cupy as cp
            free, _ = cp.cuda.Device().mem_info
            chunk = max(256, min(100_000, int(free * 0.3) // bytes_per_seq))
        else:
            mem_budget = 2 * 1024**3
            chunk = max(256, min(100_000, mem_budget // bytes_per_seq))

        chunks = range(0, n_lmers, chunk)
        if verbose:
            from tqdm import tqdm
            chunks = tqdm(
                chunks, desc="to_deltasvm",
                total=(n_lmers + chunk - 1) // chunk,
            )

        powers = 4 ** np.arange(l - 1, -1, -1)
        pos_idx = np.arange(l)

        kernel = self.kernel
        adj_coefs = to_cpu(self.coefficients).astype(np.float64)
        if kernel.normalize:
            sv_diag = to_cpu(self._get_sv_diag()).astype(np.float64)
            adj_coefs = adj_coefs / np.sqrt(np.clip(sv_diag, 1e-10, None))

        saved_norm = kernel.normalize
        saved_coefs = self.coefficients
        coef_dtype = to_cpu(saved_coefs).dtype
        kernel.normalize = False
        self.coefficients = self._match_device(
            adj_coefs.astype(coef_dtype)
        )
        try:
            for start in chunks:
                end = min(start + chunk, n_lmers)
                indices = np.arange(start, end)
                bases = (indices[:, None] // powers[None, :]) % 4
                n = end - start
                x = np.zeros((n, 4, l), dtype=coef_dtype)
                x[np.arange(n)[:, None], bases, pos_idx[None, :]] = 1.0
                x = self._match_device(x)
                scores = self(x).flatten()
                if self._on_gpu or self._on_mlx:
                    scores = to_cpu(scores)
                weights[start:end] = scores - self.bias
        finally:
            kernel.normalize = saved_norm
            self.coefficients = saved_coefs

        result = DeltaSVM(
            weights, l, l,
            include_rc=False, bias=self.bias, device=device,
        )

        return result

