from __future__ import annotations

import numpy as np
from numba import njit, prange

from gkmsvm.backend import get_array_module, is_gpu, is_mlx, to_cpu, to_gpu, to_mlx
from gkmsvm.kernels.base import GkmKernel
from gkmsvm.kernels.direct import DirectGkmKernel
from gkmsvm.kernels.esttrunc import EstTruncGkmKernel
from gkmsvm.kernels.packed import PackedGkmKernel
from gkmsvm.kernels.rbf import RbfGkmKernel, RbfKernel
from gkmsvm.kernels.weighted import (
    CenterWeightedGkmKernel,
    CenterWeightedRbfGkmKernel,
)


@njit(parallel=True, cache=True)
def _normalized_dot(raw, diag_x, diag_sv, norm, coef, gamma):
    """[B]: sum_s coef_s f(k), k = raw[b, s] / sqrt(diag_x[b] diag_sv[s]) if norm.

    f(k) = exp(gamma (k - 1)) when gamma is not NaN (RBF), else k.
    """
    B, S = raw.shape
    rbf = not np.isnan(gamma)
    out = np.zeros(B)
    for b in prange(B):
        acc = 0.0
        for s in range(S):
            k = raw[b, s]
            if norm:
                k /= max(np.sqrt(diag_x[b] * diag_sv[s]), 1e-10)
            if rbf:
                k = np.exp(gamma * (k - 1.0))
            acc += coef[s] * k
        out[b] = acc
    return out

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
        # Packed SV windows, their base-kernel self-kernels and the l-mer
        # index, on the model's device (built lazily, cleared on device moves)
        self._sv_cache: dict = {}
        # Score through the cached l-mer index (kernels/lmer_index.py) when it
        # applies; False forces the masked join / brute-force kernel
        self.use_lmer_index = True
        # Exact l-mer weight table for non-RBF kernels (lmer_table.py):
        # "auto" uses it when it applies and there are enough SVs to pay for
        # building it; True/False force it on or off
        self.use_lmer_table: bool | str = "auto"
        self._lmer_table: np.ndarray | bool | None = None
        self._lmer_table_dev = None
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
        had_gpu, had_mlx = self._on_gpu, self._on_mlx
        self._sv_cache = {}
        self._lmer_table_dev = None
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

    def _sv_packed(self):
        """(fwd, rc or None) packed SV windows of the base kernel, or None.

        None when the base kernel has no packed path for these SVs (weighted
        kernels with N positions).
        """
        c = self._sv_cache
        if "packed" not in c:
            base = self.kernel.base
            sv = self.support_sequences
            ok = isinstance(base, PackedGkmKernel) and base.packs(sv)
            c["packed"] = base.pack_both(sv) if ok else None
        return c["packed"]

    def _sv_diag(self) -> np.ndarray:
        """Base-kernel K_raw(sv, sv) on the model's device (cached)."""
        c = self._sv_cache
        if "diag" not in c:
            packed = self._sv_packed()
            base = self.kernel.base
            if packed is not None:
                diag = base.raw_diag_packed(*packed)
            else:
                diag = base._raw_diagonal(self.support_sequences, chunk_size=1000)
            c["diag"] = diag.astype(self.support_sequences.dtype)
        return c["diag"]

    def _sv_index(self):
        """Cached l-mer index over the SVs, or None when it doesn't apply."""
        c = self._sv_cache
        if "index" not in c:
            from gkmsvm.kernels import lmer_index

            packed = self._sv_packed()
            base = self.kernel.base
            use = self.use_lmer_index and packed is not None and lmer_index.applies(base)
            c["index"] = lmer_index.LmerIndex(base, *packed) if use else None
        return c["index"]

    def _sv_raw(self, bx):
        """Base-kernel K_raw [B, S] of packed query rows bx against the SVs."""
        index = self._sv_index()
        if index is not None:
            return index.pairwise(bx)
        return self.kernel.base.raw_packed(bx, *self._sv_packed())

    def _sv_scores(self, x):
        """sum_s coef_s K(x, sv_s) [B, 1] of one-hot x (on the model's device)."""
        kernel = self.kernel
        base = kernel.base
        if self._sv_packed() is None or not base.packs(x):
            return (kernel.pairwise(x, self.support_sequences)
                    * self.coefficients).sum(axis=1, keepdims=True)
        xp = get_array_module(x)
        bx, bx_rc = base.pack_both(x)
        raw = self._sv_raw(bx)
        diag_x = base.raw_diag_packed(bx, bx_rc) if base.normalize else None
        if xp is np:
            # Normalize, finish and reduce in one parallel pass, no [B, S] temporaries
            norm = diag_x is not None
            gamma = kernel.gamma if isinstance(kernel, RbfKernel) else np.nan
            out = _normalized_dot(
                np.ascontiguousarray(raw, dtype=np.float64),
                diag_x.astype(np.float64) if norm else np.ones(1),
                self._sv_diag().astype(np.float64) if norm else np.ones(1),
                norm, self.coefficients.astype(np.float64), float(gamma))
            return out.astype(x.dtype)[:, None]
        k = raw.astype(x.dtype)
        if diag_x is not None:
            k = k / xp.clip(xp.sqrt(diag_x.astype(x.dtype)[:, None] * self._sv_diag()[None, :]),
                            1e-10, None)
        return (kernel.finish(k) * self.coefficients).sum(axis=1, keepdims=True)

    def __call__(
        self, x: np.ndarray, *,
        batch_size: int | None = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """Compute SVM decision values.

        Args:
            x: [B, 4, L] one-hot encoded DNA sequences.
            batch_size: Process inputs in batches of this size.  ``None``
                means process all at once (internally in blocks that cap the
                [batch, S] kernel values at ``_MAX_SCORE_BYTES``).  Raised to
                at least ``_min_batch_size()``.
            verbose: Show a tqdm progress bar over input batches.

        Returns:
            [B, 1] uncalibrated decision values.
        """
        batch_size = self._min_batch_size(batch_size)
        if batch_size is not None and x.shape[0] > batch_size:
            chunks = range(0, x.shape[0], batch_size)
            if verbose:
                from tqdm import tqdm
                chunks = tqdm(
                    chunks, desc="Scoring",
                    total=(x.shape[0] + batch_size - 1) // batch_size,
                )
            parts = []
            for start in chunks:
                end = min(start + batch_size, x.shape[0])
                batch = self._match_device(x[start:end])
                parts.append(to_cpu(self._score_unbatched(batch)))
            return np.concatenate(parts, axis=0)

        return self._score_unbatched(self._match_device(x))

    # "auto" builds the l-mer table only for models at least this large
    _LMER_TABLE_MIN_SVS = 500

    # Without the l-mer table or index, each kernel call pays a fixed per-SV
    # cost (the masked join indexes every SV window: ~6 s for 32k SVs on CPU),
    # so batches fill at least this many bytes of [batch, S] float64
    _MIN_BATCH_BYTES = 256 * 2**20
    # Kernel scoring computes [rows, S] kernel values in blocks of at most this
    _MAX_SCORE_BYTES = 512 * 2**20

    def _min_batch_size(self, batch_size: int | None, *, table: bool = True) -> int | None:
        """batch_size, raised when calls go through the masked join.

        table: the caller scores through the l-mer table when it applies.
        """
        if batch_size is None:
            return None
        if (table and self._lmer_table_for(np) is not None) or self._sv_index() is not None:
            return batch_size
        return max(batch_size, self._MIN_BATCH_BYTES // (8 * self.num_support_vectors))

    def _build_lmer_table(self) -> np.ndarray | None:
        """Exact table w(u) = sum_sv c_sv K_raw(u, sv) over packed l-mers (cached).

        None when the kernel has no table (RBF, weighted, no mismatch
        cutoff, l > lmer_table.MAX_TABLE_L).
        """
        if self._lmer_table is None:
            from gkmsvm.lmer_table import build_lmer_table

            kernel = self.kernel
            if not isinstance(kernel, PackedGkmKernel) or kernel._join_plan() is None:
                self._lmer_table = False
                return None
            sv = to_cpu(self.support_sequences).astype(np.float64)
            fwd, rc = kernel.pack_both(sv)
            coef = to_cpu(self.coefficients).astype(np.float64)
            if kernel.normalize:
                diag = kernel.raw_diag_packed(fwd, rc)
                coef = coef / np.sqrt(np.clip(diag, 1e-10, None))
            table = build_lmer_table(kernel, fwd, rc, coef)
            self._lmer_table = False if table is None else table
        return None if self._lmer_table is False else self._lmer_table

    def _lmer_table_for(self, xp):
        """The l-mer table on xp's device, or None when it doesn't apply."""
        use = self.use_lmer_table
        if use is False or not isinstance(self.kernel, PackedGkmKernel):
            return None
        if use == "auto" and self.num_support_vectors < self._LMER_TABLE_MIN_SVS:
            return None
        table = self._build_lmer_table()
        if table is None:
            return None
        if xp is np:
            return self._lmer_table
        if self._lmer_table_dev is None or get_array_module(self._lmer_table_dev) is not xp:
            # Metal has no float64: MLX gets a float32 copy (scores are float32 there)
            table = (self._lmer_table.astype(np.float32)
                     if getattr(xp, "__name__", "") == "mlx.core" else self._lmer_table)
            self._lmer_table_dev = xp.asarray(table)
        return self._lmer_table_dev

    def _score_lmer_table(self, x, table):
        """f(x) = sum of table weights over x's windows / sqrt(K(x, x)) + b."""
        xp = get_array_module(x)
        kernel = self.kernel
        packed, packed_rc = kernel.pack_both(x)
        raw = table[packed].sum(axis=1)
        if kernel.normalize:
            diag_x = kernel.raw_diag_packed(packed, packed_rc).astype(raw.dtype)
            raw = raw / xp.sqrt(xp.clip(diag_x, 1e-10, None))
        return raw.astype(x.dtype)[:, None] + self.bias

    def _score_unbatched(self, x: np.ndarray) -> np.ndarray:
        """Score one batch on the model's device."""
        xp = get_array_module(x)
        table = self._lmer_table_for(xp)
        if table is not None and self.kernel.packs(x):
            return self._score_lmer_table(x, table)
        rows = max(1, self._MAX_SCORE_BYTES // (8 * self.num_support_vectors))
        parts = [self._sv_scores(x[s:s + rows]) for s in range(0, x.shape[0], rows)]
        scores = parts[0] if len(parts) == 1 else xp.concatenate(parts, axis=0)
        return scores + self.bias

    def score_variants(
        self, ref: np.ndarray, alt: np.ndarray, *,
        method: str = "kernel",
        batch_size: int | None = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """Variant effect scores.

        Args:
            ref: [B, 4, L] reference sequences.
            alt: [B, 4, L] alternate sequences.
            method: Scoring strategy.
                "kernel" — score(alt) - score(ref) via full kernel (default).
                "gkmexplain" — GkmExplain mutation impact score
                    (Shrikumar et al. 2019, lsgkm ``gkmexplain -m 5``).
                    See :func:`gkmsvm.explain.mutation_impact`.
            batch_size: Process inputs in batches of this size.  ``None``
                means process all at once.
            verbose: Show tqdm progress bar.

        Returns:
            [B, 1] score differences.
        """
        if method == "kernel":
            # Both alleles in the same calls share the per-call SV cost
            n = ref.shape[0]
            scores = self(get_array_module(ref).concatenate([alt, ref]),
                          batch_size=batch_size, verbose=verbose)
            return scores[:n] - scores[n:]

        if method == "gkmexplain":
            from gkmsvm.explain import mutation_impact

            return mutation_impact(
                self, ref, alt, batch_size=batch_size, verbose=verbose,
            )

        raise ValueError(
            f"method must be 'kernel' or 'gkmexplain', got {method!r}"
        )

    _MAX_LMER_TABLE_L = 14  # 4^14 ≈ 268M entries, ~1 GB

    def _deltasvm_weights_from_table(self, table, query_norm: bool) -> np.ndarray:
        """DeltaSVM weights (first base most significant) from the packed table.

        query_norm divides each weight by the l-mer's own self-kernel,
        h(0) (+ h(m(u, rc u)) with RC), as scoring that l-mer alone would.
        """
        kernel = self.kernel
        l = kernel.l
        # Packed index has position k in digit k (low first); DeltaSVM's has
        # position 0 most significant, so reverse the digit order
        w = table.reshape((4,) * l).transpose(tuple(range(l - 1, -1, -1))).ravel()
        if query_norm and kernel.normalize:
            w = w / np.sqrt(np.clip(self._lmer_self_kernels(), 1e-10, None))
        return w.astype(np.float32)

    def _lmer_self_kernels(self) -> np.ndarray:
        """K_raw(u, u) for every l-mer u, in DeltaSVM index order.

        The identical pair plus, with RC, the pair (u, rc u), which agrees at
        positions p where u_p is the complement of u_{l-1-p}.
        """
        from gkmsvm.kernels.weighted import CenterWeightedGkmKernel, _elementary_symmetric_k

        kernel = self.kernel
        l = kernel.l
        idx = np.arange(4 ** l)
        digits = (idx[:, None] // 4 ** np.arange(l - 1, -1, -1)) % 4  # [:, p] = base at p
        agree = digits == 3 - digits[:, ::-1]
        if isinstance(kernel, CenterWeightedGkmKernel):
            w = kernel._pos_weights
            self_k = np.full(len(idx), _elementary_symmetric_k(w, kernel.k))
            if kernel.include_rc:
                # e_k of the weights at agreeing positions, per l-mer
                dp = np.zeros((len(idx), kernel.k + 1))
                dp[:, 0] = 1.0
                for p in range(l):
                    v = agree[:, p] * w[p]
                    for j in range(min(p + 1, kernel.k), 0, -1):
                        dp[:, j] += v * dp[:, j - 1]
                self_k = self_k + dp[:, kernel.k]
            return self_k
        h = np.asarray(kernel._mismatch_table, dtype=np.float64)
        self_k = np.full(len(idx), h[0])
        if kernel.include_rc:
            self_k = self_k + h[(~agree).sum(axis=1)]
        return self_k

    def to_deltasvm(
        self, *, device: str = "cpu", query_norm: bool = False,
        verbose: bool = False,
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
            query_norm: Keep the per-l-mer normalization, so each weight
                equals ``gkmpredict`` on that l-mer minus bias (how
                deltaSVM weights are built from lsgkm models).
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

        table = self._build_lmer_table()
        if table is not None:
            weights = self._deltasvm_weights_from_table(table, query_norm)
            return DeltaSVM(
                weights, l, l,
                include_rc=False, bias=self.bias, device=device,
            )

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
        if kernel.normalize and not query_norm:
            sv_diag = to_cpu(self._sv_diag()).astype(np.float64)
            adj_coefs = adj_coefs / np.sqrt(np.clip(sv_diag, 1e-10, None))

        saved_norm = kernel.normalize
        saved_coefs = self.coefficients
        saved_use_table = self.use_lmer_table
        coef_dtype = to_cpu(saved_coefs).dtype
        # The loop below scores through self() with a modified kernel and
        # coefficients; keep it off the cached l-mer table
        self.use_lmer_table = False
        if not query_norm:
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
            self.use_lmer_table = saved_use_table

        result = DeltaSVM(
            weights, l, l,
            include_rc=False, bias=self.bias, device=device,
        )

        return result

