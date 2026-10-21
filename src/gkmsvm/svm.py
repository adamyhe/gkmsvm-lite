from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

from gkmsvm.backend import DeviceModel, float_dtype, get_array_module, to_cpu
from gkmsvm.kernels.direct import DirectGkmKernel
from gkmsvm.kernels.esttrunc import EstTruncGkmKernel
from gkmsvm.kernels.packed import (
    PAD_WINDOW,
    PackedGkmKernel,
    is_padded,
)
from gkmsvm.kernels.rbf import RbfGkmKernel, RbfKernel
from gkmsvm.kernels.weighted import (
    CenterWeightedGkmKernel,
    CenterWeightedRbfGkmKernel,
)

if TYPE_CHECKING:
    from gkmsvm.deltasvm import DeltaSVM


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
#   -t 0  gkm_cnt       / direct          Gapped k-mer count (pairs within d mismatches)
#   -t 1  gkm_estfull   / estimated_full  Estimated l-mer kernel, full
#   -t 2  gkm_esttrunc  / estimated       Estimated l-mer kernel, truncated (default)
#   -t 3  gkmrbf        / rbf             RBF on estimated kernel
#   -t 4  wgkm          / weighted        Center-weighted gapped k-mer
#   -t 5  wgkmrbf       / weighted_rbf    Center-weighted RBF
#   -t 6  gkm_cnt_rbf   / direct_rbf      RBF on the gapped k-mer count kernel
#                                         (kundajelab/lsgkm fork)

KERNEL_BUILDERS = {
    # No d: every pair counts (models saved before d applied to -t 0)
    "gkm_cnt": lambda p: DirectGkmKernel(
        l=p["L"],
        k=p["k"],
        d=p.get("d"),
        normalize=True,
        include_rc=p.get("include_rc", True),
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
        d=p.get("d", 3),
        M=p.get("M", 50),
        H=p.get("H", 50.0),
        normalize=True,
        include_rc=p.get("include_rc", True),
    ),
    "wgkmrbf": lambda p: CenterWeightedRbfGkmKernel(
        l=p["L"],
        k=p["k"],
        d=p.get("d", 3),
        M=p.get("M", 50),
        H=p.get("H", 50.0),
        gamma=p.get("gamma", 1.0),
        normalize=True,
        include_rc=p.get("include_rc", True),
    ),
    "gkm_cnt_rbf": lambda p: RbfKernel(
        DirectGkmKernel(
            l=p["L"],
            k=p["k"],
            d=p.get("d", 3),
            normalize=True,
            include_rc=p.get("include_rc", True),
        ),
        gamma=p.get("gamma", 1.0),
    ),
}

KERNEL_ALIASES: dict[str | int, str] = {
    "direct": "gkm_cnt",
    "estimated": "gkm_esttrunc",
    "estimated_full": "gkm_estfull",
    "rbf": "gkmrbf",
    "weighted": "wgkm",
    "weighted_rbf": "wgkmrbf",
    "direct_rbf": "gkm_cnt_rbf",
    0: "gkm_cnt",
    1: "gkm_estfull",
    2: "gkm_esttrunc",
    3: "gkmrbf",
    4: "wgkm",
    5: "wgkmrbf",
    6: "gkm_cnt_rbf",
}


def resolve_kernel_type(kernel_type: str | int) -> str:
    """Resolve a kernel type alias or ``-t N`` integer to the internal name.

    Accepts: internal names (``gkm_cnt``), aliases (``direct``),
    or LS-GKM ``-t`` integers (``0``, or ``"0"`` as the CLI passes it).
    """
    if isinstance(kernel_type, str) and kernel_type.strip().isdigit():
        kernel_type = int(kernel_type)
    if isinstance(kernel_type, int):
        canonical = KERNEL_ALIASES.get(kernel_type)
        if canonical is None:
            raise ValueError(
                f"Unknown kernel type integer {kernel_type}. "
                f"Valid: 0-6 (LS-GKM -t flag; 6 from the kundajelab fork)"
            )
        return canonical
    if kernel_type in KERNEL_BUILDERS:
        return kernel_type
    canonical = KERNEL_ALIASES.get(kernel_type)
    if canonical is not None:
        return canonical
    supported = ", ".join(f"{a!r}" for a in KERNEL_ALIASES if isinstance(a, str))
    raise NotImplementedError(
        f"Unknown kernel type {kernel_type!r}. "
        f"Aliases: {supported}. "
        f"Internal names: {', '.join(KERNEL_BUILDERS)}"
    )


# LS-GKM's MAX_SEQ_LENGTH (libsvm_gkm.h) is 2048: gkmtrain/gkmpredict read
# at most this many bases of a sequence, with a warning
LSGKM_MAX_LENGTH = 2047


def _uniform_or_lengths(lengths, shape) -> np.ndarray | None:
    """lengths as host int64 [B], or None when every row is shape[2] long."""
    if lengths is None:
        return None
    lengths = np.asarray(to_cpu(lengths), dtype=np.int64).reshape(-1)
    B, _, L = shape
    if lengths.shape[0] != B or (lengths < 0).any() or (lengths > L).any():
        raise ValueError(f"lengths must be {B} values in 0..{L}")
    return None if (lengths == L).all() else lengths


def as_batch(x, lengths=None):
    """(x [B, 4, L_max], lengths or None) from a one-hot array (with lengths of
    its rows when they differ) or a list of [4, L_i] arrays or DNA strings
    (N allowed). Warns about sequences longer than LS-GKM reads."""
    import warnings

    if isinstance(x, (list, tuple)):
        if lengths is not None:
            raise ValueError("lengths go with a padded array, not a list")
        if not len(x):
            raise ValueError("no sequences")
        if isinstance(x[0], str):
            from gkmsvm.codec import one_hot_encode

            x = [one_hot_encode(s, allow_n=True) for s in x]
        lens = np.array([a.shape[-1] for a in x], dtype=np.int64)
        xp = get_array_module(x[0])
        out = xp.zeros((len(x), 4, int(lens.max())), dtype=x[0].dtype)
        for i, a in enumerate(x):
            out[i, :, : a.shape[-1]] = a
        x, lengths = out, lens
    lengths = _uniform_or_lengths(lengths, x.shape)
    longest = int(x.shape[2] if lengths is None else lengths.max())
    if longest > LSGKM_MAX_LENGTH:
        warnings.warn(
            f"sequences up to {longest} bp: LS-GKM reads only the first "
            f"{LSGKM_MAX_LENGTH} bases of longer ones",
            stacklevel=3,
        )
    return x, lengths


def _allele_arrays(ref, alt, ref_lengths=None, alt_lengths=None):
    """(ref, alt, ref_lengths, alt_lengths): both as batches (svm.as_batch) of
    one length L, padded on the host when they differ; lengths None when
    every row is L long."""
    ref, ref_lengths = as_batch(ref, ref_lengths)
    alt, alt_lengths = as_batch(alt, alt_lengths)
    if ref.shape[0] != alt.shape[0]:
        raise ValueError(
            f"ref and alt hold {ref.shape[0]} and {alt.shape[0]} sequences"
        )
    L = max(ref.shape[2], alt.shape[2])

    def widen(x, n):
        if x.shape[2] == L:
            return x, n
        n = np.full(x.shape[0], x.shape[2], dtype=np.int64) if n is None else n
        x = to_cpu(x)
        return np.pad(x, ((0, 0), (0, 0), (0, L - x.shape[2]))), n

    ref, ref_lengths = widen(ref, ref_lengths)
    alt, alt_lengths = widen(alt, alt_lengths)
    return ref, alt, ref_lengths, alt_lengths


def _joint_lengths(L, *parts):
    """Lengths of rows from several batches of one length L (each part a
    (rows, lengths or None) pair), or None when every row is L long."""
    if all(n is None for _, n in parts):
        return None
    return np.concatenate(
        [np.full(rows, L, dtype=np.int64) if n is None else n for rows, n in parts]
    )


def _short_rows(lengths, l, what="scores"):
    """[B] bool, rows shorter than l (no windows), or None when there are none."""
    if lengths is None:
        return None
    short = lengths < l
    if not short.any():
        return None
    import warnings

    warnings.warn(
        f"{int(short.sum())} sequences are shorter than l = {l}: their {what} are NaN",
        stacklevel=3,
    )
    return short


def _nan_rows(out, short):
    """out [B, ...] with the rows in short set to NaN (on out's device)."""
    if short is None:
        return out
    xp = get_array_module(out)
    mask = xp.asarray(short.reshape(-1, *[1] * (out.ndim - 1)))
    return xp.where(mask, float("nan"), out)


def _per_position(out, lengths, short):
    """Per-position results out [B, 4, L] with positions past each row's length
    0 and rows in short NaN (on out's device)."""
    if lengths is None:
        return out
    xp = get_array_module(out)
    inside = np.arange(out.shape[2])[None, None, :] < lengths[:, None, None]
    out = xp.where(xp.asarray(inside), out, 0.0).astype(out.dtype)
    return _nan_rows(out, short)


class GkmSVM(DeviceModel):
    """Gapped k-mer SVM model for sequence classification.

    Stores support sequences, signed dual coefficients, bias, and kernel
    configuration. Produces uncalibrated decision values.

    Score = sum_i(coef_i * K(x, sv_i)) + bias
    """

    _DEVICE_ARRAYS = ("support_sequences", "coefficients")
    _FLOAT_ARRAYS = ("coefficients",)

    def __init__(
        self,
        support_sequences: np.ndarray,
        coefficients: np.ndarray,
        bias: float,
        kernel_type: str | int,
        kernel_params: dict,
        *,
        sv_lengths=None,
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
        # SVs of different lengths: support_sequences is padded to the longest
        # and sv_lengths [S] (host int64) gives each one's; None when uniform
        self.sv_lengths = _uniform_or_lengths(sv_lengths, support_sequences.shape)
        # One-hot SVs keep the caller's dtype; coefficients are float64 (float32 on MLX)
        self.coefficients = coefficients.astype(float_dtype(coefficients))
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
        # GkmExplain attributions of every l-mer window for non-RBF kernels
        # (explain._build_explain_table), on CPU, one per mode: "auto" builds
        # one when a call explains enough sequences and it fits; True/False
        # force it on or off
        self.use_explain_table: bool | str = "auto"
        self._explain_tables: dict = {}

        if device != "cpu":
            self._to_device(device)

    @property
    def num_support_vectors(self) -> int:
        return self.support_sequences.shape[0]

    @property
    def kernel_params(self) -> dict:
        return dict(self._kernel_params)

    def clear_caches(self) -> None:
        """Drop the packed SVs, self-kernels, l-mer index, l-mer table and
        explain tables.

        They are rebuilt on the next call. Frees their memory, and lets a
        timing include building them.
        """
        self._sv_cache = {}
        self._lmer_table = None
        self._lmer_table_dev = None
        self._explain_tables = {}

    def _on_device_move(self) -> None:
        """Drop device caches and free the old device's memory pool."""
        self._sv_cache = {}
        self._lmer_table_dev = None
        if self._on_mlx:
            try:
                import mlx.core as mx

                mx.clear_cache()
            except (ImportError, AttributeError):
                pass
        if self._on_gpu:
            try:
                import cupy as cp

                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except (ImportError, AttributeError):
                pass

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
        """(fwd, rc or None) packed SV windows (cached)."""
        c = self._sv_cache
        if "packed" not in c:
            c["packed"] = self.kernel.base.pack_both(
                self.support_sequences, self.sv_lengths
            )
        return c["packed"]

    def _inv_norms(self, diag_x, B):
        """1 / sqrt(K(x, x)) [B] and 1 / sqrt(K(sv, sv)) [S] on CPU (ones without
        normalization)."""
        if diag_x is None:
            return np.ones(B), np.ones(self.num_support_vectors)
        return (
            1 / np.sqrt(np.clip(np.asarray(diag_x, dtype=np.float64), 1e-20, None)),
            1
            / np.sqrt(np.clip(to_cpu(self._sv_diag()).astype(np.float64), 1e-20, None)),
        )

    def _sv_diag(self) -> np.ndarray:
        """Base-kernel K_raw(sv, sv) on the model's device (cached)."""
        c = self._sv_cache
        if "diag" not in c:
            diag = self.kernel.base.raw_diag_packed(*self._sv_packed())
            c["diag"] = diag.astype(float_dtype(diag))
        return c["diag"]

    def _sv_index(self):
        """Cached l-mer index over the SVs, or None when it doesn't apply or
        ``use_lmer_index`` is off (checked on every call)."""
        if not self.use_lmer_index:
            return None
        c = self._sv_cache
        if "index" not in c:
            from gkmsvm.kernels import lmer_index

            packed = self._sv_packed()
            base = self.kernel.base
            n_windows = sum(p.size for p in packed if p is not None)
            use = n_windows > 0 and lmer_index.applies(base, n_windows)
            c["index"] = lmer_index.LmerIndex(base, *packed) if use else None
        return c["index"]

    def _sv_raw(self, bx):
        """Base-kernel K_raw [B, S] of packed query rows bx against the SVs."""
        index = self._sv_index()
        if index is not None:
            return index.pairwise(bx)
        return self.kernel.base.raw_packed(bx, *self._sv_packed())

    def _sv_scores(self, x, lengths=None):
        """sum_s coef_s K(x, sv_s) [B, 1] of one-hot x (on the model's device);
        lengths: each row's length when they differ."""
        kernel = self.kernel
        base = kernel.base
        xp = get_array_module(x)
        bx, bx_rc = base.pack_both(x, lengths)
        diag_x = base.raw_diag_packed(bx, bx_rc) if base.normalize else None
        gamma = kernel.gamma if isinstance(kernel, RbfKernel) else np.nan
        index = self._sv_index()
        if xp is np and index is not None:
            # Fused index query + reduction: no [B, S] block at all
            inv_dx, inv_ds = self._inv_norms(diag_x, x.shape[0])
            return index.reduce(bx, inv_dx, inv_ds, self.coefficients, gamma)[:, None]
        raw = self._sv_raw(bx)
        if xp is np:
            # Normalize, finish and reduce in one parallel pass, no [B, S] temporaries
            norm = diag_x is not None
            out = _normalized_dot(
                np.ascontiguousarray(raw, dtype=np.float64),
                diag_x.astype(np.float64) if norm else np.ones(1),
                self._sv_diag().astype(np.float64) if norm else np.ones(1),
                norm,
                self.coefficients.astype(np.float64),
                float(gamma),
            )
            return out[:, None]
        k = raw.astype(float_dtype(x))
        if diag_x is not None:
            k = k / xp.clip(
                xp.sqrt(
                    diag_x.astype(float_dtype(x))[:, None] * self._sv_diag()[None, :]
                ),
                1e-10,
                None,
            )
        return (kernel.finish(k) * self.coefficients).sum(axis=1, keepdims=True)

    def __call__(
        self,
        x,
        *,
        lengths=None,
        batch_size: int | None = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """Compute SVM decision values.

        Args:
            x: [B, 4, L] one-hot encoded DNA sequences, or a list of [4, L_i]
                arrays or DNA strings of any lengths.
            lengths: [B] lengths of the rows of a padded x (sequences of
                different lengths, all-zero past each one's end); None when
                every row is L long.
            batch_size: Process inputs in batches of this size.  ``None``
                means process all at once (internally in blocks that cap the
                [batch, S] kernel values at ``_MAX_SCORE_BYTES``).  Raised to
                at least ``_min_batch_size()``.
            verbose: Show a tqdm progress bar over input batches.

        Returns:
            [B, 1] uncalibrated decision values; NaN for sequences shorter
            than the window length l (with a warning), as LS-GKM gives.
        """
        x, lengths = as_batch(x, lengths)
        short = _short_rows(lengths, self.kernel.base.l)
        if short is not None and short.all():
            return _nan_rows(np.zeros((x.shape[0], 1)), short)
        batch_size = self._min_batch_size(batch_size)
        if batch_size is not None and x.shape[0] > batch_size:
            chunks = range(0, x.shape[0], batch_size)
            if verbose:
                from tqdm import tqdm

                chunks = tqdm(
                    chunks,
                    desc="Scoring",
                    total=(x.shape[0] + batch_size - 1) // batch_size,
                )
            parts = []
            for start in chunks:
                end = min(start + batch_size, x.shape[0])
                batch = self._match_device(x[start:end])
                lb = None if lengths is None else lengths[start:end]
                parts.append(to_cpu(self._score_unbatched(batch, lb)))
            return _nan_rows(np.concatenate(parts, axis=0), short)

        return _nan_rows(self._score_unbatched(self._match_device(x), lengths), short)

    # "auto" builds the l-mer table only for models at least this large
    _LMER_TABLE_MIN_SVS = 500

    # Without the l-mer table or index, each kernel call pays a fixed per-SV
    # cost (the masked join indexes every SV window: ~6 s for 32k SVs on CPU),
    # so batches fill at least this many bytes of [batch, S] float64
    _MIN_BATCH_BYTES = 256 * 2**20
    # Kernel scoring computes [rows, S] kernel values in blocks of at most this
    _MAX_SCORE_BYTES = 512 * 2**20

    def _min_batch_size(self, batch_size: int | None) -> int | None:
        """batch_size, raised when calls go through the masked join."""
        if batch_size is None:
            return None
        if self._lmer_table_for(np) is not None or self._sv_index() is not None:
            return batch_size
        return max(batch_size, self._MIN_BATCH_BYTES // (8 * self.num_support_vectors))

    def _build_lmer_table(self) -> np.ndarray | None:
        """Exact table w(u) = sum_sv c_sv K_raw(u, sv) over packed l-mers (cached).

        None when the kernel has no table (RBF, no mismatch cutoff,
        l > lmer_table.MAX_TABLE_L).
        """
        if self._lmer_table is None:
            from gkmsvm.lmer_table import build_lmer_table

            kernel = self.kernel
            if not isinstance(kernel, PackedGkmKernel) or kernel._join_plan() is None:
                self._lmer_table = False
                return None
            sv = to_cpu(self.support_sequences).astype(np.float64)
            fwd, rc = kernel.pack_both(sv, self.sv_lengths)
            coef = to_cpu(self.coefficients).astype(np.float64)
            if kernel.normalize:
                diag = to_cpu(self._sv_diag()).astype(np.float64)
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
        if (
            self._lmer_table_dev is None
            or get_array_module(self._lmer_table_dev) is not xp
        ):
            # (MLX's asarray makes it float32: scores are float32 there)
            self._lmer_table_dev = xp.asarray(self._lmer_table)
        return self._lmer_table_dev

    def _table_raw(self, packed, table):
        """[R] sum of table weights over packed rows' windows, times their
        window weights (PAD adds 0)."""
        xp = get_array_module(packed)
        wx = self.kernel.row_weights(packed)  # 0 on PAD
        pad = packed == PAD_WINDOW if is_padded(packed) else None
        vals = table[packed] if pad is None else table[xp.where(pad, 0, packed)]
        if wx is not None:
            return (vals * xp.asarray(wx)).sum(axis=1)
        if pad is not None:
            return xp.where(pad, 0.0, vals).sum(axis=1)
        return vals.sum(axis=1)

    def _score_lmer_table(self, x, table, lengths=None):
        """f(x) = sum of table weights over x's windows (times their window
        weights) / sqrt(K(x, x)) + b."""
        xp = get_array_module(x)
        kernel = self.kernel
        packed, packed_rc = kernel.pack_both(x, lengths)
        raw = self._table_raw(packed, table)
        if kernel.normalize:
            diag_x = kernel.raw_diag_packed(packed, packed_rc).astype(raw.dtype)
            raw = raw / xp.sqrt(xp.clip(diag_x, 1e-10, None))
        return raw[:, None] + self.bias

    def _score_unbatched(self, x: np.ndarray, lengths=None) -> np.ndarray:
        """Score one batch on the model's device."""
        xp = get_array_module(x)
        table = self._lmer_table_for(xp)
        if table is not None:
            return self._score_lmer_table(x, table, lengths)
        rows = max(1, self._MAX_SCORE_BYTES // (8 * self.num_support_vectors))
        parts = [
            self._sv_scores(
                x[s : s + rows], None if lengths is None else lengths[s : s + rows]
            )
            for s in range(0, x.shape[0], rows)
        ]
        scores = parts[0] if len(parts) == 1 else xp.concatenate(parts, axis=0)
        return scores + self.bias

    def score_variants(
        self,
        ref,
        alt,
        *,
        method: str = "kernel",
        ref_lengths=None,
        alt_lengths=None,
        batch_size: int | None = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """Variant effect scores.

        Args:
            ref: [B, 4, L] reference sequences, or a list of [4, L_i] arrays
                or DNA strings of different lengths.
            alt: [B, 4, L'] alternate sequences, as ref (an indel changes
                the length).
            method: Scoring strategy.
                "kernel" — score(alt) - score(ref) via full kernel (default).
                "gkmexplain" — GkmExplain mutation impact score
                    (Shrikumar et al. 2019, lsgkm ``gkmexplain -m 5``).
                    See :func:`gkmsvm.explain.mutation_impact`.
            ref_lengths, alt_lengths: [B] lengths of the rows of padded ref
                and alt arrays when they differ.
            batch_size: Process inputs in batches of this size.  ``None``
                means process all at once.
            verbose: Show tqdm progress bar.

        Returns:
            [B, 1] score differences (NaN when an allele is shorter than l).
        """
        if method == "kernel":
            ref, alt, ref_lengths, alt_lengths = _allele_arrays(
                ref, alt, ref_lengths, alt_lengths
            )
            # Both alleles in the same calls share the per-call SV cost
            n = ref.shape[0]
            alt, ref = self._match_device(alt), self._match_device(ref)
            scores = self(
                get_array_module(ref).concatenate([alt, ref]),
                lengths=_joint_lengths(
                    ref.shape[2], (n, alt_lengths), (n, ref_lengths)
                ),
                batch_size=batch_size,
                verbose=verbose,
            )
            return scores[:n] - scores[n:]

        if method == "gkmexplain":
            from gkmsvm.explain import mutation_impact

            return mutation_impact(
                self,
                ref,
                alt,
                ref_lengths=ref_lengths,
                alt_lengths=alt_lengths,
                batch_size=batch_size,
                verbose=verbose,
            )

        raise ValueError(f"method must be 'kernel' or 'gkmexplain', got {method!r}")

    # Largest l to_deltasvm() accepts: 4^14 float64 weights = 2 GB
    _MAX_DELTASVM_L = 14

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
        # An l-mer alone is one window, weighted M for -t 4 (which normalization cancels)
        wt = kernel.window_weights(1)
        if wt is not None:
            w = w * float(wt[0])
        if query_norm and kernel.normalize:
            w = w / np.sqrt(np.clip(self._lmer_self_kernels(), 1e-10, None))
        return w.astype(np.float64)

    def _lmer_self_kernels(self) -> np.ndarray:
        """K_raw(u, u) for every l-mer u, in DeltaSVM index order.

        The identical pair plus, with RC, the pair (u, rc u), which agrees at
        positions p where u_p is the complement of u_{l-1-p}; times the
        window's weight squared (-t 4).
        """
        kernel = self.kernel.base
        l = kernel.l
        idx = np.arange(4**l)
        digits = (
            idx[:, None] // 4 ** np.arange(l - 1, -1, -1)
        ) % 4  # [:, p] = base at p
        agree = digits == 3 - digits[:, ::-1]
        h = np.asarray(kernel._mismatch_table, dtype=np.float64)
        self_k = np.full(len(idx), h[0])
        if kernel.include_rc:
            self_k = self_k + h[(~agree).sum(axis=1)]
        wt = kernel.window_weights(1)
        return self_k if wt is None else self_k * float(wt[0]) ** 2

    def to_deltasvm(
        self,
        *,
        device: str = "cpu",
        query_norm: bool = False,
        verbose: bool = False,
    ) -> DeltaSVM:
        """Convert to a DeltaSVM linear scoring model: one weight per l-mer.

        Each weight is the model's kernel sum on that l-mer alone. Kernels
        with an exact l-mer table (-t 0/1/2/4, l <= 12) read it off the
        table; others (RBF, larger l) score all 4^l l-mers through the
        kernel, which for an RBF model with many SVs takes minutes.

        Without ``query_norm`` the weights drop the l-mer's own
        normalization 1/sqrt(K(u, u)), so a DeltaSVM score sums
        unnormalized kernel values; for variant effects K(ref, ref) and
        K(alt, alt) nearly cancel (r > 0.999 for l >= 10). RBF kernels are
        defined on the normalized kernel, so they always use ``query_norm``.

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
        if l > self._MAX_DELTASVM_L:
            raise ValueError(
                f"to_deltasvm() requires l <= {self._MAX_DELTASVM_L} "
                f"(4^{l} = {4**l:,} weights would use {4**l * 4 / 1e9:.1f} GB). Got l={l}."
            )
        table = self._build_lmer_table()
        if table is not None:
            weights = self._deltasvm_weights_from_table(table, query_norm)
        else:
            weights = self._deltasvm_weights_from_kernel(query_norm, verbose)
        return DeltaSVM(weights, l, l, include_rc=False, bias=self.bias, device=device)

    # l-mers scored per call in _deltasvm_weights_from_kernel (scoring then
    # works in [rows, S] blocks of _MAX_SCORE_BYTES)
    _DELTASVM_CHUNK = 65536

    def _deltasvm_weights_from_kernel(
        self, query_norm: bool, verbose: bool
    ) -> np.ndarray:
        """DeltaSVM weights by scoring every l-mer (DeltaSVM index order).

        With RC, an l-mer and its reverse complement score the same, so only
        canonical l-mers (index <= RC index) are scored, as lsgkm's deltaSVM
        pipeline does with nrkmers.py.
        """
        base = self.kernel.base
        unnorm = not query_norm and base.normalize and base is self.kernel  # not RBF
        l = self.kernel.l
        idx_all = np.arange(4**l, dtype=np.int64)
        powers = 4 ** np.arange(l - 1, -1, -1)  # first base most significant
        digits = (idx_all[:, None] // powers) % 4
        rc = ((3 - digits[:, ::-1]) * powers).sum(axis=1)
        todo = idx_all[idx_all <= rc] if base.include_rc else idx_all
        weights = np.empty(4**l, dtype=np.float64)
        starts = range(0, len(todo), self._DELTASVM_CHUNK)
        if verbose:
            from tqdm import tqdm

            starts = tqdm(starts, desc="to_deltasvm", total=len(starts))
        for start in starts:
            idx = todo[start : start + self._DELTASVM_CHUNK]
            x = np.eye(4, dtype=np.float32)[digits[idx]].transpose(0, 2, 1)
            x = self._match_device(np.ascontiguousarray(x))
            w = to_cpu(self._score_unbatched(x)).ravel().astype(np.float64) - self.bias
            if unnorm:
                diag = to_cpu(base._raw_diagonal(x)).astype(np.float64)
                w *= np.sqrt(np.clip(diag, 1e-10, None))
            weights[idx] = w
        if base.include_rc:
            weights[rc[todo]] = weights[todo]
        return weights
