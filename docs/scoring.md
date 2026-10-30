# Scoring and variant effect prediction

## Basic scoring

All models (loaded or trained) use the same interface. Input is a `[batch, 4, length]` one-hot encoded DNA array (any dtype), a list of DNA strings, or `[batch, length]` uint8 base codes (A=0, C=1, G=2, T=3); output is `[batch, 1]` margin scores, float64 on CPU and CUDA whatever the input dtype (float32 on MLX).

```python
from gkmsvm import load_model, one_hot_encode
import numpy as np

model = load_model("model.npz")

# Encode a single sequence
x = one_hot_encode("ACGTACGTACGTACGTACGT")  # [4, 20]

# Score (add batch dimension)
score = model(x[None, ...])  # [1, 1]
print(f"Score: {score.item():.4f}")
```

### Batch scoring

Score multiple sequences at once:

```python
# From strings (N allowed; lengths may differ)
sequences = ["ACGTACGT...", "TGCATGCA...", ...]
scores = model(sequences)  # [N, 1]

# Or as a padded one-hot batch, with each row's length
from gkmsvm import encode_batch

x, mask = encode_batch(sequences, allow_n=True)  # [N, 4, max_len], [N, max_len]
scores = model(x, lengths=mask.sum(axis=1))  # [N, 1]
```

Without `lengths`, a padded tail is scored as sequence (an all-zero column reads as A). Sequences shorter than `l` score NaN, with a warning, as LS-GKM gives. Windows of a fixed length avoid padding altogether: `x, kept = extract_loci(peaks, genome, in_window=300, return_mask=True)` gives summit-centered `[N, 4, 300]` int8 windows (loci running off a chromosome end are dropped; `kept` marks the loci kept), and `codec.center_crop(seqs, 300)` crops longer sequences.

### Scoring from FASTA files

```python
from gkmsvm import read_fasta

records = read_fasta("sequences.fa")
sequences = [seq for _, seq in records]

scores = model(sequences, batch_size=1000, verbose=True)  # progress bar over batches
```

## Variant effect prediction

### Single variant

```python
ref = one_hot_encode("ACGTACGTACGTACGT")
alt = one_hot_encode("ACGTACGAACGTACGT")  # T→A at position 8

delta = model.score_variants(ref[None], alt[None]).item()  # score(alt) - score(ref)
```

### Batch variant scoring

For scoring many REF/ALT pairs efficiently:

```python
ref_seqs = np.stack([one_hot_encode(s) for s in ref_strings])
alt_seqs = np.stack([one_hot_encode(s) for s in alt_strings])

# Returns [N, 1] delta scores (alt - ref)
deltas = model.score_variants(ref_seqs, alt_seqs, batch_size=1000, verbose=True)

# GkmExplain mutation impact (lsgkm gkmexplain -m 5), the GkmExplain
# paper's dsQTL score; a linearization, so not equal to the kernel delta
impact = model.score_variants(ref_seqs, alt_seqs, method="gkmexplain")
```

Both methods score ref and alt in the same calls. Ref and alt may differ in length (indels): pass lists, or padded arrays with `ref_lengths=` / `alt_lengths=`.

### DeltaSVM (linear approximation)

DeltaSVM provides a faster but approximate variant effect score using pre-computed k-mer weights instead of full kernel evaluation:

```python
from gkmsvm import load_deltasvm_model

dsvm = load_deltasvm_model("weights.txt", l=11)
scores = dsvm(x)
deltas = dsvm.score_variants(ref_seqs, alt_seqs)
```

DeltaSVM is orders of magnitude faster than full SVM scoring but is a linear approximation. For accurate scores, use the full model's `score_variants()`.

To build a deltaSVM table from a trained model (any kernel, `l <= 14`; `-t 0/1/2/4` with l <= 12 read it off the exact l-mer table, others score all 4^l l-mers, minutes for a large RBF model):

```python
dsvm = model.to_deltasvm(query_norm=True)  # weight = gkmpredict(l-mer) - bias
deltas = dsvm.score_variants(ref_seqs, alt_seqs)
```

`query_norm=True` matches how deltaSVM weights are built from lsgkm models (`gkmpredict` on every l-mer). The default omits the per-l-mer normalization; RBF models always keep it, since the RBF is defined on the normalized kernel. Scores published with the Beer lab `deltasvm.pl` script differ for variants containing poly-A/T l-mers; see [validation](validation.md#differences-from-the-papers-scripts).

## GPU acceleration

```python
model.cuda()  # NVIDIA (CuPy)
model.mlx()  # Apple Silicon (MLX)
scores = model(x)  # inputs are moved to the model's device; outputs stay there
model.cpu()  # back to CPU
```

`device="auto"` (in the loaders and training) uses CUDA only when CuPy finds a usable GPU, else MLX if installed, else CPU. Moving a model drops its device caches (packed SVs, l-mer index), which are rebuilt on the first call. MLX runs custom Metal kernels and returns float32 scores (Metal has no float64); GkmExplain on an MLX model runs its l-mer and per-SV kernels in Metal, sums in float64 on CPU, and returns a NumPy float64 array. Apple Silicon's memory is shared between CPU and GPU, so other processes compete for it. Timings are in [performance](performance.md).

## Memory management

Peak memory is bounded internally: kernel scoring computes [rows, S] kernel values in blocks of at most 512 MB, whatever `batch_size` says, and `batch_size` bounds how many inputs move to the device at once. When scoring goes through the masked join (no l-mer table or index applies), small `batch_size` values are raised so each call fills about 256 MB of [batch, S] values, since every call indexes all SV windows. `model.sv_chunk_size` (default 2,000) only chunks SVs inside GkmExplain. See [performance](performance.md#memory).

## One-hot encoding

DNA sequences are represented as `[4, length]` one-hot arrays with channel order A=0, C=1, G=2, T=3. Internally, and in `model.support_sequences`, they are `[length]` uint8 base codes with the same numbering: inputs are converted on entry (N, an all-zero column, reads as A, as in LS-GKM; on the reverse strand it is then T). `one_hot_encode` rejects N unless `allow_n=True`, and other non-ACGT letters unless `allow_other=True` (read as N); DNA strings passed to the model accept N.

```python
from gkmsvm import one_hot_encode, one_hot_decode, reverse_complement

# Encode
x = one_hot_encode("ACGT")  # [4, 4] float32

# Decode back to string
seq = one_hot_decode(x)  # "ACGT"

# Reverse complement
rc = reverse_complement(x)  # [4, 4]
one_hot_decode(rc)  # "ACGT" (palindrome in this case)

# Batch encode with padding
from gkmsvm import encode_batch

x, mask = encode_batch(["ACGT", "ACGTACGT"])  # [2, 4, 8], shorter seqs zero-padded; mask [2, 8]
```

## Reverse-complement invariance

By default, kernels are RC-equivalent: the raw kernel adds the reverse-complement term, `K_raw(x, y) + K_raw(x, RC(y))`. This means scores are the same for a sequence and its reverse complement:

```python
x = one_hot_encode("ACGTACGTACGT")[None, ...]
rc_x = reverse_complement(x)

score = model(x).item()
rc_score = model(rc_x).item()
assert abs(score - rc_score) < 1e-10  # float64 scores
```

This is exact for every kernel type except `-t 4`/`-t 5` with an even number of windows (each RC window carries its forward window's center weight, as in LS-GKM), and for sequences with N (A on one strand, T on the other). RC equivalence is on by default and matches LS-GKM's `norc=0` setting. Models loaded from LS-GKM files respect the `norc` header if present.
