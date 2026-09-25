# Roadmap

## Completed

- [x] One-hot codec with RC, validation, batch padding
- [x] DirectGkmKernel (`-t 0` / `gkm_cnt`) with combinatorial verification
- [x] EstTruncGkmKernel (`-t 2` / `gkm_esttrunc`) ported from LS-GKM C source
- [x] GkmSVM nn.Module with chunked SV evaluation
- [x] LS-GKM model importer (plain text and gzip)
- [x] FASTA I/O backed by tangermeme
- [x] Oracle validation against Dongwon-Lee/lsgkm `gkmpredict`
- [x] 107 tests passing
- [x] NumPy/Numba CPU reference implementation with cross-validation
- [x] Numba/PyTorch threading layer pin (workqueue, adapted from scprism)
- [x] GPU kernel redesign: matmul replaces 6D broadcast (440x less intermediate memory)
- [x] GPU _apply_table histogram dispatch (17x less peak memory, prevents MPS OOM at 500 SVs)
- [x] ISM utility with window-delta optimization (19x MPS speedup via batched GPU compute)
- [x] GkmExplain attribution (mode 0 + hypothetical mode 1, 20-30x faster than ISM)
- [x] 149 tests passing

## Next
- [ ] **deltaSVM importer** — tab-separated `<kmer>\t<score>` linear model
- [ ] **gkmSVM classic importer** — legacy two-file format with opposite bias sign
- [ ] **Extend kernel modes** — `-t 3` (RBF), `-t 4` (center-weighted), `-t 5` (combined)
- [ ] **Training** — C-SVM solver with kernel-capable optimizer and cache discipline
