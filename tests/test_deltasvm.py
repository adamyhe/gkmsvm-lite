"""Tests for DeltaSVM linear k-mer scoring model."""
import random
import tempfile
from itertools import combinations
from pathlib import Path

import pytest
import torch

from gkmsvm.codec import one_hot_encode, reverse_complement
from gkmsvm.deltasvm import DeltaSVM, _kmer_to_index, _index_to_kmer
from gkmsvm.importers.deltasvm import load_deltasvm_weights


def _make_seqs(n, length, seed=42):
    rng = random.Random(seed)
    return ["".join(rng.choice("ACGT") for _ in range(length)) for _ in range(n)]


def _naive_deltasvm_score(seq, weights_dict, l, k, include_rc):
    """Brute-force deltaSVM scoring for verification."""
    score = 0.0
    W = len(seq) - l + 1
    combos = list(combinations(range(l), k))
    for w in range(W):
        window = seq[w : w + l]
        for combo in combos:
            kmer = "".join(window[p] for p in combo)
            score += weights_dict.get(kmer, 0.0)
    if include_rc:
        rc_map = {"A": "T", "T": "A", "C": "G", "G": "C"}
        rc_seq = "".join(rc_map[b] for b in reversed(seq))
        for w in range(W):
            window = rc_seq[w : w + l]
            for combo in combos:
                kmer = "".join(window[p] for p in combo)
                score += weights_dict.get(kmer, 0.0)
    return score


class TestKmerIndexing:
    def test_roundtrip(self):
        for k in [3, 5, 7]:
            for idx in range(4**k):
                kmer = _index_to_kmer(idx, k)
                assert len(kmer) == k
                assert _kmer_to_index(kmer) == idx

    def test_known_values(self):
        assert _kmer_to_index("AAA") == 0
        assert _kmer_to_index("TTT") == 63
        assert _kmer_to_index("ACG") == 0 * 16 + 1 * 4 + 2


class TestDeltaSVMConstruction:
    def test_basic(self):
        weights = torch.randn(64)
        model = DeltaSVM(weights, l=5, k=3)
        assert model.l == 5
        assert model.k == 3
        assert model._combos.shape == (10, 3)

    def test_wrong_weight_size(self):
        with pytest.raises(ValueError, match="weights must be"):
            DeltaSVM(torch.randn(32), l=5, k=3)

    def test_k_greater_than_l(self):
        with pytest.raises(ValueError, match="k.*must be <= l"):
            DeltaSVM(torch.randn(64), l=2, k=3)


class TestDeltaSVMScoring:
    def test_matches_naive_no_rc(self):
        k, l = 3, 5
        rng = random.Random(1)
        weights_dict = {}
        weights_tensor = torch.zeros(4**k)
        for idx in range(4**k):
            kmer = _index_to_kmer(idx, k)
            w = rng.gauss(0, 1)
            weights_dict[kmer] = w
            weights_tensor[idx] = w

        model = DeltaSVM(weights_tensor, l=l, k=k, include_rc=False)
        seqs = _make_seqs(3, 15, seed=2)

        for seq in seqs:
            x = one_hot_encode(seq).unsqueeze(0)
            got = model(x).item()
            expected = _naive_deltasvm_score(seq, weights_dict, l, k, False)
            assert abs(got - expected) < 1e-4, f"seq={seq}: {got} vs {expected}"

    def test_matches_naive_with_rc(self):
        k, l = 3, 5
        rng = random.Random(3)
        weights_dict = {}
        weights_tensor = torch.zeros(4**k)
        for idx in range(4**k):
            kmer = _index_to_kmer(idx, k)
            w = rng.gauss(0, 1)
            weights_dict[kmer] = w
            weights_tensor[idx] = w

        model = DeltaSVM(weights_tensor, l=l, k=k, include_rc=True)
        seqs = _make_seqs(3, 15, seed=4)

        for seq in seqs:
            x = one_hot_encode(seq).unsqueeze(0)
            got = model(x).item()
            expected = _naive_deltasvm_score(seq, weights_dict, l, k, True)
            assert abs(got - expected) < 1e-4, f"seq={seq}: {got} vs {expected}"

    def test_output_shape(self):
        model = DeltaSVM(torch.randn(64), l=5, k=3)
        x = torch.stack([one_hot_encode(s) for s in _make_seqs(3, 20)])
        assert model(x).shape == (3, 1)

    def test_bias(self):
        model_no_bias = DeltaSVM(torch.randn(64), l=5, k=3, bias=0.0)
        model_with_bias = DeltaSVM(
            model_no_bias.weights.clone(), l=5, k=3, bias=1.5
        )
        x = one_hot_encode(_make_seqs(1, 15)[0]).unsqueeze(0)
        diff = model_with_bias(x).item() - model_no_bias(x).item()
        assert abs(diff - 1.5) < 1e-6

    def test_batch_consistent(self):
        model = DeltaSVM(torch.randn(64), l=5, k=3)
        seqs = _make_seqs(4, 15, seed=5)
        x_batch = torch.stack([one_hot_encode(s) for s in seqs])
        batch_result = model(x_batch)
        for i in range(4):
            single = model(x_batch[i : i + 1])
            assert torch.allclose(batch_result[i : i + 1], single, atol=1e-6)

    def test_score_variants(self):
        model = DeltaSVM(torch.randn(64), l=5, k=3)
        ref = one_hot_encode(_make_seqs(1, 15, seed=6)[0]).unsqueeze(0)
        alt = one_hot_encode(_make_seqs(1, 15, seed=7)[0]).unsqueeze(0)
        delta = model.score_variants(ref, alt)
        expected = model(alt) - model(ref)
        assert torch.allclose(delta, expected, atol=1e-6)

    def test_short_sequence(self):
        model = DeltaSVM(torch.randn(64), l=5, k=3)
        x = one_hot_encode("ACGT").unsqueeze(0)
        result = model(x)
        assert result.shape == (1, 1)
        assert result.item() == model.bias

    def test_zero_weights(self):
        model = DeltaSVM(torch.zeros(64), l=5, k=3, bias=0.5)
        x = one_hot_encode(_make_seqs(1, 15)[0]).unsqueeze(0)
        assert abs(model(x).item() - 0.5) < 1e-6


class TestDeltaSVMImporter:
    def _write_weight_file(self, path, k, weights_dict):
        with open(path, "w") as f:
            for idx in range(4**k):
                kmer = _index_to_kmer(idx, k)
                w = weights_dict.get(kmer, 0.0)
                f.write(f"{kmer}\t{w}\n")

    def test_load_basic(self, tmp_path):
        k = 3
        rng = random.Random(10)
        weights_dict = {_index_to_kmer(i, k): rng.gauss(0, 1) for i in range(4**k)}
        path = tmp_path / "weights.txt"
        self._write_weight_file(path, k, weights_dict)

        model = load_deltasvm_weights(path, l=5)
        assert model.k == 3
        assert model.l == 5
        assert model.weights.shape == (64,)

        for kmer, w in weights_dict.items():
            idx = _kmer_to_index(kmer)
            assert abs(model.weights[idx].item() - w) < 1e-6

    def test_load_gzip(self, tmp_path):
        import gzip as gz

        k = 3
        rng = random.Random(11)
        path = tmp_path / "weights.txt.gz"
        with gz.open(path, "wt") as f:
            for idx in range(4**k):
                kmer = _index_to_kmer(idx, k)
                f.write(f"{kmer}\t{rng.gauss(0, 1)}\n")

        model = load_deltasvm_weights(path, l=5)
        assert model.k == 3

    def test_load_sparse(self, tmp_path):
        """File with only a subset of k-mers; rest should be 0."""
        path = tmp_path / "sparse.txt"
        with open(path, "w") as f:
            f.write("ACG\t1.5\n")
            f.write("TGA\t-0.3\n")

        model = load_deltasvm_weights(path, l=5)
        assert model.weights[_kmer_to_index("ACG")].item() == pytest.approx(1.5)
        assert model.weights[_kmer_to_index("TGA")].item() == pytest.approx(-0.3)
        assert model.num_kmers == 2

    def test_load_with_comments(self, tmp_path):
        path = tmp_path / "commented.txt"
        with open(path, "w") as f:
            f.write("# header comment\n")
            f.write("ACG\t1.0\n")
            f.write("\n")
            f.write("TGA\t2.0\n")

        model = load_deltasvm_weights(path, l=5)
        assert model.num_kmers == 2

    def test_invalid_base(self, tmp_path):
        path = tmp_path / "bad.txt"
        with open(path, "w") as f:
            f.write("ACN\t1.0\n")

        with pytest.raises(ValueError, match="invalid bases"):
            load_deltasvm_weights(path, l=5)

    def test_inconsistent_kmer_length(self, tmp_path):
        path = tmp_path / "bad.txt"
        with open(path, "w") as f:
            f.write("ACG\t1.0\n")
            f.write("ACGT\t2.0\n")

        with pytest.raises(ValueError, match="k-mer length"):
            load_deltasvm_weights(path, l=5)

    def test_l_less_than_k(self, tmp_path):
        path = tmp_path / "weights.txt"
        with open(path, "w") as f:
            f.write("ACGTACG\t1.0\n")

        with pytest.raises(ValueError, match="l.*must be >= k"):
            load_deltasvm_weights(path, l=3)

    def test_end_to_end(self, tmp_path):
        """Load weights, score a sequence, compare to naive."""
        k, l = 3, 5
        rng = random.Random(12)
        weights_dict = {_index_to_kmer(i, k): rng.gauss(0, 1) for i in range(4**k)}
        path = tmp_path / "weights.txt"
        self._write_weight_file(path, k, weights_dict)

        model = load_deltasvm_weights(path, l=l, include_rc=True)
        seq = _make_seqs(1, 20, seed=13)[0]
        x = one_hot_encode(seq).unsqueeze(0)

        got = model(x).item()
        expected = _naive_deltasvm_score(seq, weights_dict, l, k, True)
        assert abs(got - expected) < 1e-4


class TestDeltaSVMDevice:
    def test_to_device(self):
        model = DeltaSVM(torch.randn(64), l=5, k=3)
        x = one_hot_encode(_make_seqs(1, 15)[0]).unsqueeze(0)
        score_cpu = model(x)

        if torch.backends.mps.is_available():
            model_mps = model.to("mps")
            score_mps = model_mps(x.to("mps"))
            assert torch.allclose(
                score_cpu, score_mps.cpu(), atol=1e-5
            )

    def test_state_dict_roundtrip(self):
        model = DeltaSVM(torch.randn(64), l=5, k=3, bias=0.5)
        sd = model.state_dict()
        model2 = DeltaSVM(torch.zeros(64), l=5, k=3, bias=0.0)
        model2.load_state_dict(sd)
        x = one_hot_encode(_make_seqs(1, 15)[0]).unsqueeze(0)
        assert torch.allclose(model(x), model2(x))
