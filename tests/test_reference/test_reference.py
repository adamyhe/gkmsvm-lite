"""Cross-validation tests: NumPy/Numba reference vs PyTorch implementation."""

import gzip
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from gkmsvm.codec import one_hot_encode, reverse_complement
from gkmsvm.importers.lsgkm import load_lsgkm_model
from gkmsvm.kernels.direct import DirectGkmKernel
from gkmsvm.kernels.esttrunc import EstTruncGkmKernel
from gkmsvm.reference.kernels import (
    build_esttrunc_table,
    build_gkm_cnt_table,
    one_hot_encode_np,
    pairwise,
    reverse_complement_np,
    score,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "encode_ENCFF579AOX_10sv.model.txt.gz"


def _random_seq(length, rng):
    return "".join(rng.choice("ACGT") for _ in range(length))


class TestWeightTables:
    def test_gkm_cnt_matches_torch(self):
        for l, k in [(7, 4), (11, 7), (5, 3), (10, 5)]:
            kernel = DirectGkmKernel(l, k, normalize=False, include_rc=False)
            torch_table = kernel._mismatch_table.numpy()
            np_table = build_gkm_cnt_table(l, k)
            np.testing.assert_allclose(np_table, torch_table)

    def test_esttrunc_matches_torch(self):
        for l, k, d in [(11, 7, 3), (7, 4, 2), (9, 5, 4)]:
            kernel = EstTruncGkmKernel(l, k, d=d, normalize=False, include_rc=False)
            torch_table = kernel._mismatch_table.numpy()
            np_table = build_esttrunc_table(l, k, d, truncate=True)
            np.testing.assert_allclose(np_table, torch_table)

    def test_estfull_matches_torch(self):
        kernel = EstTruncGkmKernel(11, 7, d=3, normalize=False, include_rc=False, truncate=False)
        torch_table = kernel._mismatch_table.numpy()
        np_table = build_esttrunc_table(11, 7, d=3, truncate=False)
        np.testing.assert_allclose(np_table, torch_table)


class TestCodec:
    def test_one_hot_matches_torch(self):
        seq = "ACGTACGTAATTCCGG"
        np_enc = one_hot_encode_np(seq)
        torch_enc = one_hot_encode(seq).numpy()
        np.testing.assert_array_equal(np_enc, torch_enc)

    def test_rc_matches_torch(self):
        seq = "ACGTACGTAATTCCGG"
        np_enc = one_hot_encode_np(seq)
        np_rc = reverse_complement_np(np_enc)
        torch_rc = reverse_complement(one_hot_encode(seq)).numpy()
        np.testing.assert_array_equal(np_rc, torch_rc)


class TestKernelCrossValidation:
    """Verify reference kernel values match PyTorch for both -t 0 and -t 2."""

    @pytest.fixture
    def seqs(self):
        rng = random.Random(123)
        return [_random_seq(30, rng) for _ in range(4)]

    def _to_np(self, seqs):
        return np.stack([one_hot_encode_np(s) for s in seqs])

    def _to_torch(self, seqs):
        return torch.stack([one_hot_encode(s) for s in seqs])

    def test_gkm_cnt_no_rc_no_norm(self, seqs):
        l, k = 7, 4
        X_np, Y_np = self._to_np(seqs[:2]), self._to_np(seqs[2:])
        X_t, Y_t = self._to_torch(seqs[:2]), self._to_torch(seqs[2:])
        table = build_gkm_cnt_table(l, k)
        K_np = pairwise(X_np, Y_np, l, table, include_rc=False, normalize=False)
        kernel = DirectGkmKernel(l, k, normalize=False, include_rc=False)
        K_t = kernel.pairwise(X_t, Y_t).numpy()
        np.testing.assert_allclose(K_np, K_t, rtol=1e-10)

    def test_gkm_cnt_with_rc_normalized(self, seqs):
        l, k = 7, 4
        X_np, Y_np = self._to_np(seqs[:2]), self._to_np(seqs[2:])
        X_t, Y_t = self._to_torch(seqs[:2]), self._to_torch(seqs[2:])
        table = build_gkm_cnt_table(l, k)
        K_np = pairwise(X_np, Y_np, l, table, include_rc=True, normalize=True)
        kernel = DirectGkmKernel(l, k, normalize=True, include_rc=True)
        K_t = kernel.pairwise(X_t, Y_t).numpy()
        np.testing.assert_allclose(K_np, K_t, rtol=1e-6)

    def test_esttrunc_no_rc_no_norm(self, seqs):
        l, k, d = 11, 7, 3
        X_np, Y_np = self._to_np(seqs[:2]), self._to_np(seqs[2:])
        X_t, Y_t = self._to_torch(seqs[:2]), self._to_torch(seqs[2:])
        table = build_esttrunc_table(l, k, d, truncate=True)
        K_np = pairwise(X_np, Y_np, l, table, include_rc=False, normalize=False)
        kernel = EstTruncGkmKernel(l, k, d=d, normalize=False, include_rc=False)
        K_t = kernel.pairwise(X_t, Y_t).numpy()
        np.testing.assert_allclose(K_np, K_t, rtol=1e-6)

    def test_esttrunc_with_rc_normalized(self, seqs):
        l, k, d = 11, 7, 3
        X_np, Y_np = self._to_np(seqs[:2]), self._to_np(seqs[2:])
        X_t, Y_t = self._to_torch(seqs[:2]), self._to_torch(seqs[2:])
        table = build_esttrunc_table(l, k, d, truncate=True)
        K_np = pairwise(X_np, Y_np, l, table, include_rc=True, normalize=True)
        kernel = EstTruncGkmKernel(l, k, d=d, normalize=True, include_rc=True)
        K_t = kernel.pairwise(X_t, Y_t).numpy()
        np.testing.assert_allclose(K_np, K_t, rtol=1e-6)

    def test_variable_length_sequences(self, seqs):
        rng = random.Random(456)
        short = _random_seq(20, rng)
        long = _random_seq(50, rng)
        l, k = 7, 4
        X_np = np.stack([one_hot_encode_np(short)])
        Y_np = np.stack([one_hot_encode_np(long)])
        X_t = one_hot_encode(short).unsqueeze(0)
        Y_t = one_hot_encode(long).unsqueeze(0)
        table = build_gkm_cnt_table(l, k)
        K_np = pairwise(X_np, Y_np, l, table, include_rc=True, normalize=True)
        kernel = DirectGkmKernel(l, k, normalize=True, include_rc=True)
        K_t = kernel.pairwise(X_t, Y_t).numpy()
        np.testing.assert_allclose(K_np, K_t, rtol=1e-6)


class TestSVMScoring:
    """Verify reference SVM scoring matches PyTorch GkmSVM."""

    def test_score_matches_torch_gkm_cnt(self):
        rng = random.Random(789)
        l, k = 7, 4
        sv_seqs = [_random_seq(30, rng) for _ in range(5)]
        query_seqs = [_random_seq(30, rng) for _ in range(3)]
        coefs = np.array([0.5, -0.3, 0.8, -0.2, 0.1], dtype=np.float64)
        bias = -0.15

        X_np = np.stack([one_hot_encode_np(s) for s in query_seqs])
        SV_np = np.stack([one_hot_encode_np(s) for s in sv_seqs])
        table = build_gkm_cnt_table(l, k)
        scores_np = score(X_np, SV_np, coefs, bias, l, table)

        from gkmsvm.svm import GkmSVM

        sv_t = torch.from_numpy(SV_np).float()
        coefs_t = torch.from_numpy(coefs).float()
        model = GkmSVM(sv_t, coefs_t, bias, "gkm_cnt", {"L": l, "k": k, "include_rc": True})
        X_t = torch.from_numpy(X_np).float()
        scores_t = model(X_t).squeeze(1).detach().numpy()

        np.testing.assert_allclose(scores_np, scores_t, atol=1e-4)

    def test_score_matches_torch_esttrunc(self):
        rng = random.Random(101)
        l, k, d = 11, 7, 3
        sv_seqs = [_random_seq(30, rng) for _ in range(5)]
        query_seqs = [_random_seq(30, rng) for _ in range(3)]
        coefs = np.array([0.5, -0.3, 0.8, -0.2, 0.1], dtype=np.float64)
        bias = -0.15

        X_np = np.stack([one_hot_encode_np(s) for s in query_seqs])
        SV_np = np.stack([one_hot_encode_np(s) for s in sv_seqs])
        table = build_esttrunc_table(l, k, d)
        scores_np = score(X_np, SV_np, coefs, bias, l, table)

        from gkmsvm.svm import GkmSVM

        sv_t = torch.from_numpy(SV_np).float()
        coefs_t = torch.from_numpy(coefs).float()
        model = GkmSVM(
            sv_t, coefs_t, bias, "gkm_esttrunc",
            {"L": l, "k": k, "d": d, "include_rc": True},
        )
        X_t = torch.from_numpy(X_np).float()
        scores_t = model(X_t).squeeze(1).detach().numpy()

        np.testing.assert_allclose(scores_np, scores_t, atol=1e-4)


class TestEncodeOracle:
    """Verify reference implementation matches gkmpredict oracle scores."""

    ORACLE = {
        "query_0": 0.944171,
        "query_1": 0.922764,
        "query_2": 0.977893,
        "random": -0.123703,
    }

    @pytest.fixture
    def model_data(self):
        model = load_lsgkm_model(FIXTURE)
        sv_np = model.support_sequences.numpy().astype(np.float64)
        coefs_np = model.coefficients.numpy().astype(np.float64)
        bias = model.bias
        return sv_np, coefs_np, bias

    @pytest.fixture
    def queries(self):
        sv_seqs = []
        with gzip.open(FIXTURE, "rt") as f:
            in_sv = False
            for line in f:
                line = line.strip()
                if line.startswith("SV"):
                    in_sv = True
                    continue
                if not in_sv:
                    continue
                sv_seqs.append(line.split()[-1])
                if len(sv_seqs) >= 3:
                    break
        rng = random.Random(42)
        rand_seq = "".join(rng.choice("ACGT") for _ in range(300))
        return {
            "query_0": sv_seqs[0],
            "query_1": sv_seqs[1],
            "query_2": sv_seqs[2],
            "random": rand_seq,
        }

    def test_matches_gkmpredict(self, model_data, queries):
        sv_np, coefs_np, bias = model_data
        l, k, d = 11, 7, 3
        table = build_esttrunc_table(l, k, d)

        for name, seq in queries.items():
            x = one_hot_encode_np(seq)[np.newaxis]
            s = score(x, sv_np, coefs_np, bias, l, table)
            expected = self.ORACLE[name]
            assert s[0] == pytest.approx(expected, abs=1e-4), (
                f"{name}: expected {expected}, got {s[0]}"
            )
