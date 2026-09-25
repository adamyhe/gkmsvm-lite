import pytest
import torch

from gkmsvm.codec import (
    encode_batch,
    one_hot_decode,
    one_hot_encode,
    reverse_complement,
    validate,
)


class TestOneHotEncode:
    def test_basic(self):
        t = one_hot_encode("ACGT")
        assert t.shape == (4, 4)
        assert t.dtype == torch.float32
        expected = torch.eye(4, dtype=torch.float32)
        assert torch.equal(t, expected)

    def test_single_base(self):
        for i, base in enumerate("ACGT"):
            t = one_hot_encode(base)
            assert t.shape == (4, 1)
            assert t[i, 0] == 1.0
            assert t.sum() == 1.0

    def test_case_insensitive(self):
        upper = one_hot_encode("ACGT")
        lower = one_hot_encode("acgt")
        mixed = one_hot_encode("AcGt")
        assert torch.equal(upper, lower)
        assert torch.equal(upper, mixed)

    def test_invalid_base_raises(self):
        with pytest.raises(ValueError, match="Invalid base 'N'"):
            one_hot_encode("ACNGT")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            one_hot_encode("")

    def test_dtype(self):
        t = one_hot_encode("ACGT", dtype=torch.float64)
        assert t.dtype == torch.float64


class TestOneHotDecode:
    def test_roundtrip(self):
        for seq in ["ACGT", "AAAA", "TGCA", "GCGCGCGC"]:
            assert one_hot_decode(one_hot_encode(seq)) == seq

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            one_hot_decode(torch.zeros(3, 4))

    def test_3d_raises(self):
        with pytest.raises(ValueError, match="shape"):
            one_hot_decode(torch.zeros(2, 4, 4))


class TestReverseComplement:
    def test_basic(self):
        seq = "ACGT"
        t = one_hot_encode(seq)
        rc_t = reverse_complement(t)
        rc_seq = one_hot_decode(rc_t)
        assert rc_seq == "ACGT"  # ACGT is its own RC

    def test_asymmetric(self):
        seq = "AAAC"
        t = one_hot_encode(seq)
        rc_t = reverse_complement(t)
        rc_seq = one_hot_decode(rc_t)
        assert rc_seq == "GTTT"

    def test_double_rc_is_identity(self):
        seq = "ACGTACGT"
        t = one_hot_encode(seq)
        assert torch.equal(reverse_complement(reverse_complement(t)), t)

    def test_batch(self):
        batch, _ = encode_batch(["AAAC", "TTTG"])
        rc = reverse_complement(batch)
        assert rc.shape == batch.shape
        assert one_hot_decode(rc[0]) == "GTTT"
        assert one_hot_decode(rc[1]) == "CAAA"

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="2D or 3D"):
            reverse_complement(torch.zeros(4))

    def test_wrong_channels_raises(self):
        with pytest.raises(ValueError, match="Channel dimension must be 4"):
            reverse_complement(torch.zeros(3, 10))


class TestValidate:
    def test_valid(self):
        t = one_hot_encode("ACGT")
        validate(t)  # should not raise

    def test_valid_batch(self):
        batch, _ = encode_batch(["ACGT", "TGCA"])
        validate(batch[:, :, :4])  # unpadded portion

    def test_wrong_channels(self):
        with pytest.raises(ValueError, match="Channel dimension"):
            validate(torch.zeros(3, 10))

    def test_not_one_hot(self):
        t = torch.zeros(4, 4)
        with pytest.raises(ValueError):
            validate(t)

    def test_multi_hot(self):
        t = one_hot_encode("ACGT")
        t[1, 0] = 1.0  # make position 0 multi-hot
        with pytest.raises(ValueError):
            validate(t)


class TestEncodeBatch:
    def test_same_length(self):
        batch, mask = encode_batch(["ACGT", "TGCA"])
        assert batch.shape == (2, 4, 4)
        assert mask.shape == (2, 4)
        assert mask.all()

    def test_variable_length(self):
        batch, mask = encode_batch(["AC", "ACGT"])
        assert batch.shape == (2, 4, 4)
        assert mask[0].tolist() == [True, True, False, False]
        assert mask[1].tolist() == [True, True, True, True]
        assert one_hot_decode(batch[1]) == "ACGT"

    def test_padding_value(self):
        batch, mask = encode_batch(["AC", "ACGT"])
        padded = batch[0, :, 2:]  # positions 2-3 of first sequence
        assert (padded == 0.0).all()

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            encode_batch([])

    def test_single_sequence(self):
        batch, mask = encode_batch(["ACGT"])
        assert batch.shape == (1, 4, 4)
        assert mask.all()
