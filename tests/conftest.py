import pytest

from gkmsvm.codec import one_hot_encode


@pytest.fixture
def short_sequences():
    return ["ACGT", "AAAA", "CCCC", "TGCA"]


@pytest.fixture
def short_tensors(short_sequences):
    return [one_hot_encode(s) for s in short_sequences]


@pytest.fixture
def medium_sequence():
    return "ACGTACGTACGT"


@pytest.fixture
def medium_tensor(medium_sequence):
    return one_hot_encode(medium_sequence)
