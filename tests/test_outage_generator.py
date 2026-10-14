import numpy as np
import pytest

from dasr_har.outages.generator import generate_mask


@pytest.mark.parametrize("pattern", ["point", "single_burst", "multi_burst", "periodic"])
@pytest.mark.parametrize("rate", [0.1, 0.3, 0.5])
def test_exact_budget(pattern, rate):
    length = 250
    mask = generate_mask(length, rate, pattern, np.random.default_rng(3407))
    assert int((~mask).sum()) == int(np.floor(length * rate))


def test_seed_reproducibility():
    first = generate_mask(250, 0.3, "multi_burst", np.random.default_rng(2026))
    second = generate_mask(250, 0.3, "multi_burst", np.random.default_rng(2026))
    assert np.array_equal(first, second)

