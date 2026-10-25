import numpy as np
import pytest

from wpf.data import etth1


@pytest.fixture
def synthetic_etth1():
    """Factory for an ETTh1-shaped float32 array: 7 channels with distinct means and scales."""

    def make(rows: int = etth1.TEST_END + 100, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        loc = np.arange(1, 8)
        return rng.normal(loc=loc, scale=loc, size=(rows, 7)).astype(np.float32)

    return make
