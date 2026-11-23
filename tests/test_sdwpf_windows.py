import numpy as np
import pytest

from wpf.data import sdwpf_windows as W
from wpf.data import splits


@pytest.fixture
def tiny():
    """6 / 1 / 2 days: train [0, 864), val [864, 1008), test [1008, 1296)."""
    return splits.make_split((6, 1, 2))


def test_fill_missing_carries_last_reading_forward_and_zeroes_a_leading_gap():
    dense = np.array([[np.nan, np.nan, 3.0, np.nan, 5.0, np.nan], [1.0, np.nan, np.nan, 4.0, 4.0, 4.0]])
    filled = W.fill_missing(dense)
    np.testing.assert_array_equal(filled, [[0.0, 0.0, 3.0, 3.0, 5.0, 5.0], [1.0, 1.0, 1.0, 4.0, 4.0, 4.0]])
    assert np.isnan(dense[0, 0])  # input untouched


def test_scaler_uses_valid_readings_of_the_range_only():
    dense = np.array([[1.0, 100.0, 3.0, 50.0], [5.0, 7.0, np.nan, 9.0]])
    valid = np.array([[True, False, True, True], [True, True, False, False]])
    scaler = W.fit_scaler(dense, valid, (0, 3))  # valid in [0, 3): 1, 3, 5, 7
    assert scaler.mean == 4.0 and scaler.std == pytest.approx(np.sqrt(5.0))
    np.testing.assert_allclose(scaler.inverse(scaler.transform(dense[0])), dense[0])
    with pytest.raises(ValueError, match="no valid"):
        W.fit_scaler(dense, np.zeros_like(valid), (0, 4))


def test_scaler_constant_series_gets_unit_std():
    assert W.fit_scaler(np.full((1, 5), 2.0), np.ones((1, 5), bool), (0, 5)).std == 1.0


def test_train_index_keeps_only_windows_with_a_valid_target_step(tiny):
    horizon, stride = 4, 8
    valid = np.ones((2, tiny.n_steps), bool)
    origins = splits.train_origins(tiny, 1.0, horizon, stride)
    valid[1, origins[0] : origins[0] + horizon] = False  # turbine 1, first window: no valid target
    valid[0, origins[1] : origins[1] + horizon - 1] = False  # turbine 0, second window: one valid step left
    index = W.train_index(valid, tiny, 1.0, horizon, stride)
    assert index.shape == (2 * len(origins) - 1, 2)
    assert not ((index[:, 0] == 1) & (index[:, 1] == origins[0])).any()
    assert ((index[:, 0] == 0) & (index[:, 1] == origins[1])).any()
    assert index[:, 1].min() >= splits.LOOKBACK and index[:, 1].max() + horizon <= tiny.train[1]


def test_train_index_respects_subset_and_stride(tiny):
    valid = np.ones((1, tiny.n_steps), bool)
    start, end = splits.subset_range(tiny, 0.5)
    index = W.train_index(valid, tiny, 0.5, 4, stride=3)
    assert index[:, 1].min() == start + splits.LOOKBACK and index[:, 1].max() + 4 <= end
    assert set(np.diff(index[:, 1])) == {3}


def test_dataset_items_match_the_series_and_the_mask(tiny):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    series = rng.normal(size=(2, tiny.n_steps)).astype(np.float32)
    valid = rng.random(series.shape) > 0.3
    horizon = 5
    index = W.train_index(valid, tiny, 1.0, horizon, stride=50)
    ds = W.SDWPFWindows(series, valid, index, horizon)
    assert len(ds) == len(index)
    turbine, origin = index[3]
    x, y, mask = ds[3]
    assert x.shape == (splits.LOOKBACK, 1) and y.shape == mask.shape == (horizon, 1)
    assert x.dtype == y.dtype == mask.dtype == torch.float32
    np.testing.assert_array_equal(x[:, 0].numpy(), series[turbine, origin - splits.LOOKBACK : origin])
    np.testing.assert_array_equal(y[:, 0].numpy(), series[turbine, origin : origin + horizon])
    np.testing.assert_array_equal(mask[:, 0].numpy(), valid[turbine, origin : origin + horizon].astype(np.float32))


def test_dataset_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        W.SDWPFWindows(np.zeros((1, 10)), np.ones((2, 10), bool), np.zeros((0, 2), int), 2)


def test_eval_arrays_use_the_shared_origins(tiny):
    series = np.arange(2 * tiny.n_steps, dtype=np.float32).reshape(2, -1)
    valid = np.ones(series.shape, bool)
    valid[1, 868] = False
    arrays = W.eval_arrays(series, valid, tiny, "val", 8)
    origins = splits.eval_origins(tiny, "val")
    np.testing.assert_array_equal(arrays["origins"], origins)
    assert arrays["x"].shape == (2, len(origins), splits.LOOKBACK)
    assert arrays["y"].shape == arrays["mask"].shape == (2, len(origins), 8)
    assert arrays["x"][0, 0, -1] == origins[0] - 1 and arrays["y"][1, 0, 0] == tiny.n_steps + origins[0]
    assert not arrays["mask"][1, 0, 868 - origins[0]]
