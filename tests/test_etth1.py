from pathlib import Path

import numpy as np
import pytest

from wpf.data import etth1

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "data" / "etth1" / "ETTh1.csv"


def synthetic(rows: int = etth1.TEST_END + 100, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=[1, 2, 3, 4, 5, 6, 7], scale=[1, 2, 3, 4, 5, 6, 7], size=(rows, 7)).astype(np.float32)


def test_split_sizes_match_official_code():
    s = etth1.make_splits(synthetic(), seq_len=336)
    assert s.train.shape == (8640, 7)
    assert s.val.shape == (2880 + 336, 7)
    assert s.test.shape == (2880 + 336, 7)


def test_scaler_fit_on_train_only():
    data = synthetic()
    s = etth1.make_splits(data, seq_len=336)
    # train rows are standardised
    assert np.allclose(s.train.mean(axis=0), 0, atol=1e-3)
    assert np.allclose(s.train.std(axis=0), 1, atol=1e-3)
    # the same stats are applied to test rows, not refit there
    expected_test = (data[etth1.VAL_END - 336 : etth1.TEST_END] - s.scaler.mean) / s.scaler.std
    assert np.allclose(s.test, expected_test, atol=1e-5)
    assert np.allclose(s.scaler.inverse(s.train), data[: etth1.TRAIN_END], atol=1e-3)


def test_first_val_target_starts_at_split_boundary():
    data = synthetic()
    seq_len, pred_len = 336, 96
    s = etth1.make_splits(data, seq_len=seq_len)
    _, y = etth1.make_windows(s.val, seq_len, pred_len)
    first_target = s.scaler.inverse(y[0])
    assert np.allclose(first_target, data[etth1.TRAIN_END : etth1.TRAIN_END + pred_len], atol=1e-3)


@pytest.mark.parametrize("pred_len", [96, 192, 336, 720])
def test_window_shapes_and_count(pred_len):
    rows, seq_len = 1200, 336
    series = synthetic(rows=rows)
    x, y = etth1.make_windows(series, seq_len, pred_len)
    n = rows - seq_len - pred_len + 1
    assert x.shape == (n, seq_len, 7)
    assert y.shape == (n, pred_len, 7)
    i = n - 1
    assert np.array_equal(x[i], series[i : i + seq_len])
    assert np.array_equal(y[i], series[i + seq_len : i + seq_len + pred_len])


def test_window_dataset_len():
    ds = etth1.WindowDataset(synthetic(rows=500), seq_len=336, pred_len=96)
    assert len(ds) == 500 - 336 - 96 + 1


@pytest.mark.skipif(not CSV.exists(), reason="run scripts/download_etth1.sh first")
def test_real_file_shape():
    data = etth1.load_raw(CSV)
    assert data.shape == (17420, 7)
    s = etth1.make_splits(data, seq_len=336)
    assert s.train.shape[0] == 8640
