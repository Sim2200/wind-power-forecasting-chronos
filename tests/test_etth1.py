from pathlib import Path

import numpy as np
import pytest

from wpf.data import etth1

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "data" / "etth1" / "ETTh1.csv"
SEQ_LEN = 336


def test_split_sizes_match_official_code(synthetic_etth1):
    s = etth1.make_splits(synthetic_etth1(), SEQ_LEN)
    assert (s.train.shape, s.val.shape, s.test.shape) == ((8640, 7), (2880 + SEQ_LEN, 7), (2880 + SEQ_LEN, 7))


def test_scaler_fit_on_train_only(synthetic_etth1):
    data = synthetic_etth1()
    s = etth1.make_splits(data, SEQ_LEN)
    expected_test = (data[etth1.VAL_END - SEQ_LEN : etth1.TEST_END] - s.scaler.mean) / s.scaler.std
    # train rows are standardised; test rows use the same statistics rather than their own
    assert np.allclose(s.train.mean(axis=0), 0, atol=1e-3)
    assert np.allclose(s.train.std(axis=0), 1, atol=1e-3)
    assert np.allclose(s.test, expected_test, atol=1e-5)
    assert np.allclose(s.scaler.inverse(s.train), data[: etth1.TRAIN_END], atol=1e-3)


def test_first_val_target_starts_at_split_boundary(synthetic_etth1):
    data = synthetic_etth1()
    pred_len = 96
    s = etth1.make_splits(data, SEQ_LEN)
    _, y = etth1.make_windows(s.val, SEQ_LEN, pred_len)
    first_target = s.scaler.inverse(y[0])
    assert np.allclose(first_target, data[etth1.TRAIN_END : etth1.TRAIN_END + pred_len], atol=1e-3)


@pytest.mark.parametrize("pred_len", [96, 192, 336, 720])
def test_window_shapes_and_count(synthetic_etth1, pred_len):
    rows = 1200
    series = synthetic_etth1(rows=rows)
    n = rows - SEQ_LEN - pred_len + 1
    last = n - 1
    x, y = etth1.make_windows(series, SEQ_LEN, pred_len)
    assert (x.shape, y.shape) == ((n, SEQ_LEN, 7), (n, pred_len, 7))
    assert np.array_equal(x[last], series[last : last + SEQ_LEN])
    assert np.array_equal(y[last], series[last + SEQ_LEN : last + SEQ_LEN + pred_len])


def test_window_dataset_len(synthetic_etth1):
    rows, pred_len = 500, 96
    ds = etth1.WindowDataset(synthetic_etth1(rows=rows), SEQ_LEN, pred_len)
    assert len(ds) == rows - SEQ_LEN - pred_len + 1


@pytest.mark.skipif(not CSV.exists(), reason="run scripts/download_etth1.sh first")
def test_real_file_shape():
    data = etth1.load_raw(CSV)
    s = etth1.make_splits(data, SEQ_LEN)
    assert data.shape == (17420, 7)
    assert s.train.shape[0] == etth1.TRAIN_END
