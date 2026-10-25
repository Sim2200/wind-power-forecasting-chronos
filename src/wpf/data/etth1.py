"""ETTh1 loading, splitting, scaling and windowing.

Follows the data pipeline of the official PatchTST code (`Dataset_ETT_hour`):
- all 7 columns are used (features='M'); the date column is dropped
- 12/4/4 "months" of 30 days: train rows [0, 8640), val [8640, 11520),
  test [11520, 14400); the remaining rows of the file are unused
- a StandardScaler is fit on the train rows only and applied everywhere
- val and test slices start seq_len rows early, so the first window of each
  split draws its history from the previous split and its target starts
  exactly at the split boundary
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_LOG = logging.getLogger(__name__)

HOURS_PER_MONTH = 30 * 24
TRAIN_END = 12 * HOURS_PER_MONTH  # 8640
VAL_END = 16 * HOURS_PER_MONTH  # 11520
TEST_END = 20 * HOURS_PER_MONTH  # 14400
ETTH1_BORDERS = (TRAIN_END, VAL_END, TEST_END)
COLUMNS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


@dataclass
class ScalerStats:
    mean: np.ndarray  # shape (n_vars,)
    std: np.ndarray  # shape (n_vars,)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


@dataclass
class Splits:
    """Scaled train/val/test arrays; val and test carry `seq_len` rows of lookback at the front."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    scaler: ScalerStats


def load_raw(csv_path: str | Path) -> np.ndarray:
    """Read 'ETTh1.csv' and return the 7 feature columns as float32, (T, 7)."""
    _LOG.debug("csv_path='%s'", csv_path)
    df = pd.read_csv(csv_path)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns {missing}")
    return df[COLUMNS].to_numpy(dtype=np.float32)


def fit_scaler(train: np.ndarray) -> ScalerStats:
    """Per-column mean and std over the train rows (population std, as sklearn's `StandardScaler`)."""
    return ScalerStats(mean=train.mean(axis=0), std=train.std(axis=0))


def make_splits(data: np.ndarray, seq_len: int, borders: tuple[int, int, int] = ETTH1_BORDERS) -> Splits:
    """Chronological split at the row `borders` (train end, val end, test end) with train-only scaling.

    `data` is a (T, n_vars) array. Val and test include the `seq_len` rows before
    their boundary so that windowing can start right at it.
    """
    _LOG.debug("data.shape=%s seq_len=%s borders=%s", data.shape, seq_len, borders)
    train_end, val_end, test_end = borders
    if data.shape[0] < test_end:
        raise ValueError(f"need at least {test_end} rows, got {data.shape[0]}")
    # Fit on the train rows only, so no validation or test statistics leak into training.
    scaler = fit_scaler(data[:train_end])
    scaled = scaler.transform(data)
    # Val and test start seq_len rows early: their first window takes its history from the previous split.
    return Splits(
        train=scaled[0:train_end],
        val=scaled[train_end - seq_len : val_end],
        test=scaled[val_end - seq_len : test_end],
        scaler=scaler,
    )


def make_windows(series: np.ndarray, seq_len: int, pred_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows with stride 1.

    Returns x of shape (n, seq_len, n_vars) and y of shape (n, pred_len, n_vars),
    where x[i] = series[i : i + seq_len] and y[i] = series[i + seq_len : i + seq_len + pred_len].
    The windows are views (no copy) via stride tricks.
    """
    _LOG.debug("series.shape=%s seq_len=%s pred_len=%s", series.shape, seq_len, pred_len)
    if series.shape[0] < seq_len + pred_len:
        raise ValueError("series too short for one window")
    all_windows = np.lib.stride_tricks.sliding_window_view(series, seq_len + pred_len, axis=0)
    # sliding_window_view puts the window axis last: (n, n_vars, seq_len + pred_len)
    all_windows = np.moveaxis(all_windows, -1, 1)
    return all_windows[:, :seq_len], all_windows[:, seq_len:]


class WindowDataset:
    """Minimal torch-style dataset over `make_windows` output.

    torch is imported lazily so the numpy-only parts of this module (and their
    tests) work on machines without it.
    """

    def __init__(self, series: np.ndarray, seq_len: int, pred_len: int):
        self.x, self.y = make_windows(series, seq_len, pred_len)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int):
        import torch

        return torch.from_numpy(np.ascontiguousarray(self.x[i])), torch.from_numpy(np.ascontiguousarray(self.y[i]))
