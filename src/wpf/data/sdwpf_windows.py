"""SDWPF windows for PatchTST training and evaluation: gap filling, scaling, the training index and the datasets.

Builds on DS-01 (`sdwpf.to_dense` arrays of Patv and `valid`) and DS-02 (`splits`). Everything here is for the
univariate task: one turbine's Patv history of L = 288 steps in, its next H steps out.

Technical decisions (also listed in the PT-01 pull request):
- One model is shared across the 134 turbines. A training sample is one (turbine, origin) window of shape
  (L, 1): under channel independence this is the same computation as stacking turbines as channels, but it
  shuffles across turbines and keeps the paper's batch-size meaning (128 sequences per step).
- Inputs: missing readings (Patv NaN) are filled by carrying the last reading forward within the turbine; a
  gap at the start of a series becomes 0. Unknown and abnormal readings stay as they are (DS-01 decision).
- Targets: every step is kept; `valid` goes along as a mask and the loss and the metrics only count valid
  steps. A training window with no valid target step is left out of the index since it has no gradient.
- Scaling: Patv is divided by one global mean / std fitted on the valid readings of the model's own training
  subset (a 5% model sees only 5% of the data), applied before windowing; RevIN still normalizes each window.
  Predictions are mapped back to kW before anything is saved or scored.
- Training windows follow `splits.train_origins` (fully inside the subset, stride configurable). Evaluation
  windows are the shared origins from `splits.eval_origins`, in the (turbines, origins, H) layout DS-04 scores.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from wpf.data import splits

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Scaler:
    mean: float
    std: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


def fill_missing(dense: np.ndarray) -> np.ndarray:
    """Carry the last reading forward along the step axis of a (turbines, steps) array; leading NaN -> 0."""
    _LOG.debug("dense.shape=%s nan=%d", dense.shape, int(np.isnan(dense).sum()))
    idx = np.where(np.isnan(dense), 0, np.arange(dense.shape[1]))
    np.maximum.accumulate(idx, axis=1, out=idx)
    filled = np.take_along_axis(dense, idx, axis=1)
    return np.where(np.isnan(filled), 0.0, filled)


def fit_scaler(dense: np.ndarray, valid: np.ndarray, steps: tuple[int, int]) -> Scaler:
    """Mean and population std of the valid readings in `steps` (the model's training subset)."""
    _LOG.debug("steps=%s", steps)
    start, end = steps
    values = dense[:, start:end][valid[:, start:end]]
    if values.size == 0:
        raise ValueError("no valid reading in the scaler range")
    std = float(values.std())
    return Scaler(mean=float(values.mean()), std=std if std > 0 else 1.0)


def train_index(valid: np.ndarray, split: splits.Split, fraction: float, horizon: int, stride: int = 1) -> np.ndarray:
    """(turbine, origin) pairs of the training windows with at least one valid target step, shape (n, 2)."""
    _LOG.debug("fraction=%s horizon=%s stride=%s", fraction, horizon, stride)
    origins = splits.train_origins(split, fraction, horizon, stride)
    cumulative = np.concatenate([np.zeros((valid.shape[0], 1), dtype=np.int64), valid.cumsum(axis=1)], axis=1)
    valid_targets = cumulative[:, origins + horizon] - cumulative[:, origins]  # (turbines, origins)
    turbine, position = np.nonzero(valid_targets > 0)
    return np.stack([turbine, origins[position]], axis=1)


class SDWPFWindows:
    """Torch-style dataset of (x, y, mask) for the windows in `index`; indexes lazily, nothing is materialized.

    `series` is the filled, scaled (turbines, steps) float32 array and `valid` the matching bool mask.
    Items are x (L, 1), y (H, 1) and mask (H, 1) as float32; the trailing axis is the single channel.
    """

    def __init__(self, series: np.ndarray, valid: np.ndarray, index: np.ndarray, horizon: int):
        if series.shape != valid.shape:
            raise ValueError("series and valid must have the same shape")
        self.series = series.astype(np.float32, copy=False)
        self.valid = valid
        self.index = index
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        import torch

        turbine, origin = self.index[i]
        x = self.series[turbine, origin - splits.LOOKBACK : origin]
        y = self.series[turbine, origin : origin + self.horizon]
        mask = self.valid[turbine, origin : origin + self.horizon]
        return (
            torch.from_numpy(x[:, None].copy()),
            torch.from_numpy(y[:, None].copy()),
            torch.from_numpy(mask[:, None].astype(np.float32)),
        )


def eval_arrays(series: np.ndarray, valid: np.ndarray, split: splits.Split, period: str, horizon: int) -> dict[str, np.ndarray]:
    """The shared evaluation windows of `period`: x (turbines, origins, L), y and mask (turbines, origins, H)."""
    _LOG.debug("period='%s' horizon=%s", period, horizon)
    origins = splits.eval_origins(split, period)
    x, y = splits.windows_at(series, origins, horizon)
    _, mask = splits.windows_at(valid, origins, horizon)
    return {"x": x.astype(np.float32, copy=False), "y": y.astype(np.float32, copy=False), "mask": mask, "origins": origins}
