"""Shared evaluation metrics for every SDWPF model: MAE, RMSE (kW), MASE and WQL, per horizon.

Inputs are the arrays `wpf.data.splits.windows_at` produces: `y` and `valid` of shape (turbines, origins, H_max)
(H_max = 64), predictions of the same shape (point) or with a leading quantile axis (Q, turbines, origins,
H_max). Every model writes its forecasts in this layout and calls `evaluate`, so all numbers come from one
implementation.

Technical decisions (also listed in the DS-04 pull request):
- Only valid readings are scored (`valid` from DS-01: not missing, unknown or abnormal). Invalid readings are
  dropped from every sum and every denominator; predictions there are ignored.
- "Per horizon" means per forecast task: the metric for H = 6 / 36 / 64 pools the first H steps of every window
  (DS-02 uses one origin set for all horizons, so the three tasks share windows). It is not per step ahead.
- MAE, RMSE and WQL pool all valid readings of all turbines, so a turbine with more valid readings counts more.
- MASE is computed per turbine and then averaged over turbines: MASE_i = MAE_i / scale_i with scale_i = mean
  |y_t - y_{t-m}| over the full train series [0, 24624) (m = 144, one day; `splits.mase_train_range`), using
  only pairs where both readings are valid. The scale is the same for every model and training subset, so MASE
  is comparable across them. Per turbine because turbines differ a lot in power and variability; a turbine with
  no valid reading in the evaluation windows, or with scale 0 or NaN, is left out of the average (the number
  used is returned as `mase_turbines`).
- WQL = mean over q of 2 * sum(pinball_q) / sum(|y|), the GluonTS / Chronos definition, over q = 0.1..0.9.
  Point models (persistence, LSTM) have no WQL and report NaN.
- A prediction that is not finite on a valid reading is an error, not a skipped reading: a model that emits NaN
  must not look better by being scored on fewer readings.
- Empty selections (no valid reading, or sum |y| = 0) give NaN rather than 0.
"""
from __future__ import annotations

import logging

import numpy as np

from wpf.data import splits

_LOG = logging.getLogger(__name__)

SEASON = 144  # m: one day of 10-minute readings
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def mase_scale(train: np.ndarray, train_valid: np.ndarray, m: int = SEASON) -> np.ndarray:
    """Per-turbine seasonal-naive error on the training series: mean |y_t - y_{t-m}| over valid pairs.

    `train` and `train_valid` are (turbines, steps) over `splits.mase_train_range`. NaN where a turbine has no
    valid pair.
    """
    _LOG.debug("train.shape=%s m=%d", train.shape, m)
    if train.shape != train_valid.shape or train.shape[1] <= m:
        raise ValueError("train and train_valid must have the same shape with more than m steps")
    both = train_valid[:, m:] & train_valid[:, :-m]
    diff = np.where(both, np.abs(train[:, m:].astype(np.float64) - train[:, :-m]), 0.0)
    n = both.sum(axis=1)
    return np.where(n > 0, diff.sum(axis=1) / np.maximum(n, 1), np.nan)


def _check_shapes(y: np.ndarray, valid: np.ndarray, pred: np.ndarray) -> None:
    if y.ndim != 3 or valid.shape != y.shape or pred.shape[-3:] != y.shape:
        raise ValueError(f"y, valid and pred must share the shape (turbines, origins, H); got {y.shape}, {valid.shape}, {pred.shape}")


def _check(y: np.ndarray, valid: np.ndarray, pred: np.ndarray) -> None:
    _check_shapes(y, valid, pred)
    if not np.isfinite(y[valid]).all():
        raise ValueError("y has a non-finite value on a valid reading")
    if not np.isfinite(pred[..., valid]).all():
        raise ValueError("pred has a non-finite value on a valid reading")


def point_metrics(pred: np.ndarray, y: np.ndarray, valid: np.ndarray, scale: np.ndarray) -> dict:
    """MAE, RMSE and MASE of a point forecast over the valid readings of (turbines, origins, H) arrays."""
    _LOG.debug("pred.shape=%s", pred.shape)
    _check(y, valid, pred)
    err = np.where(valid, np.abs(pred.astype(np.float64) - np.where(valid, y, 0.0)), 0.0)
    n = valid.sum()
    if n == 0:
        return {"mae": np.nan, "rmse": np.nan, "mase": np.nan, "mase_turbines": 0, "n_valid": 0}
    n_t = valid.sum(axis=(1, 2))
    mae_t = err.sum(axis=(1, 2)) / np.maximum(n_t, 1)
    use = (n_t > 0) & np.isfinite(scale) & (scale > 0)
    return {
        "mae": float(err.sum() / n),
        "rmse": float(np.sqrt((err**2).sum() / n)),
        "mase": float((mae_t[use] / scale[use]).mean()) if use.any() else np.nan,
        "mase_turbines": int(use.sum()),
        "n_valid": int(n),
    }


def wql(quantile_pred: np.ndarray, y: np.ndarray, valid: np.ndarray, quantiles: tuple[float, ...] = QUANTILES) -> float:
    """Weighted quantile loss: mean over q of 2 * sum(pinball_q) / sum(|y|), over valid readings.

    `quantile_pred` is (Q, turbines, origins, H) with Q = len(quantiles).
    """
    _LOG.debug("quantile_pred.shape=%s", quantile_pred.shape)
    if quantile_pred.shape[0] != len(quantiles):
        raise ValueError("quantile_pred needs one leading entry per quantile")
    _check(y, valid, quantile_pred)
    denom = np.abs(y[valid].astype(np.float64)).sum()
    if valid.sum() == 0 or denom == 0:
        return float("nan")
    y_v = y[valid].astype(np.float64)
    total = 0.0
    for q, p in zip(quantiles, quantile_pred):
        gap = y_v - p[valid].astype(np.float64)
        total += 2.0 * np.maximum(q * gap, (q - 1.0) * gap).sum() / denom
    return float(total / len(quantiles))


def evaluate(
    y: np.ndarray,
    valid: np.ndarray,
    scale: np.ndarray,
    pred: np.ndarray | None = None,
    quantile_pred: np.ndarray | None = None,
    horizons: tuple[int, ...] = splits.HORIZONS,
) -> dict[int, dict]:
    """Metrics for each horizon H on the first H steps of every window.

    Give `pred` (point forecast; for a quantile model pass its median) and/or `quantile_pred` (WQL). At least one
    is required; `wql` is NaN without `quantile_pred`. Returns {H: {mae, rmse, mase, mase_turbines, n_valid, wql}}.
    """
    _LOG.debug("y.shape=%s horizons=%s", y.shape, horizons)
    if pred is None and quantile_pred is None:
        raise ValueError("give pred and/or quantile_pred")
    for forecast in (pred, quantile_pred):
        if forecast is not None:
            _check_shapes(y, valid, forecast)
    if max(horizons) > y.shape[-1]:
        raise ValueError("a horizon is longer than the windows")
    out = {}
    for h in horizons:
        y_h, v_h = y[..., :h], valid[..., :h]
        res = {"mae": np.nan, "rmse": np.nan, "mase": np.nan, "mase_turbines": 0, "n_valid": int(v_h.sum())}
        if pred is not None:
            res = point_metrics(pred[..., :h], y_h, v_h, scale)
        res["wql"] = wql(quantile_pred[..., :h], y_h, v_h) if quantile_pred is not None else float("nan")
        out[h] = res
    return out
