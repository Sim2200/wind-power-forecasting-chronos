"""Persistence and seasonal-naive baselines on the shared SDWPF evaluation windows (DS-05).

Both forecasters return (turbines, origins, H) kW arrays in the layout `metrics.evaluate` scores, for the origins of
`splits.eval_origins`. Inputs come from `metrics.eval_inputs`, scoring from `metrics.evaluate` and writing from
`metrics.save_results`; nothing about windows or metrics is defined here.

Technical decisions (also listed in the DS-05 pull request):
- Persistence repeats the last VALID reading of the 288-step history (valid = not missing, unknown or abnormal). A
  window with no valid reading in its history predicts 0 (the turbine is treated as offline) and is counted, because
  `evaluate` raises on a non-finite forecast. This differs from the DS-04 smoke run (`metrics.persistence_forecast`),
  which repeated the last reading after a last-value fill, flagged readings included; the smoke run stays as it was.
- Seasonal naive forecasts step t + h with the stored reading at t + h - 144 (one day earlier; m = `metrics.SEASON`).
  That index lies inside the history (h < 144), so no window needs extra data. Flagged readings are used as stored,
  consistent with the decision that flagged rows stay in inputs. A missing reading (Patv NaN) has no stored value, so
  the series is last-value filled first (`metrics.last_value_fill`, the fill PatchTST and LSTM see; 0 before a
  turbine's first reading). The filled readings on valid targets are counted per horizon.
- Counts go next to the metrics in `results/baseline_counts_<period>.json`; the four result files keep the shared
  layout (string horizon keys only) so EX-03 reads them like every other model's.
- No tuning is involved, so the test period is scored in the same run as val; no setting depends on either.

Usage:
    python -m wpf.baselines --period both --out results/
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from wpf import metrics
from wpf.data import splits

_LOG = logging.getLogger(__name__)

NAMES = ("persistence", "seasonal_naive")


def last_valid_history(dense: np.ndarray, valid: np.ndarray, origins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Last valid reading in the history window of every (turbine, origin), and where there is none.

    `dense` and `valid` are (turbines, steps). Returns (value, no_valid), both (turbines, origins); `value` is 0
    where `no_valid` is True.
    """
    _LOG.debug("dense.shape=%s n_origins=%d", dense.shape, len(origins))
    x, _ = splits.windows_at(dense, origins, 1)
    v, _ = splits.windows_at(valid, origins, 1)
    has_valid = v.any(axis=2)
    last = splits.LOOKBACK - 1 - np.argmax(v[..., ::-1], axis=2)  # index of the latest valid step in the window
    value = np.take_along_axis(x, last[..., None], axis=2)[..., 0].astype(np.float64)
    return np.where(has_valid, value, 0.0), ~has_valid


def persistence(dense: np.ndarray, valid: np.ndarray, origins: np.ndarray, horizon: int = max(splits.HORIZONS)) -> tuple[np.ndarray, np.ndarray]:
    """(pred, no_valid): the last valid reading repeated over `horizon` steps, and the windows that had none."""
    _LOG.debug("horizon=%d", horizon)
    value, no_valid = last_valid_history(dense, valid, origins)
    return np.repeat(value[..., None], horizon, axis=2), no_valid


def seasonal_naive(dense: np.ndarray, origins: np.ndarray, horizon: int = max(splits.HORIZONS), m: int = metrics.SEASON) -> tuple[np.ndarray, np.ndarray]:
    """(pred, source_missing): the reading one season (m steps) before each target step.

    `source_missing` is (turbines, origins, horizon), True where that reading was NaN and the filled value was used.
    """
    _LOG.debug("horizon=%d m=%d", horizon, m)
    if not 0 < horizon <= m <= splits.LOOKBACK:
        raise ValueError("need 0 < horizon <= m <= LOOKBACK so every source reading lies in the history")
    origins = np.asarray(origins)
    if origins.min() < splits.LOOKBACK:
        raise ValueError("an origin's history falls outside the series")
    source = origins[:, None] + np.arange(horizon) - m  # (origins, horizon), all before the origin
    dense = dense.astype(np.float64)
    return metrics.last_value_fill(dense)[:, source], np.isnan(dense[:, source])


def run(dense: np.ndarray, valid: np.ndarray, split: splits.Split, period: str) -> tuple[dict[str, dict], dict[str, dict]]:
    """({name: evaluate output}, {name: counts}) of both baselines on the shared windows of `period`."""
    _LOG.debug("period='%s'", period)
    y, v, scale = metrics.eval_inputs(dense, valid, split, period)
    origins = splits.eval_origins(split, period)
    horizon = y.shape[-1]
    pred_p, no_valid = persistence(dense, valid, origins, horizon)
    pred_s, missing = seasonal_naive(dense, origins, horizon)
    results = {
        "persistence": metrics.evaluate(y, v, scale, pred=pred_p),
        "seasonal_naive": metrics.evaluate(y, v, scale, pred=pred_s),
    }
    counts = {
        "persistence": {"windows": int(no_valid.size), "no_valid_history": int(no_valid.sum())},
        "seasonal_naive": {
            "windows": int(no_valid.size),
            "filled_source_on_valid_targets": {str(h): int((missing[..., :h] & v[..., :h]).sum()) for h in splits.HORIZONS},
            "valid_targets": {str(h): int(v[..., :h].sum()) for h in splits.HORIZONS},
        },
    }
    return results, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the persistence and seasonal-naive baselines on the shared SDWPF windows.")
    parser.add_argument("--period", default="both", choices=("val", "test", "both"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--clean", type=Path, default=Path("data/sdwpf/clean.parquet"), help="DS-01 parquet")
    parser.add_argument("--splits", type=Path, default=Path("results/ds02_splits.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    from wpf.data import sdwpf  # lazy: pandas/pyarrow are only needed here

    frame = sdwpf.load_clean(args.clean)
    dense, valid = sdwpf.to_dense(frame, "Patv").astype(np.float64), sdwpf.to_dense(frame, "valid").astype(bool)
    split = splits.load_splits(args.splits)
    if dense.shape[1] != split.n_steps:
        raise ValueError(f"{args.clean} has {dense.shape[1]} steps per turbine, the split expects {split.n_steps}")
    for period in ("val", "test") if args.period == "both" else (args.period,):
        results, counts = run(dense, valid, split, period)
        for name in NAMES:
            metrics.save_results(args.out / f"baseline_{name}_{period}.json", results[name])
        (args.out / f"baseline_counts_{period}.json").write_text(json.dumps(counts, indent=1) + "\n", encoding="utf-8")
        _LOG.info("wrote the %s baseline files to %s", period, args.out)
        for name in NAMES:
            print(period, name, {h: round(r["mae"], 2) for h, r in results[name].items()}, "mase", {h: round(r["mase"], 4) for h, r in results[name].items()})
        print(period, "counts", json.dumps(counts))


if __name__ == "__main__":
    main()
