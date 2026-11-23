"""EX-03: is the test-split ranking uniform over the test weeks, and are the gaps larger than the origin-to-origin noise?

Reads the saved predictions of the grid (output/ex_<cell>_pred.npz), builds the persistence forecast, and writes
results/ex03_test_breakdown.json with
- `weeks`: MASE per model (mean over the seeds), per test week (7 weeks of 7 days), at H = 6 / 36 / 64;
- `bootstrap`: for the 100% rows, zero-shot and persistence, the MASE on val and test with a 95% interval from 1000
  resamples of the forecast origins (seed 2021 predictions), and the same interval for the difference to zero-shot Chronos,
  paired on the resample.

    python scripts/analysis/test_breakdown.py [--output output] [--out results/ex03_test_breakdown.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from wpf import grid, metrics  # noqa: E402
from wpf.data import sdwpf, splits  # noqa: E402

_LOG = logging.getLogger(__name__)
HORIZONS = (6, 36, 64)
WEEK = 7 * splits.SLOTS_PER_DAY
BOOTSTRAP_MODELS = ("patchtst_100", "patchtst_point_100", "lstm_100", "chronos_ft_100", "chronos_lora_100", "chronos_zeroshot", "persistence")


def forecasts(output: Path, patv: np.ndarray, split: splits.Split, period: str) -> dict[str, list[np.ndarray]]:
    """{model_pct: [point forecast per seed]} for the period, plus persistence."""
    out = {}
    for cell in grid.cells():
        path = output / f"{cell.name}_pred.npz"
        if not path.exists():
            _LOG.warning("missing %s", path)
            continue
        key = cell.model if cell.pct is None else f"{cell.model}_{cell.pct}"
        with np.load(path) as f:
            out.setdefault(key, []).append(f[period].astype(np.float64))
    out["persistence"] = [metrics.persistence_forecast(patv, splits.eval_origins(split, period))]
    return out


def mase_on(pred: np.ndarray, y: np.ndarray, valid: np.ndarray, scale: np.ndarray, idx: np.ndarray, h: int) -> float:
    return metrics.point_metrics(pred[:, idx, :h], y[:, idx, :h], valid[:, idx, :h], scale)["mase"]


def weekly(preds: dict, y: np.ndarray, valid: np.ndarray, scale: np.ndarray, origins: np.ndarray) -> dict:
    week = (origins - origins[0]) // WEEK
    out = {}
    for key, runs in preds.items():
        out[key] = {}
        for h in HORIZONS:
            per_week = []
            for w in range(int(week.max()) + 1):
                idx = np.flatnonzero(week == w)
                per_week.append(round(float(np.mean([mase_on(p, y, valid, scale, idx, h) for p in runs])), 4))
            out[key][str(h)] = per_week
    return out


def _per_origin(pred: np.ndarray, y: np.ndarray, valid: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Absolute-error sum and valid count per (turbine, origin) over the first h steps: the pieces MASE is built from."""
    err = np.where(valid[..., :h], np.abs(pred[..., :h] - np.where(valid[..., :h], y[..., :h], 0.0)), 0.0)
    return err.sum(axis=2), valid[..., :h].sum(axis=2)


def _mase_from(err: np.ndarray, cnt: np.ndarray, scale: np.ndarray, idx: np.ndarray) -> float:
    n_t = cnt[:, idx].sum(axis=1)
    mae_t = err[:, idx].sum(axis=1) / np.maximum(n_t, 1)
    use = (n_t > 0) & np.isfinite(scale) & (scale > 0)
    return float((mae_t[use] / scale[use]).mean())


def bootstrap(preds: dict, y: np.ndarray, valid: np.ndarray, scale: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n_origins = y.shape[1]
    samples = [rng.integers(0, n_origins, n_origins) for _ in range(n)]
    pieces = {key: {h: _per_origin(preds[key][0], y, valid, h) for h in HORIZONS} for key in BOOTSTRAP_MODELS if key in preds}
    draws = {key: {h: np.array([_mase_from(*pieces[key][h], scale, s) for s in samples]) for h in HORIZONS} for key in pieces}
    full = np.arange(n_origins)
    out = {}
    for key in pieces:
        out[key] = {}
        for h in HORIZONS:
            d = draws[key][h]
            row = {"mase": round(_mase_from(*pieces[key][h], scale, full), 4),
                   "ci": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]}
            if key != "chronos_zeroshot" and "chronos_zeroshot" in draws:
                diff = d - draws["chronos_zeroshot"][h]
                row["diff_vs_zeroshot"] = round(float(_mase_from(*pieces[key][h], scale, full) - _mase_from(*pieces["chronos_zeroshot"][h], scale, full)), 4)
                row["diff_ci"] = [round(float(np.percentile(diff, 2.5)), 4), round(float(np.percentile(diff, 97.5)), 4)]
            out[key][str(h)] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output", default="output")
    ap.add_argument("--clean", default="data/sdwpf/clean.parquet")
    ap.add_argument("--splits", default="results/ds02_splits.json")
    ap.add_argument("--out", default="results/ex03_test_breakdown.json")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    frame = sdwpf.load_clean(a.clean)
    patv, valid = sdwpf.to_dense(frame, "Patv").astype(np.float64), sdwpf.to_dense(frame, "valid").astype(bool)
    split = splits.load_splits(a.splits)
    result = {"weeks": {}, "bootstrap": {}, "week_days": 7, "bootstrap_resamples": 1000, "bootstrap_seed_runs": "first seed of each cell"}
    for period in ("val", "test"):
        y, v, scale = metrics.eval_inputs(patv, valid, split, period)
        preds = forecasts(Path(a.output), patv, split, period)
        if period == "test":
            result["weeks"] = weekly(preds, y, v, scale, splits.eval_origins(split, period))
        result["bootstrap"][period] = bootstrap(preds, y, v, scale)
        _LOG.info("%s: %d models", period, len(preds))
    Path(a.out).write_text(json.dumps(result, indent=1) + "\n")
    for key, row in result["bootstrap"]["test"].items():
        _LOG.info("test %-20s H=64 %.3f [%.3f, %.3f] diff vs zero-shot %s", key, row["64"]["mase"], *row["64"]["ci"], row["64"].get("diff_ci"))
    for key in ("patchtst_100", "lstm_100", "chronos_ft_100", "chronos_zeroshot", "persistence"):
        _LOG.info("weeks %-18s H=64 %s", key, result["weeks"][key]["64"])


if __name__ == "__main__":
    main()
