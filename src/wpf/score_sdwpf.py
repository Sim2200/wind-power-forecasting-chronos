"""Score saved SDWPF predictions with the shared evaluation code (DS-04).

`wpf.train_sdwpf` writes `<run>_pred.npz` with kW point forecasts `val` and `test` of shape (turbines, origins, H)
and, for a quantile model, `val_q` / `test_q` of shape (Q, turbines, origins, H) with the `quantiles` array. This
module turns such a file into the per-horizon metrics every model reports (MAE, RMSE, MASE, n_valid, WQL) through
`metrics.eval_inputs` and `metrics.evaluate`, and writes one `results/<name>_<period>.json` per period with
`metrics.save_results`, the layout EX-03 reads. Horizons longer than the saved H are left out; WQL is null
without quantile arrays.

Usage:
    python -m wpf.score_sdwpf --pred output/sdwpf_5pct_pred.npz --name patchtst_5pct --out results/
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from wpf import metrics
from wpf.data import splits

_LOG = logging.getLogger(__name__)

PERIODS = ("val", "test")


def score(patv: np.ndarray, valid: np.ndarray, split: splits.Split, pred: np.ndarray, period: str,
          quantile_pred: np.ndarray | None = None, quantiles: tuple[float, ...] = metrics.QUANTILES) -> dict[str, dict]:
    """`metrics.evaluate` on the shared windows of `period` for kW predictions (turbines, origins, H), H <= 64, with
    `quantile_pred` (Q, turbines, origins, H) for the WQL when the model has one."""
    _LOG.debug("period='%s' pred.shape=%s", period, pred.shape)
    horizon = pred.shape[-1]
    y, v, scale = metrics.eval_inputs(patv, valid, split, period)
    horizons = tuple(h for h in splits.HORIZONS if h <= horizon)
    if not horizons:
        raise ValueError(f"predictions with H={horizon} cover none of the horizons {splits.HORIZONS}")
    return metrics.evaluate(y[..., :horizon], v[..., :horizon], scale, pred=pred, quantile_pred=quantile_pred,
                            horizons=horizons, quantiles=quantiles)


def score_file(patv: np.ndarray, valid: np.ndarray, split: splits.Split, pred_path: str | Path) -> dict[str, dict[str, dict]]:
    """{period: evaluate output} for the `val` and `test` arrays of a `_pred.npz` file, with WQL from `<period>_q`."""
    _LOG.debug("pred_path='%s'", pred_path)
    with np.load(pred_path) as f:
        quantiles = tuple(float(q) for q in f["quantiles"]) if "quantiles" in f.files else metrics.QUANTILES
        return {p: score(patv, valid, split, f[p].astype(np.float64), p,
                         quantile_pred=f[p + "_q"].astype(np.float64) if p + "_q" in f.files else None, quantiles=quantiles)
                for p in PERIODS if p in f.files}


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a _pred.npz file with the shared SDWPF metrics.")
    ap.add_argument("--pred", required=True, help="<run>_pred.npz written by wpf.train_sdwpf")
    ap.add_argument("--name", required=True, help="result files are <out>/<name>_val.json and <out>/<name>_test.json")
    ap.add_argument("--out", default="results")
    ap.add_argument("--clean", default="data/sdwpf/clean.parquet")
    ap.add_argument("--splits", default="results/ds02_splits.json")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    from wpf.data import sdwpf  # lazy: pandas/pyarrow are only needed here

    frame = sdwpf.load_clean(a.clean)
    patv, valid = sdwpf.to_dense(frame, "Patv").astype(np.float64), sdwpf.to_dense(frame, "valid").astype(bool)
    split = splits.load_splits(a.splits)
    for period, res in score_file(patv, valid, split, a.pred).items():
        path = Path(a.out) / f"{a.name}_{period}.json"
        metrics.save_results(path, res)
        _LOG.info("wrote %s", path)
        print(period, {h: round(r["mae"], 2) for h, r in res.items()}, "mase", {h: round(r["mase"], 4) for h, r in res.items()},
              "wql", {h: round(r["wql"], 4) for h, r in res.items()})


if __name__ == "__main__":
    main()
