"""EX-03: are the validation and test months different regimes? Statistics of the three DS-02 periods.

Reads the DS-01 clean parquet and the DS-02 split, and writes results/ex03_period_stats.json: per period, the share of valid
readings, mean and std of valid power (kW), mean wind speed, and the mean absolute change of power over 1, 6, 36 and 64 steps
(the error of the persistence forecast at those horizons, over valid pairs). Runs on the login node in a few seconds.

    python scripts/analysis/period_stats.py [--clean data/sdwpf/clean.parquet] [--splits results/ds02_splits.json] [--out results/ex03_period_stats.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from wpf.data import sdwpf  # noqa: E402

_LOG = logging.getLogger(__name__)
LAGS = (1, 6, 36, 64)


def period_stats(patv: np.ndarray, valid: np.ndarray, wspd: np.ndarray, start: int, end: int) -> dict:
    """Statistics of one period of the dense (turbines, steps) arrays over its valid readings."""
    p, v, w = patv[:, start:end], valid[:, start:end], wspd[:, start:end]
    out = {"steps": int(end - start), "valid_share": round(float(v.mean()), 4), "patv_mean_kw": round(float(p[v].mean()), 1),
           "patv_std_kw": round(float(p[v].std()), 1), "wspd_mean": round(float(np.nanmean(w[v])), 2), "abs_change_kw": {}}
    for lag in LAGS:
        pair = v[:, lag:] & v[:, :-lag]
        out["abs_change_kw"][str(lag)] = round(float(np.abs(p[:, lag:] - p[:, :-lag])[pair].mean()), 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clean", default="data/sdwpf/clean.parquet")
    ap.add_argument("--splits", default="results/ds02_splits.json")
    ap.add_argument("--out", default="results/ex03_period_stats.json")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    df = sdwpf.load_clean(a.clean)
    patv, valid, wspd = (sdwpf.to_dense(df, c) for c in ("Patv", "valid", "Wspd"))
    split = json.loads(Path(a.splits).read_text())["split"]
    stats = {period: period_stats(patv, valid.astype(bool), wspd, *bounds) for period, bounds in split.items()}
    Path(a.out).write_text(json.dumps(stats, indent=2) + "\n")
    for period, s in stats.items():
        _LOG.info("%-5s valid %.3f  Patv %.0f ± %.0f kW  Wspd %.2f  |Δ| %s", period, s["valid_share"], s["patv_mean_kw"], s["patv_std_kw"],
                  s["wspd_mean"], s["abs_change_kw"])


if __name__ == "__main__":
    main()
