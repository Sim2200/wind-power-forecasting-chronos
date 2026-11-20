"""Mean and standard deviation of the shared metrics over the seed runs of one model (DS-06 reports 3 seeds).

Reads `wpf.train_sdwpf` run records (their `val` / `test` sections are `metrics.evaluate` output) and writes
{"runs": [...], "seeds": [...], "val": {H: {metric: {"mean", "std", "n"}}}, "test": {...}}. The std is the sample
standard deviation (ddof = 1); with one run it is null. `wpf.chronos_eval` records work too: their seed is the `seed`
of the training YAML named in `train_config`, their model type is "chronos", and only the periods every record scored
(usually `val` alone) are summarized.

Usage:
    python -m wpf.seed_summary results/lstm_100pct_s2021.json results/lstm_100pct_s2022.json results/lstm_100pct_s2023.json \
        --out results/lstm_100pct_seeds.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml

_LOG = logging.getLogger(__name__)

METRICS = ("mae", "rmse", "mase", "wql")


def record_seed(record: dict) -> int:
    """The seed of a train_sdwpf record (`config.seed`) or of a chronos_eval record (the training YAML's `seed`)."""
    if "config" in record:
        return record["config"]["seed"]
    return yaml.safe_load(Path(record["train_config"]).read_text())["seed"]


def summarize(records: list[dict], periods: tuple[str, ...] | None = None, seeds: list[int] | None = None) -> dict:
    """Mean / std / n per period, horizon and metric over the records; a metric that is null in any record is null.

    `periods` defaults to those present in every record ("val" and "test" for train_sdwpf, usually "val" for chronos_eval).
    `seeds` overrides what the records say (the EX-01 grid passes the seed on the command line, so a Chronos record's
    training YAML still names the config's seed).
    """
    _LOG.debug("n_records=%d", len(records))
    if not records:
        raise ValueError("no records")
    if periods is None:
        periods = tuple(p for p in ("val", "test") if all(p in r for r in records))
    if seeds is not None and len(seeds) != len(records):
        raise ValueError("one seed per record")
    model_type = records[0].get("model_type", "chronos" if "train_config" in records[0] else "patchtst")
    out = {"seeds": list(seeds) if seeds is not None else [record_seed(r) for r in records], "model_type": model_type}
    for period in periods:
        horizons = list(records[0][period])
        if any(list(r[period]) != horizons for r in records):
            raise ValueError(f"the records report different horizons for {period}")
        out[period] = {}
        for h in horizons:
            entry = {"n_valid": records[0][period][h]["n_valid"]}
            if any(r[period][h]["n_valid"] != entry["n_valid"] for r in records):
                raise ValueError(f"the records score different readings at {period} H = {h}")
            for k in METRICS:
                values = [r[period][h][k] for r in records]
                if any(v is None for v in values):
                    entry[k] = None
                else:
                    arr = np.asarray(values, dtype=np.float64)
                    entry[k] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else None, "n": len(arr)}
            out[period][h] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Mean and std of the shared metrics over seed runs.")
    ap.add_argument("records", nargs="+", help="run records written by wpf.train_sdwpf")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    records = [json.loads(Path(p).read_text()) for p in a.records]
    result = {"runs": a.records, **summarize(records)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    _LOG.info("wrote %s", a.out)
    for period in (p for p in ("val", "test") if p in result):
        print(period, {h: {k: f"{v[k]['mean']:.3f}±{v[k]['std']:.3f}" if v[k] and v[k]["std"] is not None else None for k in ("mase", "wql")} for h, v in result[period].items()})


if __name__ == "__main__":
    main()
