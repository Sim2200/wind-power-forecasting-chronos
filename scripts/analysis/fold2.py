"""EX-03 second fold: the split moved back by 25 days, so the current validation months become a test period.

`results/ds02_splits_fold2.json` is `splits.make_split((146, 25, 25))`: train on the first 146 days, validate on the next 25, test
on days 171 to 196, which are exactly the DS-02 validation months (same forecast origins). PatchTST and the LSTM are retrained on
that split (`configs/{sdwpf_100pct_lr3e-5,lstm_100pct_s2021}_fold2.json`, early stopping on the fold's own validation days) so
their score on days 171 to 196 is a test score, not the selection-biased validation score of the grid. The Chronos forecasts
need no retraining: nothing of theirs was selected on those days, so their grid `val` predictions are rescored here as the
fold's test (the MASE scale is the fold's train range, so the numbers differ slightly from the grid's val column).

    python scripts/analysis/fold2.py write-split                       -> results/ds02_splits_fold2.json
    python scripts/analysis/fold2.py score-chronos [--output output]   -> results/fold2_chronos_<name>_test.json
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from wpf import metrics, score_sdwpf  # noqa: E402
from wpf.data import splits  # noqa: E402

_LOG = logging.getLogger(__name__)
DAYS = (146, 25, 25)
SPLIT_PATH = "results/ds02_splits_fold2.json"
CHRONOS = {"zeroshot": "ex_chronos_zeroshot", "ft_100pct": "ex_chronos_ft_100pct_s2021", "lora_100pct": "ex_chronos_lora_100pct_s2021"}


def fold2_split() -> splits.Split:
    return splits.make_split(DAYS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("write-split", "score-chronos"))
    ap.add_argument("--output", default="output")
    ap.add_argument("--clean", default="data/sdwpf/clean.parquet")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    split = fold2_split()
    if a.command == "write-split":
        splits.save_splits(SPLIT_PATH, split)
        _LOG.info("-> %s: %s", SPLIT_PATH, splits.to_dict(split)["split"])
        return
    from wpf.data import sdwpf

    frame = sdwpf.load_clean(a.clean)
    patv, valid = sdwpf.to_dense(frame, "Patv").astype(np.float64), sdwpf.to_dense(frame, "valid").astype(bool)
    grid_split = splits.load_splits(Path(a.results) / "ds02_splits.json")
    assert np.array_equal(splits.eval_origins(grid_split, "val"), splits.eval_origins(split, "test"))
    for name, cell in CHRONOS.items():
        with np.load(Path(a.output) / f"{cell}_pred.npz") as f:
            res = score_sdwpf.score(patv, valid, split, f["val"].astype(np.float64), "test", quantile_pred=f["val_q"].astype(np.float64),
                                    quantiles=tuple(float(q) for q in f["quantiles"]))
        path = Path(a.results) / f"fold2_chronos_{name}_test.json"
        metrics.save_results(path, res)
        _LOG.info("%s mase %s", path, {h: round(r["mase"], 3) for h, r in res.items()})


if __name__ == "__main__":
    main()
