"""The EX-03 report reads the grid table and seed summaries into one row per (model, fraction) and formats the tables."""
import json
from pathlib import Path

from wpf import grid, grid_report

RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_rows_cover_the_grid_and_the_baselines():
    rows = grid_report.rows(RESULTS)
    keys = [(r["model"], r["pct"]) for r in rows]
    for model, spec in grid.MODELS.items():
        for pct in (grid.FRACTIONS if spec.seeded else (None,)):
            assert (model, pct) in keys
    assert ("persistence", None) in keys and ("seasonal_naive", None) in keys
    seeded = next(r for r in rows if r["model"] == "patchtst" and r["pct"] == 100)
    assert seeded["n_runs"] == 3 and seeded["val"]["64"]["mase"]["n"] == 3 and seeded["su"] > 0
    zero = next(r for r in rows if r["model"] == "chronos_zeroshot")
    assert zero["n_runs"] == 1 and zero["train_seconds"] is None and zero["val"]["6"]["mase"]["std"] is None
    point = next(r for r in rows if r["model"] == "patchtst_point" and r["pct"] == 5)
    assert "wql" not in point["val"]["6"]


def test_fmt_and_tables():
    assert grid_report.fmt({"mean": 0.5934, "std": 0.0011, "n": 3}) == "0.593 ± 0.001"
    assert grid_report.fmt({"mean": 0.5934, "std": None, "n": 1}) == "0.593"
    assert grid_report.fmt(None) == "–" and grid_report.fmt({"mean": None, "std": None, "n": 1}) == "–"
    rows = [{"model": "lstm", "label": "LSTM", "pct": 25, "n_runs": 3, "su": 0.76, "gpus": 1, "trainable_parameters": 1_000_000,
             "train_seconds": 100.0, "inference_seconds": 1.0,
             "val": {h: {"mase": {"mean": 0.5, "std": 0.01, "n": 3}} for h in grid.HORIZONS},
             "test": {h: {"mase": {"mean": 0.4, "std": 0.01, "n": 3}} for h in grid.HORIZONS}}]
    table = grid_report.metric_table(rows, "val", "mase")
    assert "| LSTM | 25% | 0.500 ± 0.010 | 0.500 ± 0.010 | 0.500 ± 0.010 |" in table
    assert grid_report.metric_table(rows, "val", "wql").count("–") == 3
    cost = grid_report.cost_table(rows)
    assert "| LSTM | 25% | 3 | 0.76 | 1 | 1.00M | 100 | 1 | 0.500 ± 0.010 | 0.400 ± 0.010 |" in cost


def test_report_has_every_section(tmp_path):
    rows = grid_report.rows(RESULTS)
    text = grid_report.report(rows, 107.8)
    for section in ("## Validation: MASE", "## Validation: WQL", "## Test: MASE", "## Test: MAE (kW)", "## Cost"):
        assert section in text
    assert "107.8 SU for 46 runs" in text
