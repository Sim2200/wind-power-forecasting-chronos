import json
import os
from pathlib import Path

import numpy as np
import pytest

from wpf import baselines
from wpf.data import splits

REPO = Path(__file__).resolve().parents[1]
CLEAN = Path(os.environ.get("DS_CLEAN_PARQUET", REPO / "data" / "sdwpf" / "clean.parquet"))
FILES = [f"baseline_{name}_{period}.json" for name in baselines.NAMES for period in ("val", "test")]
COUNTS = [f"baseline_counts_{period}.json" for period in ("val", "test")]


@pytest.fixture
def series():
    """Two turbines, 400 steps. Turbine 0 is Patv = step; turbine 1 is 100 + step and invalid before step 305."""
    dense = np.stack([np.arange(400.0), 100.0 + np.arange(400.0)])
    valid = np.ones(dense.shape, bool)
    valid[0, 295:300] = False  # the last readings before origin 300 are flagged
    valid[1, :305] = False  # turbine 1 has no valid reading in the history of origin 300
    return dense, valid


def test_persistence_repeats_the_last_valid_reading_not_the_last_reading(series):
    dense, valid = series
    pred, no_valid = baselines.persistence(dense, valid, np.array([300, 310]), horizon=3)
    assert pred.shape == (2, 2, 3)
    # origin 300: turbine 0 skips the flagged steps 295-299 and takes step 294; origin 310: step 309 is valid
    assert (pred[0, 0] == 294).all() and (pred[0, 1] == 309).all()
    assert (pred[1, 1] == 409).all()  # turbine 1 at origin 310: valid steps 305-309 -> last is 309 -> 100 + 309


def test_persistence_with_no_valid_history_predicts_zero_and_is_counted(series):
    dense, valid = series
    pred, no_valid = baselines.persistence(dense, valid, np.array([300, 310]), horizon=3)
    assert (pred[1, 0] == 0).all()
    np.testing.assert_array_equal(no_valid, [[False, False], [True, False]])


def test_persistence_ignores_a_nan_reading_because_it_is_invalid():
    dense = np.arange(400.0)[None]
    valid = np.ones(dense.shape, bool)
    dense[0, 296:300], valid[0, 296:300] = np.nan, False  # missing readings are invalid
    pred, no_valid = baselines.persistence(dense, valid, np.array([300]), horizon=2)
    assert (pred == 295).all() and not no_valid.any()


def test_seasonal_naive_uses_the_reading_one_season_before_each_target_step():
    dense = np.arange(400.0)[None]
    pred, missing = baselines.seasonal_naive(dense, np.array([300]), horizon=4, m=144)
    np.testing.assert_array_equal(pred[0, 0], [156, 157, 158, 159])  # t + h - 144 for t = 300
    assert not missing.any()


def test_seasonal_naive_fills_a_missing_source_and_uses_a_flagged_one_as_stored():
    dense = np.arange(400.0)[None]
    dense[0, 157] = np.nan  # missing: last-value filled with step 156
    pred, missing = baselines.seasonal_naive(dense, np.array([300]), horizon=4, m=144)
    np.testing.assert_array_equal(pred[0, 0], [156, 156, 158, 159])
    np.testing.assert_array_equal(missing[0, 0], [False, True, False, False])
    # a flagged (valid = False) reading that is not NaN is not touched: seasonal_naive never looks at `valid`


def test_seasonal_naive_rejects_a_horizon_that_would_read_the_future():
    with pytest.raises(ValueError, match="horizon <= m"):
        baselines.seasonal_naive(np.zeros((1, 400)), np.array([300]), horizon=5, m=4)


def test_run_scores_both_forecasters_on_the_shared_windows():
    split = splits.make_split()
    rng = np.random.default_rng(0)
    dense = rng.uniform(0, 1500, size=(3, split.n_steps))
    valid = rng.random(dense.shape) > 0.2
    dense[~valid & (rng.random(dense.shape) > 0.5)] = np.nan  # some invalid readings are missing, some only flagged
    valid[:, 24624 - 288 : 24624 - 100] = False  # a long invalid stretch: windows at the start of val still have history after it
    results, counts = baselines.run(dense, valid, split, "val")
    assert set(results) == set(baselines.NAMES)
    for name in baselines.NAMES:
        assert set(results[name]) == {"6", "36", "64"}
        assert all(np.isfinite(results[name][h]["mae"]) and results[name][h]["n_valid"] > 0 for h in results[name])
    assert counts["persistence"]["windows"] == 3 * len(splits.eval_origins(split, "val"))
    filled = counts["seasonal_naive"]["filled_source_on_valid_targets"]
    assert filled["6"] <= filled["36"] <= filled["64"] <= counts["seasonal_naive"]["valid_targets"]["64"]


@pytest.mark.skipif(not CLEAN.exists(), reason="run python -m wpf.data.sdwpf first (data/sdwpf/clean.parquet)")
def test_committed_baseline_files_match_the_clean_data(tmp_path, monkeypatch):
    """Catches a stale result: if the DS-01 mask or the baselines change, regenerate with python -m wpf.baselines."""
    monkeypatch.setattr("sys.argv", ["baselines", "--clean", str(CLEAN), "--splits", str(REPO / "results" / "ds02_splits.json"), "--out", str(tmp_path)])
    baselines.main()
    for name in FILES + COUNTS:
        assert json.loads((tmp_path / name).read_text()) == json.loads((REPO / "results" / name).read_text()), name
    # same windows and mask as the DS-04 smoke run: the valid counts per horizon are identical
    smoke = json.loads((REPO / "results" / "ds04_smoke_persistence_val.json").read_text())
    got = json.loads((tmp_path / "baseline_persistence_val.json").read_text())
    assert {h: r["n_valid"] for h, r in got.items()} == {h: r["n_valid"] for h, r in smoke.items()}
