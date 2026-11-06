import numpy as np
import pytest

from wpf import metrics, score_sdwpf
from wpf.data import splits


@pytest.fixture
def data():
    """6 / 1 / 2 days, 3 turbines: a sine with some NaN and flagged readings."""
    split = splits.make_split((6, 1, 2))
    rng = np.random.default_rng(1)
    t = np.arange(split.n_steps)
    patv = 500 + 400 * np.sin(2 * np.pi * t / 144)[None, :] + rng.normal(0, 20, (3, split.n_steps))
    valid = rng.random(patv.shape) > 0.2
    patv[rng.random(patv.shape) < 0.03] = np.nan
    valid &= ~np.isnan(patv)
    return patv, valid, split


def test_perfect_forecast_scores_zero_and_shorter_horizons_drop_long_tasks(data):
    patv, valid, split = data
    y, _, _ = metrics.eval_inputs(patv, valid, split, "val")
    pred = np.nan_to_num(y[..., :40])  # H = 40 covers 6 and 36 only
    res = score_sdwpf.score(patv, valid, split, pred, "val")
    assert list(res) == ["6", "36"]
    assert res["36"]["mae"] == 0.0 and res["36"]["mase"] == 0.0 and res["36"]["mase_turbines"] == 3
    assert res["36"]["n_valid"] > 0 and res["36"]["rmse"] == 0.0
    assert np.isnan(res["6"]["wql"])


def test_score_matches_a_direct_evaluate_call_on_persistence(data):
    patv, valid, split = data
    origins = splits.eval_origins(split, "test")
    pred = metrics.persistence_forecast(patv, origins)
    y, v, scale = metrics.eval_inputs(patv, valid, split, "test")
    got, want = score_sdwpf.score(patv, valid, split, pred, "test"), metrics.evaluate(y, v, scale, pred=pred)
    assert got.keys() == want.keys() == {"6", "36", "64"}
    for h in want:
        assert {k: v for k, v in got[h].items() if k != "wql"} == {k: v for k, v in want[h].items() if k != "wql"}
        assert np.isnan(got[h]["wql"])


def test_score_rejects_a_horizon_shorter_than_every_task(data):
    patv, valid, split = data
    with pytest.raises(ValueError, match="none of the horizons"):
        score_sdwpf.score(patv, valid, split, np.zeros((3, len(splits.eval_origins(split, "val")), 5)), "val")


def test_score_file_reads_val_and_test_arrays(data, tmp_path):
    patv, valid, split = data
    preds = {p: metrics.persistence_forecast(patv, splits.eval_origins(split, p), horizon=12).astype(np.float32) for p in ("val", "test")}
    np.savez_compressed(tmp_path / "run_pred.npz", **preds)
    out = score_sdwpf.score_file(patv, valid, split, tmp_path / "run_pred.npz")
    assert set(out) == {"val", "test"} and list(out["val"]) == ["6"]
    assert out["test"]["6"]["n_valid"] == int(valid[:, [o + k for o in splits.eval_origins(split, "test") for k in range(6)]].sum())
