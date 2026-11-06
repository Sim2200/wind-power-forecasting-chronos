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


def test_score_file_uses_quantile_arrays_for_wql(data, tmp_path):
    patv, valid, split = data
    y, v, _ = metrics.eval_inputs(patv, valid, split, "val")
    median = np.nan_to_num(y[..., :12])
    perfect = np.stack([median] * 3)  # every quantile equal to y: pinball 0 at every q
    np.savez_compressed(tmp_path / "q_pred.npz", val=median.astype(np.float32), val_q=perfect.astype(np.float32),
                        quantiles=np.float32([0.25, 0.5, 0.75]))
    out = score_sdwpf.score_file(patv, valid, split, tmp_path / "q_pred.npz")
    assert list(out) == ["val"]
    assert out["val"]["6"]["wql"] == pytest.approx(0.0, abs=1e-6)  # float32 round trip of y
    assert out["val"]["6"]["mae"] == pytest.approx(0.0, abs=1e-3)  # kW, float32 round trip
    shifted = perfect + np.array([-10.0, 0.0, 10.0])[:, None, None, None]
    np.savez_compressed(tmp_path / "q2_pred.npz", val=median.astype(np.float32), val_q=shifted.astype(np.float32),
                        quantiles=np.float32([0.25, 0.5, 0.75]))
    res = score_sdwpf.score_file(patv, valid, split, tmp_path / "q2_pred.npz")["val"]["6"]
    y6, v6 = y[..., :6], v[..., :6]
    assert res["wql"] == pytest.approx(metrics.wql(shifted[..., :6], y6, v6, (0.25, 0.5, 0.75)), rel=1e-5)  # file is float32
    assert res["wql"] > 0
