import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wpf import chronos_eval, metrics, score_sdwpf
from wpf.data import splits


class FakePipeline:
    """Stands in for ChronosPipeline: quantile q of every step = last non-NaN context value + 100 * (q - 0.5)."""

    def __init__(self):
        self.calls = []

    def predict_quantiles(self, inputs, prediction_length, quantile_levels, num_samples=None):
        self.calls.append(len(inputs))
        x = inputs.clone()
        last = torch.stack([row[~torch.isnan(row)][-1] if (~torch.isnan(row)).any() else torch.tensor(0.0) for row in x])
        offsets = torch.tensor(quantile_levels) - 0.5
        q = last[:, None, None] + 100.0 * offsets[None, None, :]  # (B, 1, Q)
        q = q.expand(-1, prediction_length, -1)
        return q, q.mean(-1)


@pytest.fixture
def data():
    split = splits.make_split((6, 1, 2))
    rng = np.random.default_rng(3)
    t = np.arange(split.n_steps)
    patv = 500 + 400 * np.sin(2 * np.pi * t / 144)[None, :] + rng.normal(0, 20, (3, split.n_steps))
    valid = rng.random(patv.shape) > 0.2
    patv[rng.random(patv.shape) < 0.03] = np.nan
    valid &= ~np.isnan(patv)
    return patv, valid, split


def test_contexts_are_the_lookback_before_each_origin_in_both_modes(data):
    patv, valid, split = data
    origins = splits.eval_origins(split, "val")
    keep = chronos_eval.contexts(patv, valid, split, "val", "keep")
    assert keep.shape == (3, len(origins), splits.LOOKBACK) and keep.dtype == np.float32
    np.testing.assert_allclose(keep[1, 2], patv[1, origins[2] - splits.LOOKBACK : origins[2]].astype(np.float32))
    blank = chronos_eval.contexts(patv, valid, split, "val", "nan")
    flagged = ~valid[1, origins[2] - splits.LOOKBACK : origins[2]]
    assert np.isnan(blank[1, 2][flagged]).all() and np.isnan(blank).sum() > np.isnan(keep).sum()
    with pytest.raises(ValueError, match="mode"):
        chronos_eval.contexts(patv, valid, split, "val", "drop")


def test_forecast_layout_median_and_batching():
    pipe = FakePipeline()
    x = np.random.default_rng(0).normal(500, 100, (2, 5, 288)).astype(np.float32)
    x[0, 0, -1] = np.nan  # last value missing: the fake falls back to the last non-NaN one
    q, median = chronos_eval.forecast(pipe, x, horizon=8, batch_size=4)
    assert q.shape == (9, 2, 5, 8) and median.shape == (2, 5, 8)
    assert pipe.calls == [4, 4, 2]
    np.testing.assert_allclose(median, q[4])
    assert (np.diff(q, axis=0) > 0).all()
    np.testing.assert_allclose(median[1, 3], x[1, 3, -1])
    np.testing.assert_allclose(median[0, 0], x[0, 0, -2])
    with pytest.raises(ValueError, match="0.5"):
        chronos_eval.forecast(pipe, x, horizon=8, quantiles=(0.1, 0.9))


def test_evaluate_matches_the_shared_scorer_on_the_saved_arrays(data, tmp_path):
    patv, valid, split = data
    scores, preds = chronos_eval.evaluate(FakePipeline(), patv, valid, split, periods=("val", "test"))
    assert set(preds) == {"quantiles", "val", "val_q", "test", "test_q"} and "inference_seconds" in scores
    assert preds["val_q"].shape[0] == 9 and preds["val"].shape == preds["val_q"].shape[1:]
    for period in ("val", "test"):
        assert set(scores[period]) == {"6", "36", "64"}
        assert np.isfinite(scores[period]["64"]["wql"]) and scores[period]["64"]["n_valid"] > 0
    np.savez_compressed(tmp_path / "run_pred.npz", **preds)
    rescored = score_sdwpf.score_file(patv, valid, split, tmp_path / "run_pred.npz")
    for h in ("6", "36", "64"):
        assert rescored["val"][h]["wql"] == pytest.approx(scores["val"][h]["wql"], rel=1e-5)
        assert rescored["val"][h]["mae"] == pytest.approx(scores["val"][h]["mae"], rel=1e-5)


def test_fake_persistence_median_equals_the_persistence_forecast(data):
    """With `keep` contexts the fake's median is the last stored reading: metrics.persistence_forecast's rule."""
    patv, valid, split = data
    _, preds = chronos_eval.evaluate(FakePipeline(), patv, valid, split, periods=("val",))
    origins = splits.eval_origins(split, "val")
    persistence = metrics.persistence_forecast(patv, origins)
    np.testing.assert_allclose(preds["val"], persistence.astype(np.float32), rtol=1e-6)


def test_json_safe_turns_nan_into_null():
    out = chronos_eval._json_safe({"6": {"mae": 1.0, "wql": float("nan"), "n_valid": 3}})
    assert json.loads(json.dumps(out)) == {"6": {"mae": 1.0, "wql": None, "n_valid": 3}}
