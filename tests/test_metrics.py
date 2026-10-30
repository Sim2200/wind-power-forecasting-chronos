import numpy as np
import pytest

from wpf import metrics


def test_mase_scale_is_the_seasonal_naive_error_on_valid_pairs():
    train = np.array([[1, 3, 2, 7, 4, 4], [5, 5, 5, 5, 5, 5], [9, 9, 9, 9, 9, 9]], dtype=np.float32)
    valid = np.ones_like(train, dtype=bool)
    valid[0, 3] = False  # kills the pairs (3, 1) and (5, 3)
    valid[2] = False
    scale = metrics.mase_scale(train, valid, m=2)
    # turbine 0 pairs: |2-1| = 1 and |4-2| = 2 -> 1.5; turbine 1 is constant -> 0; turbine 2 has no valid pair -> NaN
    assert scale[0] == 1.5 and scale[1] == 0.0 and np.isnan(scale[2])


def test_mase_scale_rejects_series_not_longer_than_m():
    with pytest.raises(ValueError, match="more than m"):
        metrics.mase_scale(np.zeros((1, 2)), np.ones((1, 2), bool), m=2)


def test_point_metrics_by_hand_with_an_invalid_reading_and_nan_prediction_there():
    y = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    pred = np.array([[[2.0, 2.0], [1.0, np.nan]]])
    valid = np.array([[[True, True], [True, False]]])  # the NaN prediction sits on the invalid reading
    out = metrics.evaluate(y, valid, np.array([2.0]), pred=pred, horizons=(1, 2))
    # H = 1: errors 1 and 2 -> MAE 1.5, RMSE sqrt(5/2), MASE 1.5 / 2
    assert out[1]["mae"] == 1.5 and out[1]["rmse"] == pytest.approx(np.sqrt(2.5)) and out[1]["mase"] == 0.75
    # H = 2: errors 1, 0, 2 (the invalid one is dropped) -> MAE 1, RMSE sqrt(5/3), MASE 0.5
    assert out[2]["mae"] == 1.0 and out[2]["rmse"] == pytest.approx(np.sqrt(5 / 3)) and out[2]["mase"] == 0.5
    assert out[2]["n_valid"] == 3 and out[1]["n_valid"] == 2
    assert np.isnan(out[2]["wql"])  # a point model has no WQL


def test_mase_is_averaged_over_turbines_and_skips_a_zero_scale():
    y = np.array([[[10.0, 10.0]], [[5.0, 5.0]], [[1.0, 1.0]]])
    pred = np.array([[[11.0, 9.0]], [[8.0, 2.0]], [[1.0, 1.0]]])
    valid = np.ones(y.shape, bool)
    res = metrics.point_metrics(pred, y, valid, np.array([2.0, 1.5, 0.0]))
    assert res["mae"] == pytest.approx(4 / 3)  # pooled: (2 + 6 + 0) / 6
    assert res["mase"] == pytest.approx((1 / 2 + 3 / 1.5) / 2)  # per turbine 0.5 and 2.0; the scale-0 turbine is left out
    assert res["mase_turbines"] == 2


def test_turbine_without_valid_readings_is_left_out_of_mase():
    y = np.ones((2, 1, 2))
    valid = np.array([[[True, True]], [[False, False]]])
    res = metrics.point_metrics(np.full(y.shape, 3.0), y, valid, np.array([1.0, 1.0]))
    assert res["mae"] == 2.0 and res["mase"] == 2.0 and res["mase_turbines"] == 1


def test_wql_by_hand():
    y = np.array([[[4.0, 0.0]]])
    qpred = np.array([[[[2.0, 2.0]]], [[[6.0, 1.0]]]])  # q = 0.25 and 0.75
    valid = np.ones(y.shape, bool)
    # q = 0.25: pinballs 0.5 and 1.5 -> 2 * 2.0 / 4 = 1.0; q = 0.75: 0.5 and 0.25 -> 2 * 0.75 / 4 = 0.375
    assert metrics.wql(qpred, y, valid, quantiles=(0.25, 0.75)) == pytest.approx((1.0 + 0.375) / 2)
    valid[0, 0, 1] = False  # only the first reading (y = 4) counts: 2 * 0.5 / 4 for both quantiles
    assert metrics.wql(qpred, y, valid, quantiles=(0.25, 0.75)) == pytest.approx(0.25)


def test_wql_of_the_median_is_mae_over_sum_abs_y_and_zero_for_a_perfect_forecast():
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 100, size=(3, 4, 5))
    valid = rng.random(y.shape) > 0.3
    pred = y + rng.normal(size=y.shape)
    got = metrics.wql(pred[None], y, valid, quantiles=(0.5,))
    assert got == pytest.approx(np.abs(pred - y)[valid].sum() / y[valid].sum())
    assert metrics.wql(np.stack([y, y]), y, valid, quantiles=(0.1, 0.9)) == 0.0


def test_evaluate_takes_the_first_h_steps_and_both_kinds_of_forecast():
    y = np.arange(12, dtype=np.float64).reshape(1, 2, 6)
    valid = np.ones(y.shape, bool)
    pred = y + 1.0
    qpred = np.stack([y + 1.0] * len(metrics.QUANTILES))
    out = metrics.evaluate(y, valid, np.array([1.0]), pred=pred, quantile_pred=qpred, horizons=(2, 6))
    assert set(out) == {2, 6} and out[2]["n_valid"] == 4 and out[6]["n_valid"] == 12
    assert out[2]["mae"] == out[6]["mae"] == 1.0 and out[6]["mase"] == 1.0
    assert out[2]["wql"] == pytest.approx(_wql_shift(y[..., :2], metrics.QUANTILES))


def _wql_shift(y, quantiles):
    """WQL of a constant +1 forecast: pinball is (1 - q) for each reading, so 2 * sum(1 - q) / sum(y)."""
    return float(np.mean([2 * (1 - q) * y.size / y.sum() for q in quantiles]))


def test_no_valid_reading_gives_nan_not_zero():
    y = np.ones((1, 1, 2))
    valid = np.zeros(y.shape, bool)
    res = metrics.point_metrics(y, y, valid, np.array([1.0]))
    assert np.isnan(res["mae"]) and np.isnan(res["mase"]) and res["n_valid"] == 0
    assert np.isnan(metrics.wql(np.stack([y] * 9), y, valid))


def test_non_finite_prediction_on_a_valid_reading_is_an_error():
    y = np.ones((1, 1, 2))
    valid = np.ones(y.shape, bool)
    pred = np.array([[[1.0, np.nan]]])
    with pytest.raises(ValueError, match="non-finite"):
        metrics.evaluate(y, valid, np.array([1.0]), pred=pred, horizons=(2,))


def test_evaluate_validates_its_inputs():
    y = np.ones((1, 1, 2))
    valid = np.ones(y.shape, bool)
    with pytest.raises(ValueError, match="pred and/or"):
        metrics.evaluate(y, valid, np.array([1.0]), horizons=(2,))
    with pytest.raises(ValueError, match="longer"):
        metrics.evaluate(y, valid, np.array([1.0]), pred=y, horizons=(3,))
    with pytest.raises(ValueError, match="share the shape"):
        metrics.evaluate(y, valid[:, :, :1], np.array([1.0]), pred=y, horizons=(1,))
    with pytest.raises(ValueError, match="one leading entry"):
        metrics.wql(np.stack([y] * 3), y, valid)
