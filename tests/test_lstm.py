import pytest

torch = pytest.importorskip("torch")

from wpf.models.lstm import LSTMConfig, LSTMForecaster

L = 288


def test_point_forecast_shape_and_parameter_count():
    cfg = LSTMConfig(seq_len=L, pred_len=64, n_vars=1, hidden_size=16, n_layers=1, dropout=0.0)
    model = LSTMForecaster(cfg).eval()
    assert model(torch.randn(4, L, 1)).shape == (4, 64, 1)
    lstm = 4 * (16 * 1 + 16 * 16 + 2 * 16)  # input, hidden and two bias vectors per gate
    assert model.num_parameters() == lstm + 16 * 64 + 64


def test_quantile_head_is_monotone_after_denorm():
    torch.manual_seed(0)
    model = LSTMForecaster(LSTMConfig(seq_len=L, pred_len=24, n_vars=1, hidden_size=8, n_layers=2, n_quantiles=5)).eval()
    y = model(torch.randn(6, L, 1) * 300 + 500)
    assert y.shape == (6, 24, 1, 5)
    assert (y[..., 1:] >= y[..., :-1]).all() and not torch.allclose(y[..., 0], y[..., -1])


def test_eval_is_deterministic_and_train_uses_dropout():
    torch.manual_seed(0)
    model = LSTMForecaster(LSTMConfig(seq_len=L, pred_len=8, n_vars=1, hidden_size=8, n_layers=2, dropout=0.5))
    x = torch.randn(2, L, 1)
    model.eval()
    assert torch.equal(model(x), model(x))
    model.train()
    assert not torch.equal(model(x), model(x))


def test_gradients_reach_every_parameter():
    model = LSTMForecaster(LSTMConfig(seq_len=L, pred_len=8, n_vars=1, hidden_size=8, n_layers=1, n_quantiles=3))
    model(torch.randn(2, L, 1)).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
