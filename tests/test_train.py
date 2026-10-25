import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from wpf import train as T
from wpf.data import etth1


def test_lr_type3_schedule():
    base_lr = 1e-4
    epochs = (1, 2, 3, 4, 5)
    expected = [1e-4, 1e-4, 1e-4, 9e-5, 8.1e-5]
    assert [T.lr_type3(base_lr, e) for e in epochs] == pytest.approx(expected)


def test_early_stopping_restores_best():
    model = torch.nn.Linear(2, 1)
    stopper = T.EarlyStopping(patience=2)
    assert stopper.step(1.0, 1, model) is False
    with torch.no_grad():
        model.weight.fill_(7.0)
    assert stopper.step(2.0, 2, model) is False  # worse: counter 1
    assert stopper.step(3.0, 3, model) is True  # worse: counter 2 -> stop
    assert (stopper.best, stopper.best_epoch) == (1.0, 1)
    assert not torch.equal(stopper.best_state["weight"], model.weight)


def test_short_run_on_synthetic_data(tmp_path):
    # a small sine-plus-noise series: val MSE must fall over a few epochs on CPU
    rng = np.random.default_rng(0)
    t = np.arange(etth1.TEST_END)
    base = np.stack([np.sin(2 * np.pi * t / (24 * (i + 1))) for i in range(7)], axis=1)
    data = (base * np.arange(1, 8) + rng.normal(0, 0.1, size=base.shape)).astype(np.float32)
    csv = tmp_path / "ETTh1.csv"
    pd.DataFrame(data, columns=etth1.COLUMNS).assign(date="x").to_csv(csv, index=False)
    epochs = 3
    model_cfg = {"d_model": 8, "n_heads": 2, "d_ff": 16, "n_layers": 1, "dropout": 0.0}
    cfg = T.TrainConfig(data_path=str(csv), pred_len=24, epochs=epochs, batch_size=256, learning_rate=1e-3,
                        device="cpu", model=model_cfg)
    result = T.train(cfg, log=lambda s: None)
    vals = [h["val_mse"] for h in result["history"]]
    assert len(vals) == epochs
    assert vals[-1] < vals[0]
    assert result["best_epoch"] == int(np.argmin(vals)) + 1
    assert set(result["test"]) == set(result["test_full"]) == {"mse", "mae", "num_windows"}
    assert result["test_full"]["num_windows"] >= result["test"]["num_windows"]
