import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wpf import train as T


def test_lr_type3_schedule():
    assert [T.lr_type3(1e-4, e) for e in (1, 2, 3, 4, 5)] == pytest.approx([1e-4, 1e-4, 1e-4, 9e-5, 8.1e-5])


def test_early_stopping_restores_best():
    m = torch.nn.Linear(2, 1)
    s = T.EarlyStopping(patience=2)
    assert s.step(1.0, m) is False
    with torch.no_grad():
        m.weight.fill_(7.0)
    assert s.step(2.0, m) is False  # worse: counter 1
    assert s.step(3.0, m) is True  # worse: counter 2 -> stop
    assert s.best == 1.0 and not torch.equal(s.best_state["weight"], m.weight)


def test_short_run_on_synthetic_data(tmp_path):
    # a small sine-plus-noise series: val MSE must fall over a few epochs on CPU
    rng = np.random.default_rng(0)
    t = np.arange(etth1_rows := 14400)
    base = np.stack([np.sin(2 * np.pi * t / (24 * (i + 1))) for i in range(7)], axis=1)
    data = (base * (i_scale := np.arange(1, 8)) + rng.normal(0, 0.1, size=base.shape)).astype(np.float32)
    import pandas as pd

    csv = tmp_path / "ETTh1.csv"
    pd.DataFrame(data, columns=["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]).assign(date="x").to_csv(csv, index=False)
    cfg = T.TrainConfig(data_path=str(csv), pred_len=24, epochs=3, batch_size=256, learning_rate=1e-3, device="cpu",
                        model={"d_model": 8, "n_heads": 2, "d_ff": 16, "n_layers": 1, "dropout": 0.0})
    result = T.train(cfg, log=lambda s: None)
    vals = [h["val_mse"] for h in result["history"]]
    assert len(vals) == 3 and vals[-1] < vals[0]
    assert result["best_epoch"] == int(np.argmin(vals)) + 1
    assert set(result["test"]) == {"mse", "mae", "num_windows"}
