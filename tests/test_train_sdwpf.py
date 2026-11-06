import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from wpf import score_sdwpf
from wpf import train_sdwpf as T
from wpf.data import splits
from wpf.models.patchtst import PatchTST, PatchTSTConfig


def test_masked_mse_by_hand():
    pred = torch.tensor([[[1.0], [2.0], [3.0]]])
    y = torch.tensor([[[0.0], [0.0], [0.0]]])
    mask = torch.tensor([[[1.0], [0.0], [1.0]]])
    assert T.masked_mse(pred, y, mask).item() == pytest.approx((1 + 9) / 2)


def test_predict_returns_turbines_by_origins_by_horizon():
    cfg = PatchTSTConfig(seq_len=splits.LOOKBACK, pred_len=6, n_vars=1, d_model=8, n_heads=2, d_ff=16, n_layers=1)
    model = PatchTST(cfg).eval()
    x = np.random.default_rng(0).normal(size=(3, 5, splits.LOOKBACK)).astype(np.float32)
    out = T.predict(model, x, batch_size=4, device="cpu")
    assert out.shape == (3, 5, 6)
    np.testing.assert_allclose(out, T.predict(model, x, batch_size=100, device="cpu"), atol=1e-6)


def _synthetic_clean(path, split, n_turbines=3, seed=0):
    """A DS-01-shaped parquet: sine-plus-noise Patv per turbine with some missing and flagged rows."""
    rng = np.random.default_rng(seed)
    rows = []
    for turbine in range(1, n_turbines + 1):
        t = np.arange(split.n_steps)
        patv = 500 + 400 * np.sin(2 * np.pi * t / 144 + turbine) + rng.normal(0, 20, split.n_steps)
        valid = rng.random(split.n_steps) > 0.2
        missing = rng.random(split.n_steps) < 0.03
        patv[missing] = np.nan
        valid &= ~missing
        rows.append(pd.DataFrame({"TurbID": turbine, "step": t, "Patv": patv.astype(np.float32), "valid": valid}))
    pd.concat(rows, ignore_index=True).to_parquet(path, index=False)


def test_short_run_on_synthetic_data(tmp_path):
    split = splits.make_split((6, 1, 2))
    clean, splits_json = tmp_path / "clean.parquet", tmp_path / "splits.json"
    _synthetic_clean(clean, split)
    splits.save_splits(splits_json, split)
    cfg = T.TrainConfig(clean_path=str(clean), splits_path=str(splits_json), fraction=0.5, pred_len=12, train_stride=4,
                        epochs=3, patience=10, batch_size=64, learning_rate=1e-3, device="cpu",
                        model={"d_model": 8, "n_heads": 2, "d_ff": 16, "n_layers": 1, "dropout": 0.0})
    pred_path = tmp_path / "run_pred.npz"
    result = T.train(cfg, log=lambda s: None, save_weights=tmp_path / "run.pt", save_pred=pred_path)
    vals = [h["val_mse"] for h in result["history"]]
    assert len(vals) == 3 and vals[-1] < vals[0]
    assert result["best_epoch"] == int(np.argmin(vals)) + 1
    assert result["model"]["n_vars"] == 1 and result["model"]["seq_len"] == splits.LOOKBACK
    assert result["train_windows"] > 0
    n_origins = {p: len(splits.eval_origins(split, p)) for p in ("val", "test")}
    with np.load(pred_path) as f:
        assert f["val"].shape == (3, n_origins["val"], 12) and f["test"].shape == (3, n_origins["test"], 12)
        assert np.isfinite(f["val"]).all() and np.isfinite(f["test"]).all()
    assert set(result["val"]) == set(result["test"]) == {"6"}  # the horizons of HORIZONS that fit H = 12
    assert set(result["val"]["6"]) == {"mae", "rmse", "mase", "mase_turbines", "n_valid", "wql"}  # the DS-04 layout
    assert result["val"]["6"]["n_valid"] > 0 and result["test"]["6"]["mae"] > 0 and result["test"]["6"]["mase_turbines"] == 3
    data = T.load_data(cfg)  # the trainer reports what the shared scorer gives for its saved predictions
    rescored = score_sdwpf.score_file(data.patv, data.valid, split, pred_path)
    assert rescored["test"]["6"]["mae"] == pytest.approx(result["test"]["6"]["mae"], rel=1e-5)
    assert (tmp_path / "run.pt").exists()
