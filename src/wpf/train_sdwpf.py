"""Train PatchTST on SDWPF with a masked loss and score the shared evaluation windows.

Reuses the ETTh1 recipe from `wpf.train` (Adam 1e-4, batch 128, the 'type3' schedule, early stopping with the
best checkpoint restored) and the verified `PatchTST` model with one channel. Differences for SDWPF:
- Loss: MSE over the valid target steps only (`sdwpf_windows` decisions). Batches always have at least one.
- Validation: the 99 shared origins x 134 turbines of DS-02, masked MSE in scaled units. The same windows give
  the reported val metrics, so early stopping and model selection use the numbers that are reported.
- Test: scored once with the best weights, after training. Nothing in the loop sees it.
- Output: the result JSON (config, best epoch, history, and the shared DS-04 metrics of `wpf.score_sdwpf` for val
  and test: MAE / RMSE in kW, MASE, n_valid on the first 6 / 36 / 64 steps) and, next to it, the kW predictions
  `<out>_pred.npz` with `val` and `test` arrays of shape (turbines, origins, H), which `python -m wpf.score_sdwpf`
  turns into the `results/<name>_<period>.json` files every model commits.

Usage:
    python -m wpf.train_sdwpf --config configs/sdwpf_5pct.json --out output/sdwpf_5pct.json
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from wpf import score_sdwpf
from wpf.data import sdwpf, sdwpf_windows, splits
from wpf.models.patchtst import PatchTST, PatchTSTConfig
from wpf.train import EarlyStopping, lr_type3, set_seed

_LOG = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    clean_path: str = "data/sdwpf/clean.parquet"
    splits_path: str = "results/ds02_splits.json"
    fraction: float = 1.0  # training subset: 0.05, 0.25 or 1.0
    pred_len: int = 64
    train_stride: int = 1  # origins every n steps inside the subset; raise it to cut epoch cost
    model: dict = field(default_factory=dict)  # overrides for `PatchTSTConfig`
    batch_size: int = 128
    eval_batch_size: int = 1024
    learning_rate: float = 1e-4
    epochs: int = 100
    patience: int = 10
    seed: int = 2021
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Data:
    series: np.ndarray  # filled and scaled, (turbines, steps)
    patv: np.ndarray  # as read, kW with NaN for missing; the shared metrics score against this
    valid: np.ndarray
    scaler: sdwpf_windows.Scaler
    split: splits.Split


def load_data(cfg: TrainConfig) -> Data:
    """Dense Patv and valid mask from the DS-01 parquet, filled and scaled with the subset's statistics."""
    _LOG.debug("clean_path='%s' fraction=%s", cfg.clean_path, cfg.fraction)
    frame = sdwpf.load_clean(cfg.clean_path)
    split = splits.load_splits(cfg.splits_path)
    patv, valid = sdwpf.to_dense(frame, "Patv"), sdwpf.to_dense(frame, "valid").astype(bool)
    if patv.shape[1] != split.n_steps:
        raise ValueError(f"{cfg.clean_path} has {patv.shape[1]} steps per turbine, the split expects {split.n_steps}")
    scaler = sdwpf_windows.fit_scaler(patv, valid, splits.subset_range(split, cfg.fraction))
    series = scaler.transform(sdwpf_windows.fill_missing(patv)).astype(np.float32)
    return Data(series=series, patv=patv.astype(np.float64), valid=valid, scaler=scaler, split=split)


def masked_mse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over the steps where `mask` is 1."""
    return ((pred - y) ** 2 * mask).sum() / mask.sum()


def make_train_loader(cfg: TrainConfig, data: Data) -> DataLoader:
    index = sdwpf_windows.train_index(data.valid, data.split, cfg.fraction, cfg.pred_len, cfg.train_stride)
    dataset = sdwpf_windows.SDWPFWindows(data.series, data.valid, index, cfg.pred_len)
    _LOG.info("training windows: %d (fraction %s, stride %d)", len(dataset), cfg.fraction, cfg.train_stride)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)


@torch.no_grad()
def predict(model: torch.nn.Module, x: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    """Forecasts for x of shape (turbines, origins, L), returned as (turbines, origins, H) in scaled units."""
    model.eval()
    flat = torch.from_numpy(x.reshape(-1, x.shape[-1], 1))
    out = [model(flat[i : i + batch_size].to(device)).cpu() for i in range(0, len(flat), batch_size)]
    return torch.cat(out)[..., 0].numpy().reshape(x.shape[0], x.shape[1], -1)



def train(cfg: TrainConfig, log=print, save_weights: str | Path | None = None, save_pred: str | Path | None = None) -> dict:
    """Train with early stopping on the shared validation windows, then score val and test once with the best weights."""
    _LOG.debug("cfg=%s", cfg)
    set_seed(cfg.seed)
    data = load_data(cfg)
    train_loader = make_train_loader(cfg, data)
    val = sdwpf_windows.eval_arrays(data.series, data.valid, data.split, "val", cfg.pred_len)
    val_y, val_mask = torch.from_numpy(val["y"]), torch.from_numpy(val["mask"].astype(np.float32))
    model_cfg = PatchTSTConfig(seq_len=splits.LOOKBACK, pred_len=cfg.pred_len, n_vars=1, **cfg.model)
    model = PatchTST(model_cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    stopper = EarlyStopping(cfg.patience)
    history = []
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = torch.zeros((), device=cfg.device)
        for x, y, mask in train_loader:
            optimizer.zero_grad()
            loss = masked_mse(model(x.to(cfg.device)), y.to(cfg.device), mask.to(cfg.device))
            loss.backward()
            optimizer.step()
            total += loss.detach()
        train_loss = total.item() / len(train_loader)
        val_pred = torch.from_numpy(predict(model, val["x"], cfg.eval_batch_size, cfg.device))
        val_loss = masked_mse(val_pred, val_y, val_mask).item()
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss, "lr": optimizer.param_groups[0]["lr"],
                        "seconds": round(time.time() - t0, 1)})
        log(json.dumps(history[-1]))
        if stopper.step(val_loss, epoch, model):
            log(f"early stopping at epoch {epoch}")
            break
        for g in optimizer.param_groups:
            g["lr"] = lr_type3(cfg.learning_rate, epoch + 1)
    model.load_state_dict(stopper.best_state)
    if save_weights is not None:
        torch.save(stopper.best_state, save_weights)
    scores, preds = {}, {}
    for period, arrays in (("val", val), ("test", sdwpf_windows.eval_arrays(data.series, data.valid, data.split, "test", cfg.pred_len))):
        pred_kw = data.scaler.inverse(predict(model, arrays["x"], cfg.eval_batch_size, cfg.device))
        preds[period] = pred_kw.astype(np.float32)
        scores[period] = score_sdwpf.score(data.patv, data.valid, data.split, pred_kw.astype(np.float64), period)
    if save_pred is not None:
        np.savez_compressed(save_pred, **preds)
    return {
        "config": asdict(cfg),
        "model": asdict(model_cfg),
        "parameters": model.num_parameters(),
        "train_windows": len(train_loader.dataset),
        "scaler": asdict(data.scaler),
        "best_val_mse": stopper.best,
        "best_epoch": stopper.best_epoch,
        "epochs_run": len(history),
        "train_seconds": round(time.time() - t0, 1),
        "val": scores["val"],
        "test": scores["test"],
        "history": history,
        "device": cfg.device,
    }


def load_config(path: str) -> TrainConfig:
    return TrainConfig(**json.loads(Path(path).read_text()))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train PatchTST on SDWPF.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="result JSON; weights and predictions are written next to it")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = train(load_config(a.config), save_weights=out.with_suffix(".pt"), save_pred=out.with_name(out.stem + "_pred.npz"))
    for period in ("val", "test"):  # NaN (no WQL for a point model) is not JSON; write null like metrics.save_results
        result[period] = {h: {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()} for h, r in result[period].items()}
    out.write_text(json.dumps(result, indent=2))
    h = str(max(int(k) for k in result["val"]))
    print(f"val mae kW {result['val'][h]['mae']:.1f} mase {result['val'][h]['mase']:.3f}, test mae kW {result['test'][h]['mae']:.1f} (H={h}) -> {out}")


if __name__ == "__main__":
    main()
