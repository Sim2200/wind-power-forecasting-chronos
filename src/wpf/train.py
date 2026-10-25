"""Train and evaluate PatchTST on ETTh1, following the official supervised recipe.

Official details reproduced: Adam with lr 1e-4, MSE loss, batch 128, the "type3"
learning-rate schedule (constant for two epochs, then x0.9 per epoch), early stopping
on validation loss with the best checkpoint restored at the end, `drop_last=True` on
every loader, and metrics computed on the normalized data.

Because the official test loader drops its last partial batch, the result reports the
test score both that way ("test", comparable to the paper) and over every test window
("test_full"). The best weights are saved next to the result for re-scoring.

Usage:
    python -m wpf.train --config configs/etth1_h96.json --out results/etth1_h96.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from wpf.data import etth1
from wpf.models.patchtst import PatchTST, PatchTSTConfig

_LOG = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    data_path: str = "data/etth1/ETTh1.csv"
    seq_len: int = 336
    pred_len: int = 96
    model: dict = field(default_factory=dict)  # overrides for `PatchTSTConfig`
    batch_size: int = 128
    learning_rate: float = 1e-4
    epochs: int = 100
    patience: int = 100
    seed: int = 2021
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    """Seed every RNG the training path uses: Python, NumPy, torch and CUDA."""
    _LOG.debug("seed=%s", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_type3(base_lr: float, epoch: int) -> float:
    """Official 'type3' schedule: constant for epochs 1-2, then x0.9 per epoch; `epoch` is 1-based."""
    return base_lr if epoch < 3 else base_lr * 0.9 ** (epoch - 3)


class EarlyStopping:
    """Track the best validation loss; stop after `patience` epochs without improvement."""

    def __init__(self, patience: int):
        self.patience = patience
        self.best = float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.best_state: dict | None = None

    def step(self, val_loss: float, epoch: int, model: nn.Module) -> bool:
        """Record `val_loss` for `epoch`; return True when training should stop."""
        if val_loss < self.best:
            self.best = val_loss
            self.best_epoch = epoch
            self.counter = 0
            self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience


def make_loaders(cfg: TrainConfig) -> dict[str, DataLoader]:
    """Loaders over the ETTh1 windows: "train", "val", "test" (official, drops the last batch) and "test_full"."""
    _LOG.debug("cfg=%s", cfg)
    data = etth1.load_raw(cfg.data_path)
    splits = etth1.make_splits(data, cfg.seq_len)

    def loader(series: np.ndarray, shuffle: bool, drop_last: bool) -> DataLoader:
        dataset = etth1.WindowDataset(series, cfg.seq_len, cfg.pred_len)
        return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=cfg.num_workers)

    # Only train is shuffled; drop_last=True on train, val and test matches the official loaders.
    return {
        "train": loader(splits.train, True, True),
        "val": loader(splits.val, False, True),
        "test": loader(splits.test, False, True),
        "test_full": loader(splits.test, False, False),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict:
    """Mean MSE and MAE per element over the loader's windows, on the normalized data (as in the official code)."""
    _LOG.debug("device='%s' num_batches=%s", device, len(loader))
    model.eval()
    # Accumulate on the device and read back once; a .item() per batch would sync every step.
    squared_error = torch.zeros((), device=device)
    abs_error = torch.zeros((), device=device)
    num_values = num_windows = 0
    for x, y in loader:
        y = y.to(device)
        diff = model(x.to(device)) - y
        squared_error += diff.pow(2).sum()
        abs_error += diff.abs().sum()
        num_values += y.numel()
        num_windows += y.shape[0]
    return {"mse": squared_error.item() / num_values, "mae": abs_error.item() / num_values, "num_windows": num_windows}


def train(cfg: TrainConfig, log=print, save_weights: str | Path | None = None) -> dict:
    """Train with early stopping on validation MSE, then score the test set once with the best weights."""
    _LOG.debug("cfg=%s save_weights=%s", cfg, save_weights)
    set_seed(cfg.seed)
    loaders = make_loaders(cfg)
    train_loader, val_loader = loaders["train"], loaders["val"]
    n_vars = train_loader.dataset.x.shape[-1]  # the data decides the channel count
    model = PatchTST(PatchTSTConfig(seq_len=cfg.seq_len, pred_len=cfg.pred_len, n_vars=n_vars, **cfg.model)).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.MSELoss()
    stopper = EarlyStopping(cfg.patience)
    history = []
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = torch.zeros((), device=cfg.device)
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x.to(cfg.device)), y.to(cfg.device))
            loss.backward()
            optimizer.step()
            total += loss.detach()
        train_loss = total.item() / len(train_loader)
        val_loss = evaluate(model, val_loader, cfg.device)["mse"]
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss, "lr": optimizer.param_groups[0]["lr"]})
        log(json.dumps(history[-1]))
        if stopper.step(val_loss, epoch, model):
            log(f"early stopping at epoch {epoch}")
            break
        for g in optimizer.param_groups:  # schedule applied after the epoch, as in the official code
            g["lr"] = lr_type3(cfg.learning_rate, epoch + 1)
    # Restore the best-validation weights; the test set is scored once, with those weights only.
    model.load_state_dict(stopper.best_state)
    if save_weights is not None:
        torch.save(stopper.best_state, save_weights)
    return {
        "config": asdict(cfg),
        "parameters": model.num_parameters(),
        "best_val_mse": stopper.best,
        "best_epoch": stopper.best_epoch,
        "epochs_run": len(history),
        "train_seconds": round(time.time() - t0, 1),
        "test": evaluate(model, loaders["test"], cfg.device),
        "test_full": evaluate(model, loaders["test_full"], cfg.device),
        "history": history,
        "device": cfg.device,
    }


def load_config(path: str) -> TrainConfig:
    return TrainConfig(**json.loads(Path(path).read_text()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = train(load_config(a.config), save_weights=out.with_suffix(".pt"))
    out.write_text(json.dumps(result, indent=2))
    print(f"test mse {result['test']['mse']:.4f} mae {result['test']['mae']:.4f} -> {out}")


if __name__ == "__main__":
    main()
