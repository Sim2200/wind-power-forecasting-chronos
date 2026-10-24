"""Train and evaluate PatchTST on ETTh1, following the official supervised recipe.

Official details reproduced: Adam with lr 1e-4, MSE loss, batch 128, the "type3"
learning-rate schedule (constant for two epochs, then x0.9 per epoch), early stopping
on validation loss with the best checkpoint restored at the end, drop_last=True on
every loader, and metrics computed on the normalized data.

Usage:
    python -m wpf.train --config configs/etth1_h96.json --out results/etth1_h96.json
"""
from __future__ import annotations

import argparse
import json
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


@dataclass
class TrainConfig:
    data_path: str = "data/etth1/ETTh1.csv"
    seq_len: int = 336
    pred_len: int = 96
    model: dict = field(default_factory=dict)  # overrides for PatchTSTConfig
    batch_size: int = 128
    learning_rate: float = 1e-4
    epochs: int = 100
    patience: int = 100
    seed: int = 2021
    num_workers: int = 0
    test_drop_last: bool = True  # official loaders drop the last partial test batch
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_type3(base_lr: float, epoch: int) -> float:
    """Official 'type3' schedule; `epoch` is 1-based."""
    return base_lr if epoch < 3 else base_lr * 0.9 ** (epoch - 3)


class EarlyStopping:
    """Track the best validation loss; stop after `patience` epochs without improvement."""

    def __init__(self, patience: int):
        self.patience = patience
        self.best = float("inf")
        self.counter = 0
        self.best_state: dict | None = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best:
            self.best = val_loss
            self.counter = 0
            self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience


def make_loaders(cfg: TrainConfig):
    data = etth1.load_raw(cfg.data_path)
    splits = etth1.make_splits(data, cfg.seq_len)
    mk = lambda series, shuffle, drop_last: DataLoader(
        etth1.WindowDataset(series, cfg.seq_len, cfg.pred_len),
        batch_size=cfg.batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=cfg.num_workers,
    )
    return mk(splits.train, True, True), mk(splits.val, False, True), mk(splits.test, False, cfg.test_drop_last)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict:
    """Mean MSE and MAE over all windows, on the normalized data (as in the official code)."""
    model.eval()
    se = ae = 0.0
    n = 0
    for x, y in loader:
        pred = model(x.to(device))
        y = y.to(device)
        se += (pred - y).pow(2).sum().item()
        ae += (pred - y).abs().sum().item()
        n += y.numel()
    return {"mse": se / n, "mae": ae / n, "num_windows": len(loader.dataset) if not loader.drop_last else len(loader) * loader.batch_size}


def train(cfg: TrainConfig, log=print) -> dict:
    set_seed(cfg.seed)
    train_loader, val_loader, test_loader = make_loaders(cfg)
    model = PatchTST(PatchTSTConfig(seq_len=cfg.seq_len, pred_len=cfg.pred_len, **cfg.model)).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.MSELoss()
    stopper = EarlyStopping(cfg.patience)
    history = []
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x.to(cfg.device)), y.to(cfg.device))
            loss.backward()
            optimizer.step()
            total += loss.item()
        train_loss = total / len(train_loader)
        val_loss = evaluate(model, val_loader, cfg.device)["mse"]
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss, "lr": optimizer.param_groups[0]["lr"]})
        log(json.dumps(history[-1]))
        stop = stopper.step(val_loss, model)
        for g in optimizer.param_groups:  # schedule applied after the epoch, as in the official code
            g["lr"] = lr_type3(cfg.learning_rate, epoch + 1)
        if stop:
            log(f"early stopping at epoch {epoch}")
            break
    model.load_state_dict(stopper.best_state)
    test = evaluate(model, test_loader, cfg.device)  # the test set is scored once, with the best-val weights
    return {
        "config": asdict(cfg),
        "parameters": model.num_parameters(),
        "best_val_mse": stopper.best,
        "best_epoch": min(history, key=lambda h: h["val_mse"])["epoch"],
        "epochs_run": len(history),
        "train_seconds": round(time.time() - t0, 1),
        "test": test,
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
    result = train(load_config(a.config))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(f"test mse {result['test']['mse']:.4f} mae {result['test']['mae']:.4f} -> {a.out}")


if __name__ == "__main__":
    main()
