"""LSTM baseline for SDWPF (DS-06): the recurrent counterpart of PatchTST on the same windows.

Flow for an input of shape (B, L, C):
  RevIN norm (per window, as PatchTST) -> LSTM over the L steps -> last hidden state (B, hidden)
  -> linear head -> (B, H, C) or, with `n_quantiles` = Q > 0, non-crossing quantiles (B, H, C, Q) built the same
  way as PatchTST's head (lowest quantile plus cumulative softplus increments) -> RevIN denorm.

Channels are mixed inside the LSTM (input size C); SDWPF uses C = 1, so this matches PatchTST's one-channel setup.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from wpf.models.layers import RevIN

_LOG = logging.getLogger(__name__)


@dataclass
class LSTMConfig:
    seq_len: int = 288
    pred_len: int = 64
    n_vars: int = 1
    hidden_size: int = 128
    n_layers: int = 2
    dropout: float = 0.2  # between LSTM layers and before the head
    revin: bool = True
    revin_affine: bool = False
    n_quantiles: int = 0  # 0: point forecast (B, H, C); Q > 0: non-crossing quantiles (B, H, C, Q)


class LSTMForecaster(nn.Module):
    def __init__(self, cfg: LSTMConfig):
        super().__init__()
        _LOG.debug("cfg=%s", cfg)
        self.cfg = cfg
        self.revin = RevIN(cfg.n_vars, affine=cfg.revin_affine) if cfg.revin else None
        self.lstm = nn.LSTM(cfg.n_vars, cfg.hidden_size, cfg.n_layers, batch_first=True,
                            dropout=cfg.dropout if cfg.n_layers > 1 else 0.0)
        self.per_step = max(cfg.n_quantiles, 1)
        self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden_size, cfg.pred_len * cfg.n_vars * self.per_step))
        _LOG.debug("parameters=%s", self.num_parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, C) -> (B, H, C), or (B, H, C, Q) with `n_quantiles` = Q > 0."""
        if self.revin is not None:
            x = self.revin(x, "norm")
        out, _ = self.lstm(x)  # (B, L, hidden)
        y = self.head(out[:, -1]).view(-1, self.cfg.n_vars, self.cfg.pred_len, self.per_step)  # (B, C, H, Q or 1)
        if self.cfg.n_quantiles:
            y = torch.cat([y[..., :1], y[..., :1] + torch.cumsum(F.softplus(y[..., 1:]), dim=-1)], dim=-1)
            y = y.permute(3, 0, 2, 1)  # (Q, B, H, C): RevIN's (B, 1, C) statistics broadcast over Q
            if self.revin is not None:
                y = self.revin(y, "denorm")
            return y.permute(1, 2, 3, 0)  # (B, H, C, Q)
        y = y[..., 0].transpose(1, 2)  # (B, H, C)
        if self.revin is not None:
            y = self.revin(y, "denorm")
        return y

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
