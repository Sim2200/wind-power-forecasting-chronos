"""PatchTST with spatial context (PT-05): the centre turbine's patches and its k nearest neighbours' patches in one encoder.

Flow for an input of shape (B, L, 1 + k), channel 0 the centre turbine:
  RevIN norm per channel -> patch every channel -> (B, C, N, P) -> linear patch embedding
  + positional encoding (per patch position) + turbine-slot embedding (per channel: centre, 1st neighbour, ...)
  -> one sequence of C x N tokens -> the PatchTST encoder (post-norm BatchNorm, residual attention)
  -> the centre's N tokens -> flatten -> linear head -> (B, H, 1), or (B, H, 1, Q) non-crossing quantiles
  -> RevIN denorm with the centre channel's statistics.

Every neighbour token can attend to every centre token and vice versa, so the model sees what the wind did at the
neighbours over the same two days. With k = 0 it is plain channel-independent PatchTST on one turbine. The encoder,
the patching, the head and the quantile construction are the PatchTST ones (`wpf.models.patchtst`).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from wpf.models.layers import Patcher, RevIN, num_patches
from wpf.models.patchtst import EncoderLayer, PatchTSTConfig

_LOG = logging.getLogger(__name__)


@dataclass
class SpatialPatchTSTConfig(PatchTSTConfig):
    """`n_vars` is 1 + k: the centre turbine plus its k neighbours. RevIN affine is not supported (the centre's statistics
    are sliced out for the denormalization)."""

    seq_len: int = 288
    pred_len: int = 64
    n_vars: int = 5


class SpatialPatchTST(nn.Module):
    def __init__(self, cfg: SpatialPatchTSTConfig):
        super().__init__()
        _LOG.debug("cfg=%s", cfg)
        if cfg.revin_affine:
            raise ValueError("SpatialPatchTST denormalizes with the centre channel's statistics; revin_affine is not supported")
        self.cfg = cfg
        self.revin = RevIN(cfg.n_vars, affine=False) if cfg.revin else None
        self.patcher = Patcher(cfg.patch_len, cfg.stride)
        self.n_patches = num_patches(cfg.seq_len, cfg.patch_len, cfg.stride)
        self.per_step = max(cfg.n_quantiles, 1)
        self.embed = nn.Linear(cfg.patch_len, cfg.d_model)
        self.pos = nn.Parameter(torch.empty(self.n_patches, cfg.d_model).uniform_(-0.02, 0.02))
        self.slot = nn.Parameter(torch.empty(cfg.n_vars, cfg.d_model).uniform_(-0.02, 0.02))
        self.embed_dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(EncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_layers))
        self.head = nn.Sequential(nn.Flatten(start_dim=-2), nn.Linear(cfg.d_model * self.n_patches, cfg.pred_len * self.per_step),
                                  nn.Dropout(cfg.head_dropout))
        _LOG.debug("tokens=%d parameters=%d", cfg.n_vars * self.n_patches, self.num_parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, 1 + k) -> (B, H, 1), or (B, H, 1, Q) with `n_quantiles` = Q > 0; the forecast is for channel 0."""
        b = x.shape[0]
        if self.revin is not None:
            x = self.revin(x, "norm")
        z = self.embed(self.patcher(x.transpose(1, 2)))  # (B, C, N, D)
        z = z + self.pos[None, None] + self.slot[None, :, None]
        z = self.embed_dropout(z.reshape(b, self.cfg.n_vars * self.n_patches, self.cfg.d_model))
        scores = None
        for layer in self.layers:
            z, scores = layer(z, scores)
        centre = z[:, : self.n_patches]  # the centre turbine's tokens come first
        y = self.head(centre.transpose(1, 2)).view(b, self.cfg.pred_len, self.per_step)  # (B, H, Q or 1)
        if self.cfg.n_quantiles:
            y = torch.cat([y[..., :1], y[..., :1] + torch.cumsum(F.softplus(y[..., 1:]), dim=-1)], dim=-1)
        if self.revin is not None:  # the centre's (B, 1, 1) statistics broadcast over H and Q
            y = y * self.revin.stdev[:, :, :1] + self.revin.mean[:, :, :1]
        return y[:, :, None, :] if self.cfg.n_quantiles else y  # channel axis: (B, H, 1, Q), or the (B, H, 1) point forecast

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
