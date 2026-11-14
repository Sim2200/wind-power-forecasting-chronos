"""PatchTST (Nie et al., ICLR 2023), following the official supervised code.

Flow for an input of shape (B, L, C):
  RevIN norm -> (B, C, L) -> end-pad + patch -> (B, C, N, P)
  -> linear patch embedding + learnable positional encoding -> (B*C, N, D)
  -> transformer encoder (post-norm with BatchNorm, residual attention, GELU)
  -> (B, C, D*N) flatten -> linear head -> (B, C, H) -> (B, H, C) -> RevIN denorm

Official ETTh1 setting: L=336, P=16, stride 8 (N=42), D=16, 4 heads, d_ff=128,
3 layers, dropout 0.3, head_dropout 0, RevIN without affine.

Quantile head (PT-02): with `n_quantiles` = Q > 0 the linear head emits Q values per step, (B, H, C, Q). The
first is the lowest quantile and the others are softplus increments added cumulatively, so the quantiles
never cross; RevIN denorm is applied to every quantile with the window's statistics. Q = 0 keeps the
point head and the ETTh1 behaviour unchanged.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from wpf.models.layers import Patcher, RevIN, merge_channels, num_patches, split_channels

_LOG = logging.getLogger(__name__)


@dataclass
class PatchTSTConfig:
    seq_len: int = 336
    pred_len: int = 96
    n_vars: int = 7
    patch_len: int = 16
    stride: int = 8
    d_model: int = 16
    n_heads: int = 4
    d_ff: int = 128
    n_layers: int = 3
    dropout: float = 0.3
    head_dropout: float = 0.0
    revin: bool = True
    revin_affine: bool = False  # the official ETTh1 runs use `--affine 0`
    # Ablation (paper Table 6): False mixes channels, i.e. one token holds all C variables of a patch.
    channel_independent: bool = True
    n_quantiles: int = 0  # 0: point forecast (B, H, C); Q > 0: non-crossing quantiles (B, H, C, Q)


class MultiheadAttention(nn.Module):
    """Multi-head self-attention with residual attention (scores passed between layers)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head**-0.5
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.to_out = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, prev_scores: torch.Tensor | None = None):
        b, n, _ = x.shape
        q = self.w_q(x).view(b, n, self.n_heads, self.d_head).transpose(1, 2)  # (B, h, N, d)
        k = self.w_k(x).view(b, n, self.n_heads, self.d_head).permute(0, 2, 3, 1)  # (B, h, d, N)
        v = self.w_v(x).view(b, n, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k) * self.scale  # (B, h, N, N)
        # Residual attention: the previous layer's pre-softmax scores are added to this layer's.
        if prev_scores is not None:
            scores = scores + prev_scores
        out = torch.matmul(F.softmax(scores, dim=-1), v)  # (B, h, N, d)
        out = out.transpose(1, 2).reshape(b, n, -1)
        # The pre-softmax scores go back out so the next layer can add them.
        return self.to_out(out), scores


class BatchNormOverTokens(nn.Module):
    """BatchNorm1d applied to the feature dim of a (B, N, D) tensor, as in the official code."""

    def __init__(self, d_model: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


class EncoderLayer(nn.Module):
    """Post-norm transformer encoder layer: attention + FFN, each with residual, dropout and BatchNorm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = MultiheadAttention(d_model, n_heads, dropout)
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = BatchNormOverTokens(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = BatchNormOverTokens(d_model)

    def forward(self, x: torch.Tensor, prev_scores: torch.Tensor | None = None):
        attn_out, scores = self.attn(x, prev_scores)
        x = self.norm_attn(x + self.dropout_attn(attn_out))
        x = self.norm_ffn(x + self.dropout_ffn(self.ff(x)))
        return x, scores


class PatchTST(nn.Module):
    def __init__(self, cfg: PatchTSTConfig):
        super().__init__()
        _LOG.debug("cfg=%s", cfg)
        self.cfg = cfg
        self.revin = RevIN(cfg.n_vars, affine=cfg.revin_affine) if cfg.revin else None
        self.patcher = Patcher(cfg.patch_len, cfg.stride)
        self.n_patches = num_patches(cfg.seq_len, cfg.patch_len, cfg.stride)
        token_size = cfg.patch_len if cfg.channel_independent else cfg.patch_len * cfg.n_vars
        self.per_step = max(cfg.n_quantiles, 1)
        head_out = cfg.pred_len * self.per_step * (1 if cfg.channel_independent else cfg.n_vars)
        self.embed = nn.Linear(token_size, cfg.d_model)
        self.pos = nn.Parameter(torch.empty(self.n_patches, cfg.d_model).uniform_(-0.02, 0.02))
        self.embed_dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(
            EncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_layers)
        )
        # The flattened (D x N) size fixes the input length: seq_len cannot change after construction.
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2), nn.Linear(cfg.d_model * self.n_patches, head_out), nn.Dropout(cfg.head_dropout)
        )
        _LOG.debug("n_patches=%s parameters=%s", self.n_patches, self.num_parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, C) -> (B, H, C), or (B, H, C, Q) with `n_quantiles` = Q > 0."""
        if self.revin is not None:
            x = self.revin(x, "norm")
        z = self.patcher(x.transpose(1, 2))  # (B, C, N, P)
        if self.cfg.channel_independent:
            z = merge_channels(self.embed(z))  # (B*C, N, D): every channel is its own sequence
        else:
            z = self.embed(z.permute(0, 2, 1, 3).flatten(-2))  # (B, N, C*P) -> (B, N, D)
        z = self.embed_dropout(z + self.pos)
        scores = None
        for layer in self.layers:
            z, scores = layer(z, scores)
        if self.cfg.channel_independent:
            z = split_channels(z, self.cfg.n_vars)  # (B, C, N, D)
            y = self.head(z.transpose(2, 3))  # flatten (D, N) -> Linear -> (B, C, H * per_step)
        else:
            y = self.head(z.transpose(1, 2))  # (B, C * H * per_step)
        y = y.view(-1, self.cfg.n_vars, self.cfg.pred_len, self.per_step)  # (B, C, H, Q or 1)
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
