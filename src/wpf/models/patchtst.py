"""PatchTST (Nie et al., ICLR 2023), following the official supervised code.

Flow for an input of shape (B, L, C):
  RevIN norm -> (B, C, L) -> end-pad + patch -> (B, C, N, P)
  -> linear patch embedding + learnable positional encoding -> (B*C, N, D)
  -> transformer encoder (post-norm with BatchNorm, residual attention, GELU)
  -> (B, C, D*N) flatten -> linear head -> (B, C, H) -> (B, H, C) -> RevIN denorm

Official ETTh1 setting: L=336, P=16, stride 8 (N=42), D=16, 4 heads, d_ff=128,
3 layers, dropout 0.3, head_dropout 0, RevIN without affine.
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
        self.embed = nn.Linear(cfg.patch_len, cfg.d_model)
        self.pos = nn.Parameter(torch.empty(self.n_patches, cfg.d_model).uniform_(-0.02, 0.02))
        self.embed_dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(
            EncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_layers)
        )
        # The flattened (D x N) size fixes the input length: seq_len cannot change after construction.
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2), nn.Linear(cfg.d_model * self.n_patches, cfg.pred_len), nn.Dropout(cfg.head_dropout)
        )
        _LOG.debug("n_patches=%s parameters=%s", self.n_patches, self.num_parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, C) -> (B, H, C)."""
        if self.revin is not None:
            x = self.revin(x, "norm")
        z = self.patcher(x.transpose(1, 2))  # (B, C, N, P)
        z = self.embed(z)  # (B, C, N, D)
        z = merge_channels(z)  # (B*C, N, D): channel independence
        z = self.embed_dropout(z + self.pos)
        scores = None
        for layer in self.layers:
            z, scores = layer(z, scores)
        z = split_channels(z, self.cfg.n_vars)  # (B, C, N, D)
        y = self.head(z.transpose(2, 3))  # flatten (D, N) -> Linear -> (B, C, H)
        y = y.transpose(1, 2)  # (B, H, C)
        if self.revin is not None:
            y = self.revin(y, "denorm")
        return y

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
