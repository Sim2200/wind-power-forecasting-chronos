"""PatchTST input layers: RevIN, channel independence and patching.

Matches the official PatchTST code (layers/RevIN.py and layers/PatchTST_backbone.py):
- RevIN normalizes each series of each sample by its own mean and std over time
  (biased variance, eps added before the square root, statistics detached),
  with an optional learnable per-channel affine transform.
- Patching pads the end of the series by repeating the last value `stride` times,
  then cuts patches of length `patch_len` every `stride` steps.
- Channel independence: every variable is treated as its own series, so the
  batch and variable dimensions are merged before the transformer and split after.

Shapes use: B = batch, L = input length, C = number of variables (channels),
N = number of patches, P = patch length, D = model dimension.
"""
from __future__ import annotations

import torch
from torch import nn


class RevIN(nn.Module):
    """Reversible instance normalization over the time axis of a (B, L, C) tensor."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True, subtract_last: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise ValueError(f"mode must be 'norm' or 'denorm', got {mode!r}")

    def _get_statistics(self, x: torch.Tensor) -> None:
        dims = tuple(range(1, x.ndim - 1))  # every dim except batch and channel
        if self.subtract_last:
            self.last = x[:, -1:, :].detach()
        else:
            self.mean = torch.mean(x, dim=dims, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dims, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = x - (self.last if self.subtract_last else self.mean)
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)  # as in the official code
        x = x * self.stdev
        return x + (self.last if self.subtract_last else self.mean)


def num_patches(seq_len: int, patch_len: int, stride: int, pad_end: bool = True) -> int:
    """Number of patches for an input of `seq_len` steps (42 for 336 / 16 / 8 with end padding)."""
    n = (seq_len - patch_len) // stride + 1
    return n + 1 if pad_end else n


class Patcher(nn.Module):
    """(B, C, L) -> (B, C, N, P): end-pad by repeating the last value, then unfold."""

    def __init__(self, patch_len: int = 16, stride: int = 8, pad_end: bool = True):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.pad = nn.ReplicationPad1d((0, stride)) if pad_end else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad is not None:
            x = self.pad(x)
        return x.unfold(dimension=-1, size=self.patch_len, step=self.stride)


def merge_channels(z: torch.Tensor) -> torch.Tensor:
    """Channel independence, in: (B, C, N, D) -> (B*C, N, D), one sequence per variable."""
    b, c, n, d = z.shape
    return z.reshape(b * c, n, d)


def split_channels(z: torch.Tensor, n_vars: int) -> torch.Tensor:
    """Inverse of `merge_channels`: (B*C, N, D) -> (B, C, N, D)."""
    bc, n, d = z.shape
    return z.reshape(bc // n_vars, n_vars, n, d)
