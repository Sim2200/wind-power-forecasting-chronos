"""Spatial context for PatchTST (PT-05): each turbine's window plus the windows of its k nearest neighbours.

`nearest_neighbours` ranks the other turbines by Euclidean distance on the SDWPF layout (`turbine_locations.parquet`,
rows in TurbID order like the dense arrays). `SpatialWindows` is `SDWPFWindows` with x of shape (L, 1 + k): channel 0 is
the centre turbine, channels 1..k its neighbours from nearest to farthest; y and the mask stay the centre's. `eval_arrays`
gives the shared evaluation windows in the same layout, (turbines, origins, L, 1 + k).
"""
from __future__ import annotations

import logging

import numpy as np

from wpf.data import sdwpf_windows, splits

_LOG = logging.getLogger(__name__)


def nearest_neighbours(x: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """(turbines, k) row indices of the k nearest other turbines, nearest first, from coordinates in row order."""
    _LOG.debug("n=%d k=%d", len(x), k)
    if k < 1 or k >= len(x):
        raise ValueError(f"k must be between 1 and {len(x) - 1}, got {k}")
    xy = np.stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)], axis=1)
    dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    return np.argsort(dist, axis=1, kind="stable")[:, :k]


class SpatialWindows(sdwpf_windows.SDWPFWindows):
    """`SDWPFWindows` whose x holds the centre turbine and its neighbours as channels, (L, 1 + k)."""

    def __init__(self, series: np.ndarray, valid: np.ndarray, index: np.ndarray, horizon: int, neighbours: np.ndarray):
        super().__init__(series, valid, index, horizon)
        if neighbours.shape[0] != series.shape[0]:
            raise ValueError("neighbours must have one row per turbine")
        self.neighbours = neighbours

    def __getitem__(self, i: int):
        import torch

        turbine, origin = self.index[i]
        rows = np.concatenate([[turbine], self.neighbours[turbine]])
        x = self.series[rows, origin - splits.LOOKBACK : origin].T  # (L, 1 + k)
        y = self.series[turbine, origin : origin + self.horizon]
        mask = self.valid[turbine, origin : origin + self.horizon]
        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(y[:, None].copy()),
            torch.from_numpy(mask[:, None].astype(np.float32)),
        )


def eval_arrays(series: np.ndarray, valid: np.ndarray, split: splits.Split, period: str, horizon: int,
                neighbours: np.ndarray) -> dict[str, np.ndarray]:
    """The shared evaluation windows with neighbour channels: x (turbines, origins, L, 1 + k), y and mask (turbines, origins, H)."""
    _LOG.debug("period='%s' k=%d", period, neighbours.shape[1])
    base = sdwpf_windows.eval_arrays(series, valid, split, period, horizon)
    rows = np.concatenate([np.arange(series.shape[0])[:, None], neighbours], axis=1)  # (turbines, 1 + k)
    x = base["x"][rows]  # (turbines, 1 + k, origins, L)
    return {**base, "x": np.ascontiguousarray(x.transpose(0, 2, 3, 1))}
