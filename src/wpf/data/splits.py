"""SDWPF splits, training subsets and evaluation windows, shared by every model.

Everything here is a function of step indices only (the grid is complete: 134 turbines x 35,280 steps), so
the splits need no data and are saved as a small JSON (`results/ds02_splits.json`) that anyone can load.
`step` is the 10-minute index from `wpf.data.sdwpf`; a window is (turbine, origin) where the history is
steps [origin - L, origin) and the target is steps [origin, origin + H).

Technical decisions (also listed in the DS-02 pull request):
- Boundaries are day-aligned, 171 / 25 / 49 days (the issue's "about 171/25/49"): train [0, 24624),
  val [24624, 28224), test [28224, 35280). Exact 70/10/20 would cut a day in half; this does not.
  The MASE train series is steps [0, 24624).
- A training subset is the most recent n = round(fraction x 24624) steps of train, so 5% / 25% / 100% are
  1231 / 6156 / 24624 steps, nested, and all end where val starts.
- Evaluation origins are every 36 steps (6 h) starting exactly at the split start. One origin set is used
  for every horizon, sized for the longest (H = 64): 99 in val, 195 in test. Shorter horizons lose the few
  late origins they could have used, in exchange for the same windows at every horizon.
- Evaluation history may reach back into the previous split (the first target starts exactly at the
  boundary); targets never cross a split boundary, so there is no leakage and no window lost at the start.
- Training windows of a subset lie fully inside the subset: no lookback before its start, stride 1 by default
  (a model may raise the stride to cut cost). A 5% model therefore sees only 5% of the data.
- The index keeps every origin. Windows are not dropped for missing data: metrics mask per reading and per
  horizon (DS-04), and `window_coverage` reports how many windows have no valid target.

Usage:
    python -m wpf.data.splits --out results/ds02_splits.json \\
        [--clean data/sdwpf/clean.parquet --coverage results/ds02_window_coverage.json]
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_LOG = logging.getLogger(__name__)

SLOTS_PER_DAY = 144  # 10-minute readings
DAYS = (171, 25, 49)  # train / val / test
LOOKBACK = 288  # L: two days of history
HORIZONS = (6, 36, 64)  # H: 1 h, 6 h, about 10.7 h
ORIGIN_STRIDE = 36  # steps between forecast origins in val/test (6 h)
SUBSET_FRACTIONS = (0.05, 0.25, 1.0)  # most recent share of train


@dataclass(frozen=True)
class Split:
    """Step ranges [start, end) of the three periods."""

    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]

    @property
    def n_steps(self) -> int:
        return self.test[1]

    def period(self, name: str) -> tuple[int, int]:
        if name not in ("train", "val", "test"):
            raise ValueError(f"unknown period '{name}'")
        return getattr(self, name)


def make_split(days: tuple[int, int, int] = DAYS) -> Split:
    """Day-aligned chronological split into train / val / test."""
    _LOG.debug("days=%s", days)
    train_end = days[0] * SLOTS_PER_DAY
    val_end = train_end + days[1] * SLOTS_PER_DAY
    test_end = val_end + days[2] * SLOTS_PER_DAY
    return Split(train=(0, train_end), val=(train_end, val_end), test=(val_end, test_end))


def subset_range(split: Split, fraction: float) -> tuple[int, int]:
    """The most recent `fraction` of the training steps, ending where train ends."""
    _LOG.debug("fraction=%s", fraction)
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    train_end = split.train[1]
    return train_end - round(fraction * train_end), train_end


def eval_origins(split: Split, period: str, max_horizon: int = max(HORIZONS)) -> np.ndarray:
    """Forecast origins of val or test: every ORIGIN_STRIDE steps from the period start, with the
    longest horizon still inside the period."""
    _LOG.debug("period='%s' max_horizon=%s", period, max_horizon)
    if period not in ("val", "test"):
        raise ValueError("evaluation periods are 'val' and 'test'")
    start, end = split.period(period)
    return np.arange(start, end - max_horizon + 1, ORIGIN_STRIDE)


def train_origins(split: Split, fraction: float, horizon: int, stride: int = 1) -> np.ndarray:
    """Origins of training windows lying fully inside the subset (history and target both)."""
    _LOG.debug("fraction=%s horizon=%s stride=%s", fraction, horizon, stride)
    start, end = subset_range(split, fraction)
    return np.arange(start + LOOKBACK, end - horizon + 1, stride)


def windows_at(dense: np.ndarray, origins: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """History and target of every turbine at the given origins.

    `dense` is (turbines, steps), e.g. from `sdwpf.to_dense`. Returns x of shape (turbines, origins, LOOKBACK)
    and y of shape (turbines, origins, horizon). Use it on the series and on the valid mask alike.
    """
    _LOG.debug("dense.shape=%s n_origins=%d horizon=%s", dense.shape, len(origins), horizon)
    origins = np.asarray(origins)
    if origins.min() < LOOKBACK or origins.max() + horizon > dense.shape[1]:
        raise ValueError("an origin's history or target falls outside the series")
    x = dense[:, origins[:, None] + np.arange(-LOOKBACK, 0)]
    y = dense[:, origins[:, None] + np.arange(horizon)]
    return x, y


def to_dict(split: Split) -> dict:
    """Everything a consumer needs, as plain JSON-able data."""
    return {
        "lookback": LOOKBACK,
        "horizons": list(HORIZONS),
        "origin_stride": ORIGIN_STRIDE,
        "days": [(e - b) // SLOTS_PER_DAY for b, e in (split.train, split.val, split.test)],
        "split": {p: list(split.period(p)) for p in ("train", "val", "test")},
        "train_subsets": {str(f): list(subset_range(split, f)) for f in SUBSET_FRACTIONS},
        "eval_origins": {p: eval_origins(split, p).tolist() for p in ("val", "test")},
    }


def save_splits(path: str | Path, split: Split) -> None:
    _LOG.debug("path='%s'", path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(to_dict(split), indent=1) + "\n")


def load_splits(path: str | Path) -> Split:
    """Read a saved split and check it still matches what the code regenerates."""
    _LOG.debug("path='%s'", path)
    saved = json.loads(Path(path).read_text())
    split = make_split(tuple(saved["days"]))
    if saved != to_dict(split):
        raise ValueError(f"{path} does not match the splits defined in wpf.data.splits")
    return split


def window_coverage(valid: np.ndarray, split: Split) -> dict:
    """Per period and horizon: how many windows, share of valid target readings, windows with no valid target.

    `valid` is the (turbines, steps) bool mask. Target readings are what metrics score; the history share
    shows how much of each input is real data rather than gap.
    """
    _LOG.debug("valid.shape=%s", valid.shape)
    out: dict = {}
    for period in ("val", "test"):
        origins = eval_origins(split, period)
        out[period] = {}
        for horizon in HORIZONS:
            x, y = windows_at(valid, origins, horizon)
            per_window = y.sum(axis=2)
            out[period][str(horizon)] = {
                "windows": int(per_window.size),
                "valid_target_fraction": float(y.mean()),
                "windows_without_valid_target": int((per_window == 0).sum()),
                "windows_fully_valid_target": int((per_window == horizon).sum()),
                "valid_history_fraction": float(x.mean()),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the SDWPF split boundaries and window index.")
    parser.add_argument("--out", type=Path, default=Path("results/ds02_splits.json"))
    parser.add_argument("--clean", type=Path, default=None, help="DS-01 parquet; adds the coverage report")
    parser.add_argument("--coverage", type=Path, default=Path("results/ds02_window_coverage.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    split = make_split()
    save_splits(args.out, split)
    _LOG.info("wrote %s", args.out)
    if args.clean is not None:
        from wpf.data import sdwpf  # lazy: pandas/pyarrow are only needed for the report

        valid = sdwpf.to_dense(sdwpf.load_clean(args.clean), "valid")
        coverage = window_coverage(valid, split)
        args.coverage.parent.mkdir(parents=True, exist_ok=True)
        args.coverage.write_text(json.dumps(coverage, indent=1) + "\n")
        _LOG.info("wrote %s", args.coverage)
        print(json.dumps(coverage, indent=1))


if __name__ == "__main__":
    main()
