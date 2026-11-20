"""SDWPF splits, training subsets and evaluation windows, shared by every model.

Everything here is a function of step indices only (the grid is complete: 134 turbines x 35,280 steps), so
the splits need no data and are saved as a small JSON (`results/ds02_splits.json`) that anyone can load.
`step` is the 10-minute index from `wpf.data.sdwpf`; a window is (turbine, origin) where the history is
steps [origin - L, origin) and the target is steps [origin, origin + H).

Technical decisions (also listed in the DS-02 pull request):
- Boundaries are day-aligned, 171 / 25 / 49 days (the issue's "about 171/25/49"): train [0, 24624),
  val [24624, 28224), test [28224, 35280). Exact 70/10/20 would cut a day in half; this does not.
- A training subset is the most recent n = round(fraction x 24624) steps of train, so 5% / 25% / 100% are
  1231 / 6156 / 24624 steps, nested, and all end where val starts.
- Evaluation origins are every 36 steps (6 h) starting exactly at the split start. One origin set is used
  for every horizon, sized for the longest (H = 64): 99 in val, 195 in test. Shorter horizons lose the few
  late origins they could have used, in exchange for the same windows at every horizon.
- "Validation" means exactly those 99 shared origins x 134 turbines, for early stopping, model selection
  (PT-03) and the reported val metrics alike, so they are the same numbers. It is not every stride-1 val window.
- MASE scale: the denominator uses the full train series [0, 24624) for every model, including the 5% and
  25% ones, so MASE stays comparable across subsets. DS-04 reads the range from `mase_train_range` in the JSON.
- Evaluation history may reach back into the previous split (the first target starts exactly at the
  boundary); targets never cross a split boundary, so there is no leakage and no window lost at the start.
- Training windows of a subset lie fully inside the subset: no lookback before its start, stride 1 by default
  (a model may raise the stride to cut cost). A 5% model therefore sees only 5% of the data.
- `windows_at` copies (fancy indexing): use it for evaluation and coverage only (13k val / 26k test windows).
  Training loaders must index lazily per batch, as `etth1.WindowDataset` does with `sliding_window_view`:
  100% of train at stride 1 is 3.25M windows, about 3.75 GB per feature if materialized.
- The index keeps every origin. Windows are not dropped for missing data: metrics mask per reading and per
  horizon (DS-04). What each window contains is saved per (turbine, origin) in `window_counts`: valid target
  steps per horizon, valid history steps, and observed (not missing) history steps, so the analysis can
  filter or stratify instead of guessing. Flagged (unknown/abnormal) rows stay in input history as real data
  (DS-01 decision); only missing rows are gap-filled, which is why "observed" is saved next to "valid".
- Coverage is also reported per turbine (valid fraction and longest invalid run) for val, test and each
  training subset, since turbines offline for weeks matter most for the 8.5-day 5% subset.

Usage:
    python -m wpf.data.splits --out results/ds02_splits.json \\
        [--clean data/sdwpf/clean.parquet --coverage results/ds02_window_coverage.json \\
         --window-counts results/ds02_window_counts.npz]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
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


def mase_train_range(split: Split) -> tuple[int, int]:
    """Steps of the series behind the MASE denominator: the full train period for every model and subset."""
    return split.train


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
    """History and target of every turbine at the given origins. For evaluation and coverage only.

    `dense` is (turbines, steps), e.g. from `sdwpf.to_dense`. Returns x of shape (turbines, origins, LOOKBACK)
    and y of shape (turbines, origins, horizon). Use it on the series and on the valid mask alike.
    This copies: it is fine for the 13k / 26k evaluation windows but not for training (100% of train at
    stride 1 is 3.25M windows, about 3.75 GB per feature). Training loaders index lazily per batch.
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
        "mase_train_range": list(mase_train_range(split)),
        "train_subsets": {str(f): list(subset_range(split, f)) for f in SUBSET_FRACTIONS},
        "eval_origins": {p: eval_origins(split, p).tolist() for p in ("val", "test")},
    }


def dump_json(data: dict) -> str:
    """JSON with every list of plain numbers on one line, so long origin and per-turbine lists stay readable."""
    text = json.dumps(data, indent=1)
    return re.sub(r"\[[^\[\]{}]*\]", lambda m: " ".join(m.group(0).split()).replace("[ ", "[").replace(" ]", "]"), text)


def save_splits(path: str | Path, split: Split) -> None:
    _LOG.debug("path='%s'", path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(dump_json(to_dict(split)) + "\n")


def load_splits(path: str | Path) -> Split:
    """Read a saved split and check it still matches what the code regenerates."""
    _LOG.debug("path='%s'", path)
    saved = json.loads(Path(path).read_text())
    split = make_split(tuple(saved["days"]))
    if saved != to_dict(split):
        raise ValueError(f"{path} does not match the splits defined in wpf.data.splits")
    return split


def window_counts(valid: np.ndarray, missing: np.ndarray, split: Split) -> dict[str, np.ndarray]:
    """Per (turbine, origin) counts for val and test, each of shape (turbines, origins).

    `valid` and `missing` are the (turbines, steps) bool masks from DS-01. Keys, for period in val/test:
    `{period}_origins` (origins, shape (origins,)), `{period}_target_valid_h{H}` (valid target steps out of H),
    `{period}_history_valid` (valid steps out of LOOKBACK) and `{period}_history_observed` (steps that are not
    missing, i.e. real inputs, out of LOOKBACK).
    """
    _LOG.debug("valid.shape=%s missing.shape=%s", valid.shape, missing.shape)
    out: dict[str, np.ndarray] = {}
    for period in ("val", "test"):
        origins = eval_origins(split, period)
        out[f"{period}_origins"] = origins
        for horizon in HORIZONS:
            _, y = windows_at(valid, origins, horizon)
            out[f"{period}_target_valid_h{horizon}"] = y.sum(axis=2).astype(np.uint8)
        x_valid, _ = windows_at(valid, origins, HORIZONS[0])  # history does not depend on the horizon
        x_missing, _ = windows_at(missing, origins, HORIZONS[0])
        out[f"{period}_history_valid"] = x_valid.sum(axis=2).astype(np.uint16)
        out[f"{period}_history_observed"] = (LOOKBACK - x_missing.sum(axis=2)).astype(np.uint16)
    return out


def window_coverage(counts: dict[str, np.ndarray]) -> dict:
    """Summary of `window_counts` per period and horizon: windows, valid share of target steps, windows with
    no valid target, windows with every target step valid, and the valid / observed share of history steps."""
    out: dict = {}
    for period in ("val", "test"):
        out[period] = {}
        history_valid = counts[f"{period}_history_valid"]
        history_observed = counts[f"{period}_history_observed"]
        for horizon in HORIZONS:
            target = counts[f"{period}_target_valid_h{horizon}"]
            out[period][str(horizon)] = {
                "windows": int(target.size),
                "valid_target_fraction": float(target.sum() / (target.size * horizon)),
                "windows_without_valid_target": int((target == 0).sum()),
                "windows_fully_valid_target": int((target == horizon).sum()),
                "valid_history_fraction": float(history_valid.sum() / (history_valid.size * LOOKBACK)),
                "observed_history_fraction": float(history_observed.sum() / (history_observed.size * LOOKBACK)),
            }
    return out


def _longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of True in a 1-D bool array."""
    if not mask.any():
        return 0
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return int((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).max())


def turbine_coverage(valid: np.ndarray, split: Split, turbine_ids: np.ndarray) -> dict:
    """Valid fraction and longest invalid run (steps) per turbine, for val, test and each training subset.

    The farm-wide share hides turbines that are offline for whole weeks, which matters most for the 5% subset.
    Rows of `valid` are the turbines in `turbine_ids` order.
    """
    _LOG.debug("valid.shape=%s", valid.shape)
    ranges = {"val": split.val, "test": split.test}
    ranges.update({f"train_{round(f * 100)}pct": subset_range(split, f) for f in SUBSET_FRACTIONS})
    out = {}
    for name, (start, end) in ranges.items():
        segment = valid[:, start:end]
        out[name] = {
            "TurbID": [int(t) for t in turbine_ids],
            "valid_fraction": segment.mean(axis=1).round(6).tolist(),
            "longest_invalid_run_steps": [_longest_run(~row) for row in segment],
        }
    return out


def save_window_counts(path: str | Path, counts: dict[str, np.ndarray]) -> None:
    _LOG.debug("path='%s'", path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **counts)


def load_window_counts(path: str | Path) -> dict[str, np.ndarray]:
    _LOG.debug("path='%s'", path)
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the SDWPF split boundaries and window index.")
    parser.add_argument("--out", type=Path, default=Path("results/ds02_splits.json"))
    parser.add_argument("--clean", type=Path, default=None, help="DS-01 parquet; adds the coverage reports")
    parser.add_argument("--coverage", type=Path, default=Path("results/ds02_window_coverage.json"))
    parser.add_argument("--window-counts", type=Path, default=Path("results/ds02_window_counts.npz"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    split = make_split()
    save_splits(args.out, split)
    _LOG.info("wrote %s", args.out)
    if args.clean is not None:
        from wpf.data import sdwpf  # lazy: pandas/pyarrow are only needed for the reports

        df = sdwpf.load_clean(args.clean)
        valid, missing = sdwpf.to_dense(df, "valid"), sdwpf.to_dense(df, "missing")
        if valid.shape[1] != split.n_steps:
            raise ValueError(f"{args.clean} has {valid.shape[1]} steps per turbine, the split expects {split.n_steps}")
        counts = window_counts(valid, missing, split)
        coverage = {
            "windows": window_coverage(counts),
            "per_turbine": turbine_coverage(valid, split, np.sort(df["TurbID"].unique())),
        }
        args.coverage.parent.mkdir(parents=True, exist_ok=True)
        args.coverage.write_text(dump_json(coverage) + "\n")
        save_window_counts(args.window_counts, counts)
        _LOG.info("wrote %s and %s", args.coverage, args.window_counts)
        print(dump_json(coverage["windows"]))


if __name__ == "__main__":
    main()
