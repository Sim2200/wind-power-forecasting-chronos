"""Export the SDWPF training subsets as GluonTS Arrow files for Chronos' `scripts/training/train.py`.

One file per subset (5% / 25% / 100% of train, `wpf.data.splits.subset_range`), one series per turbine, each
entry `{"start", "target"}` like Chronos' own `convert_to_arrow` helper. `train.py` reads every path with
`FileDataset(path, freq="h")` and uses only `start` and `target` (1-D). The `.arrow` files stay in `data/`
(gitignored); `results/ds03_chronos_export.json` records what was written.

Technical decisions (also listed in the DS-03 pull request):
- Target is `Patv` (the clamped target scale from DS-01), NaN only where `missing` (Patv NaN). Unknown and
  abnormal rows keep their values. DS-01 keeps flagged rows as real history and evaluation contexts contain
  them, so training on the same kind of context avoids a train/eval mismatch. Effect: `train.py` cannot mask
  the loss separately from the context (a NaN does both: ignored by the tokenizer and set to -100 in the
  labels), so Chronos' loss includes flagged rows while PatchTST's masks them. Metrics still score valid
  readings only.
- Exactly the DS-02 subsets: 1231 / 6156 / 24624 steps, nested, ending at step 24,624. No val or test export:
  evaluation uses the DS-02 windows on the dense arrays, and a second Arrow window definition would invite drift.
- Chronos must be run with `min_past = 288` (and context 288, prediction 64). With its default `min_past = 64`
  it trains on windows with 64-287 steps of padded history. 288 matches DS-02's rule that a training window has
  the full L = 288 inside the subset: Chronos then samples the same origins as `splits.train_origins(split,
  fraction, 64)`. The arguments are stored in the manifest for CF-01 / CF-02.
- `start` is synthetic (`sdwpf.TS_ANCHOR` + step x 10 min). `train.py` hardcodes `freq="h"` and floors it;
  Chronos uses no time features, so it only labels the series. Do not read it as a calendar date.
- float32 and lz4, the Chronos helper's defaults. `gluonts` is imported only to write or read these files.
  It needs Python <= 3.13 (its pydantic v1 layer fails to import on 3.14); Zaratan's 3.10 is fine.

Usage:
    python -m wpf.data.chronos_arrow --clean data/sdwpf/clean.parquet --out-dir data/sdwpf/chronos
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from wpf.data import sdwpf, splits

_LOG = logging.getLogger(__name__)

# What train.py must be given so its sampling matches DS-02's training windows.
CHRONOS_TRAIN_ARGS = {
    "context_length": splits.LOOKBACK,
    "prediction_length": max(splits.HORIZONS),
    "min_past": splits.LOOKBACK,
}


def subset_name(fraction: float) -> str:
    return f"train_{round(fraction * 100)}pct"


def subset_series(patv: np.ndarray, split: splits.Split, fraction: float) -> np.ndarray:
    """`Patv` of every turbine over the most recent `fraction` of train, as a (turbines, steps) float32 array.

    `patv` is the dense (turbines, steps) target from `sdwpf.to_dense(df, "Patv")`: clamped, NaN where missing.
    """
    _LOG.debug("patv.shape=%s fraction=%s", patv.shape, fraction)
    start, end = splits.subset_range(split, fraction)
    return np.ascontiguousarray(patv[:, start:end], dtype=np.float32)


def subset_start(split: splits.Split, fraction: float) -> np.datetime64:
    """Synthetic start time of a subset: the DS-01 anchor plus its first step, in seconds."""
    first_step, _ = splits.subset_range(split, fraction)
    ts = sdwpf.TS_ANCHOR + pd.Timedelta(minutes=sdwpf.STEP_MINUTES * first_step)
    return ts.to_datetime64().astype("datetime64[s]")


def write_arrow(series: np.ndarray, start: np.datetime64, path: str | Path) -> None:
    """Write a (series, steps) array as a GluonTS Arrow file, one `{"start", "target"}` entry per row."""
    _LOG.debug("series.shape=%s start=%s path='%s'", series.shape, start, path)
    if series.ndim != 2:
        raise ValueError("series must be 2-D: (series, steps)")
    from gluonts.dataset.arrow import ArrowWriter  # lazy: only needed to write or read these files

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    dataset = [{"start": start, "target": row} for row in series]
    ArrowWriter(compression="lz4").write_to_file(dataset, path=Path(path))


def read_series(path: str | Path) -> np.ndarray:
    """Read a file back the way train.py does (`FileDataset(path, freq="h")`) as a (series, steps) array."""
    _LOG.debug("path='%s'", path)
    from gluonts.dataset.common import FileDataset

    return np.stack([np.asarray(entry["target"]) for entry in FileDataset(path=Path(path), freq="h")])


def export_subsets(patv: np.ndarray, split: splits.Split, out_dir: str | Path) -> dict:
    """Write one Arrow file per training subset into `out_dir`; returns the manifest."""
    _LOG.debug("patv.shape=%s out_dir='%s'", patv.shape, out_dir)
    files = {}
    for fraction in splits.SUBSET_FRACTIONS:
        name = f"{subset_name(fraction)}.arrow"
        series = subset_series(patv, split, fraction)
        write_arrow(series, subset_start(split, fraction), Path(out_dir) / name)
        files[name] = {
            "fraction": fraction,
            "steps": list(splits.subset_range(split, fraction)),
            "series": int(series.shape[0]),
            "length": int(series.shape[1]),
            "nan_fraction": round(float(np.isnan(series).mean()), 6),
        }
        _LOG.info("wrote %s: %s", name, files[name])
    return {"target": "Patv (clamped, NaN where missing)", "chronos_train_args": CHRONOS_TRAIN_ARGS, "files": files}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the SDWPF training subsets as Chronos/GluonTS Arrow files.")
    parser.add_argument("--clean", type=Path, default=Path("data/sdwpf/clean.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/sdwpf/chronos"))
    parser.add_argument("--manifest", type=Path, default=Path("results/ds03_chronos_export.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    patv = sdwpf.to_dense(sdwpf.load_clean(args.clean), "Patv")
    split = splits.make_split()
    if patv.shape[1] != split.n_steps:
        raise ValueError(f"{args.clean} has {patv.shape[1]} steps per turbine, the split expects {split.n_steps}")
    manifest = export_subsets(patv, split, args.out_dir)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(splits.dump_json(manifest) + "\n")
    _LOG.info("wrote %s", args.manifest)
    print(splits.dump_json(manifest))


if __name__ == "__main__":
    main()
