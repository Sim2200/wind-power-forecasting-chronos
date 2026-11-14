"""Export the SDWPF training subsets as GluonTS Arrow files for Chronos' `scripts/training/train.py`.

One file per subset (5% / 25% / 100% of train, `wpf.data.splits.subset_range`) plus `val.arrow`, one series per
turbine, each entry `{"start", "target"}` like Chronos' own `convert_to_arrow` helper. `train.py` reads every
path with `FileDataset(path, freq="h")` and uses only `start` and `target` (1-D). The `.arrow` files stay in
`data/` (gitignored); `results/ds03_chronos_export*.json` records what was written.

Technical decisions (also listed in the DS-03 pull request):
- Target is `Patv` (the clamped target scale from DS-01), NaN where `missing` (Patv NaN). `--flagged` decides
  what happens to `unknown` / `abnormal` rows (about 23%): `keep` (default) leaves their values, `nan` blanks
  them. Neither is hardcoded: `keep` matches the evaluation contexts, which contain flagged rows, but Chronos
  then learns curtailment zeros that are never scored; `nan` drops 23% of the context that PatchTST keeps.
  CF-01 runs both on the 5% file and reports the difference. `train.py` cannot mask the loss separately from
  the context (a NaN does both: ignored by the tokenizer and set to -100 in the labels), so under `keep`
  Chronos' loss includes flagged rows while PatchTST's masks them. Metrics still score valid readings only.
- Exactly the DS-02 subsets: 1231 / 6156 / 24624 steps, nested, ending at step 24,624.
- `val.arrow` is the val period plus the 288 steps before it (steps 24,336..28,224, 3888 long), in the same
  `--flagged` mode. It lets CF-01 pick a checkpoint by val loss, like PatchTST's best-val early stopping
  (Chronos' `train.py` only saves the last step). There is no test export, and DS-04 scores on the dense
  arrays with the DS-02 windows, so the Arrow files add no second window definition that could drift.
- Each mode has its own directory and manifest: `keep` -> `data/sdwpf/chronos/` and
  `results/ds03_chronos_export.json`, `nan` -> `data/sdwpf/chronos_nan/` and `results/ds03_chronos_export_nan.json`.
- Chronos must be run with `min_past = 288` (and context 288, prediction 64). With its default `min_past = 64`
  it trains on windows with 64-287 steps of padded history. 288 matches DS-02's rule that a training window has
  the full L = 288 inside the subset: Chronos then samples the same origins as `splits.train_origins(split,
  fraction, 64)`. The arguments are stored in the manifest for CF-01 / CF-02.
- `start` is synthetic (`sdwpf.TS_ANCHOR` + step x 10 min). `train.py` hardcodes `freq="h"` and floors it;
  Chronos uses no time features, so it only labels the series. Do not read it as a calendar date.
- float32 and lz4, the Chronos helper's defaults. `gluonts` is imported only to write or read these files.
  It needs Python <= 3.13 (its pydantic v1 layer fails to import on 3.14); Zaratan's 3.10 is fine.

Usage:
    python -m wpf.data.chronos_arrow [--flagged keep|nan] [--clean data/sdwpf/clean.parquet]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from wpf.data import sdwpf, splits

_LOG = logging.getLogger(__name__)

FLAGGED_MODES = ("keep", "nan")
VAL_NAME = "val.arrow"

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


def apply_flagged(patv: np.ndarray, flagged: np.ndarray | None, mode: str) -> np.ndarray:
    """`keep`: `patv` as is. `nan`: NaN wherever `flagged` (unknown or abnormal) is set; missing is NaN already."""
    _LOG.debug("patv.shape=%s mode='%s'", patv.shape, mode)
    if mode not in FLAGGED_MODES:
        raise ValueError(f"flagged mode must be one of {FLAGGED_MODES}, got '{mode}'")
    if mode == "keep":
        return patv
    if flagged is None or flagged.shape != patv.shape:
        raise ValueError("mode 'nan' needs a flagged mask with the shape of patv")
    return np.where(flagged, np.float32("nan"), patv).astype(np.float32)


def val_range(split: splits.Split) -> tuple[int, int]:
    """Steps of the val file: the val period plus the LOOKBACK steps before it (the first context)."""
    return split.val[0] - splits.LOOKBACK, split.val[1]


def val_series(patv: np.ndarray, split: splits.Split) -> np.ndarray:
    """`Patv` of every turbine over `val_range`, as a (turbines, steps) float32 array."""
    _LOG.debug("patv.shape=%s", patv.shape)
    start, end = val_range(split)
    return np.ascontiguousarray(patv[:, start:end], dtype=np.float32)


def step_start(step: int) -> np.datetime64:
    """Synthetic start time of a series: the DS-01 anchor plus `step` readings, in seconds."""
    ts = sdwpf.TS_ANCHOR + pd.Timedelta(minutes=sdwpf.STEP_MINUTES * step)
    return ts.to_datetime64().astype("datetime64[s]")


def subset_start(split: splits.Split, fraction: float) -> np.datetime64:
    """Synthetic start time of a subset: the DS-01 anchor plus its first step."""
    return step_start(splits.subset_range(split, fraction)[0])


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


def _entry(series: np.ndarray, steps: tuple[int, int]) -> dict:
    return {
        "steps": list(steps),
        "series": int(series.shape[0]),
        "length": int(series.shape[1]),
        "nan_fraction": round(float(np.isnan(series).mean()), 6),
    }


def export_subsets(
    patv: np.ndarray,
    split: splits.Split,
    out_dir: str | Path,
    flagged: np.ndarray | None = None,
    flagged_mode: str = "keep",
) -> dict:
    """Write one Arrow file per training subset plus `val.arrow` into `out_dir`; returns the manifest.

    `flagged` is the dense (turbines, steps) `unknown | abnormal` mask; it is only needed for `flagged_mode="nan"`.
    """
    _LOG.debug("patv.shape=%s out_dir='%s' flagged_mode='%s'", patv.shape, out_dir, flagged_mode)
    patv = apply_flagged(patv, flagged, flagged_mode)
    files = {}
    for fraction in splits.SUBSET_FRACTIONS:
        name = f"{subset_name(fraction)}.arrow"
        series = subset_series(patv, split, fraction)
        write_arrow(series, subset_start(split, fraction), Path(out_dir) / name)
        files[name] = {"fraction": fraction, **_entry(series, splits.subset_range(split, fraction))}
        _LOG.info("wrote %s: %s", name, files[name])
    series = val_series(patv, split)
    write_arrow(series, step_start(val_range(split)[0]), Path(out_dir) / VAL_NAME)
    files[VAL_NAME] = {"val_period": list(split.val), **_entry(series, val_range(split))}
    _LOG.info("wrote %s: %s", VAL_NAME, files[VAL_NAME])
    target = "Patv (clamped, NaN where missing" + (", unknown or abnormal)" if flagged_mode == "nan" else ")")
    return {"target": target, "flagged": flagged_mode, "chronos_train_args": CHRONOS_TRAIN_ARGS, "files": files}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the SDWPF training subsets and val as Chronos/GluonTS Arrow files.")
    parser.add_argument("--clean", type=Path, default=Path("data/sdwpf/clean.parquet"))
    parser.add_argument("--flagged", choices=FLAGGED_MODES, default="keep", help="keep unknown/abnormal values or set them to NaN")
    parser.add_argument("--out-dir", type=Path, help="default data/sdwpf/chronos (keep) or data/sdwpf/chronos_nan (nan)")
    parser.add_argument("--manifest", type=Path, help="default results/ds03_chronos_export.json (keep) or ..._nan.json (nan)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    suffix = "" if args.flagged == "keep" else "_nan"
    args.out_dir = args.out_dir or Path(f"data/sdwpf/chronos{suffix}")
    args.manifest = args.manifest or Path(f"results/ds03_chronos_export{suffix}.json")
    df = sdwpf.load_clean(args.clean)
    patv = sdwpf.to_dense(df, "Patv")
    flagged = sdwpf.to_dense(df, "unknown") | sdwpf.to_dense(df, "abnormal")
    split = splits.make_split()
    if patv.shape[1] != split.n_steps:
        raise ValueError(f"{args.clean} has {patv.shape[1]} steps per turbine, the split expects {split.n_steps}")
    manifest = export_subsets(patv, split, args.out_dir, flagged, args.flagged)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(splits.dump_json(manifest) + "\n")
    _LOG.info("wrote %s", args.manifest)
    print(splits.dump_json(manifest))


if __name__ == "__main__":
    main()
