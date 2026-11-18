"""SDWPF (Baidu KDD Cup 2022) loading, cleaning and the valid-reading mask.

Raw file: `sdwpf_245days_v1.csv`, 134 turbines x 245 days x 144 ten-minute slots = 4,727,520 rows.
Every turbine has the full grid, so a lost record is a row whose Patv is NaN (in this file all 10 sensor
columns are NaN together, but the rule only depends on Patv).

Technical decisions (also listed in the DS-01 pull request):
- NaN stays NaN in the saved data and is only marked invalid. Nothing is filled here: Chronos takes the
  NaNs as they are, PatchTST/LSTM apply a backward-only last-value fill per turbine at input time
  (DS-02..DS-06). Metrics and scaler statistics must use `valid` rows only.
- `missing` means Patv is NaN (a lost record, as in the SDWPF paper). A row with a good Patv and another
  sensor NaN is not missing; `sensor_nan` flags any NaN among the 9 other sensors and is informational only
  (it does not affect `valid`).
- `Patv_raw` is the untouched reading. `Patv` is the same reading with negatives set to 0: the paper's
  scoring rule, so it is the scale for targets and metrics. Inputs may use `Patv_raw` to keep the
  "turbine idle / drawing power" signal. Both are NaN on missing rows. The "unknown" rule (Patv <= 0) gives
  the same answer on either column, so clamp order does not matter.
- A NaN never trips a threshold (the comparison is False), so a gap row is only `missing`. The three flags
  can overlap; `valid` is the complement of their union. Counts are reported per category and combined,
  plus per turbine and per day, since the recency subsets depend on which turbines are missing when.
- Flagged rows in inputs: `unknown` and `abnormal` rows (about 23%) are excluded from the loss and the
  metrics, but they stay in input windows as real history (not filled, not dropped). Only `missing` rows
  are gap-filled at input time. DS-02 and PT-01 must build windows this way.
- Output is one long-format parquet, float32 features, sorted by (TurbID, step). Because the grid is
  complete, `to_dense` reshapes any column to (turbines, steps) for window building (DS-02, DS-03).
  Turbine locations (TurbID, x, y) are saved next to it as `turbine_locations.parquet`.
- SDWPF hides the calendar date. `step = (Day - 1) * 144 + slot` is the real ordering key; `ts` is that
  step on a synthetic anchor date and must not be read as a real date.
- Covariates (Wspd, Wdir, Etmp, ...) are left untouched, including outliers such as Etmp = -273 and
  Wdir = -3030: only Patv is read by the models for now. A model that reads another sensor (e.g. the LSTM
  baseline) must first clip it to its physical range and flag the clipped rows.

Usage (raw files: scripts/download_sdwpf.sh):
    python -m wpf.data.sdwpf --raw data/sdwpf/raw/sdwpf_245days_v1.csv \\
        --out data/sdwpf/clean.parquet --counts results/ds01_counts.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

_LOG = logging.getLogger(__name__)

N_TURBINES = 134
N_DAYS = 245
SLOTS_PER_DAY = 144  # 10-minute readings
N_ROWS = N_TURBINES * N_DAYS * SLOTS_PER_DAY  # 4,727,520
STEP_MINUTES = 10
TS_ANCHOR = pd.Timestamp("2020-01-01")  # synthetic: SDWPF gives no calendar date

FEATURES = ["Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Pab2", "Pab3", "Prtv", "Patv"]
SENSORS = [c for c in FEATURES if c != "Patv"]
RAW_COLUMNS = ["TurbID", "Day", "Tmstamp", *FEATURES]
LOCATION_COLUMNS = ["TurbID", "x", "y"]
FLAGS = ["missing", "unknown", "abnormal"]

WSPD_STOPPED = 2.5  # m/s: a turbine producing <= 0 above this wind speed is not behaving normally
PITCH_MAX = 89.0  # deg: blade pitch beyond this means the turbine is feathered/stopped
NDIR_MAX = 720.0  # deg: nacelle direction range
WDIR_MAX = 180.0  # deg: relative wind direction range


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """Read the raw SDWPF CSV with float32 features; raise if a column is missing."""
    _LOG.debug("csv_path='%s'", csv_path)
    dtypes = {"TurbID": "int16", "Day": "int16", **{c: "float32" for c in FEATURES}}
    df = pd.read_csv(csv_path, dtype=dtypes)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns {missing}")
    return df[RAW_COLUMNS]


def check_grid(df: pd.DataFrame, n_turbines: int = N_TURBINES, n_days: int = N_DAYS) -> None:
    """Raise unless every turbine has exactly one row per (Day, Tmstamp) of the full grid."""
    _LOG.debug("rows=%d n_turbines=%d n_days=%d", len(df), n_turbines, n_days)
    expected = n_turbines * n_days * SLOTS_PER_DAY
    if len(df) != expected:
        raise ValueError(f"expected {expected} rows, got {len(df)}")
    if df["TurbID"].nunique() != n_turbines:
        raise ValueError(f"expected {n_turbines} turbines, got {df['TurbID'].nunique()}")
    if df.duplicated(["TurbID", "Day", "Tmstamp"]).any():
        raise ValueError("duplicate (TurbID, Day, Tmstamp) rows")
    per_turbine = df.groupby("TurbID").size()
    if not (per_turbine == n_days * SLOTS_PER_DAY).all():
        raise ValueError("turbines do not all have the full grid")


def add_time(df: pd.DataFrame) -> pd.DataFrame:
    """Add `step` (0-based 10-minute index per turbine) and `ts`, drop `Tmstamp`, sort by (TurbID, step)."""
    _LOG.debug("rows=%d", len(df))
    hh = df["Tmstamp"].str.slice(0, 2).astype("int32")
    mm = df["Tmstamp"].str.slice(3, 5).astype("int32")
    slot = hh * 6 + mm // STEP_MINUTES
    out = df.drop(columns="Tmstamp").copy()
    out["step"] = ((out["Day"].astype("int32") - 1) * SLOTS_PER_DAY + slot).astype("int32")
    out["ts"] = TS_ANCHOR + pd.to_timedelta(out["step"].astype("int64") * STEP_MINUTES, unit="m")
    return out.sort_values(["TurbID", "step"], kind="stable").reset_index(drop=True)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep the reading as `Patv_raw`, set negative Patv to 0 (NaN is kept).

    Returns the cleaned frame and how many readings were clamped.
    """
    _LOG.debug("rows=%d", len(df))
    out = df.copy()
    out["Patv_raw"] = out["Patv"]
    n_negative = int((out["Patv"] < 0).sum())
    out["Patv"] = out["Patv"].clip(lower=0)  # clip keeps NaN
    return out, n_negative


def build_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Add bool columns `missing`, `unknown`, `abnormal`, `valid` (= none of the three) and `sensor_nan`.

    `missing` is Patv NaN; `sensor_nan` (any other sensor NaN) is informational and not part of `valid`.
    Call this on the output of `clean`.
    """
    _LOG.debug("rows=%d", len(df))
    out = df.copy()
    out["missing"] = out["Patv"].isna()
    out["sensor_nan"] = out[SENSORS].isna().any(axis=1)
    out["unknown"] = _unknown_stopped(out) | _unknown_pitch(out)
    out["abnormal"] = _abnormal_ndir(out) | _abnormal_wdir(out)
    out["valid"] = ~out[FLAGS].any(axis=1)
    return out


def _unknown_stopped(df: pd.DataFrame) -> pd.Series:
    return (df["Patv"] <= 0) & (df["Wspd"] > WSPD_STOPPED)


def _unknown_pitch(df: pd.DataFrame) -> pd.Series:
    return (df[["Pab1", "Pab2", "Pab3"]] > PITCH_MAX).any(axis=1)


def _abnormal_ndir(df: pd.DataFrame) -> pd.Series:
    return df["Ndir"].abs() > NDIR_MAX


def _abnormal_wdir(df: pd.DataFrame) -> pd.Series:
    return df["Wdir"].abs() > WDIR_MAX


def category_counts(df: pd.DataFrame, n_negative_clamped: int = 0) -> dict[str, int]:
    """Row counts per mask category (they overlap), their union, and the valid total."""
    _LOG.debug("rows=%d n_negative_clamped=%d", len(df), n_negative_clamped)
    m, u, a = (df[f] for f in FLAGS)
    counts = {
        "rows": len(df),
        "negative_patv_set_to_zero": n_negative_clamped,
        "missing": int(m.sum()),
        "sensor_nan": int(df["sensor_nan"].sum()),
        "sensor_nan_but_patv_present": int((df["sensor_nan"] & ~m).sum()),
        "unknown": int(u.sum()),
        "unknown_stopped": int(_unknown_stopped(df).sum()),
        "unknown_pitch": int(_unknown_pitch(df).sum()),
        "abnormal": int(a.sum()),
        "abnormal_ndir": int(_abnormal_ndir(df).sum()),
        "abnormal_wdir": int(_abnormal_wdir(df).sum()),
        "missing_and_unknown": int((m & u).sum()),
        "missing_and_abnormal": int((m & a).sum()),
        "unknown_and_abnormal": int((u & a).sum()),
        "invalid_any": int((m | u | a).sum()),
        "valid": int(df["valid"].sum()),
    }
    if counts["valid"] + counts["invalid_any"] != counts["rows"]:
        raise ValueError("valid + invalid_any does not add up to the row count")
    return counts


def valid_fractions(df: pd.DataFrame) -> dict[str, dict[str, list]]:
    """Share of valid and of missing readings per turbine and per day, as column-wise lists.

    Farm-wide totals hide which turbines are missing when, and the recency subsets depend on exactly that.
    """
    _LOG.debug("rows=%d", len(df))
    out = {}
    for name, key in (("per_turbine", "TurbID"), ("per_day", "Day")):
        g = df.groupby(key)[["valid", "missing"]].mean().round(6)
        out[name] = {
            key: g.index.tolist(),
            "valid_fraction": g["valid"].tolist(),
            "missing_fraction": g["missing"].tolist(),
        }
    return out


def load_locations(csv_path: str | Path, n_turbines: int = N_TURBINES) -> pd.DataFrame:
    """Read the SDWPF turbine location file (TurbID, x, y), one row per turbine, sorted by TurbID."""
    _LOG.debug("csv_path='%s' n_turbines=%d", csv_path, n_turbines)
    loc = pd.read_csv(csv_path, dtype={"TurbID": "int16", "x": "float32", "y": "float32"})
    missing = [c for c in LOCATION_COLUMNS if c not in loc.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns {missing}")
    loc = loc[LOCATION_COLUMNS].sort_values("TurbID").reset_index(drop=True)
    if len(loc) != n_turbines or loc["TurbID"].duplicated().any():
        raise ValueError(f"expected {n_turbines} distinct turbines in {csv_path}, got {len(loc)} rows")
    return loc


def to_dense(df: pd.DataFrame, column: str) -> np.ndarray:
    """Reshape a column of the sorted long frame to (turbines, steps); needs the complete grid."""
    _LOG.debug("column='%s'", column)
    n_turbines = df["TurbID"].nunique()
    if len(df) % n_turbines:
        raise ValueError("rows are not a multiple of the turbine count")
    ids = df["TurbID"].to_numpy().reshape(n_turbines, -1)
    steps = df["step"].to_numpy().reshape(n_turbines, -1)
    # Sorted by (TurbID, step) on a complete grid: one turbine per row, steps consecutive.
    if not ((ids == ids[:, :1]).all() and (np.diff(steps, axis=1) == 1).all()):
        raise ValueError("frame is not sorted by (TurbID, step) on a complete grid")
    return df[column].to_numpy().reshape(n_turbines, -1)


def save_clean(df: pd.DataFrame, path: str | Path) -> None:
    _LOG.debug("path='%s'", path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_clean(path: str | Path) -> pd.DataFrame:
    _LOG.debug("path='%s'", path)
    return pd.read_parquet(path)


def build(raw_csv: str | Path) -> tuple[pd.DataFrame, dict]:
    """Raw CSV -> cleaned frame with mask, plus the counts (per category, per turbine, per day)."""
    df = load_raw(raw_csv)
    check_grid(df)
    df = add_time(df)
    df, n_negative = clean(df)
    df = build_mask(df)
    return df, {**category_counts(df, n_negative), **valid_fractions(df)}


def _dump_counts(counts: dict) -> str:
    """JSON with the per-turbine and per-day lists on one line each, so the file stays readable."""
    text = json.dumps(counts, indent=2)
    return re.sub(r"\[[^\[\]{}]*\]", lambda m: " ".join(m.group(0).split()).replace("[ ", "[").replace(" ]", "]"), text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean the SDWPF CSV and build the valid-reading mask.")
    parser.add_argument("--raw", type=Path, default=Path("data/sdwpf/raw/sdwpf_245days_v1.csv"))
    parser.add_argument("--locations", type=Path, default=Path("data/sdwpf/raw/sdwpf_baidukddcup2022_turb_location.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/sdwpf/clean.parquet"))
    parser.add_argument("--counts", type=Path, default=Path("results/ds01_counts.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    df, counts = build(args.raw)
    loc = load_locations(args.locations)
    if set(loc["TurbID"]) != set(df["TurbID"].unique()):
        raise ValueError("turbine ids in the location file and the readings differ")
    save_clean(df, args.out)
    save_clean(loc, args.out.parent / "turbine_locations.parquet")
    args.counts.parent.mkdir(parents=True, exist_ok=True)
    args.counts.write_text(_dump_counts(counts) + "\n")
    _LOG.info("wrote %s, turbine_locations.parquet and %s", args.out, args.counts)
    print(json.dumps({k: v for k, v in counts.items() if not isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
