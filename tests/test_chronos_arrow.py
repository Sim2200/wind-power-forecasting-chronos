import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wpf.data import chronos_arrow, sdwpf, splits

REPO = Path(__file__).resolve().parents[1]
# data/ is gitignored, so a worktree may not have it; the main checkout's copy is found via DS_CLEAN_PARQUET.
CLEAN = Path(os.environ.get("DS_CLEAN_PARQUET", REPO / "data" / "sdwpf" / "clean.parquet"))


@pytest.fixture
def tiny():
    """3 / 1 / 2 days: train [0, 432); subsets of 22 / 108 / 432 steps starting at 410 / 324 / 0."""
    return splits.make_split((3, 1, 2))


@pytest.fixture
def gluonts_ok():
    """Skip when gluonts cannot be imported (e.g. it fails to import on Python 3.14)."""
    try:
        from gluonts.dataset.arrow import ArrowWriter  # noqa: F401
        from gluonts.dataset.common import FileDataset  # noqa: F401
    except Exception as e:  # ImportError or a pydantic error at import time
        pytest.skip(f"gluonts is not usable on this Python: {e!r}")


def _tiny_frame(n_days: int = 6) -> pd.DataFrame:
    """Raw-format frame for 2 turbines with healthy readings, plus four hand-placed special readings.

    steps (inside the 5% subset 410..431 unless noted): turbine 1 step 420 has Patv = -5 (clamped to 0),
    turbine 1 step 430 is a lost record (all sensors NaN), turbine 2 step 425 has Pab1 = 95 (flagged unknown,
    Patv = 300 stays), turbine 2 step 100 has only Etmp NaN (not missing; only in the 100% subset).
    """
    steps = np.arange(n_days * 144)
    rows = pd.DataFrame(
        {
            "TurbID": np.repeat([1, 2], len(steps)),
            "step": np.tile(steps, 2),
        }
    )
    rows["Day"] = rows["step"] // 144 + 1
    slot = rows["step"] % 144
    rows["Tmstamp"] = (slot // 6).map("{:02d}".format) + ":" + ((slot % 6) * 10).map("{:02d}".format)
    for col, value in dict(Wspd=6.0, Wdir=0.0, Etmp=20.0, Itmp=25.0, Ndir=100.0, Pab1=1.0, Pab2=1.0, Pab3=1.0, Prtv=0.0, Patv=500.0).items():
        rows[col] = np.float32(value)

    def at(turb: int, step: int) -> pd.Series:
        return (rows["TurbID"] == turb) & (rows["step"] == step)

    rows.loc[at(1, 420), "Patv"] = -5.0
    rows.loc[at(1, 430), sdwpf.FEATURES] = np.nan
    rows.loc[at(2, 425), "Pab1"] = 95.0
    rows.loc[at(2, 425), "Patv"] = 300.0
    rows.loc[at(2, 100), "Etmp"] = np.nan
    return rows.drop(columns="step")


def _tiny_dense() -> tuple[np.ndarray, np.ndarray]:
    """Dense Patv and the unknown | abnormal mask of the tiny frame. Flagged readings: turbine 1 step 420 (Patv 0
    while the wind blows) and turbine 2 step 425 (Pab1 = 95)."""
    df = sdwpf.build_mask(sdwpf.clean(sdwpf.add_time(_tiny_frame()))[0])
    return sdwpf.to_dense(df, "Patv"), sdwpf.to_dense(df, "unknown") | sdwpf.to_dense(df, "abnormal")


def _tiny_patv() -> np.ndarray:
    return _tiny_dense()[0]


def test_subset_series_equals_dense_slice_and_ends_at_train_end(tiny):
    patv = np.arange(2)[:, None] * 1000 + np.arange(tiny.n_steps)[None, :].astype(np.float32)  # encodes (turbine, step)
    s = chronos_arrow.subset_series(patv, tiny, 0.05)
    assert s.shape == (2, 22) and s.dtype == np.float32
    assert np.array_equal(s[1], patv[1, 410:432]) and s[0, -1] == 431  # ends at the last train step, before val


def test_subsets_are_nested_tails_with_the_ds02_lengths(tiny):
    patv = np.random.default_rng(0).normal(size=(2, tiny.n_steps)).astype(np.float32)
    s5, s25, s100 = (chronos_arrow.subset_series(patv, tiny, f) for f in splits.SUBSET_FRACTIONS)
    assert (s5.shape[1], s25.shape[1], s100.shape[1]) == (22, 108, 432)
    assert np.array_equal(s25[:, -22:], s5) and np.array_equal(s100[:, -108:], s25)
    full = splits.make_split()
    lengths = [chronos_arrow.subset_series(np.zeros((1, full.n_steps), np.float32), full, f).shape[1] for f in splits.SUBSET_FRACTIONS]
    assert lengths == [1231, 6156, 24624]


def test_subset_start_is_the_anchor_plus_first_step(tiny):
    # step 410 = 4100 minutes = 2 days 20 h 20 min; step 324 = 3240 minutes = 2 days 6 h
    assert chronos_arrow.subset_start(tiny, 0.05) == np.datetime64("2020-01-03T20:20:00")
    assert chronos_arrow.subset_start(tiny, 0.25) == np.datetime64("2020-01-03T06:00:00")
    assert chronos_arrow.subset_start(tiny, 1.0) == np.datetime64("2020-01-01T00:00:00")


def test_target_is_clamped_patv_nan_only_where_missing_flagged_rows_keep_values(tiny):
    s = chronos_arrow.subset_series(_tiny_patv(), tiny, 0.05)  # steps 410..431
    assert s[0, 420 - 410] == 0.0  # negative reading clamped (Patv_raw would be -5)
    assert np.isnan(s[0, 430 - 410])  # lost record
    assert s[1, 425 - 410] == 300.0  # flagged unknown row keeps its value
    assert np.isnan(s).sum() == 1  # nothing else is NaN
    full = chronos_arrow.subset_series(_tiny_patv(), tiny, 1.0)
    assert full[1, 100] == 500.0  # a sensor NaN with a good Patv is not a gap


def test_apply_flagged_keep_is_identity_and_nan_blanks_only_flagged_readings(tiny):
    patv, flagged = _tiny_dense()
    assert chronos_arrow.apply_flagged(patv, flagged, "keep") is patv
    out = chronos_arrow.apply_flagged(patv, flagged, "nan")
    assert out.dtype == np.float32
    assert np.isnan(out[0, 420]) and np.isnan(out[1, 425])  # unknown rows: 0 while windy, pitch > 89
    assert np.isnan(out[0, 430])  # missing stays NaN
    assert np.isnan(out).sum() == 3  # nothing else
    assert np.array_equal(out[~np.isnan(out)], patv[~np.isnan(out)])  # every other reading is untouched


def test_apply_flagged_validates_mode_and_mask():
    patv, flagged = _tiny_dense()
    with pytest.raises(ValueError, match="one of"):
        chronos_arrow.apply_flagged(patv, flagged, "drop")
    with pytest.raises(ValueError, match="flagged mask"):
        chronos_arrow.apply_flagged(patv, None, "nan")


def test_val_series_is_val_plus_one_context_before_it(tiny):
    patv = np.arange(2)[:, None] * 1000 + np.arange(tiny.n_steps)[None, :].astype(np.float32)
    assert chronos_arrow.val_range(tiny) == (144, 576)  # val [432, 576) minus the 288 steps before it
    v = chronos_arrow.val_series(patv, tiny)
    assert v.shape == (2, 432) and v.dtype == np.float32
    assert v[0, 0] == 144 and v[0, 288] == tiny.val[0] and v[0, -1] == 575  # ends at the last val step, before test
    full = chronos_arrow.val_series(np.zeros((1, splits.make_split().n_steps), np.float32), splits.make_split())
    assert full.shape == (1, 3888)
    assert chronos_arrow.step_start(144) == np.datetime64("2020-01-02T00:00:00")


def test_write_arrow_rejects_1d(tmp_path):
    with pytest.raises(ValueError, match="2-D"):
        chronos_arrow.write_arrow(np.zeros(5, np.float32), np.datetime64("2000-01-01T00:00:00"), tmp_path / "x.arrow")


def test_export_round_trip_through_gluonts_and_manifest(tiny, tmp_path, gluonts_ok):
    patv = _tiny_patv()
    manifest = chronos_arrow.export_subsets(patv, tiny, tmp_path)
    assert sorted(manifest["files"]) == ["train_100pct.arrow", "train_25pct.arrow", "train_5pct.arrow", "val.arrow"]
    assert manifest["flagged"] == "keep" and manifest["target"] == "Patv (clamped, NaN where missing)"
    assert manifest["chronos_train_args"] == {"context_length": 288, "prediction_length": 64, "min_past": 288}
    five = manifest["files"]["train_5pct.arrow"]
    assert (five["series"], five["length"], five["steps"]) == (2, 22, [410, 432])
    assert five["nan_fraction"] == round(1 / 44, 6)  # one lost record in 2 x 22 readings
    val = manifest["files"]["val.arrow"]
    assert (val["series"], val["length"], val["steps"], val["val_period"]) == (2, 432, [144, 576], [432, 576])
    for name, info in manifest["files"].items():
        back = chronos_arrow.read_series(tmp_path / name)  # the way train.py reads it
        if name == "val.arrow":
            expected = chronos_arrow.val_series(patv, tiny)
        else:
            expected = chronos_arrow.subset_series(patv, tiny, info["fraction"])
        assert back.shape == expected.shape
        assert np.array_equal(back, expected, equal_nan=True)


def test_export_nan_mode_blanks_flagged_readings_in_every_file(tiny, tmp_path, gluonts_ok):
    patv, flagged = _tiny_dense()
    manifest = chronos_arrow.export_subsets(patv, tiny, tmp_path, flagged, "nan")
    assert manifest["flagged"] == "nan" and "unknown or abnormal" in manifest["target"]
    assert manifest["files"]["train_5pct.arrow"]["nan_fraction"] == round(3 / 44, 6)  # lost record + 2 flagged readings
    back = chronos_arrow.read_series(tmp_path / "train_5pct.arrow")
    assert np.isnan(back[0, 420 - 410]) and np.isnan(back[1, 425 - 410]) and np.isnan(back).sum() == 3
    val_back = chronos_arrow.read_series(tmp_path / "val.arrow")
    assert np.array_equal(val_back, chronos_arrow.val_series(chronos_arrow.apply_flagged(patv, flagged, "nan"), tiny), equal_nan=True)


@pytest.mark.skipif(not CLEAN.exists(), reason="run python -m wpf.data.sdwpf first (data/sdwpf/clean.parquet)")
@pytest.mark.parametrize("mode,manifest_name", [("keep", "ds03_chronos_export.json"), ("nan", "ds03_chronos_export_nan.json")])
def test_real_file_export_matches_committed_manifest(tmp_path, gluonts_ok, mode, manifest_name):
    """Also catches a stale manifest: if the DS-01 data changes, regenerate results/ds03_chronos_export*.json (both --flagged modes)."""
    df = sdwpf.load_clean(CLEAN)
    patv = sdwpf.to_dense(df, "Patv")
    flagged = sdwpf.to_dense(df, "unknown") | sdwpf.to_dense(df, "abnormal")
    manifest = chronos_arrow.export_subsets(patv, splits.make_split(), tmp_path, flagged, mode)
    assert {n: (i["series"], i["length"]) for n, i in manifest["files"].items()} == {
        "train_5pct.arrow": (134, 1231),
        "train_25pct.arrow": (134, 6156),
        "train_100pct.arrow": (134, 24624),
        "val.arrow": (134, 3888),
    }
    for name, info in manifest["files"].items():  # every entry passes train.py's length / missing-share filter
        series = chronos_arrow.read_series(tmp_path / name)
        assert series.shape[1] >= 288 + 64 and np.isnan(series).mean(axis=1).max() <= 0.9
    committed = json.loads((REPO / "results" / manifest_name).read_text())
    assert committed == json.loads(splits.dump_json(manifest))
