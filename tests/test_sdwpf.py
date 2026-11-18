import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wpf.data import sdwpf

REPO = Path(__file__).resolve().parents[1]
# data/ is gitignored, so a worktree may not have it; the main checkout's copy is found via DS01_RAW_CSV.
CSV = Path(os.environ.get("DS01_RAW_CSV", REPO / "data" / "sdwpf" / "raw" / "sdwpf_245days_v1.csv"))


def _row(**kw) -> dict:
    """One healthy reading; keyword arguments override single columns."""
    base = dict(Wspd=6.0, Wdir=0.0, Etmp=20.0, Itmp=25.0, Ndir=100.0, Pab1=1.0, Pab2=1.0, Pab3=1.0, Prtv=0.0, Patv=500.0)
    base.update(kw)
    return base


def _mask(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).astype("float32")
    return sdwpf.build_mask(sdwpf.clean(df)[0])


def _tiny_grid(n_turbines: int = 2, n_days: int = 2) -> pd.DataFrame:
    """Raw-format frame with the full grid, shuffled so sorting is exercised."""
    slots = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(0, 60, 10)]
    rows = [
        dict(TurbID=t, Day=d, Tmstamp=s, **_row(Patv=float(t * 1000 + d * 100 + i)))
        for t in range(1, n_turbines + 1)
        for d in range(1, n_days + 1)
        for i, s in enumerate(slots)
    ]
    df = pd.DataFrame(rows).astype({c: "float32" for c in sdwpf.FEATURES})  # float32 like load_raw
    return df.sample(frac=1, random_state=0).reset_index(drop=True)


def test_negative_patv_set_to_zero_and_counted():
    df = pd.DataFrame([_row(Patv=-5.0), _row(Patv=10.0), _row(Patv=np.nan)]).astype("float32")
    out, n_negative = sdwpf.clean(df)
    assert n_negative == 1
    assert out["Patv"].tolist()[:2] == [0.0, 10.0]
    assert np.isnan(out["Patv"].iloc[2])  # NaN is kept, not turned into 0
    # the untouched reading stays available for inputs
    assert out["Patv_raw"].tolist()[:2] == [-5.0, 10.0] and np.isnan(out["Patv_raw"].iloc[2])


def test_unknown_stopped_boundary():
    m = _mask([_row(Patv=0.0, Wspd=2.5), _row(Patv=0.0, Wspd=2.6), _row(Patv=1.0, Wspd=10.0)])
    assert m["unknown"].tolist() == [False, True, False]  # Wspd must be strictly above 2.5


def test_unknown_stopped_sees_clamped_negative_patv():
    # clamp comes first: a negative reading becomes 0 and then trips Patv <= 0 with Wspd > 2.5
    assert _mask([_row(Patv=-3.0, Wspd=8.0)])["unknown"].tolist() == [True]


def test_unknown_pitch_boundary_any_blade():
    m = _mask([_row(Pab1=89.0), _row(Pab2=89.5), _row(Pab3=90.0)])
    assert m["unknown"].tolist() == [False, True, True]


def test_abnormal_direction_boundaries():
    m = _mask([_row(Ndir=720.0), _row(Ndir=-721.0), _row(Wdir=180.0), _row(Wdir=-180.5)])
    assert m["abnormal"].tolist() == [False, True, False, True]


def test_nan_row_is_only_missing():
    m = _mask([_row(Wspd=np.nan, Wdir=np.nan, Ndir=np.nan, Pab1=np.nan, Patv=np.nan)])
    assert m[["missing", "unknown", "abnormal", "valid"]].iloc[0].tolist() == [True, False, False, False]


def test_missing_means_patv_nan_not_any_sensor_nan():
    m = _mask([_row(Etmp=np.nan), _row(Patv=np.nan), _row()])
    # a good Patv with another sensor missing is still scored; only a lost Patv is "missing"
    assert m["missing"].tolist() == [False, True, False]
    assert m["sensor_nan"].tolist() == [True, False, False]  # informational, not part of valid
    assert m["valid"].tolist() == [True, False, True]


def test_valid_is_complement_of_union_and_counts_add_up():
    rows = [
        _row(),  # valid
        _row(Patv=0.0, Wspd=5.0),  # unknown
        _row(Ndir=900.0),  # abnormal
        _row(Patv=0.0, Wspd=5.0, Wdir=200.0),  # unknown and abnormal
        _row(Wspd=np.nan, Wdir=np.nan, Etmp=np.nan, Itmp=np.nan, Ndir=np.nan, Pab1=np.nan, Pab2=np.nan,
             Pab3=np.nan, Prtv=np.nan, Patv=np.nan),  # missing
    ]
    rows.append(_row(Itmp=np.nan))  # sensor NaN with a good Patv: valid, counted as sensor_nan only
    c = sdwpf.category_counts(_mask(rows), n_negative_clamped=7)
    assert (c["rows"], c["valid"], c["invalid_any"]) == (6, 2, 4)
    assert (c["missing"], c["unknown"], c["abnormal"]) == (1, 2, 2)
    assert (c["sensor_nan"], c["sensor_nan_but_patv_present"]) == (2, 1)
    assert c["unknown_and_abnormal"] == 1 and c["missing_and_unknown"] == 0
    assert c["negative_patv_set_to_zero"] == 7


def test_add_time_steps_sorted_and_tmstamp_dropped():
    raw = _tiny_grid()
    out = sdwpf.add_time(raw)
    assert "Tmstamp" not in out.columns
    for _, g in out.groupby("TurbID"):
        assert g["step"].tolist() == list(range(2 * sdwpf.SLOTS_PER_DAY))
    first = out[out["TurbID"] == 1].iloc[0]
    assert first["ts"] == sdwpf.TS_ANCHOR and first["Patv"] == 1000.0 + 100.0
    # Day 2, 00:10 is step 145
    row = out[(out["TurbID"] == 2) & (out["step"] == 145)].iloc[0]
    assert row["ts"] == sdwpf.TS_ANCHOR + pd.Timedelta(minutes=1450) and row["Patv"] == 2000.0 + 200.0 + 1


def test_check_grid_accepts_full_and_rejects_gaps_and_duplicates():
    raw = _tiny_grid()
    sdwpf.check_grid(raw, n_turbines=2, n_days=2)
    with pytest.raises(ValueError, match="rows"):
        sdwpf.check_grid(raw.iloc[1:], n_turbines=2, n_days=2)
    dup = pd.concat([raw.iloc[1:], raw.iloc[[1]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        sdwpf.check_grid(dup, n_turbines=2, n_days=2)


def test_to_dense_matches_per_turbine_series():
    df = sdwpf.add_time(_tiny_grid())
    dense = sdwpf.to_dense(df, "Patv")
    assert dense.shape == (2, 2 * sdwpf.SLOTS_PER_DAY)
    assert np.array_equal(dense[1], df[df["TurbID"] == 2]["Patv"].to_numpy())
    with pytest.raises(ValueError, match="sorted"):
        sdwpf.to_dense(df.sample(frac=1, random_state=1), "Patv")


def test_valid_fractions_per_turbine_and_day_hand_counts():
    raw = _tiny_grid()  # 2 turbines x 2 days x 144 slots
    raw.loc[(raw["TurbID"] == 1) & (raw["Day"] == 1), "Patv"] = np.nan  # turbine 1 loses all of day 1
    df = sdwpf.build_mask(sdwpf.clean(sdwpf.add_time(raw))[0])
    f = sdwpf.valid_fractions(df)
    assert f["per_turbine"]["TurbID"] == [1, 2]
    assert f["per_turbine"]["valid_fraction"] == [0.5, 1.0]
    assert f["per_turbine"]["missing_fraction"] == [0.5, 0.0]
    assert f["per_day"]["Day"] == [1, 2]
    assert f["per_day"]["valid_fraction"] == [0.5, 1.0]  # day 1: turbine 1 gone, turbine 2 fine


def test_counts_json_keeps_lists_on_one_line():
    text = sdwpf._dump_counts({"rows": 3, "per_day": {"Day": [1, 2, 3], "valid_fraction": [0.5, 1.0, 0.25]}})
    assert '"Day": [1, 2, 3]' in text and '"valid_fraction": [0.5, 1.0, 0.25]' in text


def test_load_locations_checks_columns_and_count(tmp_path):
    good = tmp_path / "loc.csv"
    good.write_text("TurbID,x,y\n3,30.0,3.5\n1,10.0,1.5\n2,20.0,2.5\n")
    loc = sdwpf.load_locations(good, n_turbines=3)
    assert loc["TurbID"].tolist() == [1, 2, 3] and loc["x"].tolist() == [10.0, 20.0, 30.0]  # sorted by TurbID
    with pytest.raises(ValueError, match="expected 4"):
        sdwpf.load_locations(good, n_turbines=4)
    bad = tmp_path / "bad.csv"
    bad.write_text("TurbID,x\n1,10.0\n")
    with pytest.raises(ValueError, match="columns"):
        sdwpf.load_locations(bad, n_turbines=1)


def test_parquet_round_trip(tmp_path):
    df = sdwpf.build_mask(sdwpf.clean(sdwpf.add_time(_tiny_grid()))[0])
    sdwpf.save_clean(df, tmp_path / "clean.parquet")
    back = sdwpf.load_clean(tmp_path / "clean.parquet")
    pd.testing.assert_frame_equal(df, back)
    assert back["Patv"].dtype == np.float32 and back["valid"].dtype == bool


@pytest.mark.skipif(not CSV.exists(), reason="raw SDWPF CSV not in data/sdwpf/raw/")
def test_real_file_grid_and_missing_count():
    df, counts = sdwpf.build(CSV)
    assert counts["rows"] == sdwpf.N_ROWS == 4_727_520
    # pinned, so a change to a rule cannot shift the counts silently
    assert counts["missing"] == 49_518
    assert counts["valid"] == 3_595_859
    assert counts["negative_patv_set_to_zero"] == 1_263_062
    assert counts["valid"] + counts["invalid_any"] == counts["rows"]
    assert df["Patv_raw"].min() < 0 and df["Patv"].min() == 0  # raw kept, target clamped
    assert len(counts["per_turbine"]["TurbID"]) == 134 and len(counts["per_day"]["Day"]) == 245
    assert sdwpf.to_dense(df, "valid").shape == (sdwpf.N_TURBINES, sdwpf.N_DAYS * sdwpf.SLOTS_PER_DAY)
