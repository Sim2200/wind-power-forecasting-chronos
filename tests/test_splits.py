import json
import os
from pathlib import Path

import numpy as np
import pytest

from wpf.data import splits

REPO = Path(__file__).resolve().parents[1]
# data/ is gitignored, so a worktree may not have it; the main checkout's copy is found via DS_CLEAN_PARQUET.
CLEAN = Path(os.environ.get("DS_CLEAN_PARQUET", REPO / "data" / "sdwpf" / "clean.parquet"))
L, H_MAX = splits.LOOKBACK, max(splits.HORIZONS)


@pytest.fixture
def split():
    return splits.make_split()


@pytest.fixture
def tiny():
    """3 / 1 / 2 days: train [0, 432), val [432, 576), test [576, 864); short enough to check by hand."""
    return splits.make_split((3, 1, 2))


def test_boundaries_are_day_aligned_171_25_49(split):
    assert split.train == (0, 24_624) and split.val == (24_624, 28_224) and split.test == (28_224, 35_280)
    assert split.n_steps == 245 * 144 == 35_280
    assert all(b % 144 == 0 for b in (split.val[0], split.test[0], split.test[1]))


def test_subsets_are_most_recent_and_nested(split):
    ranges = [splits.subset_range(split, f) for f in splits.SUBSET_FRACTIONS]
    assert ranges == [(23_393, 24_624), (18_468, 24_624), (0, 24_624)]  # 1231 / 6156 / 24624 steps
    assert all(end == split.val[0] for _, end in ranges)  # adjacent to val
    assert ranges[2][0] <= ranges[1][0] <= ranges[0][0]


def test_subset_fraction_validated(split):
    for bad in (0, -0.1, 1.5):
        with pytest.raises(ValueError):
            splits.subset_range(split, bad)


def test_eval_origin_counts_and_spacing(split):
    val, test = splits.eval_origins(split, "val"), splits.eval_origins(split, "test")
    assert (len(val), len(test)) == (99, 195)
    assert val[0] == split.val[0] and test[0] == split.test[0]  # first target starts at the boundary
    assert (np.diff(test) == 36).all()
    assert test[-1] + H_MAX <= split.test[1] < test[-1] + 36 + H_MAX  # longest horizon fits, one more would not
    with pytest.raises(ValueError):
        splits.eval_origins(split, "train")


@pytest.mark.parametrize("period", ["val", "test"])
@pytest.mark.parametrize("horizon", splits.HORIZONS)
def test_no_target_crosses_a_split_and_history_stays_in_data(split, period, horizon):
    origins = splits.eval_origins(split, period)
    start, end = split.period(period)
    assert origins.min() - L >= 0  # history exists
    assert origins.min() >= start and origins.max() + horizon <= end  # targets stay inside the period


def test_first_val_history_reaches_into_train(split):
    first = splits.eval_origins(split, "val")[0]
    assert first - L == 24_624 - 288 and first - L < split.train[1]


def test_train_origins_lie_inside_subset(split):
    o = splits.train_origins(split, 0.05, 64)
    start, end = splits.subset_range(split, 0.05)
    assert len(o) == 1231 - 288 - 64 + 1 == 880
    assert o[0] - L == start and o[-1] + 64 == end  # history and target both inside the subset
    assert len(splits.train_origins(split, 0.05, 64, stride=10)) == 88


def test_windows_at_matches_manual_slicing(tiny):
    dense = np.arange(2)[:, None] * 1000 + np.arange(tiny.n_steps)[None, :]  # value encodes (turbine, step)
    origins = splits.eval_origins(tiny, "val")
    assert origins.tolist() == [432, 468, 504]
    x, y = splits.windows_at(dense, origins, 6)
    assert (x.shape, y.shape) == ((2, 3, L), (2, 3, 6))
    assert np.array_equal(x[1, 2], dense[1, 504 - L : 504])
    assert np.array_equal(y[0, 1], dense[0, 468:474])


def test_windows_at_rejects_out_of_range_origins(tiny):
    dense = np.zeros((2, tiny.n_steps))
    with pytest.raises(ValueError):
        splits.windows_at(dense, np.array([L - 1]), 6)
    with pytest.raises(ValueError):
        splits.windows_at(dense, np.array([tiny.n_steps - 5]), 6)


def test_json_round_trip_and_tamper_detection(split, tmp_path):
    path = tmp_path / "splits.json"
    splits.save_splits(path, split)
    assert splits.load_splits(path) == split
    saved = json.loads(path.read_text())
    assert saved["eval_origins"]["test"][:2] == [28_224, 28_260]
    saved["eval_origins"]["test"][1] += 1
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="does not match"):
        splits.load_splits(path)


def _tiny_masks(tiny):
    """Turbine 0 never valid and missing for steps [0, 432); turbine 1 always valid except a gap [450, 460)."""
    valid = np.ones((2, tiny.n_steps), dtype=bool)
    valid[0, :] = False
    valid[1, 450:460] = False
    missing = np.zeros_like(valid)
    missing[0, :432] = True
    return valid, missing


def test_window_counts_hand_counts(tiny):
    valid, missing = _tiny_masks(tiny)
    c = splits.window_counts(valid, missing, tiny)
    assert c["val_origins"].tolist() == [432, 468, 504]
    # turbine 1's invalid gap is steps [450, 460); a target [o, o + H) loses those steps only if it contains them
    assert c["val_target_valid_h6"].tolist() == [[0, 0, 0], [6, 6, 6]]  # [432,438), [468,474), [504,510)
    assert c["val_target_valid_h36"].tolist() == [[0, 0, 0], [26, 36, 36]]  # [432,468) holds the gap
    assert c["val_target_valid_h64"].tolist() == [[0, 0, 0], [54, 64, 64]]  # [432,496) holds it, [468,532) does not
    # histories [144,432) miss the gap; [180,468) and [216,504) contain all 10 gap steps
    assert c["val_history_valid"].tolist() == [[0, 0, 0], [288, 278, 278]]
    # turbine 0 is missing for steps < 432: histories [144,432), [180,468), [216,504) hold 288, 252, 216 missing steps
    assert c["val_history_observed"].tolist() == [[0, 36, 72], [288, 288, 288]]
    assert c["val_history_valid"].dtype == np.uint16 and c["val_target_valid_h64"].dtype == np.uint8


def test_window_coverage_summarizes_counts(tiny):
    valid, missing = _tiny_masks(tiny)
    cov = splits.window_coverage(splits.window_counts(valid, missing, tiny))
    c = cov["val"]["6"]
    assert c["windows"] == 2 * 3  # 2 turbines x 3 origins
    assert c["valid_target_fraction"] == 0.5
    assert (c["windows_without_valid_target"], c["windows_fully_valid_target"]) == (3, 3)
    assert c["valid_history_fraction"] == pytest.approx((288 + 278 + 278) / (6 * 288))
    assert c["observed_history_fraction"] == pytest.approx((0 + 36 + 72 + 3 * 288) / (6 * 288))
    assert cov["val"]["64"]["windows_fully_valid_target"] == 2  # turbine 1 at origins 468 and 504
    assert cov["test"]["64"]["windows"] == 2 * 7


def test_longest_run():
    assert splits._longest_run(np.array([False, False])) == 0
    assert splits._longest_run(np.array([True, True, True])) == 3
    assert splits._longest_run(np.array([True, False, True, True, False, True])) == 2


def test_turbine_coverage_hand_counts(tiny):
    valid, _ = _tiny_masks(tiny)
    t = splits.turbine_coverage(valid, tiny, np.array([1, 2]))
    assert set(t) == {"val", "test", "train_5pct", "train_25pct", "train_100pct"}
    assert t["val"]["TurbID"] == [1, 2]
    assert t["val"]["valid_fraction"] == [0.0, round(134 / 144, 6)]  # the 10-step gap is inside val
    assert t["val"]["longest_invalid_run_steps"] == [144, 10]
    assert t["train_100pct"]["valid_fraction"] == [0.0, 1.0]
    assert t["train_100pct"]["longest_invalid_run_steps"] == [432, 0]


def test_window_counts_npz_round_trip(tiny, tmp_path):
    valid, missing = _tiny_masks(tiny)
    c = splits.window_counts(valid, missing, tiny)
    splits.save_window_counts(tmp_path / "counts.npz", c)
    back = splits.load_window_counts(tmp_path / "counts.npz")
    assert back.keys() == c.keys() and all(np.array_equal(back[k], c[k]) for k in c)


def test_mase_scale_is_the_full_train_range_for_every_subset(split):
    assert splits.mase_train_range(split) == (0, 24_624)
    assert splits.to_dict(split)["mase_train_range"] == [0, 24_624]
    # the 5% and 25% subsets are shorter, but their MASE denominator is still the full train range
    assert splits.subset_range(split, 0.05)[0] > splits.mase_train_range(split)[0]


def test_dump_json_keeps_lists_on_one_line_and_loads_back(split):
    text = splits.dump_json(splits.to_dict(split))
    assert json.loads(text) == splits.to_dict(split)
    assert text.count("\n") < 60  # ~300 origins would otherwise take ~300 lines


def test_committed_splits_json_is_current():
    # needs no data: the file must equal what the code generates
    assert splits.load_splits(REPO / "results" / "ds02_splits.json") == splits.make_split()


@pytest.mark.skipif(not CLEAN.exists(), reason="run python -m wpf.data.sdwpf first (data/sdwpf/clean.parquet)")
def test_committed_coverage_files_match_the_clean_data():
    """Catches a stale coverage file: if the DS-01 mask changes, these results must be regenerated."""
    from wpf.data import sdwpf

    df = sdwpf.load_clean(CLEAN)
    valid, missing = sdwpf.to_dense(df, "valid"), sdwpf.to_dense(df, "missing")
    assert valid.shape == (134, 35_280)
    split = splits.make_split()
    counts = splits.window_counts(valid, missing, split)
    assert counts["test_origins"].size * 134 == 134 * 195
    expected = {
        "windows": splits.window_coverage(counts),
        "per_turbine": splits.turbine_coverage(valid, split, np.sort(df["TurbID"].unique())),
    }
    committed = json.loads((REPO / "results" / "ds02_window_coverage.json").read_text())
    assert committed == json.loads(splits.dump_json(expected))
    saved = splits.load_window_counts(REPO / "results" / "ds02_window_counts.npz")
    assert saved.keys() == counts.keys() and all(np.array_equal(saved[k], counts[k]) for k in counts)
