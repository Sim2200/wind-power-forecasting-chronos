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


def test_window_coverage_hand_counts(tiny):
    valid = np.ones((2, tiny.n_steps), dtype=bool)
    valid[0, :] = False  # turbine 0 never valid, turbine 1 always
    cov = splits.window_coverage(valid, tiny)
    for horizon in splits.HORIZONS:
        c = cov["val"][str(horizon)]
        assert c["windows"] == 2 * 3  # 2 turbines x 3 origins
        assert c["valid_target_fraction"] == 0.5 and c["valid_history_fraction"] == 0.5
        assert c["windows_without_valid_target"] == 3 and c["windows_fully_valid_target"] == 3
    assert cov["test"]["64"]["windows"] == 2 * 7


@pytest.mark.skipif(not CLEAN.exists(), reason="run python -m wpf.data.sdwpf first (data/sdwpf/clean.parquet)")
def test_real_file_coverage_runs():
    from wpf.data import sdwpf

    valid = sdwpf.to_dense(sdwpf.load_clean(CLEAN), "valid")
    assert valid.shape == (134, 35_280)
    cov = splits.window_coverage(valid, splits.make_split())
    assert cov["test"]["64"]["windows"] == 134 * 195
    assert 0 < cov["test"]["64"]["valid_target_fraction"] < 1
