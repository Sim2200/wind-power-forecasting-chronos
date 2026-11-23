"""The second fold (EX-03): the split moved back 25 days, its test origins are the grid's validation origins, and the fold
configs differ from the grid configs only in the split file."""
import json
import sys
from pathlib import Path

from wpf.data import splits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import fold2  # noqa: E402


def test_fold2_test_origins_are_the_grid_val_origins():
    grid = splits.load_splits(ROOT / "results" / "ds02_splits.json")
    fold = splits.load_splits(ROOT / "results" / "ds02_splits_fold2.json")
    assert fold == fold2.fold2_split()
    assert fold.test == grid.val and fold.val[1] == grid.train[1]
    assert list(splits.eval_origins(fold, "test")) == list(splits.eval_origins(grid, "val"))


def test_fold2_configs_only_change_the_split():
    for base in ("sdwpf_100pct_lr3e-5", "lstm_100pct_s2021"):
        a = json.loads((ROOT / "configs" / f"{base}.json").read_text())
        b = json.loads((ROOT / "configs" / f"{base}_fold2.json").read_text())
        assert b.pop("splits_path") == "results/ds02_splits_fold2.json" and a == b
