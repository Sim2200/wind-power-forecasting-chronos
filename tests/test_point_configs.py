"""The PT-04 point configs mirror the quantile grid configs with only the head (and so the loss) changed."""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from wpf import train_sdwpf as T

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
GRID = {"5pct": "sdwpf_5pct_lr3e-5", "25pct": "sdwpf_25pct_lr3e-5", "100pct": "sdwpf_100pct_lr3e-5"}


@pytest.mark.parametrize("frac", ["smoke", "5pct", "25pct", "100pct"])
def test_point_config_is_a_point_patchtst(frac):
    cfg = T.load_config(str(CONFIGS / f"sdwpf_point_{frac}.json"))
    assert cfg.quantiles is None and cfg.model_type == "patchtst"
    model_cfg, model_class = T.build_model_config(cfg)
    assert model_cfg.n_quantiles == 0 and model_class.__name__ == "PatchTST"


@pytest.mark.parametrize("frac", list(GRID))
def test_point_grid_config_differs_from_quantile_grid_config_only_in_the_head(frac):
    point = json.loads((CONFIGS / f"sdwpf_point_{frac}.json").read_text())
    quant = json.loads((CONFIGS / f"{GRID[frac]}.json").read_text())
    assert "quantiles" not in point and quant.pop("quantiles")
    assert point == quant
    assert point["learning_rate"] == pytest.approx(3e-5)
