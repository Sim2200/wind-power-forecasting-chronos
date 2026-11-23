"""PT-05: neighbour lookup, the spatial windows and the spatial PatchTST forward."""
import json
from pathlib import Path

import numpy as np
import pytest

from wpf.data import spatial, splits

ROOT = Path(__file__).resolve().parents[1]


def test_nearest_neighbours_on_a_line():
    x, y = np.array([0.0, 1.0, 2.0, 10.0]), np.zeros(4)
    nn = spatial.nearest_neighbours(x, y, 2)
    assert nn.tolist() == [[1, 2], [0, 2], [1, 0], [2, 1]]  # nearest first, never itself
    with pytest.raises(ValueError):
        spatial.nearest_neighbours(x, y, 4)


def test_spatial_windows_put_the_centre_first_and_keep_its_target():
    pytest.importorskip("torch")
    split = splits.make_split((6, 1, 2))
    steps, turbines = split.n_steps, 3
    series = (np.arange(turbines)[:, None] * 1000 + np.arange(steps)[None, :]).astype(np.float32)
    valid = np.ones_like(series, dtype=bool)
    neighbours = np.array([[2, 1], [0, 2], [1, 0]])
    index = np.array([[1, 500]])
    x, y, mask = spatial.SpatialWindows(series, valid, index, 64, neighbours)[0]
    assert x.shape == (splits.LOOKBACK, 3) and y.shape == (64, 1) and mask.shape == (64, 1)
    assert x[0, 0] == 1000 + 500 - splits.LOOKBACK and x[0, 1] == 0 + 500 - splits.LOOKBACK and x[0, 2] == 2000 + 500 - splits.LOOKBACK
    assert y[0, 0] == 1500
    arrays = spatial.eval_arrays(series, valid, split, "val", 64, neighbours)
    n_origins = len(splits.eval_origins(split, "val"))
    assert arrays["x"].shape == (turbines, n_origins, splits.LOOKBACK, 3) and arrays["y"].shape == (turbines, n_origins, 64)
    np.testing.assert_array_equal(arrays["x"][1, :, :, 1], arrays["x"][0, :, :, 0])  # turbine 1's first neighbour is turbine 0


def test_spatial_patchtst_forward_shapes_and_centre_denorm():
    torch = pytest.importorskip("torch")
    from wpf.models.spatial_patchtst import SpatialPatchTST, SpatialPatchTSTConfig

    cfg = SpatialPatchTSTConfig(n_vars=5, d_model=16, n_heads=4, d_ff=32, n_layers=1, dropout=0.0)
    model = SpatialPatchTST(cfg).eval()
    x = torch.randn(3, 288, 5)
    assert model(x).shape == (3, 64, 1)
    model_q = SpatialPatchTST(SpatialPatchTSTConfig(n_vars=5, d_model=16, n_heads=4, d_ff=32, n_layers=1, dropout=0.0, n_quantiles=9)).eval()
    y = model_q(x)
    assert y.shape == (3, 64, 1, 9) and bool((y[..., 1:] >= y[..., :-1]).all())
    # Rescaling a neighbour channel changes nothing after RevIN; rescaling the centre scales the forecast with it.
    x2 = x.clone()
    x2[:, :, 1] *= 10
    assert torch.allclose(model(x), model(x2), atol=1e-5)
    x3 = x.clone()
    x3[:, :, 0] = x[:, :, 0] * 2 + 7
    assert torch.allclose(model(x3), model(x) * 2 + 7, atol=1e-3)
    with pytest.raises(ValueError, match="revin_affine"):
        SpatialPatchTST(SpatialPatchTSTConfig(n_vars=5, revin_affine=True))


def test_spatial_config_is_the_grid_recipe_plus_neighbours():
    grid = json.loads((ROOT / "configs" / "sdwpf_100pct_lr3e-5.json").read_text())
    spatial_cfg = json.loads((ROOT / "configs" / "sdwpf_spatial_100pct_k4.json").read_text())
    assert spatial_cfg.pop("model_type") == "patchtst_spatial" and spatial_cfg.pop("neighbours") == 4
    assert spatial_cfg == grid


def test_build_model_config_checks_neighbours():
    pytest.importorskip("torch")
    from wpf import train_sdwpf

    cfg = train_sdwpf.TrainConfig(model_type="patchtst_spatial", neighbours=4, quantiles=[0.1, 0.5, 0.9])
    model_cfg, model_class = train_sdwpf.build_model_config(cfg)
    assert model_cfg.n_vars == 5 and model_cfg.n_quantiles == 3 and model_class.__name__ == "SpatialPatchTST"
    with pytest.raises(ValueError):
        train_sdwpf.build_model_config(train_sdwpf.TrainConfig(model_type="patchtst_spatial"))
    with pytest.raises(ValueError):
        train_sdwpf.build_model_config(train_sdwpf.TrainConfig(model_type="patchtst", neighbours=2))
