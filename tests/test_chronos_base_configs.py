"""The CF-03 chronos-t5-base configs are the CF-01 grid recipe with only the model (and, for timing, the step count) changed."""
from pathlib import Path

import yaml

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _load(name):
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())


def test_base_5pct_config_is_the_small_recipe_with_the_base_model():
    small, base = _load("chronos_ft_5pct_lr1e-5_nan"), _load("chronos_ft_base_5pct_nan")
    assert base["model_id"] == "amazon/chronos-t5-base" and base["tf32"] is False
    changed = {k for k in set(small) | set(base) if small.get(k) != base.get(k)}
    assert changed == {"model_id", "tf32", "output_dir"}


def test_scaling_config_only_shortens_the_run():
    base, scaling = _load("chronos_ft_base_5pct_nan"), _load("chronos_ft_base_scaling")
    changed = {k for k in set(base) | set(scaling) if base.get(k) != scaling.get(k)}
    assert changed == {"max_steps", "save_steps", "output_dir"} and scaling["max_steps"] == 200


def test_effective_batch_splits_over_the_gpu_counts():
    for name in ("chronos_ft_base_5pct_nan", "chronos_ft_base_scaling"):
        batch = _load(name)["per_device_train_batch_size"]
        assert batch == 32 and all(batch % g == 0 for g in (1, 2, 4))
