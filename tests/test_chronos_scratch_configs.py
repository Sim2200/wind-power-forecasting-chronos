"""The CF-05 random-initialization configs are the CF-01 grid recipe without the pretrained weights."""
from pathlib import Path

import yaml

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _load(name):
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())


def test_scratch_configs_are_the_recipe_without_pretrained_weights():
    small = _load("chronos_ft_5pct_lr1e-5_nan")
    for lr, value in (("1e-3", 0.001), ("1e-5", 0.00001)):
        scratch = _load(f"chronos_scratch_5pct_lr{lr}_nan")
        assert scratch["random_init"] is True and scratch["learning_rate"] == value
        changed = {k for k in set(small) | set(scratch) if small.get(k) != scratch.get(k)}
        assert changed <= {"random_init", "learning_rate", "output_dir"}
