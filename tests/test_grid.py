from pathlib import Path

import pytest

from wpf import grid

ROOT = Path(__file__).resolve().parents[1]


def test_grid_has_every_cell_once_with_its_config():
    cells = grid.cells()
    assert len(cells) == 5 * 3 * 3 + 1
    assert len({c.name for c in cells}) == len(cells)
    for c in cells:
        cfg = c.config_path()
        assert cfg is None or (ROOT / cfg).exists(), cfg
        assert (ROOT / c.spec.script).exists()
    assert {c.name for c in cells if c.seed is None} == {"ex_chronos_zeroshot"}
    assert grid.Cell("lstm", 25, 2023).name == "ex_lstm_25pct_s2023"


def test_every_cell_runs_on_the_v100():
    mig, v100 = grid.cells("mig"), grid.cells("v100")
    assert len(mig) == 0 and len(v100) == len(grid.cells()) == 46  # one GPU type: comparable GPU hours across models


def test_cell_env_and_shell_lines():
    c = grid.cells("v100")[0]
    env = c.env()
    assert env["NAME"] == c.name and env["CFG"] == "sdwpf_5pct_lr3e-5" and env["SEED"] == "2021" and env["SCRIPT"] == "slurm/sdwpf.sbatch"
    lora_cell = grid.Cell("chronos_lora", 100, 2022)
    lora = lora_cell.env()
    assert lora["PERIODS"] == "val test" and lora["RANK"] == "8" and lora["CFG"] == "chronos_lora_100pct_lr1e-5_nan"
    lines = grid.shell_lines(lora_cell).split("\n")
    assert "export PERIODS='val test'" in lines and "export SEED=2022" in lines
    zero = grid.Cell("chronos_zeroshot", None, None).env()
    assert "CFG" not in zero and zero["MODEL"] == "amazon/chronos-t5-small" and zero["SCRIPT"] == "slurm/chronos_eval.sbatch"


def test_submit_commands_and_missing_only(tmp_path):
    cmds = grid.submit_commands(output=tmp_path)
    assert len(cmds) == 1
    assert cmds[0][1] == "--array=0-45" and "--gres=gpu:v100:1" in cmds[0] and "--export=ALL,GRID_PARTITION=v100" in cmds[0]
    assert cmds[0][-1] == "slurm/grid.sbatch"
    grid.cells("v100")[0].record(tmp_path).write_text("{}")  # two cells done
    grid.cells("v100")[2].record(tmp_path).write_text("{}")
    cmds = grid.submit_commands(missing_only=True, output=tmp_path)
    assert len(cmds) == 1 and cmds[0][1] == "--array=" + ",".join(str(i) for i in range(46) if i not in (0, 2))
    assert grid.array_spec([4]) == "4" and grid.array_spec([3, 4, 5]) == "3-5"
    with pytest.raises(ValueError):
        grid.array_spec([])
