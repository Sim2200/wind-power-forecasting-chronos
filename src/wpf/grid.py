"""The experiment grid (EX-01): every cell of {model} x {5, 25, 100%} x {3 seeds} as one Slurm array task.

Models and the recipe each one settled on earlier:
- `patchtst`: the quantile PatchTST, lr 3e-5 (PT-03), `configs/sdwpf_{5,25,100}pct_lr3e-5.json`
- `patchtst_point`: the point head on the same recipe (PT-04), `configs/sdwpf_point_{5,25,100}pct.json`
- `lstm`: the DS-06 LSTM, `configs/lstm_{5,25,100}pct_s2021.json` (its seed is overridden)
- `chronos_ft`: full fine-tuning of chronos-t5-small, lr 1e-5, 1000 steps, nan export (CF-01), V100 (does not fit a MIG slice)
- `chronos_lora`: LoRA rank 8 on the same recipe (CF-02), MIG
- `chronos_zeroshot`: the pretrained model, no seeds (DS-07), one cell

Every cell runs the model's existing sbatch script (`bash slurm/<script>` inside the array task's allocation) with
`CFG`, `SEED`, `NAME` and `PERIODS="val test"`, so each run scores the validation windows and, exactly once, the test
windows, and its record output/<name>.json carries train_seconds, inference_seconds, the GPU and the parameter count.
Names are uniform, `ex_<model>_<pct>pct_s<seed>`, so the EX-03 analysis can find every cell.

    python -m wpf.grid list [--partition mig|v100]       the cells, one line each (the dry run)
    python -m wpf.grid status                            which cells have a record in output/
    python -m wpf.grid submit [--dry-run] [--missing]    submit the MIG array and the V100 array (--missing: only
                                                         cells without a record, for re-running failed cells)
    python -m wpf.grid cell --partition mig --index 3    the shell lines slurm/grid.sbatch evaluates for one task
"""
from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

_LOG = logging.getLogger(__name__)

FRACTIONS = (5, 25, 100)
SEEDS = (2021, 2022, 2023)
PARTITIONS = {
    "mig": ["--gpus=a100_1g.5gb:1"],
    "v100": ["--gpus=v100:1", "--gres=gpu:v100:1"],
}
SU_PER_HOUR = {"mig": 7, "v100": 12}


@dataclass(frozen=True)
class Model:
    script: str
    config: str  # format with pct
    partition: str
    env: dict = field(default_factory=dict)
    seeded: bool = True


MODELS = {
    "patchtst": Model("slurm/sdwpf.sbatch", "sdwpf_{pct}pct_lr3e-5", "mig"),
    "patchtst_point": Model("slurm/sdwpf.sbatch", "sdwpf_point_{pct}pct", "mig"),
    "lstm": Model("slurm/sdwpf.sbatch", "lstm_{pct}pct_s2021", "mig"),
    "chronos_ft": Model("slurm/chronos_ft.sbatch", "chronos_ft_{pct}pct_lr1e-5_nan", "v100", {"PERIODS": "val test"}),
    "chronos_lora": Model("slurm/chronos_lora.sbatch", "chronos_lora_{pct}pct_lr1e-5_nan", "mig", {"PERIODS": "val test", "RANK": "8"}),
    "chronos_zeroshot": Model("slurm/chronos_eval.sbatch", "", "mig",
                              {"MODEL": "amazon/chronos-t5-small", "CONTEXT": "nan", "PERIODS": "val test"}, seeded=False),
}


@dataclass(frozen=True)
class Cell:
    model: str
    pct: int | None
    seed: int | None

    @property
    def name(self) -> str:
        if self.seed is None:
            return f"ex_{self.model}"
        return f"ex_{self.model}_{self.pct}pct_s{self.seed}"

    @property
    def spec(self) -> Model:
        return MODELS[self.model]

    @property
    def partition(self) -> str:
        return self.spec.partition

    def env(self) -> dict[str, str]:
        """The environment the model's sbatch script reads, plus SCRIPT for slurm/grid.sbatch."""
        out = {"NAME": self.name, "SCRIPT": self.spec.script, **self.spec.env}
        if self.seed is not None:
            out["CFG"] = self.spec.config.format(pct=self.pct)
            out["SEED"] = str(self.seed)
        return out

    def config_path(self) -> Path | None:
        if self.seed is None:
            return None
        cfg = self.spec.config.format(pct=self.pct)
        return Path("configs") / (cfg + (".json" if self.spec.script == "slurm/sdwpf.sbatch" else ".yaml"))

    def record(self, output: str | Path = "output") -> Path:
        return Path(output) / f"{self.name}.json"


def cells(partition: str | None = None) -> list[Cell]:
    """Every cell in a fixed order (the array index); `partition` keeps one partition's cells."""
    out = []
    for model, spec in MODELS.items():
        if spec.seeded:
            out += [Cell(model, pct, seed) for pct in FRACTIONS for seed in SEEDS]
        else:
            out.append(Cell(model, None, None))
    if partition is not None:
        out = [c for c in out if c.partition == partition]
    return out


def shell_lines(cell: Cell) -> str:
    """`export K=V` lines for `eval` in slurm/grid.sbatch."""
    return "\n".join(f"export {k}={shlex.quote(v)}" for k, v in cell.env().items())


def array_spec(indices: list[int]) -> str:
    """Slurm --array value for the indices: a range when contiguous, else a list."""
    if not indices:
        raise ValueError("no cells")
    if indices == list(range(indices[0], indices[-1] + 1)):
        return f"{indices[0]}-{indices[-1]}" if len(indices) > 1 else str(indices[0])
    return ",".join(map(str, indices))


def submit_commands(missing_only: bool = False, output: str | Path = "output", sbatch: str = "slurm/grid.sbatch") -> list[list[str]]:
    """One sbatch command per partition, arrays over that partition's cell indices (all, or those without a record)."""
    _LOG.debug("missing_only=%s", missing_only)
    commands = []
    for partition, flags in PARTITIONS.items():
        part_cells = cells(partition)
        indices = [i for i, c in enumerate(part_cells) if not (missing_only and c.record(output).exists())]
        if not indices:
            continue
        commands.append(["sbatch", f"--array={array_spec(indices)}", *flags, f"--export=ALL,GRID_PARTITION={partition}", sbatch])
    return commands


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list"); p.add_argument("--partition", choices=tuple(PARTITIONS))
    sub.add_parser("status")
    p = sub.add_parser("submit"); p.add_argument("--dry-run", action="store_true"); p.add_argument("--missing", action="store_true")
    p = sub.add_parser("cell"); p.add_argument("--partition", required=True, choices=tuple(PARTITIONS)); p.add_argument("--index", type=int, required=True)
    a = ap.parse_args()
    if a.cmd == "list":
        for partition in ([a.partition] if a.partition else PARTITIONS):
            for i, c in enumerate(cells(partition)):
                env = " ".join(f"{k}={shlex.quote(v)}" for k, v in c.env().items() if k != "SCRIPT")
                print(f"{partition}[{i}] {c.name}: {env} bash {c.spec.script}")
    elif a.cmd == "status":
        done = [c for c in cells() if c.record().exists()]
        for c in cells():
            print(f"{'done   ' if c.record().exists() else 'missing'} {c.name}")
        print(f"{len(done)} of {len(cells())} cells have a record")
    elif a.cmd == "submit":
        for cmd in submit_commands(a.missing):
            print(" ".join(cmd))
            if not a.dry_run:
                subprocess.run(cmd, check=True)
    elif a.cmd == "cell":
        print(shell_lines(cells(a.partition)[a.index]))


if __name__ == "__main__":
    main()
