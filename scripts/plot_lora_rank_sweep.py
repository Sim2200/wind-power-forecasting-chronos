"""Figure for the LoRA rank sweep (#59): validation MASE and WQL against the number of trainable parameters.

Reads the rank-sweep records written by slurm/chronos_lora.sbatch (results/chronos_lora_5pct_lr1e-5_nan_r{4,8,16,32}.json
with their *_val.json metric files) plus the zero-shot and full fine-tuning rows of the same 5% recipe as reference
lines, and writes docs/figures/lora_rank_sweep.png. Needs matplotlib (not in requirements.txt: run it on a laptop).

    python scripts/plot_lora_rank_sweep.py [--results results] [--out docs/figures/lora_rank_sweep.png]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_LOG = logging.getLogger(__name__)

RANKS = (4, 8, 16, 32)
HORIZONS = ("6", "36", "64")
METRICS = ("mase", "wql")


def load(results: Path, name: str) -> tuple[dict, dict]:
    """(record, validation metrics) of one run name."""
    return json.loads((results / f"{name}.json").read_text()), json.loads((results / f"{name}_val.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="docs/figures/lora_rank_sweep.png")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    results = Path(a.results)

    runs = [load(results, f"chronos_lora_5pct_lr1e-5_nan_r{r}") for r in RANKS]
    params = [rec["train_info"]["trainable_parameters"] for rec, _ in runs]
    _, zero_shot = load(results, "chronos_small_zeroshot_nan")
    full_rec, full_ft = load(results, "chronos_ft_5pct_lr1e-5_nan")
    full_params = full_rec["parameters"]["trainable"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for ax, metric in zip(axes, METRICS):
        for h, color in zip(HORIZONS, ("C0", "C1", "C2")):
            ax.plot(params, [val[h][metric] for _, val in runs], marker="o", color=color, label=f"LoRA, H = {h}")
            ax.axhline(zero_shot[h][metric], color=color, linestyle=":", linewidth=1)
            ax.plot([full_params], [full_ft[h][metric]], marker="s", color=color, linestyle="none", markerfacecolor="none")
        ax.set_xscale("log")
        ax.set_xlabel("trainable parameters")
        ax.set_ylabel(f"validation {metric.upper()}")
        ax.set_xticks(params + [full_params])
        ax.set_xticklabels([f"r{r}\n{p / 1e6:.2f}M" for r, p in zip(RANKS, params)] + [f"full FT\n{full_params / 1e6:.1f}M"])
        ax.grid(alpha=0.3)
    axes[0].plot([], [], color="gray", linestyle=":", label="zero-shot (dotted)")
    axes[0].plot([], [], color="gray", marker="s", linestyle="none", markerfacecolor="none", label="full fine-tune, lr 1e-5 (square)")
    axes[0].legend(fontsize=8, loc="center left")
    fig.suptitle("chronos-t5-small on the 5% subset, lr 1e-5, 1000 steps, seed 2021: LoRA rank 4 to 32")
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    _LOG.info("wrote %s", a.out)


if __name__ == "__main__":
    main()
