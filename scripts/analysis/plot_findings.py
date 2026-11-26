"""The three figures behind the EX-03 "how solid is the reversal" section, drawn from the JSON files, no compute:
docs/figures/ex03_weekly.png (test MASE per week at H = 36 and 64), ex03_bootstrap.png (paired bootstrap difference to zero-shot
Chronos with its 95% interval, validation and test) and ex03_two_periods.png (MASE on the two held-out periods: fold 2's test days
171 to 196 and the grid's test days 196 to 245).

    python scripts/analysis/plot_findings.py [--results results] [--figures docs/figures]
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

# the grid palette (plot_grid.py), validated for colour-vision deficiency; zero-shot and persistence are the references
COLORS = {"patchtst_100": "#2a78d6", "lstm_100": "#1baf7a", "chronos_ft_100": "#eda100", "chronos_lora_100": "#e87ba4",
          "chronos_zeroshot": "#008300", "persistence": "#6b6a63"}
MARKERS = {"patchtst_100": "o", "lstm_100": "D", "chronos_ft_100": "^", "chronos_lora_100": "v", "chronos_zeroshot": "*", "persistence": "x"}
LABELS = {"patchtst_100": "PatchTST (quantile), 100%", "lstm_100": "LSTM, 100%", "chronos_ft_100": "Chronos-small, full fine-tune, 100%",
          "chronos_lora_100": "Chronos-small, LoRA r8, 100%", "chronos_zeroshot": "Chronos-small, zero-shot", "persistence": "Persistence"}
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e1e0d9"
HORIZONS = ("36", "64")

# fold-2 records: the grid's validation days (171 to 196) scored as a test period nothing was chosen on
FOLD2 = {"patchtst_100": "fold2_patchtst_100pct", "lstm_100": "fold2_lstm_100pct", "chronos_ft_100": "fold2_chronos_ft_100pct",
         "chronos_lora_100": "fold2_chronos_lora_100pct", "chronos_zeroshot": "fold2_chronos_zeroshot"}
GRID_SEEDS = {"patchtst_100": "ex_patchtst_100pct_seeds", "lstm_100": "ex_lstm_100pct_seeds", "chronos_ft_100": "ex_chronos_ft_100pct_seeds",
              "chronos_lora_100": "ex_chronos_lora_100pct_seeds"}


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)


def weekly(breakdown: dict, out: Path) -> None:
    weeks = breakdown["weeks"]
    n = len(weeks["chronos_zeroshot"]["64"])
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(11, 3.8), sharey=True)
    x = list(range(1, n + 1))
    for ax, h in zip(axes, HORIZONS):
        for model, color in COLORS.items():
            style = "--" if model == "chronos_zeroshot" else ":" if model == "persistence" else "-"
            ax.plot(x, weeks[model][h], color=color, linestyle=style, marker=MARKERS[model], markersize=5 if model != "chronos_zeroshot" else 8,
                    linewidth=1.8, label=LABELS[model])
        ax.set_title(f"H = {h}", fontsize=10, color=INK, loc="left")
        ax.set_xticks(x)
        ax.set_xlabel(f"test week ({breakdown['week_days']} days each)", color=MUTED, fontsize=9)
        _style(ax)
    axes[0].set_ylabel("test MASE (mean over seeds)", color=MUTED, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Test MASE per week: the from-scratch models lead only in the calmest weeks", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    _LOG.info("-> %s", out)


def bootstrap(breakdown: dict, out: Path) -> None:
    boot = breakdown["bootstrap"]
    models = [m for m in COLORS if m != "chronos_zeroshot"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, period in zip(axes, ("val", "test")):
        for j, h in enumerate(("6", "36", "64")):
            for i, model in enumerate(models):
                stat = boot[period][model][h]
                y = j * (len(models) + 1) + i
                lo, hi = stat["diff_ci"]
                ax.errorbar(stat["diff_vs_zeroshot"], y, xerr=[[stat["diff_vs_zeroshot"] - lo], [hi - stat["diff_vs_zeroshot"]]], color=COLORS[model],
                            marker=MARKERS[model], markersize=6, linewidth=1.8, capsize=3, label=LABELS[model] if (j == 0 and period == "val") else None)
        ax.axvline(0, color=COLORS["chronos_zeroshot"], linestyle="--", linewidth=1.4)
        ticks = [j * (len(models) + 1) + (len(models) - 1) / 2 for j in range(3)]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"H = {h}" for h in ("6", "36", "64")])
        ax.invert_yaxis()
        ax.set_title({"val": "validation (days 171 to 196)", "test": "test (days 196 to 245)"}[period], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("MASE difference to zero-shot Chronos (negative: better), 95% bootstrap interval", color=MUTED, fontsize=8.5)
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle(f"Paired difference to zero-shot Chronos, {breakdown['bootstrap_resamples']} resamples of the forecast origins "
                 f"({breakdown['bootstrap_seed_runs']})", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    _LOG.info("-> %s", out)


def _fold2(results: Path, model: str, h: str) -> tuple[float, float | None]:
    """Fold-2 test MASE at `h`: mean ± std over the seed files present (seed 2021 has no suffix), std None for one seed."""
    values = []
    for suffix in ("", "_s2022", "_s2023"):
        path = results / f"{FOLD2[model]}{suffix}_test.json"
        if path.exists():
            values.append(json.loads(path.read_text())[h]["mase"])
    mean = sum(values) / len(values)
    std = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5 if len(values) > 1 else None
    return mean, std


def _grid_test(results: Path, model: str, h: str) -> tuple[float, float | None]:
    if model == "chronos_zeroshot":
        return json.loads((results / "ex_chronos_zeroshot_test.json").read_text())[h]["mase"], None
    stat = json.loads((results / f"{GRID_SEEDS[model]}.json").read_text())["test"][h]["mase"]
    return stat["mean"], stat["std"]


def two_periods(results: Path, out: Path) -> None:
    models = list(FOLD2)
    periods = (("fold 2 test: days 171 to 196", _fold2), ("grid test: days 196 to 245", _grid_test))
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(11, 3.9), sharey=True)
    width = 0.8 / len(models)
    for ax, h in zip(axes, HORIZONS):
        for p, (title, score) in enumerate(periods):
            for i, model in enumerate(models):
                mean, std = score(results, model, h)
                x = p + (i - (len(models) - 1) / 2) * width
                ax.bar(x, mean, width * 0.9, color=COLORS[model], yerr=std, capsize=2, ecolor=INK, label=LABELS[model] if (p == 0 and h == HORIZONS[0]) else None)
                ax.annotate(f"{mean:.3f}", (x, mean), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=6.5, color=MUTED)
        ax.set_xticks(range(len(periods)))
        ax.set_xticklabels([t for t, _ in periods], fontsize=9)
        ax.set_title(f"H = {h}", fontsize=10, color=INK, loc="left")
        ax.set_ylim(0.4, 0.75)
        _style(ax)
    axes[0].set_ylabel("MASE on the held-out period", color=MUTED, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Two held-out periods, two rankings: the from-scratch models win days 171 to 196, Chronos wins days 196 to 245 "
                 "(bars: mean over seeds where several ran, whiskers: std)", x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    _LOG.info("-> %s", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="docs/figures")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    results, figures = Path(a.results), Path(a.figures)
    figures.mkdir(parents=True, exist_ok=True)
    breakdown = json.loads((results / "ex03_test_breakdown.json").read_text())
    weekly(breakdown, figures / "ex03_weekly.png")
    bootstrap(breakdown, figures / "ex03_bootstrap.png")
    two_periods(results, figures / "ex03_two_periods.png")


if __name__ == "__main__":
    main()
