"""EX-03 figures from the grid (wpf.grid_report rows): data-efficiency curves and accuracy against cost.

    python scripts/analysis/plot_grid.py [--results results] [--figures docs/figures]

Writes docs/figures/ex03_data_efficiency_{val,test}.png (MASE against the training fraction, one panel per horizon, with the
zero-shot and persistence references as dashed lines) and docs/figures/ex03_accuracy_vs_cost.png (test MASE at H = 64 against
SU per run and against trainable parameters). Needs matplotlib (not in requirements.txt: run it on a laptop).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from wpf import grid_report  # noqa: E402

_LOG = logging.getLogger(__name__)

# One fixed hue per model (colorblind-checked categorical palette, assigned by entity, never by rank).
COLORS = {"patchtst": "#2a78d6", "patchtst_point": "#eb6834", "lstm": "#1baf7a", "chronos_ft": "#eda100", "chronos_lora": "#e87ba4"}
MARKERS = {"patchtst": "o", "patchtst_point": "s", "lstm": "D", "chronos_ft": "^", "chronos_lora": "v"}
REFERENCES = {"chronos_zeroshot": ("#008300", "--"), "persistence": ("#6b6a63", ":")}
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e1e0d9"
HORIZONS = grid_report.HORIZONS


def _style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)


def data_efficiency(rows: list[dict], period: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=False)
    for ax, h in zip(axes, HORIZONS):
        for model, color in COLORS.items():
            pts = sorted((r["pct"], r[period][h]["mase"]) for r in rows if r["model"] == model and period in r)
            if not pts:
                continue
            x = [p for p, _ in pts]
            ax.errorbar(x, [s["mean"] for _, s in pts], yerr=[s["std"] or 0 for _, s in pts], color=color, marker=MARKERS[model],
                        markersize=5, linewidth=2, capsize=2, label=grid_report.LABELS[model])
        for model, (color, style) in REFERENCES.items():
            ref = next((r for r in rows if r["model"] == model and period in r), None)
            if ref:
                ax.axhline(ref[period][h]["mase"]["mean"], color=color, linestyle=style, linewidth=1.5, label=grid_report.LABELS[model])
        ax.set_xscale("log")
        ax.set_xticks([5, 25, 100])
        ax.set_xticklabels(["5%", "25%", "100%"])
        ax.minorticks_off()
        ax.set_title(f"H = {h} ({int(h) * 10} min)", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("training turbines", color=MUTED, fontsize=9)
        _style(ax)
    axes[0].set_ylabel(f"{period} MASE (lower is better)", color=MUTED, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Error against training data, {period} split, mean ± std over 3 seeds", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    _LOG.info("-> %s", out)


def accuracy_vs_cost(rows: list[dict], out: Path, horizon: str = "64") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    grid_rows = [r for r in rows if r["model"] in COLORS and "test" in r]
    panels = (("train_su", "training SU (V100, 12 SU per GPU-hour)"), ("su", "SU per run, training plus scoring"), ("trainable_parameters", "trainable parameters"))
    for ax, (key, xlabel) in zip(axes, panels):
        for model, color in COLORS.items():
            pts = [r for r in grid_rows if r["model"] == model and r[key]]
            ax.plot([r[key] for r in pts], [r["test"][horizon]["mase"]["mean"] for r in pts], color=color, marker=MARKERS[model],
                    markersize=6, linewidth=1.2, alpha=0.9, label=grid_report.LABELS[model])
            for r in pts:
                if model.startswith("chronos") and r["pct"] != 5:
                    continue  # the Chronos runs share one x; one label per model keeps the cluster legible
                ax.annotate(f"{r['pct']}%", (r[key], r["test"][horizon]["mase"]["mean"]), textcoords="offset points", xytext=(5, 3),
                            fontsize=7.5, color=MUTED)
        zero = next((r for r in rows if r["model"] == "chronos_zeroshot"), None)
        if zero and zero.get(key):
            ax.plot([zero[key]], [zero["test"][horizon]["mase"]["mean"]], marker="*", markersize=11, color=REFERENCES["chronos_zeroshot"][0],
                    linestyle="none", label=grid_report.LABELS["chronos_zeroshot"])
        pers = next((r for r in rows if r["model"] == "persistence"), None)
        if pers:
            ax.axhline(pers["test"][horizon]["mase"]["mean"], color=REFERENCES["persistence"][0], linestyle=":", linewidth=1.5,
                       label=grid_report.LABELS["persistence"])
        ax.set_xscale("log")
        if key in ("su", "train_su"):
            ax.set_xticks([0.25, 0.5, 1, 2, 5])
            ax.set_xticklabels(["0.25", "0.5", "1", "2", "5"])
            ax.minorticks_off()
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
        _style(ax)
    axes[0].set_ylabel(f"test MASE at H = {horizon}", color=MUTED, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Accuracy against cost at H = {horizon}: labels give the training fraction (Chronos: 5% labelled, 25% and 100% overlap it); zero-shot has no training cost", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    _LOG.info("-> %s", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="docs/figures")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    rows = grid_report.rows(a.results)
    figures = Path(a.figures)
    figures.mkdir(parents=True, exist_ok=True)
    for period in ("val", "test"):
        data_efficiency(rows, period, figures / f"ex03_data_efficiency_{period}.png")
    accuracy_vs_cost(rows, figures / "ex03_accuracy_vs_cost.png")


if __name__ == "__main__":
    main()
