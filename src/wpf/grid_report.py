"""EX-03: the grid's numbers in one place for the analysis (tables, curves, cost).

Reads results/ex_grid.json (one row per cell, written by wpf.grid collect), the seed summaries results/ex_<model>_<pct>pct_seeds.json,
the Chronos-T5-base runs of CF-03 and CF-06 (validation and, where it was scored, test), and returns one row per
(model, fraction): mean and std over seeds of MASE, WQL and MAE at H = 6 / 36 / 64 on validation and test, plus the mean
SU, trainable parameters, train and inference seconds of the cells.

    python -m wpf.grid_report [--results results] [--out docs/analysis/ex03_tables.md] [--job NAME=JOBID ...]

The SU of a base run comes from sacct on the record's slurm_job_id; the CF-03 records predate that field, so their job ids are
given with --job (BASE_JOBS holds them). Run it on Zaratan for the SU column; elsewhere the column reads "–" for those runs.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from wpf import grid

_LOG = logging.getLogger(__name__)

HORIZONS = grid.HORIZONS
METRICS = ("mase", "wql", "mae")
# Display order and names; the grid models first, then the single-run references.
LABELS = {"patchtst": "PatchTST (quantile)", "patchtst_point": "PatchTST (point)", "lstm": "LSTM",
          "chronos_ft": "Chronos-small, full fine-tune", "chronos_lora": "Chronos-small, LoRA r8",
          "chronos_zeroshot": "Chronos-small, zero-shot", "chronos_base_ft": "Chronos-base, full fine-tune (2 GPUs)",
          "chronos_base_zeroshot": "Chronos-base, zero-shot", "persistence": "Persistence", "seasonal_naive": "Seasonal naive (1 day)"}
BASE_RUNS = {  # CF-03 and CF-06 records outside the grid: (model key, fraction, record name); test exists only where PERIODS had it
    ("chronos_base_zeroshot", None): "chronos_base_zeroshot_nan",
    ("chronos_base_ft", 5): "chronos_ft_base_5pct_nan_g2",
    ("chronos_base_ft", 100): "chronos_ft_base_100pct_nan_g2",
}


BASE_JOBS = {"chronos_base_zeroshot_nan": "24123439", "chronos_ft_base_5pct_nan_g2": "24123428"}  # CF-03 jobs (no id in the record)


def _stat(mean: float | None, std: float | None = None, n: int = 1) -> dict:
    return {"mean": mean, "std": std, "n": n}


def _period_from_summary(summary: dict, period: str) -> dict:
    return {h: {m: _stat(summary[period][h][m]["mean"], summary[period][h][m]["std"], summary[period][h][m]["n"])
                for m in METRICS if summary[period][h].get(m)} for h in HORIZONS}


def _period_from_scores(scores: dict) -> dict:
    return {h: {m: _stat(scores[h].get(m)) for m in METRICS if scores[h].get(m) is not None} for h in HORIZONS if h in scores}


def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 2) if values else None


def rows(results: str | Path = "results", jobs: dict[str, str] | None = None) -> list[dict]:
    """One row per (model, fraction): metrics (mean, std, n) per period and horizon, and the mean cost of its cells."""
    results = Path(results)
    jobs = {**BASE_JOBS, **(jobs or {})}
    table = json.loads((results / "ex_grid.json").read_text())
    out = []
    for model, spec in grid.MODELS.items():
        fractions = grid.FRACTIONS if spec.seeded else (None,)
        for pct in fractions:
            cells = [c for c in table["cells"] if c["model"] == model and c["pct"] == pct]
            if not cells:
                _LOG.warning("no cells for %s %s", model, pct)
                continue
            row = {"model": model, "label": LABELS[model], "pct": pct, "n_runs": len(cells),
                   "su": _mean([c["su"] for c in cells]), "trainable_parameters": cells[0]["trainable_parameters"],
                   "train_seconds": _mean([c["train_seconds"] for c in cells]),
                   "inference_seconds": _mean([c["inference_seconds"] for c in cells]), "gpus": 1}
            if spec.seeded:
                summary = json.loads((results / f"ex_{model}_{pct}pct_seeds.json").read_text())
                row["val"], row["test"] = _period_from_summary(summary, "val"), _period_from_summary(summary, "test")
            else:
                row["val"], row["test"] = _period_from_scores(cells[0]["val"]), _period_from_scores(cells[0]["test"])
            out.append(row)
    for (model, pct), name in BASE_RUNS.items():
        record = results / f"{name}.json"
        if not record.exists():
            _LOG.info("base run %s not there yet", name)
            continue
        rec = json.loads(record.read_text())
        row = {"model": model, "label": LABELS[model], "pct": pct, "n_runs": 1, "su": None,
               "trainable_parameters": rec["parameters"]["trainable"], "train_seconds": rec.get("train_seconds"),
               "inference_seconds": rec.get("inference_seconds"), "gpus": 2 if rec.get("train_seconds") else 1}
        for period in ("val", "test"):
            if period in rec:
                row[period] = _period_from_scores(rec[period])
        job = jobs.get(name) or rec.get("slurm_job_id")
        if job:
            acct = grid.sacct(job)
            if acct:
                row["su"] = acct["su"]
        out.append(row)
    for b in table["baselines"]:
        out.append({"model": b["model"], "label": LABELS[b["model"]], "pct": None, "n_runs": 1, "su": 0.0, "trainable_parameters": 0,
                    "train_seconds": None, "inference_seconds": None, "gpus": 0,
                    "val": _period_from_scores(b["val"]), "test": _period_from_scores(b["test"])})
    return out


def fmt(stat: dict | None, digits: int = 3) -> str:
    """`0.593 ± 0.001` for a seed mean, `0.593` for a single run, `–` when the metric does not exist."""
    if not stat or stat["mean"] is None:
        return "–"
    if stat["n"] > 1 and stat["std"] is not None:
        return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"
    return f"{stat['mean']:.{digits}f}"


def _pct(row: dict) -> str:
    return f"{row['pct']}%" if row["pct"] is not None else "–"


def metric_table(rows_: list[dict], period: str, metric: str) -> str:
    """Markdown: one line per row, the metric at H = 6 / 36 / 64 for one period."""
    digits = 0 if metric == "mae" else 3
    lines = [f"| model | train data | H = 6 | H = 36 | H = 64 |", "|---|---|---|---|---|"]
    for r in rows_:
        if period not in r:
            continue
        cells = [fmt(r[period].get(h, {}).get(metric), digits) for h in HORIZONS]
        lines.append(f"| {r['label']} | {_pct(r)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cost_table(rows_: list[dict]) -> str:
    """Markdown: SU per run, trainable parameters, train and inference seconds, next to validation and test MASE at H = 64."""
    lines = ["| model | train data | runs | SU per run | GPUs | trainable params | train s | inference s | val MASE H=64 | test MASE H=64 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows_:
        su = "–" if r["su"] is None else f"{r['su']:.2f}"
        params = f"{r['trainable_parameters'] / 1e6:.2f}M" if r["trainable_parameters"] else "0"
        train = "–" if r["train_seconds"] is None else f"{r['train_seconds']:.0f}"
        infer = "–" if r["inference_seconds"] is None else f"{r['inference_seconds']:.0f}"
        val = fmt(r.get("val", {}).get("64", {}).get("mase"))
        test = fmt(r.get("test", {}).get("64", {}).get("mase"))
        lines.append(f"| {r['label']} | {_pct(r)} | {r['n_runs']} | {su} | {r['gpus']} | {params} | {train} | {infer} | {val} | {test} |")
    return "\n".join(lines)


def report(rows_: list[dict], su_total: float) -> str:
    parts = ["# EX-03 tables: the grid at H = 6 / 36 / 64", "",
             "Generated by `python -m wpf.grid_report` from `results/ex_grid.json`, the seed summaries and the Chronos-T5-base records. "
             "Mean ± std over seeds 2021, 2022, 2023 where a model was run three times; single runs show the value alone. "
             "Validation picked every recipe; the test split was scored once per final model. "
             f"Grid cost: {su_total:.1f} SU for {sum(r['n_runs'] for r in rows_ if r['model'] in grid.MODELS)} runs.", ""]
    for period, title in (("val", "Validation"), ("test", "Test")):
        for metric, name in (("mase", "MASE"), ("wql", "WQL"), ("mae", "MAE (kW)")):
            parts += [f"## {title}: {name}", "", metric_table(rows_, period, metric), ""]
    parts += ["## Cost", "", cost_table(rows_), ""]
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="docs/analysis/ex03_tables.md")
    ap.add_argument("--job", action="append", default=[], metavar="NAME=JOBID", help="Slurm job id of a base run whose record has none")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    rows_ = rows(a.results, dict(j.split("=", 1) for j in a.job))
    su_total = json.loads((Path(a.results) / "ex_grid.json").read_text())["su_total"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(report(rows_, su_total))
    _LOG.info("%d rows -> %s", len(rows_), a.out)


if __name__ == "__main__":
    main()
