"""Score a Chronos checkpoint, zero-shot or fine-tuned, on the shared SDWPF evaluation windows.

The same windows and metrics as PatchTST (`wpf.score_sdwpf`, DS-04): contexts are the 288 kW readings before each
of the shared origins of a period, one per turbine, and `ChronosPipeline.predict_quantiles` gives the 9 deciles of
the 64 steps after it. The median is the point forecast, as for PatchTST's quantile head. Outputs:
- `<out>.json`: the run record (model path, context mode, parameters, inference seconds, metrics per period);
- `<out>_pred.npz`: `val` (median, turbines x origins x H), `val_q` (Q x turbines x origins x H), `quantiles`,
  and the same for `test` when asked, the layout `wpf.score_sdwpf` reads;
- `results/<name>_<period>.json` through `metrics.save_results` when `--results` is given.

Technical decisions:
- Context mode `keep` (default) gives Chronos the readings as stored, flagged rows included, NaN only where the
  reading is missing: the same history PatchTST sees. `nan` blanks the unknown and abnormal rows as well, the
  DS-03 `--flagged nan` export; a checkpoint fine-tuned on that export is scored with it.
- The point forecast is the 0.5 quantile of the samples, not the sample mean `predict_quantiles` also returns, so
  the point and quantile metrics describe the same forecast.
- Validation is the default and the only period CF-01 and DS-07 report; `--periods val test` is for the grid, which
  scores test once per final model.

Usage:
    python -m wpf.chronos_eval --model amazon/chronos-t5-small --name chronos_small_zeroshot --out output/chronos_small_zeroshot.json
    python -m wpf.chronos_eval --model output/chronos/chronos_ft_5pct/run-0/checkpoint-final --name chronos_ft_5pct \\
        --out output/chronos_ft_5pct.json --results results/
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from wpf import metrics, score_sdwpf
from wpf.data import splits

_LOG = logging.getLogger(__name__)

CONTEXT_MODES = ("keep", "nan")
QUANTILES = metrics.QUANTILES


def contexts(patv: np.ndarray, valid: np.ndarray, split: splits.Split, period: str, mode: str = "keep") -> np.ndarray:
    """(turbines, origins, L) kW histories before the shared origins of `period`, NaN where missing (`keep`) or
    where not valid (`nan`)."""
    _LOG.debug("period='%s' mode='%s'", period, mode)
    if mode not in CONTEXT_MODES:
        raise ValueError(f"mode must be one of {CONTEXT_MODES}")
    series = patv if mode == "keep" else np.where(valid, patv, np.nan)
    origins = splits.eval_origins(split, period)
    x, _ = splits.windows_at(series.astype(np.float32), origins, max(splits.HORIZONS))
    return x


def load_pipeline(model: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """A `ChronosPipeline` from a Hugging Face id or a local checkpoint directory (`checkpoint-final`)."""
    from chronos import ChronosPipeline  # lazy: only this module needs the package

    _LOG.info("loading %s on %s", model, device)
    return ChronosPipeline.from_pretrained(model, device_map=device, dtype=torch.float32)


def num_parameters(pipeline) -> dict[str, int]:
    params = list(pipeline.model.parameters())
    return {"total": sum(p.numel() for p in params), "trainable": sum(p.numel() for p in params if p.requires_grad)}


@torch.no_grad()
def forecast(pipeline, x: np.ndarray, horizon: int = max(splits.HORIZONS), batch_size: int = 64,
             num_samples: int | None = None, quantiles: tuple[float, ...] = QUANTILES) -> tuple[np.ndarray, np.ndarray]:
    """Quantile forecasts for contexts x of shape (..., L): (Q, ..., H) quantiles and (..., H) median, in kW."""
    _LOG.debug("x.shape=%s horizon=%d batch_size=%d", x.shape, horizon, batch_size)
    if 0.5 not in quantiles:
        raise ValueError("quantiles must include 0.5, the median is the point forecast")
    flat = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32).reshape(-1, x.shape[-1]))
    out = []
    for i in range(0, len(flat), batch_size):
        q, _ = pipeline.predict_quantiles(flat[i : i + batch_size], prediction_length=horizon,
                                          quantile_levels=list(quantiles), num_samples=num_samples)
        out.append(q.float().cpu())  # (B, H, Q)
    q = torch.cat(out).numpy()
    q = np.moveaxis(q, -1, 0).reshape(len(quantiles), *x.shape[:-1], horizon)
    return q, q[quantiles.index(0.5)]


def evaluate(pipeline, patv: np.ndarray, valid: np.ndarray, split: splits.Split, periods: tuple[str, ...] = ("val",),
             mode: str = "keep", batch_size: int = 64, num_samples: int | None = None,
             quantiles: tuple[float, ...] = QUANTILES) -> tuple[dict, dict[str, np.ndarray]]:
    """Shared metrics per period and the prediction arrays for the npz, from one pipeline."""
    _LOG.debug("periods=%s mode='%s'", periods, mode)
    scores, preds = {}, {"quantiles": np.asarray(quantiles, dtype=np.float32)}
    t0 = time.time()
    for period in periods:
        q, median = forecast(pipeline, contexts(patv, valid, split, period, mode), batch_size=batch_size,
                             num_samples=num_samples, quantiles=quantiles)
        preds[period], preds[period + "_q"] = median.astype(np.float32), q.astype(np.float32)
        scores[period] = score_sdwpf.score(patv, valid, split, median.astype(np.float64), period,
                                           quantile_pred=q.astype(np.float64), quantiles=quantiles)
    scores["inference_seconds"] = round(time.time() - t0, 1)
    return scores, preds


def _json_safe(results: dict) -> dict:
    return {h: {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()} for h, r in results.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a Chronos model on the shared SDWPF windows.")
    ap.add_argument("--model", required=True, help="Hugging Face id or a checkpoint directory")
    ap.add_argument("--name", required=True, help="result files are <results>/<name>_<period>.json")
    ap.add_argument("--out", required=True, help="run record JSON; predictions are written next to it as <stem>_pred.npz")
    ap.add_argument("--results", default=None, help="directory for the shared-metric files; omit to write none")
    ap.add_argument("--periods", nargs="+", default=["val"], choices=("val", "test"))
    ap.add_argument("--context", default="keep", choices=CONTEXT_MODES)
    ap.add_argument("--clean", default="data/sdwpf/clean.parquet")
    ap.add_argument("--splits", default="results/ds02_splits.json")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-samples", type=int, default=None, help="default: the checkpoint's own setting (20)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--train-seconds", type=float, default=None, help="recorded as is, for fine-tuned checkpoints")
    ap.add_argument("--train-config", default=None, help="path of the train.py config, recorded as is")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    from wpf.data import sdwpf  # lazy: pandas/pyarrow are only needed here

    frame = sdwpf.load_clean(a.clean)
    patv, valid = sdwpf.to_dense(frame, "Patv").astype(np.float64), sdwpf.to_dense(frame, "valid").astype(bool)
    split = splits.load_splits(a.splits)
    pipeline = load_pipeline(a.model, a.device)
    scores, preds = evaluate(pipeline, patv, valid, split, tuple(a.periods), a.context, a.batch_size, a.num_samples)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_name(out.stem + "_pred.npz"), **preds)
    record = {"model": a.model, "context": a.context, "periods": a.periods, "num_samples": a.num_samples or pipeline.model.config.num_samples,
              "parameters": num_parameters(pipeline), "device": a.device, "train_seconds": a.train_seconds,
              "train_config": a.train_config, "inference_seconds": scores.pop("inference_seconds")}
    record.update({p: _json_safe(scores[p]) for p in a.periods})
    out.write_text(json.dumps(record, indent=2) + "\n")
    for period in a.periods:
        if a.results is not None:
            metrics.save_results(Path(a.results) / f"{a.name}_{period}.json", scores[period])
        r = scores[period]
        print(period, "mae", {h: round(v["mae"], 1) for h, v in r.items()}, "mase", {h: round(v["mase"], 3) for h, v in r.items()},
              "wql", {h: round(v["wql"], 4) for h, v in r.items()})
    print(f"inference {record['inference_seconds']} s -> {out}")


if __name__ == "__main__":
    main()
