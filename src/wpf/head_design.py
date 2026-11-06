"""PT-03: choose the PatchTST head design for the grid from the pilot runs' validation metrics.

Two designs, full training data, one seed:
- `shared`: one 64-step head (`configs/sdwpf_100pct.json`) scored at H = 6 / 36 / 64 on its first H steps.
- `per_horizon`: one model per horizon (`configs/sdwpf_100pct_h6.json`, `_h36.json`, and the 64-step model again
  for H = 64), each scored on its own H.

Rule: `per_horizon` is chosen only if its validation WQL is lower than the shared head's by more than `margin`
(relative) at every horizon it adds a model for (6 and 36). Otherwise the shared head wins: one model per grid cell
instead of three, so a third of the training cost. Validation only; the test numbers in the run records are not read.

Usage:
    python -m wpf.head_design --shared results/pt03_100pct.json --h6 results/pt03_100pct_h6.json \
        --h36 results/pt03_100pct_h36.json --out results/pt03_head_design.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

_LOG = logging.getLogger(__name__)

METRICS = ("mae", "rmse", "mase", "wql")


def _val(record: dict, horizon: int) -> dict:
    if "val" not in record or str(horizon) not in record["val"]:
        raise ValueError(f"the run record has no validation metrics for H = {horizon}")
    return record["val"][str(horizon)]


def compare(shared: dict, per_horizon: dict[int, dict], margin: float = 0.02) -> dict:
    """{chosen, margin, horizons: {H: {shared, per_horizon, wql_gain}}, cost: {...}} from `wpf.train_sdwpf` records.

    `per_horizon` maps H to the record of the model trained with pred_len H; for the shared head's own H it may be
    the shared record itself.
    """
    _LOG.debug("margin=%s horizons=%s", margin, sorted(per_horizon))
    horizons, better = {}, []
    for h, record in sorted(per_horizon.items()):
        s, p = _val(shared, h), _val(record, h)
        gain = (s["wql"] - p["wql"]) / s["wql"] if s["wql"] else float("nan")
        horizons[str(h)] = {"shared": {k: s[k] for k in METRICS}, "per_horizon": {k: p[k] for k in METRICS},
                            "wql_gain": gain, "n_valid": s["n_valid"]}
        if record is not shared:
            better.append(gain > margin)
    chosen = "per_horizon" if better and all(better) else "shared"
    seconds = {"shared": shared["train_seconds"],
               "per_horizon": sum(r["train_seconds"] for r in {id(r): r for r in per_horizon.values()}.values())}
    return {"chosen": chosen, "margin": margin, "horizons": horizons,
            "train_seconds": seconds, "parameters": {"shared": shared["parameters"],
                                                     "per_horizon": {str(h): r["parameters"] for h, r in per_horizon.items()}}}


def main() -> None:
    ap = argparse.ArgumentParser(description="Choose the PatchTST head design from the PT-03 pilot runs.")
    ap.add_argument("--shared", required=True, help="run record of the 64-step head")
    ap.add_argument("--h6", required=True, help="run record of the pred_len 6 model")
    ap.add_argument("--h36", required=True, help="run record of the pred_len 36 model")
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--out", default="results/pt03_head_design.json")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    shared, h6, h36 = (json.loads(Path(p).read_text()) for p in (a.shared, a.h6, a.h36))
    result = compare(shared, {6: h6, 36: h36, 64: shared}, a.margin)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    _LOG.info("wrote %s", a.out)
    for h, r in result["horizons"].items():
        print(f"H={h}: shared wql {r['shared']['wql']:.4f} mase {r['shared']['mase']:.3f} | per-horizon wql "
              f"{r['per_horizon']['wql']:.4f} mase {r['per_horizon']['mase']:.3f} | gain {r['wql_gain']:+.1%}")
    print("chosen:", result["chosen"])


if __name__ == "__main__":
    main()
