"""LoRA fine-tuning of Chronos (CF-02) through the repository's own scripts/training/train.py.

Runs train.py's `main` with its YAML config unchanged, but with `load_model` wrapped so the T5 it builds gets LoRA
adapters (wpf.lora) before the Trainer sees it. After training the adapters are merged into the weights and the plain
model is saved as run-N/checkpoint-final with peft's adapter files removed (the adapter alone is kept in run-N/adapter), so slurm/chronos_lora.sbatch
scores it with wpf.chronos_eval exactly like a full fine-tune. run-N/checkpoint-final/lora.json records the rank,
alpha, dropout, targets and parameter counts for the run record (`--train-info`).

    python scripts/train_lora.py --config configs/chronos_lora_5pct_nan.yaml --rank 8 [--alpha 16] [--dropout 0.05]
                                 [--output-dir output/chronos/<name>] [--chronos-repo ~/scratch.msml612/chronos-forecasting]

`--output-dir` replaces the config's output_dir, so runs that share a config (the rank sweep) get separate run-N
directories: train.py numbers run-N by listing output_dir, and two jobs starting together otherwise pick the same one.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

from wpf import lora

_LOG = logging.getLogger("train_lora")


def import_train(chronos_repo: str):
    """The train.py module from the chronos-forecasting clone, importable (it defines its logger only under __main__)."""
    sys.path.insert(0, str(Path(chronos_repo).expanduser() / "scripts" / "training"))
    import train  # noqa: E402  (chronos-forecasting/scripts/training/train.py)

    train.logger = logging.getLogger("chronos_train")
    return train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, help="train.py YAML config (configs/chronos_lora_*.yaml)")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=None, help="default 2 * rank")
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--targets", default=",".join(lora.TARGETS))
    ap.add_argument("--output-dir", default=None, help="override the config's output_dir (one directory per run name)")
    ap.add_argument("--seed", type=int, default=None, help="override the config's seed")
    ap.add_argument("--chronos-repo", default=os.environ.get("CHRONOS_REPO", "~/scratch.msml612/chronos-forecasting"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    targets = tuple(a.targets.split(","))
    train = import_train(a.chronos_repo)
    state: dict = {}

    original_load_model = train.load_model

    def load_model_with_lora(**kw):
        base = original_load_model(**kw)
        state["expected"] = lora.expected_lora_parameters(base, a.rank, targets)
        state["base_total"] = lora.count_parameters(base)["total"]
        model = lora.apply_lora(base, a.rank, a.alpha, a.dropout, targets)
        state["model"] = model
        counts = lora.count_parameters(model)
        _LOG.info("LoRA rank %d: %d trainable of %d parameters (%.2f%%)", a.rank, counts["trainable"], counts["total"],
                  100 * counts["trainable"] / counts["total"])
        assert counts["trainable"] == state["expected"], (counts, state["expected"])
        return model

    train.load_model = load_model_with_lora
    config = yaml.safe_load(Path(a.config).read_text())
    output_dir = a.output_dir or config["output_dir"]
    argv = ["--config", a.config] + (["--output-dir", a.output_dir] if a.output_dir else [])
    argv += ["--seed", str(a.seed)] if a.seed is not None else []
    train.app(argv, standalone_mode=False)  # typer: run train.main with the YAML (CLI options override it), do not sys.exit

    if not train.is_main_process():
        return
    run = sorted(Path(output_dir).glob("run-*"), key=lambda p: int(p.name.split("-")[1]))[-1]
    final = run / "checkpoint-final"
    model = state["model"]
    model.save_pretrained(run / "adapter")  # peft: the adapter weights alone
    plain = lora.merge(model)
    plain.save_pretrained(final)  # over train.py's adapter-only save of the same directory
    removed = lora.strip_adapter_files(final)  # else from_pretrained would load the adapter on top of the merged weights
    _LOG.info("removed adapter files from the merged checkpoint: %s", removed)
    info = {"rank": a.rank, "alpha": 2 * a.rank if a.alpha is None else a.alpha, "dropout": a.dropout, "targets": list(targets),
            "trainable_parameters": state["expected"], "base_parameters": state["base_total"],
            "trainable_fraction": state["expected"] / state["base_total"], "config": a.config}
    (final / "lora.json").write_text(json.dumps(info, indent=2) + "\n")
    _LOG.info("merged checkpoint -> %s (%s)", final, info)


if __name__ == "__main__":
    main()
