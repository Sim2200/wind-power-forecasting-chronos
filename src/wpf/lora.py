"""LoRA for Chronos (CF-02): wrap the T5 attention projections with `peft`, count what trains, merge after training.

LoRA keeps every pretrained weight W0 frozen and learns a low-rank update BA (rank r) per wrapped matrix, so a
wrapped (d_in x d_out) matrix adds r * (d_in + d_out) trainable parameters. After training the update is folded into
the weights (`merge`), so the saved checkpoint is a plain T5 that `wpf.chronos_eval` scores unchanged.
"""
from __future__ import annotations

import logging

from torch import nn

_LOG = logging.getLogger(__name__)

TARGETS = ("q", "k", "v", "o")  # T5 attention projections; the module names in T5Attention


def apply_lora(model: nn.Module, rank: int = 8, alpha: int | None = None, dropout: float = 0.05,
               targets: tuple[str, ...] = TARGETS) -> nn.Module:
    """A `peft` model with LoRA adapters on `targets` and everything else frozen; `alpha` defaults to 2 * rank."""
    from peft import LoraConfig, TaskType, get_peft_model  # lazy: only CF-02 needs peft

    cfg = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=rank, lora_alpha=2 * rank if alpha is None else alpha,
                     lora_dropout=dropout, target_modules=list(targets), bias="none")
    wrapped = get_peft_model(model, cfg)
    _LOG.debug("lora %s", count_parameters(wrapped))
    return wrapped


def count_parameters(model: nn.Module) -> dict[str, int]:
    """{"trainable", "total"} parameter counts of any module (a `peft` model counts its adapters as trainable)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"trainable": trainable, "total": total}


def expected_lora_parameters(model: nn.Module, rank: int, targets: tuple[str, ...] = TARGETS) -> int:
    """r * (d_in + d_out) summed over the Linear modules named in `targets`: what `apply_lora` should add."""
    return sum(rank * (m.in_features + m.out_features) for name, m in model.named_modules()
               if isinstance(m, nn.Linear) and name.rsplit(".", 1)[-1] in targets)


def merge(model: nn.Module) -> nn.Module:
    """Fold the adapters into the base weights and return the plain base model (`peft`'s merge_and_unload)."""
    return model.merge_and_unload()


ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin", "README.md")


def strip_adapter_files(checkpoint_dir) -> list[str]:
    """Remove `peft`'s adapter files from a merged checkpoint directory, so `from_pretrained` cannot apply the update twice.

    train.py saves the (peft) model first, which writes the adapter alone; the merged plain model is saved over it, and
    transformers' PEFT integration would otherwise load `adapter_config.json` on top of the already-merged weights.
    Returns the names removed.
    """
    from pathlib import Path

    removed = []
    for name in ADAPTER_FILES:
        f = Path(checkpoint_dir) / name
        if f.exists():
            f.unlink()
            removed.append(name)
    return removed
