"""Optimizer construction and learning-rate schedules."""

from __future__ import annotations

import math

import torch
from torch import nn

from ..config import OptimConfig


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> torch.optim.AdamW:
    """AdamW with decay applied to matmul weights only.

    ``fused=True`` when CUDA offers it: it collapses the per-parameter element-
    wise update into one kernel, worth a few percent of step time on small
    models where the optimizer is not amortized behind large GEMMs.
    """
    groups = model.param_groups(cfg.weight_decay)  # type: ignore[operator]
    use_fused = torch.cuda.is_available()
    return torch.optim.AdamW(
        groups,
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        fused=use_fused,
    )


def lr_at_step(step: int, cfg: OptimConfig, max_steps: int) -> float:
    """Learning rate for a given *global* step.

    Computed from the global step rather than incremented per call, so a resumed
    run lands on exactly the LR the uninterrupted run would have had. A stateful
    ``LRScheduler`` that is stepped once per call silently restarts its schedule
    if the checkpoint round-trip misses it — and the resulting run trains fine,
    just to a worse loss.
    """
    if step < cfg.warmup_steps:
        # Linear warmup from lr/warmup_steps, not from 0: a literal zero first
        # step wastes a step and makes step-0 diagnostics useless.
        return cfg.lr * (step + 1) / cfg.warmup_steps

    min_lr = cfg.lr * cfg.min_lr_ratio

    if cfg.schedule == "constant":
        return cfg.lr

    if cfg.schedule == "cosine":
        progress = (step - cfg.warmup_steps) / max(1, max_steps - cfg.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return min_lr + 0.5 * (cfg.lr - min_lr) * (1 + math.cos(math.pi * progress))

    if cfg.schedule == "wsd":
        # Warmup-Stable-Decay. The stable phase holds a constant LR, so any
        # checkpoint taken during it is a legitimate branch point: annealing
        # from it for a few percent of the total budget yields a usable model.
        # That is how M4 gets an ablation budget out of a compute budget that
        # does not have one.
        decay_steps = max(1, int(cfg.wsd_decay_fraction * max_steps))
        decay_start = max_steps - decay_steps
        if step < decay_start:
            return cfg.lr
        progress = (step - decay_start) / decay_steps
        progress = min(1.0, max(0.0, progress))
        # 1-sqrt decay: empirically beats linear and cosine for the WSD tail.
        return cfg.lr - (cfg.lr - min_lr) * math.sqrt(progress)

    raise ValueError(f"unknown schedule {cfg.schedule!r}")


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
