"""Precision strategies.

The two hardware targets this project runs on disagree about numerics:

* **T4 (Turing, sm_75)** — the Kaggle GPU. fp16 only. bf16 is emulated at best
  and unusable in practice, so training needs loss scaling and the operational
  baggage that comes with it.
* **TPU / any Ampere+ box** — bf16, no scaler, no skipped steps.

Rather than sprinkling ``if fp16`` through the trainer, precision is one config
field selecting a strategy object. The trainer calls the same six methods either
way, and the fp16-specific machinery (loss scale, inf/nan step accounting) is
reported as first-class metrics rather than hidden inside GradScaler.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch
from torch import nn


class PrecisionStrategy(ABC):
    """Interface the trainer codes against."""

    name: str
    autocast_dtype: torch.dtype | None

    def __init__(self, device_type: str) -> None:
        self.device_type = device_type

    @abstractmethod
    def autocast(self) -> AbstractContextManager:
        """Context manager wrapping the forward pass."""

    @abstractmethod
    def backward(self, loss: torch.Tensor) -> None:
        """Scale (if needed) and run the backward pass."""

    @abstractmethod
    def clip_grad_norm(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, max_norm: float
    ) -> torch.Tensor:
        """Unscale gradients if needed, then clip. Returns the pre-clip norm.

        The optimizer is an argument rather than an implementation detail
        because ``GradScaler`` tracks unscale state *per optimizer object*: if
        ``unscale_`` is called with anything other than the optimizer later
        passed to ``step``, the scaler cannot tell the gradients were already
        unscaled and divides by the scale a second time. That produces a run
        that trains, just far worse, with no error anywhere.
        """

    @abstractmethod
    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        """Step the optimizer. Returns False if the step was skipped (non-finite grads)."""

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        return None

    def metrics(self) -> dict[str, float]:
        """Per-step numerics telemetry for the run record."""
        return {}


class FullPrecision(PrecisionStrategy):
    """fp32. The reference path: no autocast, no scaler, fully deterministic.

    Used by CI (which is CPU-only) and by any test that needs bitwise
    comparisons, so the correctness suite never has to reason about scaler state.
    """

    name = "fp32"
    autocast_dtype = None

    def autocast(self) -> AbstractContextManager:
        return nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def clip_grad_norm(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, max_norm: float
    ) -> torch.Tensor:
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        optimizer.step()
        return True


def bf16_natively_supported(device: int = 0) -> bool:
    """True only where bf16 runs on hardware, i.e. Ampere (sm_80) or newer.

    Deliberately does *not* use ``torch.cuda.is_bf16_supported()``. Recent
    PyTorch returns True from that when bf16 is merely **emulated**, so on a
    Turing T4 it reports::

        Tesla T4 sm_75 ... bf16=True

    which is true in the sense that the ops run, and useless in the sense that
    they run through a software path far slower than the fp16 tensor cores
    sitting unused next to them. A guard built on that check does not fire on
    the exact hardware it exists to protect.

    Compute capability is the authoritative signal and does not drift between
    PyTorch releases: bf16 tensor cores arrive with sm_80.
    """
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability(device)
    return major >= 8


class BFloat16(PrecisionStrategy):
    """bf16 autocast. No loss scaling: bf16 has fp32's exponent range."""

    name = "bf16"
    autocast_dtype = torch.bfloat16

    def __init__(self, device_type: str) -> None:
        super().__init__(device_type)
        if device_type == "cuda" and not bf16_natively_supported():
            # Fail at startup rather than 40 minutes into a run. On a T4 this is
            # the single most likely config mistake.
            emulated = torch.cuda.is_bf16_supported()
            raise RuntimeError(
                "precision=bf16 requested but this GPU has no native bf16 "
                f"({torch.cuda.get_device_name(0)}, capability "
                f"{torch.cuda.get_device_capability(0)}). Turing (T4) needs precision=fp16."
                + (
                    " torch.cuda.is_bf16_supported() reports True here because bf16 is"
                    " emulated in software; running on it would be far slower than fp16."
                    if emulated
                    else ""
                )
            )

    def autocast(self) -> AbstractContextManager:
        return torch.autocast(device_type=self.device_type, dtype=torch.bfloat16)

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def clip_grad_norm(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, max_norm: float
    ) -> torch.Tensor:
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        optimizer.step()
        return True


class Float16(PrecisionStrategy):
    """fp16 autocast with dynamic loss scaling — the T4 path.

    fp16's smallest normal is ~6e-5, and gradients in a small model routinely sit
    below that, so unscaled they flush to zero and training silently stalls.
    GradScaler multiplies the loss by a large factor, then unscales before the
    optimizer step, backing the factor off whenever it produces inf/nan.

    Two things this class adds over bare GradScaler:

    * The current scale and a running count of skipped steps are exposed as
      metrics. A healthy run skips a handful of steps early and then almost
      never; a run whose skip rate climbs is diverging, and that is visible here
      several thousand steps before the loss curve shows it.
    * ``clip_grad_norm`` unscales first. Clipping scaled gradients would apply a
      threshold that moves with the loss scale — a bug that produces a run which
      trains, just worse, and is nearly invisible without a comparison run.
    """

    name = "fp16"
    autocast_dtype = torch.float16

    def __init__(
        self,
        device_type: str,
        init_scale: float = 2.0**16,
        growth_interval: int = 2000,
    ) -> None:
        super().__init__(device_type)
        self.scaler = torch.amp.GradScaler(
            device_type,
            init_scale=init_scale,
            growth_interval=growth_interval,
            enabled=True,
        )
        self.skipped_steps = 0
        self.total_steps = 0

    def autocast(self) -> AbstractContextManager:
        return torch.autocast(device_type=self.device_type, dtype=torch.float16)

    def backward(self, loss: torch.Tensor) -> None:
        self.scaler.scale(loss).backward()

    def clip_grad_norm(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, max_norm: float
    ) -> torch.Tensor:
        # Must be the same optimizer object that step() receives, or the scaler
        # unscales twice. See PrecisionStrategy.clip_grad_norm.
        self.scaler.unscale_(optimizer)
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        scale_before = self.scaler.get_scale()
        self.scaler.step(optimizer)
        self.scaler.update()
        self.total_steps += 1
        # GradScaler skips the step by reducing the scale; there is no direct
        # "did you step" signal, so a shrinking scale is the observable proxy.
        skipped = self.scaler.get_scale() < scale_before
        if skipped:
            self.skipped_steps += 1
        return not skipped

    def state_dict(self) -> dict[str, Any]:
        return {
            "scaler": self.scaler.state_dict(),
            "skipped_steps": self.skipped_steps,
            "total_steps": self.total_steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.scaler.load_state_dict(state["scaler"])
        self.skipped_steps = state.get("skipped_steps", 0)
        self.total_steps = state.get("total_steps", 0)

    def metrics(self) -> dict[str, float]:
        return {
            "loss_scale": float(self.scaler.get_scale()),
            "skipped_steps": float(self.skipped_steps),
            "skip_rate": float(self.skipped_steps / max(self.total_steps, 1)),
        }


def build_precision(precision: str, device_type: str) -> PrecisionStrategy:
    """Factory: one config string in, one strategy out."""
    if device_type != "cuda" and precision in ("fp16", "bf16"):
        # CPU autocast exists but is a different (and much slower) code path with
        # different numerics; silently accepting it would make CPU tests
        # unrepresentative of the GPU run they are supposed to validate.
        if precision == "fp16":
            raise ValueError(
                "precision=fp16 requires CUDA; use precision=fp32 on CPU "
                "(this is what CI does) or bf16 on a supported accelerator."
            )
    strategies = {"fp32": FullPrecision, "bf16": BFloat16, "fp16": Float16}
    if precision not in strategies:
        raise ValueError(f"unknown precision {precision!r}, expected one of {list(strategies)}")
    return strategies[precision](device_type)
