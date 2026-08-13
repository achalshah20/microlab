"""Seeding and the project's determinism policy.

What we guarantee:

* **Data order** is a pure function of ``(seed, step)`` — see ``data.loader``.
  It does not depend on process start, worker count, or how many times the run
  was preempted.
* **Initialization and dropout** are reproducible from ``seed`` alone.
* **Resume** reproduces the uninterrupted run to float tolerance, and bitwise on
  the fp32 path.

What we do not guarantee, deliberately:

* **cuDNN/cuBLAS algorithm selection.** Forcing deterministic algorithms costs
  10-30% throughput on a T4 and disables the memory-efficient attention kernel
  we rely on. At ~4e18 FLOPs/week of quota, that is not a trade worth making.
  ``strict=True`` turns it on anyway for debugging a suspected nondeterminism bug.
* **Reduction order across devices.** Multi-GPU lands in M2; float addition is
  not associative and all-reduce order is not fixed.
"""

from __future__ import annotations

import hashlib
import os
import random
import subprocess
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, strict: bool = False) -> None:
    """Seed Python, NumPy and torch. ``strict`` also forces deterministic kernels."""
    if not 0 <= seed < 2**31:
        raise ValueError(f"seed must fit in int32, got {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if strict:
        # cuBLAS needs this set before the first CUDA context to make GEMM
        # reductions deterministic; setting it later is silently ignored.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def derive_seed(base_seed: int, *parts: Any) -> int:
    """Derive a stable child seed from a base seed and arbitrary labels.

    Uses BLAKE2b rather than :func:`hash`, whose string hashing is randomized
    per process — a child seed derived from ``hash("val")`` would differ across
    a preemption boundary and quietly change which data a resumed run sees.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(str(base_seed).encode())
    for p in parts:
        h.update(b"\x00")
        h.update(str(p).encode())
    return int.from_bytes(h.digest(), "big") % (2**31 - 1)


def git_revision() -> dict[str, str]:
    """Current commit SHA and dirty flag, for the run record.

    A run whose config you have but whose code you don't is not reproducible, so
    the SHA is recorded with every run and the dirty flag is recorded honestly.
    """
    def _run(args: list[str]) -> str | None:
        try:
            return subprocess.run(
                args, capture_output=True, text=True, timeout=5, check=True
            ).stdout.strip()
        except (subprocess.SubprocessError, OSError):
            return None

    sha = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {
        "sha": sha or "unknown",
        "dirty": "unknown" if status is None else str(bool(status)).lower(),
    }


def environment_info() -> dict[str, Any]:
    """Everything about the box that could change numerics."""
    info: dict[str, Any] = {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "git": git_revision(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = list(torch.cuda.get_device_capability(0))
        info["gpu_count"] = torch.cuda.device_count()
        # Recorded as two fields because they disagree on Turing: PyTorch counts
        # software emulation as support, so a T4 reports True. The run record is
        # evidence about the hardware a run actually used, and a single
        # ambiguous flag makes it evidence for the wrong conclusion.
        info["bf16_native"] = torch.cuda.get_device_capability(0)[0] >= 8
        info["bf16_reported_by_torch"] = torch.cuda.is_bf16_supported()
    return info


def rng_state() -> dict[str, Any]:
    """Capture every RNG stream that affects training, for checkpointing."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def load_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG streams captured by :func:`rng_state`.

    CUDA state is skipped when resuming onto a box with a different GPU count
    (a real scenario: 2xT4 Kaggle session resuming on a 1xP100 one). Dropout
    then diverges from the original run, which is why the resume-equivalence
    test pins the device count.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu() if torch.is_tensor(state["torch"]) else state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        saved = state["cuda"]
        if len(saved) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(saved)
