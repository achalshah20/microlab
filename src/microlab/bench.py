"""Benchmark harness: throughput, memory, MFU, and decode latency.

This is the M0 seed of what becomes M2's profiling work. It deliberately does
*not* read a dataset — it feeds random token ids — so it runs on any box, in CI,
and against any config without a prepared shard. Data loading is measured
separately, because mixing the two produces a number that improves when you
change the disk and gets attributed to the kernel.

Reporting conventions:

* **Median, not mean.** On a shared free-tier instance a handful of steps are
  arbitrarily slow because someone else's job is on the same host. The median
  step is the honest description of the steady state; p90 is reported alongside
  so the tail is visible rather than smoothed away.
* **Warmup is discarded.** The first steps include CUDA context setup, cuDNN
  algorithm selection and (if enabled) torch.compile, which are real costs but
  not per-step costs.
* **MFU is reported as null when the peak is unknown**, never as a guess.
"""

from __future__ import annotations

import statistics
import time
from typing import Any

import torch

from .config import Config
from .model.transformer import Transformer
from .precision import build_precision
from .train.optim import build_optimizer
from .train.trainer import resolve_device
from .utils.hardware import describe_device, mfu


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _stats(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "min": ordered[0],
        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "max": ordered[-1],
        "n": len(ordered),
    }


def benchmark_training(
    cfg: Config,
    steps: int = 30,
    warmup: int = 5,
) -> dict[str, Any]:
    """Time full training steps and a forward-only pass at the same shapes."""
    device = resolve_device(cfg.train.device)
    precision = build_precision(cfg.train.precision, device.type)
    model = Transformer(cfg.model).to(device)
    model.train()
    optimizer = build_optimizer(model, cfg.optim)

    b, s = cfg.data.batch_size, cfg.data.seq_len
    gen = torch.Generator(device="cpu").manual_seed(cfg.train.seed)
    x = torch.randint(0, cfg.model.vocab_size, (b, s), generator=gen).to(device)
    y = torch.randint(0, cfg.model.vocab_size, (b, s), generator=gen).to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    step_times: list[float] = []
    fwd_times: list[float] = []

    for i in range(steps + warmup):
        # Forward-only timing, so the backward cost is a derived quantity rather
        # than an assumed 2x. On small models with a large vocab it is not 2x.
        _sync(device)
        t0 = time.perf_counter()
        with torch.no_grad(), precision.autocast():
            model(x, targets=y)
        _sync(device)
        fwd = time.perf_counter() - t0

        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with precision.autocast():
            _, loss = model(x, targets=y)
        precision.backward(loss)
        precision.clip_grad_norm(model, optimizer, cfg.optim.grad_clip)
        precision.step(optimizer)
        _sync(device)
        full = time.perf_counter() - t0

        if i >= warmup:
            fwd_times.append(fwd)
            step_times.append(full)

    tokens_per_step = b * s
    step_stats = _stats(step_times)
    tokens_per_s = tokens_per_step / step_stats["median"]
    fpt = model.flops_per_token(s)

    result: dict[str, Any] = {
        "shape": {"batch_size": b, "seq_len": s, "tokens_per_step": tokens_per_step},
        "step_time_s": step_stats,
        "forward_time_s": _stats(fwd_times),
        "backward_fraction": 1.0 - (_stats(fwd_times)["median"] / step_stats["median"]),
        "tokens_per_s": tokens_per_s,
        "flops_per_token_train": fpt,
        "achieved_tflops": tokens_per_s * fpt / 1e12,
        "mfu": mfu(tokens_per_s, fpt, cfg.train.precision, device),
    }
    if device.type == "cuda":
        result["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        result["peak_mem_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
    return result


def benchmark_decode(
    cfg: Config,
    prompt_len: int = 128,
    new_tokens: int = 64,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Prefill latency (TTFT) and per-token decode throughput with the KV cache.

    M6 replaces this with a real serving engine benchmarked against vLLM. Having
    the measurement now means that comparison starts from a baseline measured on
    the same hardware with the same model, instead of a number recalled from a
    different machine.
    """
    device = resolve_device(cfg.train.device)
    model = Transformer(cfg.model).to(device).eval()
    total = prompt_len + new_tokens
    if total > cfg.model.max_seq_len:
        prompt_len = max(1, cfg.model.max_seq_len - new_tokens - 1)
        total = prompt_len + new_tokens

    gen = torch.Generator(device="cpu").manual_seed(cfg.train.seed)
    idx = torch.randint(0, cfg.model.vocab_size, (batch_size, prompt_len), generator=gen).to(device)

    with torch.no_grad():
        caches = model.init_caches(batch_size, total)
        model(idx, caches=caches)  # warm the kernels
        _sync(device)

        caches = model.init_caches(batch_size, total)
        t0 = time.perf_counter()
        logits, _ = model(idx, caches=caches)
        _sync(device)
        ttft = time.perf_counter() - t0

        step_times = []
        token = logits[:, -1:, :].argmax(dim=-1)
        for _ in range(new_tokens):
            t0 = time.perf_counter()
            logits, _ = model(token, caches=caches)
            _sync(device)
            step_times.append(time.perf_counter() - t0)
            token = logits[:, -1:, :].argmax(dim=-1)

    stats = _stats(step_times)
    return {
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "new_tokens": new_tokens,
        "ttft_s": ttft,
        "decode_step_s": stats,
        "decode_tokens_per_s": batch_size / stats["median"],
    }


def run_benchmark(cfg: Config, steps: int = 30, warmup: int = 5) -> dict[str, Any]:
    """Full benchmark report for one config."""
    device = resolve_device(cfg.train.device)
    model = Transformer(cfg.model)
    report = {
        "device": describe_device(device) if device.type == "cuda" else {"name": "cpu"},
        "precision": cfg.train.precision,
        "torch": torch.__version__,
        "model": {
            "params_total": model.num_params(non_embedding=False),
            "params_non_embedding": model.num_params(non_embedding=True),
            "d_model": cfg.model.d_model,
            "n_layers": cfg.model.n_layers,
            "n_heads": cfg.model.n_heads,
            "n_kv_heads": cfg.model.n_kv_heads,
            "vocab_size": cfg.model.vocab_size,
        },
        "training": benchmark_training(cfg, steps=steps, warmup=warmup),
        "decode": benchmark_decode(cfg),
    }
    del model
    return report
