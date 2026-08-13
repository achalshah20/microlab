"""Peak-FLOPs table for MFU accounting.

Model FLOPs Utilization is only meaningful against a correct denominator, and
the denominator is easy to get wrong in ways that flatter the result:

* Quoting the **sparse** tensor-core number (NVIDIA markets 2x figures with
  structured sparsity, which we do not use) halves the reported MFU.
* Quoting the **fp16** peak while training in fp32 — or the reverse — moves the
  number by 8x on a T4.
* Using a boost clock the card never sustains under load. The figures below are
  the official boost-clock peaks; a T4 in a shared Kaggle instance thermally
  throttles below them, so a *measured* MFU of 35% on the M2 gate is against a
  denominator the hardware may not actually reach.

Numbers are dense TFLOP/s, single device.
"""

from __future__ import annotations

import torch

# (device substring) -> {dtype family: dense TFLOP/s}
_PEAK_TFLOPS: dict[str, dict[str, float]] = {
    # Turing. fp16 via tensor cores; NO bf16, NO FlashAttention-2.
    "T4": {"fp16": 65.13, "bf16": 0.0, "fp32": 8.1},
    "P100": {"fp16": 19.05, "bf16": 0.0, "fp32": 9.5},  # Pascal, no tensor cores
    "V100": {"fp16": 125.0, "bf16": 0.0, "fp32": 15.7},
    "A100": {"fp16": 312.0, "bf16": 312.0, "fp32": 19.5},
    "L4": {"fp16": 121.0, "bf16": 121.0, "fp32": 30.3},
    "L40S": {"fp16": 362.0, "bf16": 362.0, "fp32": 91.6},
    "H100": {"fp16": 989.0, "bf16": 989.0, "fp32": 67.0},  # SXM
    "A10G": {"fp16": 125.0, "bf16": 125.0, "fp32": 31.2},
    "RTX 4090": {"fp16": 165.2, "bf16": 165.2, "fp32": 82.6},
}


def device_name(device: torch.device | str = "cuda") -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(torch.device(device).index or 0)


def peak_flops(precision: str, device: torch.device | str = "cuda") -> float | None:
    """Dense peak FLOP/s for this device and precision, or None if unknown.

    Returning None rather than a guess is deliberate: an MFU computed against a
    made-up denominator is worse than no MFU, because it will be quoted.
    """
    name = device_name(device)
    if name == "cpu":
        return None
    for key, table in _PEAK_TFLOPS.items():
        if key.lower() in name.lower():
            tflops = table.get(precision)
            if not tflops:
                return None
            return tflops * 1e12
    return None


def mfu(
    tokens_per_second: float,
    flops_per_token: float,
    precision: str,
    device: torch.device | str = "cuda",
    n_devices: int = 1,
) -> float | None:
    """Model FLOPs Utilization in ``[0, 1]``, or None when peak is unknown."""
    peak = peak_flops(precision, device)
    if not peak:
        return None
    return (tokens_per_second * flops_per_token) / (peak * n_devices)


def describe_device(device: torch.device | str = "cuda") -> dict[str, object]:
    name = device_name(device)
    info: dict[str, object] = {"name": name}
    if name != "cpu":
        idx = torch.device(device).index or 0
        props = torch.cuda.get_device_properties(idx)
        info.update(
            {
                "capability": f"{props.major}.{props.minor}",
                "total_memory_gb": round(props.total_memory / 1e9, 2),
                "multi_processor_count": props.multi_processor_count,
                # Reported separately because they disagree on Turing: PyTorch
                # says bf16 is "supported" when it is emulated in software. Only
                # the native flag should drive a precision decision.
                "bf16_native": props.major >= 8,
                "bf16_reported_by_torch": torch.cuda.is_bf16_supported(),
                # sm_75 is Turing: fp16-only training and no FlashAttention-2.
                "is_turing": props.major == 7 and props.minor == 5,
            }
        )
    return info
