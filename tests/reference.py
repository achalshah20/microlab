"""Independent reference implementations for correctness tests.

The roadmap's stated M0 risk is silent correctness bugs, and the mitigation it
names is diffing against an equivalent HF model. HF ``transformers`` is a heavy
CI dependency and pins its own torch, so the primary reference here is a
from-scratch implementation written in the most obvious way possible: explicit
loops, complex arithmetic for RoPE, materialized attention matrices, no fused
anything. ``tests/test_hf_parity.py`` additionally diffs against
``LlamaForCausalLM`` when transformers happens to be installed.

The point is that this file shares no code with ``src/microlab`` — it was
written from the equations, not from the implementation. A bug reproduced in
both would have to be a bug in the author's understanding, which is exactly the
class of error a self-consistent test suite cannot catch.
"""

from __future__ import annotations

import math

import torch


def reference_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm via an explicit per-vector loop."""
    out = torch.empty_like(x, dtype=torch.float32)
    flat = x.reshape(-1, x.shape[-1]).float()
    flat_out = out.reshape(-1, x.shape[-1])
    for i in range(flat.shape[0]):
        v = flat[i]
        rms = math.sqrt(float((v * v).sum()) / v.numel() + eps)
        flat_out[i] = v / rms
    return out * weight


def reference_rope(x: torch.Tensor, theta: float, position_offset: int = 0) -> torch.Tensor:
    """RoPE via complex multiplication, non-interleaved (half-split) layout.

    ``x`` is ``[B, H, S, D]``. Element ``j`` of the first half pairs with element
    ``j + D/2`` of the second half to form one complex number, which is rotated
    by ``position * theta ** (-2j/D)``.
    """
    b, h, s, d = x.shape
    half = d // 2
    out = torch.zeros_like(x, dtype=torch.float64)
    xf = x.double()
    for pos in range(s):
        abs_pos = pos + position_offset
        for j in range(half):
            freq = theta ** (-2.0 * j / d)
            angle = abs_pos * freq
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            real = xf[:, :, pos, j]
            imag = xf[:, :, pos, j + half]
            out[:, :, pos, j] = real * cos_a - imag * sin_a
            out[:, :, pos, j + half] = real * sin_a + imag * cos_a
    return out.to(x.dtype)


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_rep: int,
    causal: bool = True,
) -> torch.Tensor:
    """Materialized causal attention with an explicit mask.

    ``q`` is ``[B, Hq, S, D]``, ``k``/``v`` are ``[B, Hkv, S, D]``, and each KV
    head serves ``n_rep`` consecutive query heads.
    """
    b, hq, s, d = q.shape
    out = torch.zeros_like(q, dtype=torch.float64)
    qf, kf, vf = q.double(), k.double(), v.double()
    scale = 1.0 / math.sqrt(d)
    for bi in range(b):
        for h in range(hq):
            kv_head = h // n_rep
            scores = (qf[bi, h] @ kf[bi, kv_head].T) * scale  # [S, S]
            if causal:
                mask = torch.triu(torch.ones(s, s, dtype=torch.bool), diagonal=1)
                scores = scores.masked_fill(mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            out[bi, h] = weights @ vf[bi, kv_head]
    return out.to(q.dtype)


def reference_swiglu(
    x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor
) -> torch.Tensor:
    """``down(silu(gate(x)) * up(x))`` with weights in ``nn.Linear`` orientation."""
    gate = x @ w_gate.T
    up = x @ w_up.T
    silu = gate * torch.sigmoid(gate)
    return (silu * up) @ w_down.T


def reference_lr_cosine(
    step: int, base_lr: float, warmup: int, max_steps: int, min_ratio: float
) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    min_lr = base_lr * min_ratio
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
