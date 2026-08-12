"""Rotary position embeddings (RoPE).

Convention: **non-interleaved** (a.k.a. GPT-NeoX / HF-Llama layout). The head
dimension is split in half and the two halves are treated as the real and
imaginary parts of ``head_dim // 2`` complex numbers::

    x = [x_0 ... x_{h/2-1} | x_{h/2} ... x_{h-1}]
         \\_____ real _____/  \\____ imaginary ____/

The alternative (original RoFormer / GPT-J) layout pairs *adjacent* elements
instead. The two are related by a permutation of the head dimension, so a model
trained with one and evaluated with the other produces plausible-looking but
subtly wrong outputs — it degrades quality without crashing. We commit to the
non-interleaved layout here and test against an independent complex-arithmetic
reference in ``tests/test_rope.py``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def build_rope_cache(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10_000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Precompute the RoPE cos/sin tables.

    Returns two tensors of shape ``[max_seq_len, head_dim]``, each holding the
    half-width table duplicated so it can be broadcast against a full head
    without a further ``cat`` at call time.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

    # Inverse frequencies are always built in fp64 then cast down. At seq_len
    # 8k+ (M5's context extension) fp32 accumulation of position * inv_freq
    # loses enough precision to visibly rotate the last few positions.
    exponent = torch.arange(0, head_dim, 2, device=device, dtype=torch.float64) / head_dim
    inv_freq = 1.0 / (theta**exponent)  # [head_dim/2]
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float64)  # [S]
    freqs = torch.outer(positions, inv_freq)  # [S, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [S, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: Tensor) -> Tensor:
    """``[x1, x2] -> [-x2, x1]`` over the final dimension."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_offset: int = 0,
) -> tuple[Tensor, Tensor]:
    """Apply RoPE to query and key tensors.

    Args:
        q: ``[B, n_heads, S, head_dim]``
        k: ``[B, n_kv_heads, S, head_dim]``
        cos, sin: ``[max_seq_len, head_dim]`` caches from :func:`build_rope_cache`.
        position_offset: index of the first token in ``q``/``k``. Non-zero during
            cached decoding, where the incoming tensors hold one step but sit at
            absolute position ``offset``.

    The rotation is computed at *at least* fp32 and cast back to the input
    dtype. Under fp16 the cos/sin product otherwise loses ~3 bits of mantissa on
    the query, which is small per-layer but compounds across depth. Promoting
    rather than hard-casting to fp32 matters for the fp64 reference tests: a
    literal ``.float()`` would silently downcast them and cap the achievable
    agreement at fp32, hiding real precision bugs behind a loose tolerance.
    """
    seq_len = q.shape[-2]
    if position_offset + seq_len > cos.shape[0]:
        raise ValueError(
            f"RoPE cache holds {cos.shape[0]} positions but got "
            f"offset={position_offset} + seq_len={seq_len}"
        )

    compute_dtype = torch.promote_types(
        torch.promote_types(q.dtype, cos.dtype), torch.float32
    )

    # [S, head_dim] -> [1, 1, S, head_dim] so it broadcasts over batch and heads.
    window = slice(position_offset, position_offset + seq_len)
    cos_s = cos[window].unsqueeze(0).unsqueeze(0).to(compute_dtype)
    sin_s = sin[window].unsqueeze(0).unsqueeze(0).to(compute_dtype)

    q_f, k_f = q.to(compute_dtype), k.to(compute_dtype)
    q_out = q_f * cos_s + rotate_half(q_f) * sin_s
    k_out = k_f * cos_s + rotate_half(k_f) * sin_s
    return q_out.to(q.dtype), k_out.to(k.dtype)


class RotaryEmbedding(nn.Module):
    """Owns the RoPE cache as a non-persistent buffer.

    Non-persistent because the cache is a pure function of
    ``(head_dim, max_seq_len, theta)``, all of which live in the config. Keeping
    it out of the state dict means checkpoints stay valid when M5 extends the
    context window via RoPE scaling.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        cos, sin = build_rope_cache(head_dim, max_seq_len, theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, q: Tensor, k: Tensor, position_offset: int = 0) -> tuple[Tensor, Tensor]:
        return apply_rope(q, k, self.cos, self.sin, position_offset)

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, max_seq_len={self.max_seq_len}, theta={self.theta}"
