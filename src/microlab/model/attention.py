"""Grouped-query causal self-attention built on PyTorch SDPA."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from .rope import RotaryEmbedding


@dataclass
class KVCacheLayer:
    """Preallocated key/value buffers for one layer.

    ``k``/``v`` are ``[B, n_kv_heads, max_seq_len, head_dim]``; ``length`` is how
    many positions are currently populated.
    """

    k: Tensor
    v: Tensor
    length: int = 0

    def append(self, k_new: Tensor, v_new: Tensor) -> tuple[Tensor, Tensor]:
        new_len = k_new.shape[-2]
        end = self.length + new_len
        if end > self.k.shape[-2]:
            raise ValueError(f"KV cache overflow: {end} > {self.k.shape[-2]}")
        self.k[:, :, self.length : end] = k_new
        self.v[:, :, self.length : end] = v_new
        self.length = end
        return self.k[:, :, :end], self.v[:, :, :end]


def _select_sdpa_backends() -> list[SDPBackend]:
    """Backends we allow SDPA to choose from, in preference order.

    FlashAttention-2 is deliberately absent. It requires Ampere or newer; the
    T4s this project runs on are Turing (sm_75), where requesting the flash
    backend either falls back silently or raises depending on the torch build.
    The memory-efficient backend is the fast path that actually exists on
    Turing, and the math backend is kept as a correctness fallback for CPU and
    for head dims the fused kernels reject.
    """
    return [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


class CausalSelfAttention(nn.Module):
    """Multi-head causal attention with grouped-query (GQA) key/value sharing.

    With ``n_kv_heads == n_heads`` this is ordinary MHA; with ``n_kv_heads == 1``
    it is MQA. GQA shrinks the KV cache by ``n_heads / n_kv_heads``, which is the
    dominant memory term during M6's paged-cache serving, so it is in from day
    one rather than retrofitted.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        rope: RotaryEmbedding,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.rope = rope

        # Fused QKV projection: one GEMM instead of three. The output is split
        # q | k | v, with k and v narrower than q under GQA.
        self.qkv_dim = (n_heads + 2 * n_kv_heads) * self.head_dim
        self.wqkv = nn.Linear(d_model, self.qkv_dim, bias=bias)
        self.wo = nn.Linear(d_model, d_model, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)

    def _repeat_kv(self, x: Tensor) -> Tensor:
        """``[B, n_kv_heads, S, D] -> [B, n_heads, S, D]`` without copying.

        ``expand`` produces a stride-0 view. SDPA's memory-efficient backend
        handles non-contiguous inputs, so no materialization is needed; a
        ``repeat_interleave`` here would cost ``n_rep``x the KV memory.
        """
        if self.n_rep == 1:
            return x
        b, n_kv, s, d = x.shape
        expanded = x[:, :, None, :, :].expand(b, n_kv, self.n_rep, s, d)
        return expanded.reshape(b, n_kv * self.n_rep, s, d)

    def forward(
        self,
        x: Tensor,
        cache: KVCacheLayer | None = None,
    ) -> Tensor:
        """Args: ``x`` is ``[B, S, d_model]``. Returns the same shape."""
        b, s, _ = x.shape

        qkv = self.wqkv(x)  # [B, S, (n_heads + 2*n_kv_heads) * head_dim]
        q, k, v = qkv.split(
            [
                self.n_heads * self.head_dim,
                self.n_kv_heads * self.head_dim,
                self.n_kv_heads * self.head_dim,
            ],
            dim=-1,
        )
        # [B, S, H*D] -> [B, H, S, D]
        q = q.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)

        position_offset = cache.length if cache is not None else 0
        q, k = self.rope(q, k, position_offset=position_offset)

        if cache is not None:
            k, v = cache.append(k, v)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # During cached decoding q is a single step attending to the whole
        # cache, so every position is legal and the causal mask must be off.
        # Leaving is_causal=True here would mask out all but the first key and
        # is the classic silent decode bug.
        is_causal = q.shape[-2] > 1

        with sdpa_kernel(_select_sdpa_backends()):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )

        # [B, H, S, D] -> [B, S, H*D]
        out = out.transpose(1, 2).contiguous().view(b, s, self.d_model)
        return self.resid_dropout(self.wo(out))

    def empty_cache(self, batch_size: int, max_seq_len: int, device, dtype) -> KVCacheLayer:
        shape = (batch_size, self.n_kv_heads, max_seq_len, self.head_dim)
        return KVCacheLayer(
            k=torch.zeros(shape, device=device, dtype=dtype),
            v=torch.zeros(shape, device=device, dtype=dtype),
        )

    def extra_repr(self) -> str:
        return (
            f"n_heads={self.n_heads}, n_kv_heads={self.n_kv_heads}, "
            f"head_dim={self.head_dim}, n_rep={self.n_rep}"
        )
