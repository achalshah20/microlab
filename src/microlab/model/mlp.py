"""SwiGLU feed-forward network."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


def swiglu_hidden_dim(d_model: int, multiple_of: int = 64, ffn_factor: float = 8 / 3) -> int:
    """Hidden width for a SwiGLU MLP, rounded up to ``multiple_of``.

    A gated MLP has three weight matrices instead of two, so the usual ``4 *
    d_model`` is scaled by 2/3 to hold the parameter count fixed against a
    standard FFN. Rounding to a multiple of 64 keeps the GEMM shapes friendly to
    tensor cores — on a T4, a hidden dim of 683 runs measurably slower than 704
    for the same FLOPs.
    """
    hidden = int(ffn_factor * d_model)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


class SwiGLU(nn.Module):
    """``w2(silu(w1(x)) * w3(x))``, the LLaMA-style gated FFN."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        # Gate and up projections are fused into one GEMM, then split. Same math,
        # one kernel launch instead of two, and a single fused kernel to write in M2.
        self.w13 = nn.Linear(d_model, 2 * hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.w13(x).chunk(2, dim=-1)
        return self.dropout(self.w2(F.silu(gate) * up))

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, hidden_dim={self.hidden_dim}"
