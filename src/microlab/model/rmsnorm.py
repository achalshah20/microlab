"""Root-mean-square layer normalization."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """RMSNorm (Zhang & Sennrich, 2019).

    The normalization statistic is always computed in fp32 regardless of the
    autocast dtype. Under fp16 on Turing this matters: ``x.pow(2).mean()`` over a
    2048-wide activation can overflow fp16's 65504 ceiling once activations grow
    past ~256 in magnitude, which shows up as a NaN loss several thousand steps
    into a run rather than immediately.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x_fp32 = x.float()
        # [..., d] -> [..., 1]
        rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_fp32 * rms).to(dtype) * self.weight

    def extra_repr(self) -> str:
        return f"dim={tuple(self.weight.shape)}, eps={self.eps}"
