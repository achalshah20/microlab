"""The decoder-only transformer."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import ModelConfig
from .attention import CausalSelfAttention, KVCacheLayer
from .mlp import SwiGLU, swiglu_hidden_dim
from .rmsnorm import RMSNorm
from .rope import RotaryEmbedding


class Block(nn.Module):
    """Pre-norm transformer block: ``x + attn(norm(x))``, then ``x + mlp(norm(x))``.

    Pre-norm rather than post-norm because the residual stream stays unnormalized
    end to end, which is what keeps the loss stable without a long warmup at
    depth — and what makes M4's 500M run survivable on a schedule we can't
    babysit continuously.
    """

    def __init__(self, cfg: ModelConfig, rope: RotaryEmbedding) -> None:
        super().__init__()
        hidden = cfg.ffn_hidden_dim or swiglu_hidden_dim(cfg.d_model, cfg.ffn_multiple_of)
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            rope=rope,
            dropout=cfg.dropout,
            bias=cfg.bias,
        )
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, hidden, dropout=cfg.dropout, bias=cfg.bias)

    def forward(self, x: Tensor, cache: KVCacheLayer | None = None) -> Tensor:
        x = x + self.attn(self.attn_norm(x), cache=cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    """Decoder-only transformer: RMSNorm + RoPE + GQA + SwiGLU, pre-norm."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.d_model // cfg.n_heads

        # One RoPE cache shared by every layer. It is a pure function of
        # (head_dim, max_seq_len, theta), so per-layer copies would waste memory
        # and drift if a layer were ever rebuilt.
        self.rope = RotaryEmbedding(self.head_dim, cfg.max_seq_len, cfg.rope_theta)

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg, self.rope) for _ in range(cfg.n_layers)])
        self.norm_out = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            # Weight tying: the head and the embedding are the same parameter
            # object, so this must happen before init_weights so both views see
            # the same initialization.
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_module)
        # Residual-output projections get an extra 1/sqrt(2L) so the variance of
        # the residual stream stays O(1) with depth instead of growing linearly.
        residual_scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        for block in self.blocks:
            torch.nn.init.normal_(block.attn.wo.weight, mean=0.0, std=cfg.init_std * residual_scale)
            torch.nn.init.normal_(block.ffn.w2.weight, mean=0.0, std=cfg.init_std * residual_scale)

    def _init_module(self, module: nn.Module) -> None:
        std = self.cfg.init_std
        if isinstance(module, nn.Linear):
            # Truncated at 3 sigma: an untruncated normal puts ~0.3% of a large
            # embedding matrix beyond 3 sigma, and those outliers are where fp16
            # overflow starts.
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
        caches: list[KVCacheLayer] | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Args:
            idx: ``[B, S]`` int64 token ids.
            targets: ``[B, S]`` int64 next-token labels, or None for inference.
                ``-100`` entries are ignored by the loss (used by M5's SFT masking).
            caches: per-layer KV caches for incremental decoding.

        Returns ``(logits, loss)`` where logits are ``[B, S, vocab_size]``.
        """
        b, s = idx.shape
        if s > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {s} exceeds max_seq_len={self.cfg.max_seq_len}")

        x = self.drop(self.tok_emb(idx))
        for i, block in enumerate(self.blocks):
            x = block(x, cache=caches[i] if caches is not None else None)
        x = self.norm_out(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Cross-entropy in fp32: under fp16 autocast the logsumexp over a
            # 32k vocab loses enough precision to bias the loss by ~1e-3, which
            # is the same order as the improvements M3's scaling fits measure.
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    # ---- introspection -------------------------------------------------

    def num_params(self, non_embedding: bool = True) -> int:
        """Parameter count. ``non_embedding=True`` excludes the token embedding.

        Scaling-law fits (M3) are conventionally stated in non-embedding
        parameters, and mixing the two conventions is a common way to get a
        fit that looks fine and predicts wrong.
        """
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def flops_per_token(self, seq_len: int | None = None, backward: bool = True) -> float:
        """Analytic FLOPs per token, for MFU accounting.

        Counts a multiply-accumulate as 2 FLOPs. Three terms:

        * ``2 * N`` over non-embedding parameters — every block matmul.
        * ``2 * vocab * d_model`` for the LM head. This is counted explicitly
          because ``num_params(non_embedding=True)`` excludes it: under weight
          tying the head *is* the embedding matrix. Omitting it understates
          FLOPs by ~16% for the M0 config (large vocab, small d_model), which
          would inflate every reported MFU by the same factor.
        * ``2 * 2 * L * S * d_model`` for the QK^T and AV score matmuls, which
          are quadratic in sequence length and are what a folk ``6N`` estimate
          silently drops.

        Backward is 2x forward, so training is 3x.
        """
        s = seq_len or self.cfg.max_seq_len
        n = self.num_params(non_embedding=True)
        dense = 2 * n
        lm_head = 2 * self.cfg.vocab_size * self.cfg.d_model
        attn = 2 * 2 * self.cfg.n_layers * s * self.cfg.d_model
        fwd = dense + lm_head + attn
        return fwd * 3 if backward else fwd

    def param_groups(self, weight_decay: float) -> list[dict]:
        """Split parameters into decay / no-decay groups.

        Weight decay applies to matmul weights only. Norm gains and biases are
        1-D and decaying them pulls the normalization scale toward zero, which
        costs a measurable amount of loss for no regularization benefit.
        """
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append((name, p))
        return [
            {"params": [p for _, p in decay], "weight_decay": weight_decay, "name": "decay"},
            {"params": [p for _, p in no_decay], "weight_decay": 0.0, "name": "no_decay"},
        ]

    # ---- generation ----------------------------------------------------

    def init_caches(self, batch_size: int, max_seq_len: int) -> list[KVCacheLayer]:
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        return [b.attn.empty_cache(batch_size, max_seq_len, device, dtype) for b in self.blocks]

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
        generator: torch.Generator | None = None,
        max_valid_id: int | None = None,
    ) -> Tensor:
        """Sample continuations for ``idx`` ``[B, S]``, returning ``[B, S + n]``.

        Uses the KV cache: the prompt is one prefill pass, then each new token is
        a single-step forward. ``generator`` makes sampling reproducible, which
        matters because samples go into the run record as evidence.

        ``max_valid_id`` masks off ids the tokenizer cannot decode. Vocabularies
        are routinely padded up to a multiple of 64/128 so the LM-head GEMM hits
        tensor-core-friendly shapes, which leaves a tail of ids that are valid
        model outputs but have no bytes behind them. An untrained model samples
        them at chance rate; masking is the difference between a clean sample
        and a crash in the decoder.
        """
        was_training = self.training
        self.eval()
        try:
            b, s = idx.shape
            total = s + max_new_tokens
            if total > self.cfg.max_seq_len:
                raise ValueError(f"generation length {total} exceeds max_seq_len")

            caches = self.init_caches(b, total)
            logits, _ = self(idx, caches=caches)  # prefill
            out = idx

            for _ in range(max_new_tokens):
                next_logits = logits[:, -1, :].float()
                if max_valid_id is not None:
                    next_logits[:, max_valid_id:] = float("-inf")
                if temperature == 0.0:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                else:
                    next_logits = next_logits / temperature
                    if top_k is not None:
                        k = min(top_k, next_logits.size(-1))
                        kth = next_logits.topk(k, dim=-1).values[:, -1:]
                        next_logits = next_logits.masked_fill(next_logits < kth, float("-inf"))
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1, generator=generator)

                out = torch.cat((out, next_token), dim=1)
                if eos_id is not None and bool((next_token == eos_id).all()):
                    break
                logits, _ = self(next_token, caches=caches)
            return out
        finally:
            self.train(was_training)
