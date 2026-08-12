"""Component-level correctness against the independent references."""

from __future__ import annotations

import math

import pytest
import torch

from microlab.model.attention import CausalSelfAttention
from microlab.model.mlp import SwiGLU, swiglu_hidden_dim
from microlab.model.rmsnorm import RMSNorm
from microlab.model.rope import RotaryEmbedding, apply_rope, build_rope_cache

from .reference import (
    reference_attention,
    reference_rmsnorm,
    reference_rope,
    reference_swiglu,
)


class TestRMSNorm:
    def test_matches_reference(self):
        norm = RMSNorm(16, eps=1e-5)
        torch.nn.init.normal_(norm.weight, mean=1.0, std=0.1)
        x = torch.randn(3, 5, 16)
        expected = reference_rmsnorm(x, norm.weight.detach(), 1e-5)
        torch.testing.assert_close(norm(x), expected, rtol=1e-5, atol=1e-6)

    def test_scale_invariance(self):
        """RMSNorm output is invariant to input scale (up to eps)."""
        norm = RMSNorm(32, eps=1e-8)
        x = torch.randn(2, 4, 32)
        torch.testing.assert_close(norm(x), norm(x * 100.0), rtol=1e-4, atol=1e-5)

    def test_statistic_computed_in_fp32(self):
        """A half-precision input whose squared sum overflows fp16 must survive.

        ``x.pow(2).mean()`` on a 256-wide vector of magnitude 300 exceeds fp16's
        65504 ceiling. Computing the statistic in the input dtype would return
        inf here and NaN out.
        """
        norm = RMSNorm(256, eps=1e-5).half()
        x = torch.full((1, 1, 256), 300.0, dtype=torch.float16)
        out = norm(x)
        assert torch.isfinite(out).all()


class TestRoPE:
    @pytest.mark.parametrize("offset", [0, 1, 7])
    def test_matches_complex_reference(self, offset):
        head_dim, seq = 16, 8
        # fp64 cache so the comparison is limited by the implementation, not by
        # the cache's own rounding.
        cos, sin = build_rope_cache(head_dim, 64, theta=10_000.0, dtype=torch.float64)
        q = torch.randn(2, 3, seq, head_dim, dtype=torch.float64)
        k = torch.randn(2, 2, seq, head_dim, dtype=torch.float64)

        q_out, k_out = apply_rope(q, k, cos, sin, position_offset=offset)
        torch.testing.assert_close(
            q_out, reference_rope(q, 10_000.0, offset), rtol=1e-9, atol=1e-9
        )
        torch.testing.assert_close(
            k_out, reference_rope(k, 10_000.0, offset), rtol=1e-9, atol=1e-9
        )

    def test_preserves_norm(self):
        """Rotation is orthogonal, so per-head vector norms are unchanged."""
        cos, sin = build_rope_cache(32, 16, dtype=torch.float64)
        q = torch.randn(2, 4, 16, 32, dtype=torch.float64)
        q_out, _ = apply_rope(q, q, cos, sin)
        torch.testing.assert_close(q_out.norm(dim=-1), q.norm(dim=-1), rtol=1e-10, atol=1e-10)

    def test_fp32_cache_accurate_to_fp32(self):
        """The production cache is fp32; it should agree with fp64 to fp32 precision."""
        cos32, sin32 = build_rope_cache(32, 16)
        cos64, sin64 = build_rope_cache(32, 16, dtype=torch.float64)
        assert cos32.dtype == torch.float32
        torch.testing.assert_close(cos32.double(), cos64, rtol=0, atol=1e-7)
        torch.testing.assert_close(sin32.double(), sin64, rtol=0, atol=1e-7)

    def test_relative_position_property(self):
        """The defining property: ``<RoPE(q,m), RoPE(k,n)>`` depends only on ``m-n``.

        This is what RoPE is *for*. A layout bug (interleaved vs half-split) can
        still pass a norm test and an equality-at-position-0 test but breaks
        this one.
        """
        head_dim = 16
        cos_d, sin_d = build_rope_cache(head_dim, 64, dtype=torch.float64)
        q = torch.randn(1, 1, 1, head_dim, dtype=torch.float64)
        k = torch.randn(1, 1, 1, head_dim, dtype=torch.float64)

        def dot(m: int, n: int) -> float:
            qm, _ = apply_rope(q, q, cos_d, sin_d, position_offset=m)
            kn, _ = apply_rope(k, k, cos_d, sin_d, position_offset=n)
            return float((qm * kn).sum())

        assert dot(5, 3) == pytest.approx(dot(12, 10), abs=1e-9)
        assert dot(9, 1) == pytest.approx(dot(20, 12), abs=1e-9)
        assert dot(5, 3) != pytest.approx(dot(5, 1), abs=1e-6)

    def test_offset_beyond_cache_raises(self):
        cos, sin = build_rope_cache(8, 4)
        q = torch.randn(1, 1, 3, 8)
        with pytest.raises(ValueError, match="RoPE cache"):
            apply_rope(q, q, cos, sin, position_offset=2)

    def test_cache_is_not_persistent(self):
        """The cache must stay out of the state dict so M5 can rescale RoPE."""
        rope = RotaryEmbedding(8, 16)
        assert "cos" not in rope.state_dict()
        assert "sin" not in rope.state_dict()


class TestAttention:
    def _module(self, n_heads=4, n_kv_heads=2, d_model=32, max_seq=16):
        rope = RotaryEmbedding(d_model // n_heads, max_seq)
        return CausalSelfAttention(d_model, n_heads, n_kv_heads, rope, dropout=0.0)

    def test_causal_mask_blocks_future(self):
        """Perturbing token t+1.. must not change the output at token t.

        This is the strongest cheap test of causality: it exercises the real
        SDPA path rather than inspecting a mask tensor, so it catches an
        ``is_causal`` flag that got dropped somewhere in the call chain.
        """
        attn = self._module().eval()
        x = torch.randn(1, 12, 32)
        with torch.no_grad():
            base = attn(x)
            perturbed = x.clone()
            perturbed[:, 6:] += 10.0
            after = attn(perturbed)
        torch.testing.assert_close(base[:, :6], after[:, :6], rtol=1e-5, atol=1e-6)
        assert not torch.allclose(base[:, 6:], after[:, 6:])

    def test_matches_reference_attention(self):
        attn = self._module().eval()
        x = torch.randn(2, 10, 32, dtype=torch.float32)
        with torch.no_grad():
            got = attn(x)

            # Recompute through the reference using the module's own projections.
            b, s, _ = x.shape
            qkv = attn.wqkv(x)
            q, k, v = qkv.split(
                [
                    attn.n_heads * attn.head_dim,
                    attn.n_kv_heads * attn.head_dim,
                    attn.n_kv_heads * attn.head_dim,
                ],
                dim=-1,
            )
            q = q.view(b, s, attn.n_heads, attn.head_dim).transpose(1, 2)
            k = k.view(b, s, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
            v = v.view(b, s, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
            q, k = attn.rope(q, k)
            ref = reference_attention(q, k, v, n_rep=attn.n_rep, causal=True)
            ref = attn.wo(ref.transpose(1, 2).reshape(b, s, attn.d_model))
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)

    def test_gqa_equals_mha_when_kv_heads_match(self):
        """With n_kv_heads == n_heads the GQA path must reduce to plain MHA."""
        attn = self._module(n_heads=4, n_kv_heads=4).eval()
        assert attn.n_rep == 1
        x = torch.randn(2, 8, 32)
        with torch.no_grad():
            got = attn(x)
            b, s, _ = x.shape
            qkv = attn.wqkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(b, s, 4, 8).transpose(1, 2)
            k = k.view(b, s, 4, 8).transpose(1, 2)
            v = v.view(b, s, 4, 8).transpose(1, 2)
            q, k = attn.rope(q, k)
            ref = reference_attention(q, k, v, n_rep=1)
            ref = attn.wo(ref.transpose(1, 2).reshape(b, s, 32))
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)

    def test_kv_repeat_maps_heads_to_correct_group(self):
        """Query head h must read KV head h // n_rep, not h % n_kv_heads.

        Both are shape-correct; only one is right. Getting it wrong trains fine
        and costs quality, which is the worst possible failure mode.
        """
        attn = self._module(n_heads=4, n_kv_heads=2)
        kv = torch.arange(2 * 3, dtype=torch.float32).view(1, 2, 3, 1).expand(1, 2, 3, 4)
        repeated = attn._repeat_kv(kv.contiguous())
        assert repeated.shape == (1, 4, 3, 4)
        for h in range(4):
            torch.testing.assert_close(repeated[0, h], kv[0, h // attn.n_rep])

    def test_incremental_decode_matches_full_forward(self):
        """Token-by-token decoding with the KV cache must equal a full forward.

        This catches both the ``is_causal`` bug in single-step decode and any
        RoPE position-offset error, each of which produces fluent-but-wrong
        output rather than a crash.
        """
        attn = self._module(max_seq=32).eval()
        x = torch.randn(1, 9, 32)
        with torch.no_grad():
            full = attn(x)
            cache = attn.empty_cache(1, 32, x.device, x.dtype)
            stepwise = torch.cat([attn(x[:, i : i + 1], cache=cache) for i in range(9)], dim=1)
        torch.testing.assert_close(full, stepwise, rtol=1e-4, atol=1e-5)

    def test_rejects_invalid_head_config(self):
        rope = RotaryEmbedding(8, 16)
        with pytest.raises(ValueError, match="not divisible"):
            CausalSelfAttention(32, 5, 1, rope)
        with pytest.raises(ValueError, match="not divisible"):
            CausalSelfAttention(32, 4, 3, rope)


class TestSwiGLU:
    def test_matches_reference(self):
        mlp = SwiGLU(16, 32).eval()
        x = torch.randn(2, 5, 16)
        w_gate, w_up = mlp.w13.weight.chunk(2, dim=0)
        with torch.no_grad():
            got = mlp(x)
            ref = reference_swiglu(x, w_gate, w_up, mlp.w2.weight)
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize(
        "d_model,multiple_of,expected",
        [(512, 64, 1408), (256, 64, 704), (768, 256, 2048), (64, 32, 192)],
    )
    def test_hidden_dim_rounding(self, d_model, multiple_of, expected):
        hidden = swiglu_hidden_dim(d_model, multiple_of)
        assert hidden == expected
        assert hidden % multiple_of == 0
        assert hidden >= 8 / 3 * d_model - multiple_of

    def test_parameter_count_close_to_standard_ffn(self):
        """The 8/3 factor exists to match a 4x non-gated FFN's parameter count."""
        d = 512
        gated = 3 * d * swiglu_hidden_dim(d, 64)
        standard = 2 * d * (4 * d)
        assert abs(gated - standard) / standard < 0.06

    def test_silu_applied_to_gate_only(self):
        """A zero gate must zero the output regardless of the up projection."""
        mlp = SwiGLU(8, 16, bias=False).eval()
        with torch.no_grad():
            w_gate, _ = mlp.w13.weight.chunk(2, dim=0)
            w_gate.zero_()
            out = mlp(torch.randn(1, 3, 8))
        # silu(0) == 0, so the product is zero whatever the up branch produced.
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-7)


def test_rope_cache_matches_manual_computation():
    """Cache construction, independent of the apply path."""
    cos, sin = build_rope_cache(8, 5, theta=10_000.0)
    assert cos.shape == (5, 8)
    for pos in range(5):
        for j in range(4):
            angle = pos * (10_000.0 ** (-2.0 * j / 8))
            assert cos[pos, j].item() == pytest.approx(math.cos(angle), abs=1e-6)
            assert cos[pos, j + 4].item() == pytest.approx(math.cos(angle), abs=1e-6)
            assert sin[pos, j].item() == pytest.approx(math.sin(angle), abs=1e-6)
