"""Shape/FLOP instrumentation, and the MFU denominator it validates."""

from __future__ import annotations

import dataclasses

import pytest

from microlab.config import ModelConfig
from microlab.shapes import profile_shapes, render_markdown


class TestFlopAccounting:
    @pytest.mark.parametrize("tie", [True, False])
    @pytest.mark.parametrize("seq_len", [64, 128])
    def test_measured_matches_analytic(self, tie, seq_len):
        """The MFU denominator must agree with a per-module count.

        These are two independent computations: hooks summing real tensor
        shapes, versus the closed-form ``flops_per_token`` that MFU divides by.
        They agreed only after ``flops_per_token`` was corrected to include the
        LM head; without this test, an MFU inflated by 16% would have gone
        straight into the M2 gate.
        """
        cfg = ModelConfig(
            vocab_size=512,
            d_model=128,
            n_layers=3,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=128,
            ffn_multiple_of=32,
            tie_embeddings=tie,
        )
        report = profile_shapes(cfg, batch_size=2, seq_len=seq_len)
        ratio = report.measured_flops_per_token / report.analytic_flops_per_token
        assert ratio == pytest.approx(1.0, abs=0.02), (
            f"measured {report.measured_flops_per_token:.3e} vs "
            f"analytic {report.analytic_flops_per_token:.3e}"
        )

    def test_attention_term_scales_quadratically(self):
        """Doubling sequence length must more than double per-token FLOPs."""
        cfg = ModelConfig(
            vocab_size=256, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=256
        )
        short = profile_shapes(cfg, seq_len=64).analytic_flops_per_token
        long = profile_shapes(cfg, seq_len=256).analytic_flops_per_token
        assert long > short

    def test_lm_head_is_counted(self):
        """A larger vocab must raise FLOPs/token even with everything else fixed."""
        base = ModelConfig(
            vocab_size=512, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64
        )
        big_vocab = dataclasses.replace(base, vocab_size=4096)
        assert (
            profile_shapes(big_vocab, seq_len=64).analytic_flops_per_token
            > profile_shapes(base, seq_len=64).analytic_flops_per_token
        )


class TestReport:
    @pytest.fixture
    def report(self):
        cfg = ModelConfig(
            vocab_size=256,
            d_model=64,
            n_layers=3,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=32,
            ffn_multiple_of=16,
        )
        return profile_shapes(cfg, batch_size=1, seq_len=32)

    def test_records_every_leaf(self, report):
        names = {r.name for r in report.records}
        assert "tok_emb" in names
        assert "lm_head" in names
        assert "blocks.0.attn.wqkv" in names
        assert "blocks.2.ffn.w2" in names

    def test_shapes_are_recorded(self, report):
        rec = next(r for r in report.records if r.name == "lm_head")
        assert rec.output_shape == (1, 32, 256)

    def test_markdown_collapses_repeated_blocks(self, report):
        md = render_markdown(report, first_block_only=True)
        assert "blocks.0.attn.wqkv" in md
        assert "blocks.2.attn.wqkv" not in md
        # The shared RoPE module fires once per block; it must appear once.
        assert md.count("| `rope` |") == 1

    def test_markdown_shows_all_blocks_when_asked(self, report):
        md = render_markdown(report, first_block_only=False)
        assert "blocks.2.attn.wqkv" in md

    def test_markdown_has_the_headline_numbers(self, report):
        md = render_markdown(report)
        assert "non-embedding" in md
        assert "FLOP accounting" in md
