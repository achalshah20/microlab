"""Whole-model correctness: causality, tying, counting, and generation."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from microlab.model.transformer import Transformer


@pytest.fixture
def model(tiny_model_cfg):
    return Transformer(tiny_model_cfg).eval()


class TestForward:
    def test_shapes(self, model, tiny_model_cfg):
        x = torch.randint(0, tiny_model_cfg.vocab_size, (3, 12))
        logits, loss = model(x, targets=x)
        assert logits.shape == (3, 12, tiny_model_cfg.vocab_size)
        assert loss.ndim == 0

    def test_no_targets_gives_no_loss(self, model, tiny_model_cfg):
        x = torch.randint(0, tiny_model_cfg.vocab_size, (2, 8))
        _, loss = model(x)
        assert loss is None

    def test_untrained_loss_is_near_uniform_entropy(self, tiny_model_cfg):
        """A correctly initialized model starts at ~ln(V), not far above or below.

        Far above means the init is too large; far below on *random* targets
        means something is leaking the answer.
        """
        import math

        cfg = dataclasses.replace(tiny_model_cfg, tie_embeddings=False)
        m = Transformer(cfg).eval()
        x = torch.randint(0, cfg.vocab_size, (8, 16))
        y = torch.randint(0, cfg.vocab_size, (8, 16))
        _, loss = m(x, targets=y)
        assert abs(loss.detach().item() - math.log(cfg.vocab_size)) < 0.3

    def test_causality_end_to_end(self, model, tiny_model_cfg):
        """Changing a future token must not change any earlier logit."""
        x = torch.randint(0, tiny_model_cfg.vocab_size, (1, 16))
        with torch.no_grad():
            base, _ = model(x)
            altered = x.clone()
            altered[0, 8:] = (altered[0, 8:] + 5) % tiny_model_cfg.vocab_size
            after, _ = model(altered)
        torch.testing.assert_close(base[:, :8], after[:, :8], rtol=1e-5, atol=1e-6)
        assert not torch.allclose(base[:, 8:], after[:, 8:])

    def test_gradient_causality(self, model, tiny_model_cfg):
        """d(loss at position t)/d(embedding at position > t) must be exactly zero.

        A stronger statement than the forward test: it proves no information
        path exists, rather than that one happens not to fire on this input.
        """
        x = torch.randint(0, tiny_model_cfg.vocab_size, (1, 10))
        emb = model.tok_emb(x).detach().requires_grad_(True)

        h = model.drop(emb)
        for block in model.blocks:
            h = block(h)
        logits = model.lm_head(model.norm_out(h))
        logits[0, 4].sum().backward()

        assert emb.grad is not None
        assert torch.count_nonzero(emb.grad[0, 5:]) == 0
        assert torch.count_nonzero(emb.grad[0, :5]) > 0

    def test_rejects_sequence_longer_than_max(self, model, tiny_model_cfg):
        x = torch.randint(0, tiny_model_cfg.vocab_size, (1, tiny_model_cfg.max_seq_len + 1))
        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            model(x)

    def test_ignore_index_masks_loss(self, model, tiny_model_cfg):
        """-100 targets are excluded — the mechanism M5's SFT masking relies on."""
        x = torch.randint(0, tiny_model_cfg.vocab_size, (2, 8))
        y_all = x.clone()
        y_masked = x.clone()
        y_masked[:, :4] = -100
        _, loss_all = model(x, targets=y_all)
        _, loss_masked = model(x, targets=y_masked)
        assert not torch.isclose(loss_all, loss_masked)
        assert torch.isfinite(loss_masked)

    def test_fully_masked_batch_does_not_crash(self, model, tiny_model_cfg):
        x = torch.randint(0, tiny_model_cfg.vocab_size, (1, 4))
        _, loss = model(x, targets=torch.full_like(x, -100))
        assert torch.isnan(loss) or loss == 0  # documented degenerate case


class TestStructure:
    def test_weight_tying_shares_storage(self, tiny_model_cfg):
        cfg = dataclasses.replace(tiny_model_cfg, tie_embeddings=True)
        m = Transformer(cfg)
        assert m.lm_head.weight is m.tok_emb.weight
        m.tok_emb.weight.data.fill_(0.5)
        assert torch.all(m.lm_head.weight == 0.5)

    def test_untied_weights_are_independent(self, tiny_model_cfg):
        m = Transformer(dataclasses.replace(tiny_model_cfg, tie_embeddings=False))
        assert m.lm_head.weight is not m.tok_emb.weight

    def test_param_count_excludes_embeddings_when_asked(self, tiny_model_cfg):
        """Non-embedding count drops the input embedding *and* an untied head.

        This is the convention scaling-law fits use, and it is why M3 must state
        which count it reports: with an untied head at a 32k vocab, the two
        conventions differ by more than the effect being measured.
        """
        emb_params = tiny_model_cfg.vocab_size * tiny_model_cfg.d_model

        untied = Transformer(dataclasses.replace(tiny_model_cfg, tie_embeddings=False))
        assert untied.num_params(False) - untied.num_params(True) == 2 * emb_params

        tied = Transformer(dataclasses.replace(tiny_model_cfg, tie_embeddings=True))
        assert tied.num_params(False) - tied.num_params(True) == emb_params

    def test_param_count_matches_manual_sum(self, tiny_model_cfg):
        m = Transformer(tiny_model_cfg)
        assert m.num_params(non_embedding=False) == sum(p.numel() for p in m.parameters())

    def test_param_groups_exclude_1d_from_decay(self, tiny_model_cfg):
        groups = Transformer(tiny_model_cfg).param_groups(weight_decay=0.1)
        decay, no_decay = groups[0], groups[1]
        assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
        assert all(p.dim() >= 2 for p in decay["params"])
        assert all(p.dim() < 2 for p in no_decay["params"])
        assert len(no_decay["params"]) > 0  # the RMSNorm gains

    def test_all_parameters_are_in_exactly_one_group(self, tiny_model_cfg):
        m = Transformer(tiny_model_cfg)
        groups = m.param_groups(0.1)
        grouped = [p for g in groups for p in g["params"]]
        assert len(grouped) == len(list(m.parameters()))
        assert len({id(p) for p in grouped}) == len(grouped)

    def test_residual_projections_are_scaled_down(self, tiny_model_cfg):
        """1/sqrt(2L) on output projections keeps residual variance O(1) in depth."""
        cfg = dataclasses.replace(tiny_model_cfg, n_layers=8, d_model=256)
        m = Transformer(cfg)
        wo_std = m.blocks[0].attn.wo.weight.std().item()
        qkv_std = m.blocks[0].attn.wqkv.weight.std().item()
        assert wo_std < qkv_std * 0.6

    def test_flops_per_token_includes_attention_term(self, tiny_model_cfg):
        m = Transformer(tiny_model_cfg)
        n = m.num_params(non_embedding=True)
        short = m.flops_per_token(seq_len=16, backward=False)
        long = m.flops_per_token(seq_len=512, backward=False)
        assert short > 2 * n  # not just the 2N dense term
        assert long > short  # quadratic term grows with sequence length

    def test_flops_backward_is_three_times_forward(self, tiny_model_cfg):
        m = Transformer(tiny_model_cfg)
        fwd = m.flops_per_token(128, backward=False)
        assert m.flops_per_token(128, backward=True) == pytest.approx(3 * fwd)

    def test_rope_cache_shared_across_layers(self, tiny_model_cfg):
        m = Transformer(tiny_model_cfg)
        assert all(b.attn.rope is m.rope for b in m.blocks)


class TestGeneration:
    def test_generate_extends_sequence(self, model, tiny_model_cfg):
        idx = torch.randint(0, tiny_model_cfg.vocab_size, (2, 4))
        out = model.generate(idx, max_new_tokens=6, temperature=0.0)
        assert out.shape == (2, 10)
        torch.testing.assert_close(out[:, :4], idx)

    def test_greedy_is_deterministic(self, model, tiny_model_cfg):
        idx = torch.randint(0, tiny_model_cfg.vocab_size, (1, 4))
        a = model.generate(idx, max_new_tokens=8, temperature=0.0)
        b = model.generate(idx, max_new_tokens=8, temperature=0.0)
        torch.testing.assert_close(a, b)

    def test_sampling_is_reproducible_with_a_generator(self, model, tiny_model_cfg):
        idx = torch.randint(0, tiny_model_cfg.vocab_size, (1, 4))
        a = model.generate(
            idx, max_new_tokens=8, temperature=1.0, generator=torch.Generator().manual_seed(7)
        )
        b = model.generate(
            idx, max_new_tokens=8, temperature=1.0, generator=torch.Generator().manual_seed(7)
        )
        torch.testing.assert_close(a, b)

    def test_cached_generation_matches_uncached_logits(self, model, tiny_model_cfg):
        """Greedy decode with the KV cache must equal recomputing the full prefix.

        This is the model-level version of the attention cache test and is the
        check that catches a RoPE offset bug during decoding — which produces
        fluent, wrong text rather than an error.
        """
        idx = torch.randint(0, tiny_model_cfg.vocab_size, (1, 5))
        with torch.no_grad():
            cached = model.generate(idx, max_new_tokens=6, temperature=0.0)

            manual = idx.clone()
            for _ in range(6):
                logits, _ = model(manual)
                manual = torch.cat([manual, logits[:, -1:].argmax(dim=-1)], dim=1)
        torch.testing.assert_close(cached, manual)

    def test_top_k_restricts_support(self, model, tiny_model_cfg):
        idx = torch.randint(0, tiny_model_cfg.vocab_size, (1, 4))
        out = model.generate(
            idx, max_new_tokens=20, temperature=1.0, top_k=1,
            generator=torch.Generator().manual_seed(0),
        )
        greedy = model.generate(idx, max_new_tokens=20, temperature=0.0)
        # top_k=1 is greedy by construction.
        torch.testing.assert_close(out, greedy)

    def test_max_valid_id_masks_padded_vocab(self, model, tiny_model_cfg):
        """Padded vocab entries must never be emitted."""
        limit = 10
        out = model.generate(
            torch.zeros(1, 2, dtype=torch.long),
            max_new_tokens=30,
            temperature=1.0,
            max_valid_id=limit,
            generator=torch.Generator().manual_seed(0),
        )
        assert int(out[0, 2:].max()) < limit

    def test_eos_stops_generation(self, model):
        idx = torch.zeros(1, 2, dtype=torch.long)
        # Take whatever the greedy path emits first and declare it the EOS, so
        # the test does not depend on steering an untrained model's argmax.
        first_token = int(model.generate(idx, max_new_tokens=1, temperature=0.0)[0, -1])
        out = model.generate(idx, max_new_tokens=20, temperature=0.0, eos_id=first_token)
        assert out.shape[1] == 3, "generation did not stop at the EOS token"
        assert int(out[0, -1]) == first_token

    def test_generate_restores_training_mode(self, tiny_model_cfg):
        m = Transformer(tiny_model_cfg)
        m.train()
        m.generate(torch.zeros(1, 2, dtype=torch.long), max_new_tokens=2, temperature=0.0)
        assert m.training, "generate() must not leave the model in eval mode"

    def test_rejects_generation_past_context(self, model, tiny_model_cfg):
        idx = torch.zeros(1, tiny_model_cfg.max_seq_len - 2, dtype=torch.long)
        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            model.generate(idx, max_new_tokens=10)
