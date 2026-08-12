"""Trainer behaviour: determinism, resume equivalence, and accumulation.

The resume tests are the ones that matter. M4 runs across ~50 preempted
sessions, and a resume that is *nearly* right produces a run that trains to a
slightly worse loss with nothing in the logs to explain it.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from microlab.config import OptimConfig
from microlab.train.optim import lr_at_step
from microlab.train.trainer import Trainer

from .reference import reference_lr_cosine


def _with(cfg, **train_kwargs):
    """Copy a config, overriding train fields (and giving it a fresh run_id)."""
    return dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, **train_kwargs))


def _losses(trainer: Trainer) -> dict[int, float]:
    return {
        row["step"]: row["loss"]
        for row in trainer.record.read_metrics()
        if "loss" in row and "val_loss" not in row
    }


def _preempt_after(trainer: Trainer, n_steps: int) -> Trainer:
    """Make ``trainer`` stop after ``n_steps``, the way a preemption signal does.

    Note this keeps ``max_steps`` unchanged. Simply configuring a shorter run
    would *not* be an equivalent test: ``max_steps`` is an input to the cosine
    schedule, so a 5-step run and the first 5 steps of a 10-step run genuinely
    have different learning rates and are supposed to diverge. A real preempted
    session restarts with the same config, and that is what this reproduces.
    """
    original = trainer.train_step

    def wrapped():
        metrics = original()
        if trainer.step >= n_steps:
            trainer._stop_requested = True
            trainer._stop_reason = "SIGTERM"
        return metrics

    trainer.train_step = wrapped  # type: ignore[method-assign]
    return trainer


class TestDeterminism:
    def test_two_runs_with_same_seed_match(self, train_cfg):
        a = Trainer(_with(train_cfg, run_id="det_a")).train()
        b = Trainer(_with(train_cfg, run_id="det_b")).train()
        assert a["best_val_loss"] == b["best_val_loss"]

    def test_weights_match_bitwise(self, train_cfg):
        ta = Trainer(_with(train_cfg, run_id="w_a", max_steps=5))
        ta.train()
        tb = Trainer(_with(train_cfg, run_id="w_b", max_steps=5))
        tb.train()
        for (na, pa), (nb, pb) in zip(
            ta.model.named_parameters(), tb.model.named_parameters(), strict=True
        ):
            assert na == nb
            assert torch.equal(pa, pb), f"{na} diverged between identical runs"

    def test_different_seed_diverges(self, train_cfg):
        a = Trainer(_with(train_cfg, run_id="s_a", seed=1, max_steps=5))
        a.train()
        b = Trainer(_with(train_cfg, run_id="s_b", seed=2, max_steps=5))
        b.train()
        assert not torch.equal(a.model.tok_emb.weight, b.model.tok_emb.weight)


class TestResume:
    @pytest.mark.parametrize("dropout", [0.0, 0.1])
    def test_resumed_run_matches_uninterrupted(self, train_cfg, dropout):
        """The M0 gate's core claim.

        The dropout=0.1 case additionally proves the RNG streams round-trip: with
        dropout active, a resume that restores weights and optimizer state but
        not RNG state produces different masks and visibly different losses.
        """
        cfg = dataclasses.replace(
            train_cfg, model=dataclasses.replace(train_cfg.model, dropout=dropout)
        )

        full = Trainer(_with(cfg, run_id="full", max_steps=10))
        full.train()
        full_losses = _losses(full)

        # Interrupted: same config, killed at step 5, new process resumes.
        first = _preempt_after(Trainer(_with(cfg, run_id="split", max_steps=10)), 5)
        summary = first.train()
        assert summary["reason"] == "signal_SIGTERM"
        del first
        second = Trainer(_with(cfg, run_id="split", max_steps=10))
        assert second.step == 5, "did not resume from the checkpoint"
        second.train()
        split_losses = _losses(second)

        for step in range(6, 11):
            assert split_losses[step] == pytest.approx(full_losses[step], rel=1e-6, abs=1e-9), (
                f"step {step}: resumed {split_losses[step]} != uninterrupted {full_losses[step]}"
            )

        for (name, p_full), (_, p_split) in zip(
            full.model.named_parameters(), second.model.named_parameters(), strict=True
        ):
            torch.testing.assert_close(p_full, p_split, rtol=0, atol=0, msg=f"{name} diverged")

    def test_resume_restores_step_and_token_count(self, train_cfg):
        first = _preempt_after(Trainer(_with(train_cfg, run_id="counts", max_steps=12)), 6)
        first.train()
        tokens = first.tokens_seen
        del first

        second = Trainer(_with(train_cfg, run_id="counts", max_steps=12))
        assert second.step == 6
        assert second.tokens_seen == tokens

    def test_resume_restores_optimizer_moments(self, train_cfg):
        """Adam's moments are state; dropping them restarts the optimizer warmup
        and shows up as a loss bump right after every preemption."""
        first = _preempt_after(Trainer(_with(train_cfg, run_id="moments", max_steps=10)), 5)
        first.train()
        ref = {
            id_: {k: v.clone() for k, v in st.items() if torch.is_tensor(v)}
            for id_, st in enumerate(first.optimizer.state.values())
        }
        del first

        second = Trainer(_with(train_cfg, run_id="moments", max_steps=10, auto_resume=True))
        got = {
            id_: {k: v for k, v in st.items() if torch.is_tensor(v)}
            for id_, st in enumerate(second.optimizer.state.values())
        }
        assert got and len(got) == len(ref)
        for key in ref:
            for name in ref[key]:
                torch.testing.assert_close(got[key][name], ref[key][name], rtol=0, atol=0)

    def test_completed_run_does_not_train_further(self, train_cfg):
        Trainer(_with(train_cfg, run_id="done", max_steps=5)).train()
        again = Trainer(_with(train_cfg, run_id="done", max_steps=5))
        summary = again.train()
        assert summary["reason"] == "already_complete"
        assert summary["step"] == 5

    def test_auto_resume_off_starts_fresh(self, train_cfg):
        Trainer(_with(train_cfg, run_id="fresh", max_steps=5)).train()
        again = Trainer(_with(train_cfg, run_id="fresh", max_steps=5, auto_resume=False))
        assert again.step == 0

    def test_run_record_is_continuous_across_sessions(self, train_cfg):
        first = _preempt_after(Trainer(_with(train_cfg, run_id="chain", max_steps=10)), 5)
        first.train()
        second = Trainer(_with(train_cfg, run_id="chain", max_steps=10))
        second.train()

        rows = second.record.read_metrics()
        steps = sorted({r["step"] for r in rows if "loss" in r})
        assert steps == list(range(1, 11)), "metric history has a hole across the session boundary"
        assert second.record.session_index == 1, "session numbering must count sessions, not lines"

        events = [
            __import__("json").loads(line)
            for line in (second.record.dir / "sessions.jsonl").read_text().strip().split("\n")
        ]
        assert [e["event"] for e in events].count("begin") == 2
        assert events[1]["event"] == "end" and events[1]["reason"] == "signal_SIGTERM"
        assert events[2]["resumed_from_step"] == 5


class TestGradientAccumulation:
    def test_accumulation_matches_large_batch(self, train_cfg):
        """4x1 accumulated must equal 1x4 in one batch, to float tolerance.

        If the 1/accum scaling is wrong, the effective learning rate scales with
        the accumulation factor and this diverges immediately.
        """
        big = Trainer(
            dataclasses.replace(
                _with(train_cfg, run_id="accum_big", max_steps=3),
                data=dataclasses.replace(train_cfg.data, batch_size=8, grad_accum_steps=1),
            )
        )
        big.train()

        small = Trainer(
            dataclasses.replace(
                _with(train_cfg, run_id="accum_small", max_steps=3),
                data=dataclasses.replace(train_cfg.data, batch_size=4, grad_accum_steps=2),
            )
        )
        small.train()

        for (name, p_big), (_, p_small) in zip(
            big.model.named_parameters(), small.model.named_parameters(), strict=True
        ):
            torch.testing.assert_close(p_big, p_small, rtol=1e-4, atol=1e-6, msg=name)

    def test_token_accounting_includes_accumulation(self, train_cfg):
        cfg = dataclasses.replace(
            _with(train_cfg, run_id="tok", max_steps=3),
            data=dataclasses.replace(train_cfg.data, batch_size=4, grad_accum_steps=2),
        )
        t = Trainer(cfg)
        t.train()
        assert t.tokens_seen == 3 * 4 * 2 * cfg.data.seq_len


class TestTrainingProgress:
    def test_loss_decreases(self, train_cfg):
        """A working loop must actually learn the synthetic grammar."""
        t = Trainer(_with(train_cfg, run_id="learn", max_steps=60, eval_interval=0))
        t.train()
        losses = _losses(t)
        early = sum(losses[s] for s in range(1, 11)) / 10
        late = sum(losses[s] for s in range(51, 61)) / 10
        assert late < early * 0.9, f"loss did not fall: {early:.3f} -> {late:.3f}"

    def test_metrics_are_recorded(self, train_cfg):
        t = Trainer(_with(train_cfg, run_id="metrics", max_steps=3))
        t.train()
        row = next(r for r in t.record.read_metrics() if "loss" in r)
        for key in ("lr", "grad_norm", "step_time_s", "tokens_per_s", "epoch", "tokens_seen"):
            assert key in row, f"missing metric {key}"

    def test_nonfinite_loss_stops_the_run(self, train_cfg):
        """A poisoned run must stop and exit non-zero, not keep burning quota."""
        t = Trainer(_with(train_cfg, run_id="nan", max_steps=10))
        original = t.compiled_model

        class Poisoned(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x, targets=None, caches=None):
                logits, loss = self.inner(x, targets=targets, caches=caches)
                return logits, loss * float("nan") if loss is not None else loss

        t.compiled_model = Poisoned(original)
        summary = t.train()
        assert summary["reason"] == "nonfinite_loss"
        assert any(r.get("event") == "nonfinite_loss" for r in t.record.read_metrics())

    def test_config_change_across_sessions_is_recorded(self, train_cfg):
        Trainer(_with(train_cfg, run_id="cfgchg", max_steps=3)).train()
        Trainer(_with(train_cfg, run_id="cfgchg", max_steps=6)).train()
        sessions = (train_cfg.train.out_dir + "/cfgchg/sessions.jsonl",)
        text = open(sessions[0]).read()
        assert "config_changed" in text
        assert "train.max_steps" in text


class TestLRSchedule:
    def test_cosine_matches_reference(self):
        cfg = OptimConfig(lr=1e-3, warmup_steps=10, min_lr_ratio=0.1, schedule="cosine")
        for step in (0, 5, 9, 10, 50, 99, 100):
            assert lr_at_step(step, cfg, 100) == pytest.approx(
                reference_lr_cosine(step, 1e-3, 10, 100, 0.1)
            )

    def test_warmup_is_linear_and_nonzero_at_step_zero(self):
        cfg = OptimConfig(lr=1e-3, warmup_steps=10)
        assert lr_at_step(0, cfg, 100) == pytest.approx(1e-4)
        assert lr_at_step(4, cfg, 100) == pytest.approx(5e-4)
        assert lr_at_step(9, cfg, 100) == pytest.approx(1e-3)

    def test_cosine_ends_at_min_lr(self):
        cfg = OptimConfig(lr=1e-3, warmup_steps=10, min_lr_ratio=0.1, schedule="cosine")
        assert lr_at_step(100, cfg, 100) == pytest.approx(1e-4)

    def test_schedule_is_a_pure_function_of_global_step(self):
        """Resume correctness: the LR must not depend on call order."""
        cfg = OptimConfig(lr=1e-3, warmup_steps=10, schedule="cosine")
        forward = [lr_at_step(s, cfg, 100) for s in range(100)]
        backward = [lr_at_step(s, cfg, 100) for s in reversed(range(100))][::-1]
        assert forward == backward

    def test_wsd_holds_then_decays(self):
        cfg = OptimConfig(lr=1e-3, warmup_steps=10, schedule="wsd", wsd_decay_fraction=0.2)
        # Stable phase: constant, so any checkpoint here is a valid branch point.
        assert lr_at_step(20, cfg, 100) == pytest.approx(1e-3)
        assert lr_at_step(79, cfg, 100) == pytest.approx(1e-3)
        # Decay phase: monotonically down to min_lr.
        decay = [lr_at_step(s, cfg, 100) for s in range(80, 101)]
        assert all(b <= a + 1e-12 for a, b in zip(decay, decay[1:], strict=False))
        assert decay[-1] == pytest.approx(1e-4)

    def test_constant_schedule(self):
        cfg = OptimConfig(lr=1e-3, warmup_steps=5, schedule="constant")
        assert lr_at_step(50, cfg, 100) == pytest.approx(1e-3)

    def test_unknown_schedule_rejected(self):
        cfg = OptimConfig(warmup_steps=1)
        cfg.schedule = "made_up"
        with pytest.raises(ValueError, match="unknown schedule"):
            lr_at_step(5, cfg, 100)

    def test_lr_is_always_finite_and_positive(self):
        for schedule in ("cosine", "wsd", "constant"):
            cfg = OptimConfig(lr=1e-3, warmup_steps=10, schedule=schedule)
            for step in range(0, 200):
                lr = lr_at_step(step, cfg, 100)
                assert math.isfinite(lr) and lr > 0
