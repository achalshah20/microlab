"""The training loop, built to be interrupted.

Every free-tier session dies at 12 hours, so the loop treats preemption as
routine rather than exceptional: state is checkpointed on a wall-clock cadence,
SIGTERM is caught and turned into a final flush, and startup resumes the
existing run instead of beginning a new one. A run is a sequence of sessions
that share one run record and one continuous metric history.
"""

from __future__ import annotations

import math
import signal
import time
from pathlib import Path
from typing import Any

import torch

from ..config import Config, to_dict
from ..data.loader import DeterministicBatcher, SequentialBatcher
from ..data.packed import PackedDataset
from ..model.transformer import Transformer
from ..precision import build_precision
from ..tokenizer.bpe import BPETokenizer
from ..utils.determinism import load_rng_state, rng_state, seed_everything
from ..utils.hardware import describe_device, mfu
from ..utils.runrecord import RunRecord
from .checkpoint import load_latest, save_checkpoint
from .optim import build_optimizer, lr_at_step, set_lr


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


class Trainer:
    """Owns the model, data, optimizer and run record for one training run."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.train.device)
        seed_everything(cfg.train.seed)

        self.record = RunRecord(cfg.train.out_dir, cfg.train.run_id)
        self.record.save_config(cfg)

        self.model = Transformer(cfg.model).to(self.device)
        self.optimizer = build_optimizer(self.model, cfg.optim)
        self.precision = build_precision(cfg.train.precision, self.device.type)

        self.train_data = PackedDataset(cfg.data.train_bin, cfg.data.seq_len)
        self.batcher = DeterministicBatcher(
            self.train_data,
            batch_size=cfg.data.batch_size,
            seed=cfg.train.seed,
            grad_accum_steps=cfg.data.grad_accum_steps,
        )
        val_path = Path(cfg.data.val_bin)
        self.val_batcher = (
            SequentialBatcher(PackedDataset(val_path, cfg.data.seq_len), cfg.data.batch_size)
            if val_path.exists()
            else None
        )
        tok_path = Path(cfg.data.tokenizer_path)
        self.tokenizer = BPETokenizer.load(tok_path) if tok_path.exists() else None
        if self.tokenizer is not None and self.tokenizer.vocab_size > cfg.model.vocab_size:
            # The reverse (model vocab larger) is fine and common — vocabularies
            # get padded for GEMM alignment. This direction is not: the shard
            # contains ids the embedding table has no row for, and training
            # would fail on whichever batch first contains one, thousands of
            # steps in.
            raise ValueError(
                f"tokenizer vocab ({self.tokenizer.vocab_size}) exceeds model vocab "
                f"({cfg.model.vocab_size}); the shard holds ids the model cannot embed"
            )
        self._check_shard_tokenizer_match()

        self.step = 0
        self.tokens_seen = 0
        self.best_val_loss: float | None = None
        self._stop_requested = False
        self._stop_reason = ""

        resumed_from = None
        if cfg.train.auto_resume:
            resumed_from = self._maybe_resume()

        # The compiled model is not checkpointed: compilation is a property of
        # the box, not the run, and a checkpoint written by a compiled model must
        # stay loadable by an eager one (its state dict keys are prefixed).
        self.compiled_model = self.model
        if cfg.train.compile:
            self.compiled_model = torch.compile(self.model)  # type: ignore[assignment]

        self.record.begin_session(resumed_from_step=resumed_from)
        self._install_signal_handlers()

    def _check_shard_tokenizer_match(self) -> None:
        """Refuse to train on shards produced by a different tokenizer.

        Encoding with tokenizer A and decoding with tokenizer B yields fluent
        nonsense and a loss curve that looks normal. The manifest records the
        hash precisely so this is catchable at startup instead of at eval time.
        """
        manifest = self.train_data.manifest
        if manifest is None or self.tokenizer is None or not manifest.tokenizer_sha:
            return
        if manifest.tokenizer_sha != self.tokenizer.sha():
            raise ValueError(
                f"shard {self.train_data.path} was packed with tokenizer "
                f"{manifest.tokenizer_sha} but {self.cfg.data.tokenizer_path} hashes to "
                f"{self.tokenizer.sha()}; re-run data preparation"
            )

    # ---- session lifecycle ----------------------------------------------

    def _install_signal_handlers(self) -> None:
        """Turn a preemption signal into a clean final checkpoint.

        The handler only sets a flag. Checkpointing from inside a signal handler
        would run concurrently with the training step that is mid-backward, and
        the state it captured would be internally inconsistent — exactly the
        corruption this whole layer exists to prevent.
        """

        def handler(signum, _frame):
            self._stop_requested = True
            self._stop_reason = signal.Signals(signum).name

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, handler)
            except ValueError:  # pragma: no cover - not the main thread (tests)
                pass

    def _maybe_resume(self) -> int | None:
        loaded = load_latest(self.record.checkpoints_dir, map_location=str(self.device))
        if loaded is None:
            return None
        payload, path = loaded

        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.precision.load_state_dict(payload.get("precision", {}))
        self.step = int(payload["step"])
        self.tokens_seen = int(payload.get("tokens_seen", 0))
        self.best_val_loss = payload.get("best_val_loss")
        if "rng" in payload:
            load_rng_state(payload["rng"])

        self.record.log_metrics(
            self.step,
            {"event": "resume", "checkpoint": path.name, "tokens_seen": self.tokens_seen},
        )
        return self.step

    def _save(self, val_loss: float | None = None) -> Path:
        payload: dict[str, Any] = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "precision": self.precision.state_dict(),
            "rng": rng_state(),
            "config": to_dict(self.cfg),
            "tokens_seen": self.tokens_seen,
            "best_val_loss": self.best_val_loss,
        }
        return save_checkpoint(
            self.record.checkpoints_dir,
            step=self.step,
            payload=payload,
            run_id=self.cfg.train.run_id,
            session=self.record.session_index,
            tokens_seen=self.tokens_seen,
            val_loss=val_loss,
            keep_last_n=self.cfg.train.keep_last_n,
        )

    # ---- the loop ---------------------------------------------------------

    def train_step(self) -> dict[str, float]:
        """One optimizer step, including all gradient-accumulation micro-steps."""
        cfg = self.cfg
        t0 = time.perf_counter()

        lr = lr_at_step(self.step, cfg.optim, cfg.train.max_steps)
        set_lr(self.optimizer, lr)

        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for micro in range(cfg.data.grad_accum_steps):
            x, y = self.batcher.batch(self.step, micro, device=self.device)
            with self.precision.autocast():
                _, loss = self.compiled_model(x, targets=y)
            # Scale by 1/accum so the gradient equals what a single large batch
            # would produce. Without this the effective LR scales with
            # grad_accum_steps, which turns a memory workaround into a
            # different experiment.
            self.precision.backward(loss / cfg.data.grad_accum_steps)
            total_loss += loss.detach().float().item()

        grad_norm = self.precision.clip_grad_norm(self.model, self.optimizer, cfg.optim.grad_clip)
        stepped = self.precision.step(self.optimizer)

        self.step += 1
        tokens = self.batcher.tokens_per_optimizer_step
        self.tokens_seen += tokens

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tokens_per_s = tokens / dt

        metrics = {
            "loss": total_loss / cfg.data.grad_accum_steps,
            "lr": lr,
            "grad_norm": float(grad_norm),
            "step_time_s": dt,
            "tokens_per_s": tokens_per_s,
            "tokens_seen": self.tokens_seen,
            "epoch": self.batcher.epoch_at_step(self.step),
            "stepped": float(stepped),
        }
        util = mfu(
            tokens_per_s,
            self.model.flops_per_token(self.cfg.data.seq_len),
            self.cfg.train.precision,
            self.device,
        )
        if util is not None:
            metrics["mfu"] = util
        if self.device.type == "cuda":
            metrics["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        metrics.update(self.precision.metrics())
        return metrics

    @torch.no_grad()
    def evaluate(self) -> float | None:
        """Mean val loss over a fixed set of batches. Same batches every time."""
        if self.val_batcher is None:
            return None
        was_training = self.model.training
        self.model.eval()
        try:
            losses = []
            n = min(self.cfg.train.eval_batches, self.val_batcher.n_batches)
            for i in range(n):
                x, y = self.val_batcher.batch(i, device=self.device)
                with self.precision.autocast():
                    _, loss = self.compiled_model(x, targets=y)
                losses.append(loss.float().item())
            return sum(losses) / len(losses) if losses else None
        finally:
            self.model.train(was_training)

    def sample(self) -> str | None:
        """Generate one sample and write it into the run record."""
        if self.tokenizer is None:
            return None
        cfg = self.cfg.train
        prompt_ids = self.tokenizer.encode(cfg.sample_prompt)
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        budget = min(cfg.sample_max_new_tokens, self.cfg.model.max_seq_len - len(prompt_ids) - 1)
        if budget <= 0:
            return None
        # Sampling uses its own generator so that drawing a sample does not
        # perturb the training RNG stream — otherwise the run diverges from an
        # identical run with sampling disabled.
        gen = torch.Generator(device=self.device).manual_seed(
            self.cfg.train.seed + self.step
        )
        out = self.model.generate(
            idx,
            max_new_tokens=budget,
            temperature=0.8,
            top_k=50,
            generator=gen,
            eos_id=self.tokenizer.eos_id,
            max_valid_id=self.tokenizer.vocab_size,
        )
        text = self.tokenizer.decode(out[0].tolist())
        self.record.save_sample(self.step, cfg.sample_prompt, text)
        return text

    def train(self) -> dict[str, Any]:
        """Run until ``max_steps``, preemption, or a stop signal."""
        cfg = self.cfg.train
        self.model.train()
        last_ckpt_time = time.time()
        reason = "completed"

        if self.step >= cfg.max_steps:
            self.record.end_session(self.step, "already_complete")
            return self.summary("already_complete")

        while self.step < cfg.max_steps:
            metrics = self.train_step()

            if self.step % cfg.log_interval == 0 or self.step == 1:
                self.record.log_metrics(self.step, metrics)

            if not math.isfinite(metrics["loss"]):
                # A non-finite loss is unrecoverable by continuing: the weights
                # are already poisoned. Stop and leave the last good checkpoint
                # in place so the run can be restarted from it. Automatic
                # rollback is M4's loss-spike detector; here we fail loudly.
                reason = "nonfinite_loss"
                self.record.log_metrics(self.step, {"event": "nonfinite_loss", **metrics})
                break

            val_due = cfg.eval_interval > 0 and self.step % cfg.eval_interval == 0
            if val_due:
                val_loss = self.evaluate()
                if val_loss is not None:
                    if self.best_val_loss is None or val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                    self.record.log_metrics(
                        self.step, {"val_loss": val_loss, "best_val_loss": self.best_val_loss}
                    )

            if cfg.sample_interval > 0 and self.step % cfg.sample_interval == 0:
                self.sample()

            time_due = (time.time() - last_ckpt_time) >= cfg.checkpoint_every_minutes * 60
            step_due = (
                cfg.checkpoint_every_steps > 0
                and self.step % cfg.checkpoint_every_steps == 0
            )
            if time_due or step_due:
                self._save(val_loss=self.best_val_loss)
                last_ckpt_time = time.time()

            if self._stop_requested:
                reason = f"signal_{self._stop_reason}"
                break

        # Always checkpoint on the way out, whatever the reason. This is the
        # save that makes a preempted session cost minutes instead of hours.
        self._save(val_loss=self.best_val_loss)
        final_val = self.evaluate()
        if final_val is not None:
            if self.best_val_loss is None or final_val < self.best_val_loss:
                self.best_val_loss = final_val
            self.record.log_metrics(self.step, {"val_loss": final_val, "event": "final_eval"})

        self.record.end_session(self.step, reason)
        return self.summary(reason)

    def summary(self, reason: str) -> dict[str, Any]:
        return {
            "run_id": self.cfg.train.run_id,
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "best_val_loss": self.best_val_loss,
            "reason": reason,
            "session": self.record.session_index,
            "device": (
                describe_device(self.device) if self.device.type == "cuda" else {"name": "cpu"}
            ),
            "params_total": self.model.num_params(non_embedding=False),
            "params_non_embedding": self.model.num_params(non_embedding=True),
            "run_dir": str(self.record.dir),
        }
