"""Typed configuration schema.

Configs are OmegaConf *structured* configs backed by dataclasses. Merging a YAML
file against the dataclass schema gives three things a plain dict does not:

1. Unknown keys are a hard error. A typo'd ``lr_scheudle`` fails at startup
   instead of silently training with the default schedule for six hours.
2. Types are checked and coerced at merge time, so ``lr: 3e-4`` parsed as the
   string ``"3e-4"`` by a YAML edge case fails immediately.
3. The resolved config round-trips to a plain dict for the checkpoint and the
   run record, so a run is reproducible from its own artifacts.

The Hydra entrypoint in ``microlab.cli`` is a thin wrapper over these functions:
core code never depends on Hydra, so tests construct :class:`Config` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2
    max_seq_len: int = 512
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    # None -> derived as 8/3 * d_model rounded up to ffn_multiple_of.
    ffn_hidden_dim: int | None = None
    ffn_multiple_of: int = 64
    tie_embeddings: bool = True
    bias: bool = False
    dropout: float = 0.0
    # Std of the truncated-normal init. Output projections are additionally
    # scaled by 1/sqrt(2 * n_layers); see Transformer.init_weights.
    init_std: float = 0.02


@dataclass
class DataConfig:
    train_bin: str = "data/tinystories/train.bin"
    val_bin: str = "data/tinystories/val.bin"
    tokenizer_path: str = "data/tinystories/tokenizer.json"
    # Per-device micro-batch. Global tokens/step = batch_size * grad_accum * seq_len.
    batch_size: int = 32
    grad_accum_steps: int = 1
    seq_len: int = 512


@dataclass
class OptimConfig:
    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    # "cosine" | "wsd" | "constant". WSD is what M4's flagship run uses; it is
    # here from the start so the schedule code path is identical across scales.
    schedule: str = "cosine"
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    # WSD only: fraction of total steps spent in the final decay phase.
    wsd_decay_fraction: float = 0.1


@dataclass
class TrainConfig:
    max_steps: int = 2000
    seed: int = 1337
    # "fp16" (T4/Turing) | "bf16" (Ampere+/TPU) | "fp32" (CPU, tests)
    precision: str = "fp16"
    device: str = "auto"
    compile: bool = False

    log_interval: int = 10
    eval_interval: int = 250
    eval_batches: int = 20
    sample_interval: int = 500
    sample_prompt: str = "Once upon a time"
    sample_max_new_tokens: int = 128

    # Checkpoint cadence is wall-clock, not steps: the binding constraint is a
    # 12-hour session limit, not a step count.
    checkpoint_every_minutes: float = 20.0
    checkpoint_every_steps: int = 0  # 0 disables; belt-and-braces for short runs
    keep_last_n: int = 3

    out_dir: str = "runs"
    run_id: str = "m0"
    # Resume from the latest checkpoint of run_id if one exists. This is what
    # makes a preempted session chain rather than restart.
    auto_resume: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


VALID_PRECISIONS = ("fp16", "bf16", "fp32")
VALID_SCHEDULES = ("cosine", "wsd", "constant")


def validate(cfg: Config) -> None:
    """Fail fast on configs that are type-correct but semantically impossible."""
    m, d, o, t = cfg.model, cfg.data, cfg.optim, cfg.train

    if m.d_model % m.n_heads != 0:
        raise ValueError(f"d_model={m.d_model} not divisible by n_heads={m.n_heads}")
    if m.n_heads % m.n_kv_heads != 0:
        raise ValueError(f"n_heads={m.n_heads} not divisible by n_kv_heads={m.n_kv_heads}")
    if (m.d_model // m.n_heads) % 2 != 0:
        raise ValueError(f"head_dim={m.d_model // m.n_heads} must be even for RoPE")
    if d.seq_len > m.max_seq_len:
        raise ValueError(f"data.seq_len={d.seq_len} exceeds model.max_seq_len={m.max_seq_len}")
    if t.precision not in VALID_PRECISIONS:
        raise ValueError(f"precision must be one of {VALID_PRECISIONS}, got {t.precision!r}")
    if o.schedule not in VALID_SCHEDULES:
        raise ValueError(f"schedule must be one of {VALID_SCHEDULES}, got {o.schedule!r}")
    if o.warmup_steps >= t.max_steps:
        raise ValueError(f"warmup_steps={o.warmup_steps} >= max_steps={t.max_steps}")
    if d.grad_accum_steps < 1 or d.batch_size < 1:
        raise ValueError("batch_size and grad_accum_steps must be >= 1")
    if not 0.0 <= o.min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {o.min_lr_ratio}")


def schema() -> DictConfig:
    """The structured-config schema, for merging YAML against."""
    return OmegaConf.structured(Config)


def from_dict(raw: dict[str, Any]) -> Config:
    """Merge a plain dict onto the schema and return a validated :class:`Config`."""
    merged = OmegaConf.merge(schema(), OmegaConf.create(raw))
    cfg: Config = OmegaConf.to_object(merged)  # type: ignore[assignment]
    validate(cfg)
    return cfg


def load(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML config file, apply ``key=value`` dotted overrides, validate.

    Overrides use Hydra/OmegaConf syntax (``optim.lr=1e-3``) so that the same
    string works whether it came through the Hydra CLI or a test.
    """
    raw = OmegaConf.load(Path(path))
    merged = OmegaConf.merge(schema(), raw)
    if overrides:
        merged.merge_with_dotlist(overrides)
    cfg: Config = OmegaConf.to_object(merged)  # type: ignore[assignment]
    validate(cfg)
    return cfg


def to_dict(cfg: Config) -> dict[str, Any]:
    """Resolved config as a plain JSON-serializable dict."""
    return OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)  # type: ignore[return-value]
