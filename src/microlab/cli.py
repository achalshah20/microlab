"""Command-line entry points.

Config composition goes through Hydra's ``compose`` API rather than the
``@hydra.main`` decorator. The decorator owns the process: it takes over
argument parsing (so subcommands are awkward), changes the working directory,
and installs its own output directory scheme — all of which fight the run-record
and session-chaining layers, which need stable relative paths across ~50
restarts. ``compose`` gives the part that is actually wanted (config groups,
defaults lists, dotted overrides) and leaves process control alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from . import config as config_mod
from .config import Config

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_config(config_name: str, overrides: list[str], config_dir: Path | None = None) -> Config:
    """Compose a config by name with Hydra, then validate it against the schema."""
    directory = (config_dir or DEFAULT_CONFIG_DIR).resolve()
    if not directory.exists():
        raise FileNotFoundError(f"config directory not found: {directory}")
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        composed = compose(config_name=config_name, overrides=overrides)
    raw = OmegaConf.to_container(composed, resolve=True)
    return config_mod.from_dict(raw)  # type: ignore[arg-type]


def cmd_train(args: argparse.Namespace) -> int:
    from .train.trainer import Trainer

    cfg = load_config(args.config, args.overrides, args.config_dir)
    print(OmegaConf.to_yaml(OmegaConf.create(config_mod.to_dict(cfg))))
    trainer = Trainer(cfg)
    summary = trainer.train()
    print(json.dumps(summary, indent=2, default=str))
    # A run stopped by a non-finite loss must not exit 0: on Kaggle the chaining
    # wrapper would otherwise treat it as a clean finish and never restart it.
    return 0 if summary["reason"] != "nonfinite_loss" else 1


def cmd_prepare(args: argparse.Namespace) -> int:
    from .data.prepare import prepare

    result = prepare(
        source=args.source,
        out_dir=args.out_dir,
        vocab_size=args.vocab_size,
        val_fraction=args.val_fraction,
        limit=args.limit,
        seed=args.seed,
        tokenizer_train_chars=args.tokenizer_train_chars,
    )
    print(json.dumps(result.__dict__, indent=2, default=str))
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    import torch

    from .model.transformer import Transformer
    from .tokenizer.bpe import BPETokenizer
    from .train.checkpoint import load_latest

    cfg = load_config(args.config, args.overrides, args.config_dir)
    run_dir = Path(cfg.train.out_dir) / cfg.train.run_id
    loaded = load_latest(run_dir / "checkpoints", map_location="cpu")
    if loaded is None:
        print(f"no checkpoints in {run_dir / 'checkpoints'}", file=sys.stderr)
        return 1
    payload, path = loaded

    model = Transformer(cfg.model)
    model.load_state_dict(payload["model"])
    model.eval()
    tokenizer = BPETokenizer.load(cfg.data.tokenizer_path)

    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)
    gen = torch.Generator().manual_seed(args.seed)
    out = model.generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=tokenizer.eos_id,
        generator=gen,
        # Without this the sampler can emit an id from the padded tail of the
        # vocabulary — real model outputs that the tokenizer has no bytes for —
        # and decode raises. The trainer's sampler already masks them; this path
        # did not, so `microlab sample` crashed on any lightly-trained model.
        max_valid_id=tokenizer.vocab_size,
    )
    print(f"# checkpoint: {path.name} (step {payload['step']})", file=sys.stderr)
    print(tokenizer.decode(out[0].tolist()))
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .bench import run_benchmark

    cfg = load_config(args.config, args.overrides, args.config_dir)
    results = run_benchmark(cfg, steps=args.steps, warmup=args.warmup)
    print(json.dumps(results, indent=2, default=str))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    return 0


def cmd_shapes(args: argparse.Namespace) -> int:
    from .shapes import profile_shapes, render_markdown

    cfg = load_config(args.config, args.overrides, args.config_dir)
    report = profile_shapes(cfg.model, batch_size=args.batch_size, seq_len=args.seq_len)
    markdown = render_markdown(report, first_block_only=not args.all_blocks)
    if args.out:
        Path(args.out).write_text(markdown)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(markdown)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="microlab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_config_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("config", nargs="?", default="m0", help="config name in configs/")
        p.add_argument("--config-dir", type=Path, default=None)
        # Overrides are NOT a positional argument. argparse cannot interleave two
        # positionals with an option between them, so
        # `train m0 --config-dir X optim.lr=1e-3` fails to parse. They are
        # collected from the leftovers in main() instead.

    p_train = sub.add_parser("train", help="train a model")
    add_config_args(p_train)
    p_train.set_defaults(func=cmd_train)

    p_prep = sub.add_parser("prepare", help="build tokenizer and packed shards")
    p_prep.add_argument("--source", default="synthetic", help="tinystories | synthetic | path")
    p_prep.add_argument("--out-dir", default="data/synthetic")
    p_prep.add_argument("--vocab-size", type=int, default=4096)
    p_prep.add_argument("--val-fraction", type=float, default=0.02)
    p_prep.add_argument("--limit", type=int, default=None)
    p_prep.add_argument("--seed", type=int, default=0)
    p_prep.add_argument(
        "--tokenizer-train-chars",
        type=int,
        default=20_000_000,
        help="cap on characters used to train the tokenizer (the slow, "
        "pure-Python step); merge frequencies are stable well below this",
    )
    p_prep.set_defaults(func=cmd_prepare)

    p_sample = sub.add_parser("sample", help="sample from the latest checkpoint")
    add_config_args(p_sample)
    p_sample.add_argument("--prompt", default="Once upon a time")
    p_sample.add_argument("--max-new-tokens", type=int, default=128)
    p_sample.add_argument("--temperature", type=float, default=0.8)
    p_sample.add_argument("--top-k", type=int, default=50)
    p_sample.add_argument("--seed", type=int, default=0)
    p_sample.set_defaults(func=cmd_sample)

    p_bench = sub.add_parser("bench", help="throughput / memory / MFU benchmark")
    add_config_args(p_bench)
    p_bench.add_argument("--steps", type=int, default=30)
    p_bench.add_argument("--warmup", type=int, default=5)
    p_bench.add_argument("--out", default=None, help="write results JSON here")
    p_bench.set_defaults(func=cmd_bench)

    p_shapes = sub.add_parser("shapes", help="annotate a forward pass with shapes and FLOPs")
    add_config_args(p_shapes)
    p_shapes.add_argument("--batch-size", type=int, default=1)
    p_shapes.add_argument("--seq-len", type=int, default=None)
    p_shapes.add_argument(
        "--all-blocks", action="store_true", help="do not collapse identical blocks"
    )
    p_shapes.add_argument("--out", default=None, help="write Markdown here instead of stdout")
    p_shapes.set_defaults(func=cmd_shapes)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, leftovers = parser.parse_known_args(argv)

    # Everything argparse did not claim is treated as a Hydra dotted override.
    # Requiring an '=' keeps a genuine typo (`--config-dirr`, `trian`) an error
    # rather than a silently ignored argument — which on a 12-hour session means
    # discovering at the end that the override never applied.
    bad = [tok for tok in leftovers if "=" not in tok or tok.startswith("-")]
    if bad:
        parser.error(f"unrecognized arguments: {' '.join(bad)}")
    args.overrides = leftovers
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
