"""CLI argument handling.

Overrides are collected from argparse's leftovers, which is flexible enough to
accept a typo as an override if it is not checked. These tests pin both
directions: real overrides parse anywhere on the line, and typos still fail.
"""

from __future__ import annotations

import pytest

from microlab.cli import build_parser, load_config, main
from microlab.tokenizer.bpe import BPETokenizer


def _parse(argv):
    parser = build_parser()
    args, leftovers = parser.parse_known_args(argv)
    return args, leftovers


class TestOverrideParsing:
    def test_overrides_after_an_option(self):
        """The exact form that broke: an option between the config and overrides."""
        args, leftovers = _parse(
            ["train", "m0", "--config-dir", "/tmp/x", "optim.lr=1e-3", "train.max_steps=5"]
        )
        assert args.config == "m0"
        assert str(args.config_dir) == "/tmp/x"
        assert leftovers == ["optim.lr=1e-3", "train.max_steps=5"]

    def test_overrides_before_an_option(self):
        args, leftovers = _parse(["train", "m0", "optim.lr=1e-3", "--config-dir", "/tmp/x"])
        assert leftovers == ["optim.lr=1e-3"]

    def test_config_name_defaults(self):
        args, _ = _parse(["train"])
        assert args.config == "m0"

    def test_typo_is_rejected_not_swallowed(self):
        with pytest.raises(SystemExit):
            main(["train", "m0", "--config-dirr", "/tmp/x"])

    def test_bare_word_is_rejected(self):
        """A stray word is a mistake, not an override."""
        with pytest.raises(SystemExit):
            main(["train", "m0", "maxsteps"])

    def test_subcommand_required(self):
        with pytest.raises(SystemExit):
            main([])


class TestSubcommands:
    @pytest.mark.parametrize("name", ["train", "prepare", "sample", "bench"])
    def test_subcommand_exists(self, name):
        args, _ = _parse([name])
        assert args.func is not None

    def test_prepare_flags(self):
        args, _ = _parse(
            ["prepare", "--source", "synthetic", "--vocab-size", "512", "--out-dir", "/tmp/d"]
        )
        assert args.source == "synthetic"
        assert args.vocab_size == 512

    def test_bench_flags(self):
        args, _ = _parse(["bench", "m0", "--steps", "3", "--warmup", "1"])
        assert args.steps == 3 and args.warmup == 1


class TestSampleCommand:
    """`microlab sample` end to end — the command the README tells people to run.

    Argument parsing is not enough here: the bug this guards against was a
    missing argument at the *call site*, not a parsing error. The trainer's
    sampler masked the padded vocabulary tail; the CLI's did not, so
    `microlab sample` raised `token id N not in vocabulary` on any model whose
    vocab was padded — which is every real config, since vocabularies are
    padded to a multiple of 64 for GEMM alignment.
    """

    def _padded_overrides(self, cli_workspace, run_id: str) -> tuple[list[str], int]:
        cfg = load_config("chain", [], cli_workspace["config_dir"])
        tokenizer = BPETokenizer.load(cfg.data.tokenizer_path)
        # Pad well past the tokenizer so an untrained model is near-certain to
        # sample an untokenizable id within a few dozen tokens.
        padded = tokenizer.vocab_size + 256
        return (
            [f"model.vocab_size={padded}", "train.max_steps=5", f"train.run_id={run_id}"],
            tokenizer.vocab_size,
        )

    def test_sample_runs_with_a_padded_vocabulary(self, cli_workspace):
        config_dir = str(cli_workspace["config_dir"])
        overrides, vocab_size = self._padded_overrides(cli_workspace, "sample_pad")
        assert vocab_size > 0

        assert main(["train", "chain", "--config-dir", config_dir, *overrides]) == 0
        assert (
            main(
                [
                    # The fixture model has max_seq_len 64; leave room for the
                    # prompt so this exercises sampling, not the length guard.
                    "sample", "chain", "--config-dir", config_dir, *overrides,
                    "--max-new-tokens", "40", "--temperature", "1.0", "--seed", "0",
                ]
            )
            == 0
        )

    def test_sample_without_a_checkpoint_fails_cleanly(self, cli_workspace):
        """Missing checkpoints should be a clear message and exit 1, not a traceback."""
        config_dir = str(cli_workspace["config_dir"])
        code = main(
            ["sample", "chain", "--config-dir", config_dir, "train.run_id=never_trained"]
        )
        assert code == 1
