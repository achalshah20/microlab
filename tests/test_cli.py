"""CLI argument handling.

Overrides are collected from argparse's leftovers, which is flexible enough to
accept a typo as an override if it is not checked. These tests pin both
directions: real overrides parse anywhere on the line, and typos still fail.
"""

from __future__ import annotations

import pytest

from microlab.cli import build_parser, main


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
