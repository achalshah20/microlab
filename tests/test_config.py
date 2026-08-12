"""Config schema: typos and impossible configs must fail at startup."""

from __future__ import annotations

import pytest
from hydra.errors import HydraException
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from microlab import config as config_mod
from microlab.cli import load_config
from microlab.config import Config, from_dict, to_dict, validate

# OmegaConf raises several distinct subclasses for schema violations
# (ConfigKeyError, ValidationError, ...); this is their common base.
ConfigError = OmegaConfBaseException


class TestSchema:
    def test_defaults_are_valid(self):
        validate(Config())

    def test_unknown_key_is_rejected(self):
        """A typo'd key must not silently train with the default value."""
        with pytest.raises(ConfigError, match="(?i)key.*not in|not a valid|full key"):
            from_dict({"optim": {"lr_scheudle": "cosine"}})

    def test_types_are_coerced_and_checked(self):
        cfg = from_dict({"optim": {"lr": "3e-4"}})
        assert isinstance(cfg.optim.lr, float)
        with pytest.raises(ConfigError):
            from_dict({"model": {"n_layers": "not a number"}})

    def test_round_trips_through_dict(self):
        cfg = from_dict({"model": {"d_model": 256, "n_heads": 8, "n_kv_heads": 4}})
        assert from_dict(to_dict(cfg)) == cfg

    def test_to_dict_is_json_serializable(self):
        import json

        json.dumps(to_dict(Config()))


class TestValidation:
    @pytest.mark.parametrize(
        "patch,match",
        [
            ({"model": {"d_model": 100, "n_heads": 8}}, "not divisible"),
            ({"model": {"n_heads": 8, "n_kv_heads": 3}}, "not divisible"),
            # d_model 12 / 4 heads = head_dim 3, which RoPE cannot split in half.
            ({"model": {"d_model": 12, "n_heads": 4, "n_kv_heads": 2}}, "must be even for RoPE"),
            ({"data": {"seq_len": 4096}}, "exceeds model.max_seq_len"),
            ({"train": {"precision": "fp8"}}, "precision must be one of"),
            ({"optim": {"schedule": "linear"}}, "schedule must be one of"),
            ({"optim": {"warmup_steps": 99999}}, "warmup_steps"),
            ({"data": {"batch_size": 0}}, "must be >= 1"),
            ({"optim": {"min_lr_ratio": 1.5}}, "min_lr_ratio"),
        ],
    )
    def test_impossible_configs_rejected(self, patch, match):
        with pytest.raises(ValueError, match=match):
            from_dict(patch)


class TestFileLoading:
    def test_load_with_overrides(self, tmp_path):
        path = tmp_path / "c.yaml"
        OmegaConf.save(OmegaConf.create({"optim": {"lr": 1e-4}}), path)
        cfg = config_mod.load(path, overrides=["optim.lr=5e-4", "model.n_layers=3"])
        assert cfg.optim.lr == pytest.approx(5e-4)
        assert cfg.model.n_layers == 3

    def test_ships_configs_are_valid(self):
        """The configs in the repo must actually load — catches drift between
        the schema and the YAML that only shows up when you start a GPU run."""
        for name in ("m0", "m0_cpu"):
            cfg = load_config(name, [])
            validate(cfg)

    def test_m0_matches_its_documented_budget(self):
        """The M0 config should be the shape the roadmap's compute estimate assumes."""
        cfg = load_config("m0", [])
        tokens_per_step = cfg.data.batch_size * cfg.data.grad_accum_steps * cfg.data.seq_len
        assert tokens_per_step == 65_536
        total_tokens = tokens_per_step * cfg.train.max_steps
        assert 5e8 < total_tokens < 1e9
        assert cfg.train.precision == "fp16", "T4 is Turing: fp16 only"

    def test_cli_overrides_apply(self):
        cfg = load_config("m0_cpu", ["train.max_steps=70", "optim.lr=0.5"])
        assert cfg.train.max_steps == 70
        assert cfg.optim.lr == 0.5

    def test_cli_rejects_unknown_override(self):
        # Hydra rejects an unknown dotted override during composition, before
        # OmegaConf ever sees it, so this raises a HydraException rather than a
        # schema error. Either way it must fail loudly.
        with pytest.raises((ConfigError, HydraException)):
            load_config("m0_cpu", ["train.max_stpes=7"])
