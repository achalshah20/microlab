"""Shared fixtures.

Everything here runs on CPU in fp32. That is deliberate: CI has no GPU, and a
correctness suite that only passes on the training hardware is a correctness
suite you run once a week instead of on every commit.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from microlab.config import Config, DataConfig, ModelConfig, OptimConfig, TrainConfig
from microlab.data.packed import ShardManifest, write_shard
from microlab.tokenizer.bpe import BPETokenizer


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)
    np.random.seed(0)
    yield


@pytest.fixture
def tiny_model_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=128,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        ffn_multiple_of=16,
        tie_embeddings=False,
        dropout=0.0,
    )


@pytest.fixture
def tiny_shards(tmp_path):
    """A tokenizer plus train/val shards, enough for a real training step."""
    from microlab.data.synthetic import generate_corpus

    docs = generate_corpus(400, seed=0)
    tokenizer = BPETokenizer()
    tokenizer.train("\n".join(docs), vocab_size=300)
    tok_path = tokenizer.save(tmp_path / "tokenizer.json")

    paths = {}
    for split, subset in (("train", docs[:360]), ("val", docs[360:])):
        ids = []
        for doc in subset:
            ids.extend(tokenizer.encode_ordinary(doc))
            ids.append(tokenizer.eos_id)
        path = tmp_path / f"{split}.bin"
        write_shard(
            path,
            np.asarray(ids, dtype=np.uint16),
            ShardManifest(
                n_tokens=len(ids),
                vocab_size=tokenizer.vocab_size,
                tokenizer_sha=tokenizer.sha(),
                source="test-fixture",
                source_split=split,
                eos_id=tokenizer.eos_id,
            ),
        )
        paths[split] = path
    return {"tokenizer": tokenizer, "tokenizer_path": tok_path, **paths}


@pytest.fixture
def train_cfg(tmp_path, tiny_shards) -> Config:
    """A complete, runnable config wired to the fixture shards."""
    vocab = tiny_shards["tokenizer"].vocab_size
    return Config(
        model=ModelConfig(
            vocab_size=vocab,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=64,
            ffn_multiple_of=16,
            tie_embeddings=True,
            dropout=0.0,
        ),
        data=DataConfig(
            train_bin=str(tiny_shards["train"]),
            val_bin=str(tiny_shards["val"]),
            tokenizer_path=str(tiny_shards["tokenizer_path"]),
            batch_size=4,
            grad_accum_steps=1,
            seq_len=32,
        ),
        optim=OptimConfig(lr=1e-3, warmup_steps=2, schedule="cosine"),
        train=TrainConfig(
            max_steps=10,
            precision="fp32",
            device="cpu",
            log_interval=1,
            eval_interval=5,
            eval_batches=2,
            sample_interval=0,
            checkpoint_every_minutes=1e9,  # only the explicit final save fires
            checkpoint_every_steps=5,
            out_dir=str(tmp_path / "runs"),
            run_id="test",
        ),
    )
