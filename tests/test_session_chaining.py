"""End-to-end preemption test: kill a real process, verify the chain recovers.

The M0 gate says, verbatim: "Kill the session mid-run and verify the chain
recovers." Everything else in the suite simulates preemption in-process, which
cannot catch the failure modes that only exist across a real process boundary —
a checkpoint still buffered in the page cache, a signal handler that never runs
because the loop is blocked, a run directory written relative to a working
directory that changed.

So this test launches the actual CLI as a subprocess, sends it SIGTERM, and then
starts a second process to finish the job. It is the smallest honest version of
what M4 does ~50 times.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from microlab.config import to_dict
from microlab.data.packed import ShardManifest, write_shard
from microlab.data.synthetic import generate_corpus
from microlab.tokenizer.bpe import BPETokenizer
from microlab.train.checkpoint import list_checkpoints

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cli_workspace(tmp_path, train_cfg):
    """A config directory and shards laid out the way the CLI expects."""
    docs = generate_corpus(2000, seed=0)
    tokenizer = BPETokenizer()
    tokenizer.train("\n".join(docs), vocab_size=400)
    tok_path = tokenizer.save(tmp_path / "tokenizer.json")

    for split, subset in (("train", docs[:1900]), ("val", docs[1900:])):
        ids: list[int] = []
        for doc in subset:
            ids.extend(tokenizer.encode_ordinary(doc))
            ids.append(tokenizer.eos_id)
        write_shard(
            tmp_path / f"{split}.bin",
            np.asarray(ids, dtype=np.uint16),
            ShardManifest(
                n_tokens=len(ids),
                vocab_size=tokenizer.vocab_size,
                tokenizer_sha=tokenizer.sha(),
                source="chaining-test",
                source_split=split,
                eos_id=tokenizer.eos_id,
            ),
        )

    raw = to_dict(train_cfg)
    raw["model"]["vocab_size"] = tokenizer.vocab_size
    raw["data"].update(
        {
            "train_bin": str(tmp_path / "train.bin"),
            "val_bin": str(tmp_path / "val.bin"),
            "tokenizer_path": str(tok_path),
            "batch_size": 8,
            "grad_accum_steps": 1,
        }
    )
    raw["train"].update(
        {
            "max_steps": 400,
            "run_id": "chain_e2e",
            "out_dir": str(tmp_path / "runs"),
            "log_interval": 1,
            "eval_interval": 0,
            "sample_interval": 0,
            # Checkpoint often: the test needs at least one to exist before the
            # kill arrives.
            "checkpoint_every_steps": 5,
            "checkpoint_every_minutes": 1e9,
            "keep_last_n": 3,
        }
    )

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    OmegaConf.save(OmegaConf.create(raw), config_dir / "chain.yaml")
    return {"config_dir": config_dir, "run_dir": tmp_path / "runs" / "chain_e2e"}


def _launch(config_dir: Path, overrides: list[str] | None = None) -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return subprocess.Popen(
        [sys.executable, "-m", "microlab.cli", "train", "chain", "--config-dir", str(config_dir)]
        + (overrides or []),
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _wait_for_checkpoint(run_dir: Path, timeout: float = 60.0) -> int:
    """Block until a complete checkpoint exists; return its step."""
    deadline = time.time() + timeout
    ckpt_dir = run_dir / "checkpoints"
    while time.time() < deadline:
        found = list_checkpoints(ckpt_dir)
        if found:
            return found[0][0]
        time.sleep(0.2)
    raise AssertionError(f"no checkpoint appeared under {ckpt_dir} within {timeout}s")


@pytest.mark.slow
def test_killed_session_resumes_and_completes(cli_workspace):
    run_dir = cli_workspace["run_dir"]

    # --- session 1: start training, then kill it like a preemption would ---
    proc = _launch(cli_workspace["config_dir"])
    try:
        step_at_kill = _wait_for_checkpoint(run_dir)
        proc.send_signal(signal.SIGTERM)
        output = proc.communicate(timeout=120)[0]
    finally:
        if proc.poll() is None:  # pragma: no cover - only on a hung test
            proc.kill()
            proc.communicate(timeout=30)

    assert proc.returncode == 0, f"killed session exited badly:\n{output}"
    assert "signal_SIGTERM" in output

    checkpoints = list_checkpoints(run_dir / "checkpoints")
    assert checkpoints, "session died without leaving a usable checkpoint"
    resume_step = checkpoints[0][0]
    assert resume_step >= step_at_kill

    # --- session 2: a fresh process must continue, not restart ---
    proc2 = _launch(cli_workspace["config_dir"], ["train.max_steps=40"])
    output2 = proc2.communicate(timeout=300)[0]
    assert proc2.returncode == 0, f"resumed session failed:\n{output2}"

    sessions = [
        json.loads(line)
        for line in (run_dir / "sessions.jsonl").read_text().strip().split("\n")
        if line.strip()
    ]
    begins = [s for s in sessions if s["event"] == "begin"]
    assert len(begins) == 2, "the second process did not register as a new session"
    assert begins[0]["resumed_from_step"] is None
    assert begins[1]["resumed_from_step"] == resume_step, "second session restarted from scratch"

    # The metric history must be one continuous record, not two fragments.
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().strip().split("\n")
        if line.strip()
    ]
    steps = sorted({r["step"] for r in rows if "loss" in r})
    assert steps[0] == 1
    assert steps[-1] == 40
    assert steps == list(range(1, 41)), f"gap in the metric history: {steps}"

    # Both sessions wrote into the same record, which is what makes
    # "trained across N preempted sessions" a checkable claim.
    assert {r["session"] for r in rows} == {0, 1}


@pytest.mark.slow
def test_second_process_does_not_duplicate_completed_work(cli_workspace):
    """Re-running a finished run must be a no-op, not a second training run."""
    proc = _launch(cli_workspace["config_dir"], ["train.max_steps=10"])
    assert proc.communicate(timeout=300)[0] is not None
    assert proc.returncode == 0

    proc2 = _launch(cli_workspace["config_dir"], ["train.max_steps=10"])
    output = proc2.communicate(timeout=120)[0]
    assert proc2.returncode == 0
    assert "already_complete" in output

    run_dir = cli_workspace["run_dir"]
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().strip().split("\n")
        if line.strip()
    ]
    train_steps = [r["step"] for r in rows if "loss" in r]
    assert len(train_steps) == len(set(train_steps)), "steps were trained twice"
