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

import pytest

from microlab.train.checkpoint import list_checkpoints

REPO_ROOT = Path(__file__).resolve().parents[1]


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
