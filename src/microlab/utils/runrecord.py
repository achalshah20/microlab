"""The run record: one directory per run, append-only, survives preemption.

Layout::

    runs/<run_id>/
        config.yaml       resolved config, written once
        env.json          torch/CUDA/GPU/git info, one entry per session
        metrics.jsonl     append-only, one JSON object per logged step
        sessions.jsonl    append-only, one entry per session start/end
        samples/          generated text, named by step
        checkpoints/      see train.checkpoint

The append-only design is the point. A run that spans ~50 preempted sessions
(M4) must produce *one* continuous history, not 50 fragments to stitch together
afterwards. Every writer opens in append mode and flushes, so a session killed
mid-write loses at most the final line rather than the file.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from ..config import Config, to_dict
from .determinism import environment_info


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


class RunRecord:
    """Append-only artifacts for a single run across all of its sessions."""

    def __init__(self, out_dir: str | Path, run_id: str) -> None:
        self.dir = Path(out_dir) / run_id
        self.run_id = run_id
        self.checkpoints_dir = self.dir / "checkpoints"
        self.samples_dir = self.dir / "samples"
        for d in (self.dir, self.checkpoints_dir, self.samples_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.session_index = self._next_session_index()
        self._session_start = time.time()

    # ---- session bookkeeping -------------------------------------------

    def _next_session_index(self) -> int:
        """Count ``begin`` events only.

        sessions.jsonl also holds ``end`` and ``config_changed`` records, so
        counting lines would inflate the session number — and the session index
        is what M4's write-up reports as "trained across N preempted sessions".
        """
        path = self.dir / "sessions.jsonl"
        if not path.exists():
            return 0
        count = 0
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    if json.loads(line).get("event") == "begin":
                        count += 1
                except json.JSONDecodeError:
                    # Truncated final line from a killed session; not a begin.
                    continue
        return count

    def begin_session(self, resumed_from_step: int | None) -> None:
        """Record the start of a session. Called once per process."""
        self._session_start = time.time()
        self._append(
            "sessions.jsonl",
            {
                "session": self.session_index,
                "event": "begin",
                "wall_clock": time.time(),
                "resumed_from_step": resumed_from_step,
                "pid": os.getpid(),
                "env": environment_info(),
            },
        )

    def end_session(self, step: int, reason: str) -> None:
        """Record a clean end. A preempted session simply never writes this,
        which is itself the signal that the session was killed."""
        self._append(
            "sessions.jsonl",
            {
                "session": self.session_index,
                "event": "end",
                "wall_clock": time.time(),
                "duration_s": time.time() - self._session_start,
                "step": step,
                "reason": reason,
            },
        )

    # ---- artifacts ------------------------------------------------------

    def save_config(self, cfg: Config) -> None:
        """Write the resolved config once; on resume, verify it has not changed.

        A silently edited config across a preemption boundary is the kind of bug
        that invalidates a 50-session run after the fact. We do not block the
        resume (sometimes you legitimately raise max_steps), but the change is
        recorded so the run's own history explains the discontinuity.
        """
        path = self.dir / "config.yaml"
        resolved = to_dict(cfg)
        if path.exists():
            previous = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
            if previous != resolved:
                self._append(
                    "sessions.jsonl",
                    {
                        "session": self.session_index,
                        "event": "config_changed",
                        "wall_clock": time.time(),
                        "diff": _shallow_diff(previous, resolved),
                    },
                )
        OmegaConf.save(OmegaConf.create(resolved), path)

    def log_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        payload = {"step": step, "session": self.session_index, "wall_clock": time.time()}
        payload.update(metrics)
        self._append("metrics.jsonl", payload)

    def save_sample(self, step: int, prompt: str, text: str) -> Path:
        path = self.samples_dir / f"step_{step:08d}.txt"
        path.write_text(f"=== prompt ===\n{prompt}\n=== sample ===\n{text}\n")
        return path

    def read_metrics(self) -> list[dict[str, Any]]:
        """Every metric line ever written for this run, across all sessions."""
        path = self.dir / "metrics.jsonl"
        if not path.exists():
            return []
        rows = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # A session killed mid-write can leave one truncated line.
                    # Skipping it is correct; failing to read the whole history
                    # because of it is not.
                    continue
        return rows

    def _append(self, filename: str, payload: dict[str, Any]) -> None:
        with (self.dir / filename).open("a") as f:
            f.write(json.dumps(payload, default=_json_default) + "\n")
            f.flush()
            os.fsync(f.fileno())


def _shallow_diff(a: dict, b: dict, prefix: str = "") -> dict[str, Any]:
    """Flat ``{dotted.key: [old, new]}`` diff of two nested dicts."""
    out: dict[str, Any] = {}
    for key in set(a) | set(b):
        path = f"{prefix}{key}"
        av, bv = a.get(key), b.get(key)
        if isinstance(av, dict) and isinstance(bv, dict):
            out.update(_shallow_diff(av, bv, prefix=f"{path}."))
        elif av != bv:
            out[path] = [av, bv]
    return out
