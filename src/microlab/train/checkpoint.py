"""Checkpoint save/load built for a machine that will be killed without warning.

Three properties matter more here than in a normal trainer:

1. **Completeness is observable.** The payload is written and fsynced first, and
   a small ``.meta.json`` commit marker is written last. A checkpoint without
   its marker was interrupted mid-write and is ignored. Without this, a session
   killed during ``torch.save`` leaves a file that looks like the newest, best
   checkpoint and fails to load — three sessions later, when nobody remembers
   what changed.
2. **Loading falls back.** ``load_latest`` walks checkpoints newest-first and
   returns the first that loads cleanly, so one bad write costs one checkpoint
   interval, not the run. Over ~50 sessions (M4) this will be exercised.
3. **State is complete.** Model, optimizer, precision (loss scale), RNG streams,
   step, and token counts all round-trip. Anything omitted is a silent
   divergence between a resumed run and an uninterrupted one.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_VERSION = 1


@dataclass
class CheckpointMeta:
    """The commit marker. Its presence means the payload is complete."""

    step: int
    version: int
    sha256: str
    bytes: int
    wall_clock: float
    tokens_seen: int
    val_loss: float | None
    run_id: str
    session: int
    torch_version: str

    def save(self, path: Path) -> None:
        _atomic_write_bytes(path, json.dumps(self.__dict__, indent=2).encode())

    @classmethod
    def load(cls, path: Path) -> CheckpointMeta:
        return cls(**json.loads(path.read_text()))


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    _fsync_dir(path.parent)


def _fsync_dir(directory: Path) -> None:
    """Durably persist a rename.

    ``os.replace`` is atomic but the *directory entry* is not durable until the
    directory itself is synced. Skipping this is the difference between "the
    checkpoint survives a kill -9" and "the checkpoint survives a kill -9 unless
    the box also loses power", and free-tier instances do get yanked.
    """
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - not supported on some filesystems
        pass
    finally:
        os.close(fd)


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_paths(directory: Path, step: int) -> tuple[Path, Path]:
    return directory / f"step_{step:08d}.pt", directory / f"step_{step:08d}.meta.json"


def save_checkpoint(
    directory: str | Path,
    step: int,
    payload: dict[str, Any],
    run_id: str,
    session: int,
    tokens_seen: int,
    val_loss: float | None = None,
    keep_last_n: int = 3,
) -> Path:
    """Write a checkpoint and its commit marker. Returns the payload path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ckpt_path, meta_path = checkpoint_paths(directory, step)

    payload = dict(payload)
    payload["version"] = CHECKPOINT_VERSION
    payload["step"] = step

    tmp = ckpt_path.with_suffix(".pt.tmp")
    with tmp.open("wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(ckpt_path)
    _fsync_dir(directory)

    CheckpointMeta(
        step=step,
        version=CHECKPOINT_VERSION,
        sha256=_sha256_file(ckpt_path),
        bytes=ckpt_path.stat().st_size,
        wall_clock=time.time(),
        tokens_seen=tokens_seen,
        val_loss=val_loss,
        run_id=run_id,
        session=session,
        torch_version=torch.__version__,
    ).save(meta_path)

    prune_checkpoints(directory, keep_last_n)
    return ckpt_path


def list_checkpoints(directory: str | Path) -> list[tuple[int, Path, Path]]:
    """Complete checkpoints as ``(step, payload, meta)``, newest first.

    Payloads without a marker are skipped: they are interrupted writes.
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    found = []
    for meta_path in directory.glob("step_*.meta.json"):
        ckpt_path = meta_path.parent / (meta_path.name.removesuffix(".meta.json") + ".pt")
        if not ckpt_path.exists():
            continue
        try:
            step = int(meta_path.name.removeprefix("step_").removesuffix(".meta.json"))
        except ValueError:
            continue
        found.append((step, ckpt_path, meta_path))
    return sorted(found, key=lambda t: t[0], reverse=True)


def load_checkpoint(path: str | Path, map_location: str = "cpu", verify_hash: bool = False) -> dict:
    """Load one checkpoint payload, optionally verifying its content hash."""
    path = Path(path)
    meta_path = path.parent / (path.stem + ".meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"checkpoint {path} has no commit marker; treating as incomplete")

    meta = CheckpointMeta.load(meta_path)
    if verify_hash:
        actual = _sha256_file(path)
        if actual != meta.sha256:
            raise ValueError(f"checkpoint {path} failed hash check: {actual} != {meta.sha256}")

    # weights_only=False is required and safe here: the payload deliberately
    # contains non-tensor state (RNG tuples, the config dict) and these files are
    # produced by this trainer, not downloaded.
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"checkpoint version {payload.get('version')} != {CHECKPOINT_VERSION}")
    return payload


def load_latest(
    directory: str | Path,
    map_location: str = "cpu",
    verify_hash: bool = False,
) -> tuple[dict, Path] | None:
    """Newest checkpoint that actually loads, or None if there are none.

    Corrupt or truncated checkpoints are skipped rather than raised, because the
    correct response to a bad checkpoint on a preemptible box is to lose one
    interval of progress, not the run.
    """
    errors: list[str] = []
    for _step, ckpt_path, _meta in list_checkpoints(directory):
        try:
            return load_checkpoint(ckpt_path, map_location, verify_hash), ckpt_path
        except Exception as exc:  # noqa: BLE001 - any failure means "try the previous one"
            errors.append(f"{ckpt_path.name}: {type(exc).__name__}: {exc}")
            continue
    if errors:
        raise RuntimeError(
            "every checkpoint in "
            f"{directory} failed to load:\n  " + "\n  ".join(errors)
        )
    return None


def prune_checkpoints(directory: str | Path, keep_last_n: int) -> list[Path]:
    """Delete all but the newest ``keep_last_n`` checkpoints. Returns what was removed."""
    if keep_last_n <= 0:
        return []
    removed = []
    for _step, ckpt_path, meta_path in list_checkpoints(directory)[keep_last_n:]:
        # Marker first: if we are interrupted between the two unlinks, the
        # checkpoint reads as incomplete rather than as a valid file with a
        # missing payload.
        meta_path.unlink(missing_ok=True)
        ckpt_path.unlink(missing_ok=True)
        removed.extend([meta_path, ckpt_path])
    return removed
