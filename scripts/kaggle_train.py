#!/usr/bin/env python
"""Kaggle session launcher.

Wraps data preparation and training with the bookkeeping a preemptible session
needs, so re-running the notebook cell after a preemption continues the run
instead of restarting it.

Usage inside a Kaggle notebook::

    !python scripts/kaggle_train.py --config m0 --data-dir /kaggle/working/data/tinystories

What it adds over calling the CLI directly:

* Reports the session's *remaining* budget against the 12-hour limit, so you can
  see whether the next checkpoint interval fits before it starts.
* Prepares shards only if they are missing, since preparation is the expensive
  part and it survives across sessions if written to persistent storage.
* Prints the resume point up front, which is the number to check when a session
  starts — a resume point of 0 on what should be session 12 means the
  checkpoints did not persist and the run is silently restarting.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SESSION_LIMIT_HOURS = 12.0


def _run(args: list[str]) -> int:
    print(f"$ {' '.join(args)}", flush=True)
    return subprocess.call(args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="m0")
    parser.add_argument("--data-dir", default="/kaggle/working/data/tinystories")
    parser.add_argument("--source", default="tinystories")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--runs-dir", default="/kaggle/working/runs")
    parser.add_argument("overrides", nargs="*", help="extra Hydra dotted overrides")
    args = parser.parse_args()

    started = time.time()
    data_dir = Path(args.data_dir)
    train_bin = data_dir / "train.bin"

    if not train_bin.exists():
        print(f"no shards at {train_bin}; preparing (this is the slow part)", flush=True)
        code = _run(
            [
                sys.executable, "-m", "microlab.cli", "prepare",
                "--source", args.source,
                "--out-dir", str(data_dir),
                "--vocab-size", str(args.vocab_size),
            ]
        )
        if code != 0:
            return code
    else:
        print(f"reusing existing shards at {train_bin}", flush=True)

    # Report where this session is starting from before any training happens.
    from microlab.train.checkpoint import list_checkpoints  # noqa: PLC0415 - needs the install

    run_id = next(
        (o.split("=", 1)[1] for o in args.overrides if o.startswith("train.run_id=")),
        args.config,
    )
    existing = list_checkpoints(Path(args.runs_dir) / run_id / "checkpoints")
    if existing:
        print(f"resuming from step {existing[0][0]} ({len(existing)} checkpoints on disk)")
    else:
        print("no checkpoint found — this session starts from step 0")

    code = _run(
        [
            sys.executable, "-m", "microlab.cli", "train", args.config,
            f"data.train_bin={data_dir / 'train.bin'}",
            f"data.val_bin={data_dir / 'val.bin'}",
            f"data.tokenizer_path={data_dir / 'tokenizer.json'}",
            f"train.out_dir={args.runs_dir}",
        ]
        + args.overrides
    )

    elapsed = (time.time() - started) / 3600
    print(f"\nsession used {elapsed:.2f}h of the {SESSION_LIMIT_HOURS}h limit")
    if code == 0:
        print("exit 0 — run either completed or checkpointed cleanly")
    else:
        print(f"exit {code} — check the run record before relaunching")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
