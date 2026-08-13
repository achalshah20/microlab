"""Packed token shards on disk.

A shard is a flat ``uint16`` array of token ids with no per-document framing —
documents are concatenated with an EOS token between them and sequences are cut
on a fixed stride. uint16 caps the vocabulary at 65536, which every tokenizer in
this project stays under, and halves both disk footprint and page-cache pressure
against uint32.

The sidecar manifest records what produced the shard. Without it, a ``.bin``
file is an anonymous pile of integers and a run that used it is not reproducible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

TOKEN_DTYPE = np.uint16
MAX_VOCAB = np.iinfo(TOKEN_DTYPE).max + 1


@dataclass
class ShardManifest:
    """Provenance for one packed shard."""

    n_tokens: int
    dtype: str = "uint16"
    vocab_size: int = 0
    tokenizer_sha: str = ""
    source: str = ""
    source_split: str = ""
    n_documents: int = 0
    eos_id: int = 0
    created_utc: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> ShardManifest:
        return cls(**json.loads(path.read_text()))


def manifest_path(bin_path: str | Path) -> Path:
    return Path(str(bin_path).removesuffix(".bin") + ".manifest.json")


class PackedDataset:
    """Read-only view over a packed shard, cut into fixed-length samples.

    Sample ``i`` is ``tokens[i * seq_len : i * seq_len + seq_len + 1]``, split
    into ``(x, y)`` where ``y`` is ``x`` shifted by one. Windows are disjoint, so
    one epoch is exact coverage with no token seen twice and none skipped.

    The file is memory-mapped, not loaded: a 10B-token shard is 20 GB, which
    exceeds Kaggle's RAM but is fine to stream through the page cache.
    """

    def __init__(self, path: str | Path, seq_len: int) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"packed shard not found: {self.path}. "
                "Run `microlab-prepare-tinystories` (or your M1 data pipeline) first."
            )
        self.seq_len = seq_len
        self.tokens = np.memmap(self.path, dtype=TOKEN_DTYPE, mode="r")

        # The +1 is the target shift: the last sample still needs one token past
        # its own window. Dropping the remainder is deliberate — a partial final
        # sample would change length with the shard and break the epoch bijection.
        self.n_samples = max(0, (len(self.tokens) - 1) // seq_len)
        if self.n_samples == 0:
            raise ValueError(
                f"{self.path} holds {len(self.tokens)} tokens, too few for even one "
                f"sample of seq_len={seq_len}"
            )

        mpath = manifest_path(self.path)
        self.manifest = ShardManifest.load(mpath) if mpath.exists() else None

    def __len__(self) -> int:
        return self.n_samples

    @property
    def n_tokens(self) -> int:
        return int(len(self.tokens))

    def gather(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fetch samples by index. Returns ``(x, y)``, each ``[len(indices), seq_len]`` int64.

        Built with an explicit gather rather than fancy-indexing the memmap
        directly: the latter materializes an intermediate copy of the whole
        indexed span, which at batch 32 x seq 2048 is a measurable per-step cost.
        """
        idx = np.asarray(indices, dtype=np.int64)
        if np.any(idx < 0) or np.any(idx >= self.n_samples):
            raise IndexError(f"sample index out of range [0, {self.n_samples})")

        x = np.empty((len(idx), self.seq_len), dtype=np.int64)
        y = np.empty((len(idx), self.seq_len), dtype=np.int64)
        for row, i in enumerate(idx):
            start = int(i) * self.seq_len
            window = np.asarray(self.tokens[start : start + self.seq_len + 1], dtype=np.int64)
            x[row] = window[:-1]
            y[row] = window[1:]
        return x, y


def write_shard(
    path: str | Path,
    tokens: np.ndarray,
    manifest: ShardManifest,
) -> Path:
    """Write a packed shard plus its manifest, atomically.

    Atomic because a shard half-written by a preempted session is
    indistinguishable from a complete one on disk, and training on a truncated
    shard produces a loss curve that looks merely disappointing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tokens.max(initial=0) >= MAX_VOCAB:
        raise ValueError(f"token id {tokens.max()} exceeds uint16 range")

    tmp = path.with_suffix(path.suffix + ".tmp")
    tokens.astype(TOKEN_DTYPE, copy=False).tofile(tmp)
    tmp.replace(path)

    manifest.n_tokens = int(len(tokens))
    manifest.save(manifest_path(path))
    return path
