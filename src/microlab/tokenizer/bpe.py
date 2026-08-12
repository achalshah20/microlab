"""Byte-level BPE tokenizer — **placeholder implementation**.

This is deliberately a placeholder. The roadmap puts the real tokenizer in M1: a
Rust BPE trainer with Python bindings, parity-tested against tiktoken. That one
will train on tens of GB; this one trains on a few hundred MB in pure Python and
is roughly two orders of magnitude too slow to do more.

What it is *not* is sloppy: it implements the same algorithm with the same
byte-level, regex-pre-split, merge-ranked semantics, round-trips arbitrary bytes
losslessly, and serializes to a versioned JSON file with a content hash that gets
recorded in every shard manifest and run record. When M1 replaces the internals,
the interface and the on-disk format stay put.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import regex as re

# The GPT-4 pre-tokenization pattern. Pre-splitting before BPE is what stops
# merges from spanning word and category boundaries — without it, " the" and
# " the." become unrelated tokens and the vocabulary fills with punctuation
# variants of common words.
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

DEFAULT_SPECIAL_TOKENS = ("<|endoftext|>",)
FORMAT_VERSION = 1


class BPETokenizer:
    """Byte-level BPE with a fixed merge ranking.

    Token id layout: ``[0, 256)`` are raw bytes, ``[256, 256 + n_merges)`` are
    learned merges in training order, and special tokens occupy the top of the
    range. Keeping bytes at the bottom means every byte string is encodable —
    there is no unknown token and no lossy path.
    """

    def __init__(
        self,
        merges: dict[tuple[int, int], int] | None = None,
        special_tokens: dict[str, int] | None = None,
        pattern: str = SPLIT_PATTERN,
    ) -> None:
        self.pattern = pattern
        self._compiled = re.compile(pattern)
        self.merges: dict[tuple[int, int], int] = merges or {}
        self.special_tokens: dict[str, int] = special_tokens or {}
        self._rebuild_vocab()

    def _rebuild_vocab(self) -> None:
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in sorted(self.merges.items(), key=lambda kv: kv[1]):
            vocab[idx] = vocab[a] + vocab[b]
        for text, idx in self.special_tokens.items():
            vocab[idx] = text.encode("utf-8")
        self.vocab = vocab
        self._special_inverse = {v: k for k, v in self.special_tokens.items()}
        self._special_pattern = (
            re.compile("(" + "|".join(re.escape(t) for t in self.special_tokens) + ")")
            if self.special_tokens
            else None
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def eos_id(self) -> int:
        eos = self.special_tokens.get("<|endoftext|>")
        if eos is None:
            raise KeyError("tokenizer has no <|endoftext|> token")
        return eos

    # ---- training -------------------------------------------------------

    def train(
        self,
        text: str,
        vocab_size: int,
        special_tokens: Iterable[str] = DEFAULT_SPECIAL_TOKENS,
        verbose: bool = False,
    ) -> None:
        """Learn merges until the vocabulary reaches ``vocab_size``."""
        specials = list(special_tokens)
        n_merges = vocab_size - 256 - len(specials)
        if n_merges < 0:
            raise ValueError(
                f"vocab_size={vocab_size} is smaller than 256 bytes + {len(specials)} specials"
            )

        # Pre-split once, then count identical chunks rather than re-walking the
        # corpus: on natural text the chunk multiset is ~20x smaller than the
        # token stream, which is the difference between minutes and hours here.
        chunk_counts = Counter(self._compiled.findall(text))
        sequences: list[list[int]] = [list(c.encode("utf-8")) for c in chunk_counts]
        weights: list[int] = list(chunk_counts.values())

        self.merges = {}
        for i in range(n_merges):
            stats: Counter[tuple[int, int]] = Counter()
            for seq, w in zip(sequences, weights, strict=True):
                for pair in zip(seq, seq[1:], strict=False):
                    stats[pair] += w
            if not stats:
                if verbose:
                    print(f"no pairs left after {i} merges; stopping early")
                break

            pair = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))
            new_id = 256 + i
            sequences = [_merge(seq, pair, new_id) for seq in sequences]
            self.merges[pair] = new_id
            if verbose and (i + 1) % 500 == 0:
                print(f"merge {i + 1}/{n_merges}: {pair} -> {new_id} (count {stats[pair]})")

        base = 256 + len(self.merges)
        self.special_tokens = {tok: base + i for i, tok in enumerate(specials)}
        self._rebuild_vocab()

    # ---- encode / decode -------------------------------------------------

    def _encode_chunk(self, data: bytes) -> list[int]:
        ids = list(data)
        while len(ids) >= 2:
            # Apply the lowest-ranked available merge, which reproduces the
            # order merges were learned in. Applying merges greedily by
            # frequency instead would give a different (and non-round-trippable
            # against the trainer) segmentation.
            pairs = set(zip(ids, ids[1:], strict=False))
            candidate = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if candidate not in self.merges:
                break
            ids = _merge(ids, candidate, self.merges[candidate])
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode, treating special-token text as ordinary text."""
        out: list[int] = []
        for chunk in self._compiled.findall(text):
            out.extend(self._encode_chunk(chunk.encode("utf-8")))
        return out

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        """Encode text, honoring special tokens when ``allowed_special``.

        Special tokens are matched *before* pre-splitting so that
        ``<|endoftext|>`` becomes one id rather than a dozen punctuation tokens.
        """
        if not allowed_special or not self._special_pattern:
            return self.encode_ordinary(text)

        out: list[int] = []
        for part in self._special_pattern.split(text):
            if not part:
                continue
            if part in self.special_tokens:
                out.append(self.special_tokens[part])
            else:
                out.extend(self.encode_ordinary(part))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        """Decode ids back to text.

        ``errors="replace"`` because a *partial* token sequence — the normal case
        during streaming generation — can end mid-UTF-8-codepoint. Raising there
        would make the sampler crash on perfectly valid model output.
        """
        parts = []
        for i in ids:
            i = int(i)
            if i not in self.vocab:
                raise ValueError(f"token id {i} not in vocabulary of size {self.vocab_size}")
            parts.append(self.vocab[i])
        return b"".join(parts).decode("utf-8", errors="replace")

    # ---- serialization ---------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "format_version": FORMAT_VERSION,
            "pattern": self.pattern,
            # JSON has no tuple keys; merges are stored as [a, b, new_id] triples
            # sorted by rank so the file is stable and diffable.
            "merges": [
                [a, b, idx] for (a, b), idx in sorted(self.merges.items(), key=lambda kv: kv[1])
            ],
            "special_tokens": self.special_tokens,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return path

    @classmethod
    def from_dict(cls, data: dict) -> BPETokenizer:
        version = data.get("format_version")
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported tokenizer format version {version!r}")
        merges = {(a, b): idx for a, b, idx in data["merges"]}
        return cls(
            merges=merges,
            special_tokens={k: int(v) for k, v in data["special_tokens"].items()},
            pattern=data.get("pattern", SPLIT_PATTERN),
        )

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def sha(self) -> str:
        """Content hash, recorded in shard manifests and run records.

        Training against tokenizer A and evaluating with tokenizer B produces
        plausible garbage; comparing hashes catches it immediately.
        """
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` in ``ids``."""
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
