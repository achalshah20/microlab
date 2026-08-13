"""Corpus -> tokenizer -> packed shards.

Three sources, one code path:

* ``tinystories`` — the real M0 corpus, via ``datasets``. Needs network, so in
  practice this runs on Kaggle.
* ``files`` — a directory of ``.txt``, for any local corpus.
* ``synthetic`` — the offline grammar corpus, for CI and for a first run before
  you have a GPU session.

The tokenizer is trained on a bounded prefix of the corpus rather than all of
it. Beyond a few tens of MB, BPE merge frequencies are stable enough that more
data changes the vocabulary hardly at all, and the pure-Python trainer is
quadratic enough in practice that the difference is minutes versus hours. The
cap is recorded in the manifest so the decision is visible rather than implicit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..tokenizer.bpe import DEFAULT_SPECIAL_TOKENS, BPETokenizer
from .packed import ShardManifest, write_shard
from .synthetic import generate_corpus


@dataclass
class PrepareResult:
    train_tokens: int
    val_tokens: int
    vocab_size: int
    tokenizer_sha: str
    out_dir: Path
    compression_ratio: float


def load_documents(source: str, limit: int | None = None, seed: int = 0) -> tuple[list[str], str]:
    """Return ``(documents, source_description)``."""
    if source == "synthetic":
        n = limit or 20_000
        return generate_corpus(n, seed=seed), f"synthetic-grammar(n={n},seed={seed})"

    if source == "tinystories":
        try:
            from datasets import load_dataset  # noqa: PLC0415 - optional, network-only path
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "source=tinystories needs `pip install datasets`. On an offline box use "
                "source=synthetic (CI does) or source=files with a local corpus."
            ) from exc
        ds = load_dataset("roneneldan/TinyStories", split="train")
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        return [d["text"] for d in ds], f"roneneldan/TinyStories(n={len(ds)})"

    path = Path(source)
    if path.exists():
        files = sorted(path.glob("**/*.txt")) if path.is_dir() else [path]
        if not files:
            raise FileNotFoundError(f"no .txt files under {path}")
        docs = [f.read_text(encoding="utf-8", errors="replace") for f in files[: limit or None]]
        return docs, f"files({path}, n={len(docs)})"

    raise ValueError(f"unknown source {source!r}: expected 'tinystories', 'synthetic', or a path")


def prepare(
    source: str = "synthetic",
    out_dir: str | Path = "data/synthetic",
    vocab_size: int = 4096,
    val_fraction: float = 0.02,
    limit: int | None = None,
    tokenizer_train_chars: int = 20_000_000,
    seed: int = 0,
    verbose: bool = True,
) -> PrepareResult:
    """Build a tokenizer and packed train/val shards from ``source``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    docs, source_desc = load_documents(source, limit=limit, seed=seed)
    if verbose:
        print(f"loaded {len(docs)} documents from {source_desc} in {time.time() - t0:.1f}s")

    # The split is by document, before tokenization. Splitting the token stream
    # instead would cut a document in half and put its head in train and its
    # tail in val — a small, real leak that flatters val loss.
    n_val = max(1, int(len(docs) * val_fraction))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(docs))
    val_docs = [docs[i] for i in order[:n_val]]
    train_docs = [docs[i] for i in order[n_val:]]

    tok_path = out_dir / "tokenizer.json"
    t0 = time.time()
    tokenizer = BPETokenizer()
    train_text = "\n".join(train_docs)[:tokenizer_train_chars]
    tokenizer.train(train_text, vocab_size=vocab_size, special_tokens=DEFAULT_SPECIAL_TOKENS)
    tokenizer.save(tok_path)
    if verbose:
        print(
            f"trained tokenizer: vocab={tokenizer.vocab_size} "
            f"sha={tokenizer.sha()} on {len(train_text):,} chars in {time.time() - t0:.1f}s"
        )

    eos = tokenizer.eos_id
    results = {}
    for split, split_docs in (("train", train_docs), ("val", val_docs)):
        t0 = time.time()
        ids: list[int] = []
        for doc in split_docs:
            ids.extend(tokenizer.encode_ordinary(doc))
            # EOS between documents so the model learns where a story ends —
            # without it, training teaches the model to run one story into the
            # next and generation never terminates.
            ids.append(eos)
        tokens = np.asarray(ids, dtype=np.uint16)
        bin_path = out_dir / f"{split}.bin"
        write_shard(
            bin_path,
            tokens,
            ShardManifest(
                n_tokens=len(tokens),
                vocab_size=tokenizer.vocab_size,
                tokenizer_sha=tokenizer.sha(),
                source=source_desc,
                source_split=split,
                n_documents=len(split_docs),
                eos_id=eos,
                created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                extra={
                    "tokenizer_train_chars": min(len(train_text), tokenizer_train_chars),
                    "val_fraction": val_fraction,
                    "split_seed": seed,
                },
            ),
        )
        results[split] = len(tokens)
        if verbose:
            print(
                f"{split}: {len(split_docs):,} docs -> {len(tokens):,} tokens "
                f"({time.time() - t0:.1f}s) -> {bin_path}"
            )

    total_chars = sum(len(d) for d in docs)
    total_tokens = results["train"] + results["val"]
    ratio = total_chars / max(total_tokens, 1)
    if verbose:
        print(f"compression: {ratio:.2f} chars/token")

    return PrepareResult(
        train_tokens=results["train"],
        val_tokens=results["val"],
        vocab_size=tokenizer.vocab_size,
        tokenizer_sha=tokenizer.sha(),
        out_dir=out_dir,
        compression_ratio=ratio,
    )
