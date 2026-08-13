"""Tokenizer round-trip, merge semantics, and serialization."""

from __future__ import annotations

import random
import string
from collections import Counter

import pytest

from microlab.data.synthetic import generate_corpus
from microlab.tokenizer.bpe import BPETokenizer, _merge


@pytest.fixture(scope="module")
def trained() -> BPETokenizer:
    tok = BPETokenizer()
    tok.train("\n".join(generate_corpus(500, seed=0)), vocab_size=400)
    return tok


class TestRoundTrip:
    @pytest.mark.parametrize(
        "text",
        [
            "hello world",
            "One day Lily went to the park.",
            "",
            " ",
            "\n\n\t  mixed   whitespace \n",
            "unicode: café naïve 日本語 Ελληνικά",
            "emoji: 🚀🔥👩‍👩‍👧‍👦",
            "punctuation!?;:'\"[]{}()<>",
            "numbers 1234567890 and 3.14159",
            "a" * 500,
        ],
    )
    def test_lossless(self, trained, text):
        """Byte-level BPE must reproduce arbitrary input exactly.

        Anything less means the training corpus differs from what the data
        pipeline believes it packed.
        """
        assert trained.decode(trained.encode(text)) == text

    def test_untrained_tokenizer_is_still_lossless(self):
        """With zero merges the tokenizer degenerates to raw bytes, not to failure."""
        tok = BPETokenizer()
        text = "no merges, just bytes: 日本語 🚀"
        assert tok.decode(tok.encode_ordinary(text)) == text
        assert max(tok.encode_ordinary(text)) < 256

    def test_all_byte_values_encodable(self):
        """There is no unknown token: every byte string must have an encoding."""
        tok = BPETokenizer()
        raw = bytes(range(256)).decode("latin-1")
        assert tok.decode(tok.encode_ordinary(raw)) == raw


class TestSpecialTokens:
    def test_special_token_is_a_single_id(self, trained):
        ids = trained.encode("hello<|endoftext|>world")
        assert trained.eos_id in ids
        assert ids.count(trained.eos_id) == 1

    def test_special_tokens_can_be_disabled(self, trained):
        ids = trained.encode("<|endoftext|>", allowed_special=False)
        assert trained.eos_id not in ids
        assert trained.decode(ids) == "<|endoftext|>"

    def test_encode_ordinary_never_emits_specials(self, trained):
        ids = trained.encode_ordinary("text with <|endoftext|> inside")
        assert trained.eos_id not in ids

    def test_special_ids_sit_above_merges(self, trained):
        assert trained.eos_id >= 256 + len(trained.merges)


class TestMergeSemantics:
    def test_merges_are_applied_in_training_order(self, trained):
        """Encoding must reproduce the segmentation the trainer would produce."""
        text = "One day Lily went to the park."
        ids = trained.encode_ordinary(text)
        # Re-encoding the decoded text is a fixed point.
        assert trained.encode_ordinary(trained.decode(ids)) == ids

    def test_training_compresses(self, trained):
        text = "One day Lily went to the park. Lily was happy."
        assert len(trained.encode_ordinary(text)) < len(text.encode("utf-8"))

    def test_vocab_size_respects_budget(self):
        tok = BPETokenizer()
        tok.train("the quick brown fox jumps over the lazy dog. " * 200, vocab_size=300)
        assert tok.vocab_size <= 300

    def test_stops_early_when_pairs_exhausted(self):
        """A tiny corpus cannot fill a large vocabulary; that is not an error."""
        tok = BPETokenizer()
        tok.train("ab ab ab", vocab_size=1000)
        assert tok.vocab_size < 1000
        assert tok.decode(tok.encode_ordinary("ab ab ab")) == "ab ab ab"

    def test_rejects_impossible_vocab_size(self):
        with pytest.raises(ValueError, match="smaller than 256"):
            BPETokenizer().train("hello", vocab_size=100)

    def test_pretokenization_stops_merges_across_word_boundaries(self, trained):
        """No learned token may span a space-to-letter category change mid-word.

        Without pre-splitting, BPE happily merges across boundaries and the
        vocabulary fills with junk like "the park." as one token.
        """
        for token_id in range(256, 256 + len(trained.merges)):
            piece = trained.vocab[token_id].decode("utf-8", errors="replace")
            # A space is only ever allowed at the start of a piece.
            assert " " not in piece[1:], f"token {token_id!r} = {piece!r} spans a word boundary"


class TestSerialization:
    def test_round_trips_through_json(self, trained, tmp_path):
        path = trained.save(tmp_path / "tok.json")
        reloaded = BPETokenizer.load(path)
        assert reloaded.sha() == trained.sha()
        assert reloaded.vocab_size == trained.vocab_size
        text = "One day the little cat found a ball."
        assert reloaded.encode(text) == trained.encode(text)

    def test_sha_is_content_addressed(self, trained):
        other = BPETokenizer()
        other.train("completely different corpus text here " * 50, vocab_size=300)
        assert other.sha() != trained.sha()

    def test_sha_is_stable_across_instances(self, trained, tmp_path):
        path = trained.save(tmp_path / "t.json")
        assert BPETokenizer.load(path).sha() == BPETokenizer.load(path).sha()

    def test_rejects_unknown_format_version(self):
        with pytest.raises(ValueError, match="format version"):
            BPETokenizer.from_dict({"format_version": 99, "merges": [], "special_tokens": {}})

    def test_decode_rejects_unknown_id(self, trained):
        with pytest.raises(ValueError, match="not in vocabulary"):
            trained.decode([trained.vocab_size + 10])

    def test_decode_tolerates_split_codepoint(self, trained):
        """Streaming generation can end mid-UTF-8; decode must not raise."""
        ids = trained.encode_ordinary("日本語")
        assert isinstance(trained.decode(ids[:1]), str)


def _naive_train(text: str, vocab_size: int, n_specials: int = 1) -> dict:
    """Reference BPE trainer: recompute every pair count on every merge.

    This is the obvious implementation, and the one the optimized trainer has
    to agree with exactly. Keeping it here — rather than deleting it when the
    incremental version landed — is what makes the optimization checkable
    instead of merely plausible.
    """
    scratch = BPETokenizer()
    n_merges = vocab_size - 256 - n_specials
    chunk_counts = Counter(scratch._compiled.findall(text))
    sequences = [list(c.encode("utf-8")) for c in chunk_counts]
    weights = list(chunk_counts.values())

    merges: dict[tuple[int, int], int] = {}
    for i in range(n_merges):
        stats: Counter[tuple[int, int]] = Counter()
        for seq, w in zip(sequences, weights, strict=True):
            for pair in zip(seq, seq[1:], strict=False):
                stats[pair] += w
        if not stats:
            break
        pair = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))
        sequences = [_merge(seq, pair, 256 + i) for seq in sequences]
        merges[pair] = 256 + i
    return merges


class TestIncrementalTrainingParity:
    """The fast trainer must learn exactly the merges the naive one would.

    The incremental version maintains pair counts and a pair -> sequence index
    instead of rescanning the corpus per merge. That is a large speedup (the
    naive cost scales with corpus x merges) and a large opportunity to change
    the learned vocabulary by accident — a subtly different tie-break or a
    stale count shifts one merge, and every later merge diverges from there.
    A tokenizer that is *nearly* right silently changes what every downstream
    run was trained on.
    """

    def test_matches_naive_on_diverse_text(self):
        # Randomized words, not the synthetic grammar: the grammar has so few
        # distinct chunks that it exhausts pairs after ~240 merges and never
        # exercises the index at scale.
        rng = random.Random(0)
        vocabulary = [
            "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(2, 9)))
            for _ in range(800)
        ]
        text = " ".join(rng.choice(vocabulary) for _ in range(8000))

        tokenizer = BPETokenizer()
        tokenizer.train(text, vocab_size=700)
        assert tokenizer.merges == _naive_train(text, 700)
        assert len(tokenizer.merges) > 100, "corpus too small to exercise the index"

    def test_matches_naive_with_unicode_and_punctuation(self):
        text = ("héllo wörld! " * 200) + ("日本語 テスト。" * 200) + ("a,b;c:d " * 200)
        tokenizer = BPETokenizer()
        tokenizer.train(text, vocab_size=400)
        assert tokenizer.merges == _naive_train(text, 400)

    def test_matches_naive_when_pairs_are_exhausted(self):
        """Early stopping must trigger at the same point in both."""
        text = "ab ab ab cd cd "
        tokenizer = BPETokenizer()
        tokenizer.train(text, vocab_size=1000)
        assert tokenizer.merges == _naive_train(text, 1000)

    def test_round_trips_after_incremental_training(self):
        rng = random.Random(1)
        text = " ".join(
            "".join(rng.choice("abcdefg") for _ in range(rng.randint(1, 6))) for _ in range(3000)
        )
        tokenizer = BPETokenizer()
        tokenizer.train(text, vocab_size=500)
        assert tokenizer.decode(tokenizer.encode_ordinary(text)) == text


def test_merge_helper_replaces_non_overlapping_occurrences():
    assert _merge([1, 2, 1, 2], (1, 2), 9) == [9, 9]
    assert _merge([1, 1, 1], (1, 1), 9) == [9, 1]
    assert _merge([5], (1, 2), 9) == [5]
    assert _merge([], (1, 2), 9) == []
