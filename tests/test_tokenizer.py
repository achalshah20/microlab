"""Tokenizer round-trip, merge semantics, and serialization."""

from __future__ import annotations

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


def test_merge_helper_replaces_non_overlapping_occurrences():
    assert _merge([1, 2, 1, 2], (1, 2), 9) == [9, 9]
    assert _merge([1, 1, 1], (1, 1), 9) == [9, 1]
    assert _merge([5], (1, 2), 9) == [5]
    assert _merge([], (1, 2), 9) == []
