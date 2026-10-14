"""Unit tests for sentence spans aligned to generated tokens."""

from __future__ import annotations

import pytest

from halo.spans import Span, answer_spans, split_sentences, token_char_offsets

# Tokenization of "The cat. It ran." with a skipped end token in the
# middle of the id list, to prove specials never join a span.
PIECES = {10: "The", 11: " cat", 12: ".", 13: " It", 14: " ran", 15: "."}
IDS = [10, 11, 12, 99, 13, 14, 15]
ANSWER = "The cat. It ran."


class _StubTokenizer:
    """Joins fixed pieces; ids in ``special`` vanish on decode."""

    def __init__(self, pieces: dict, special: tuple[int, ...] = (99,)):
        self._pieces = pieces
        self._special = special

    def decode(self, ids, skip_special_tokens=True):
        return "".join(
            self._pieces[i]
            for i in ids
            if not (skip_special_tokens and i in self._special)
        )


def _tokenizer() -> _StubTokenizer:
    return _StubTokenizer(PIECES)


@pytest.mark.parametrize(
    "text",
    [
        "One sentence.",
        "Two sentences. Yes.",
        "What now? Indeed! Fine.",
        "No punctuation at all",
        "A.\n\nNewline split here.",
        "Trailing space.   ",
        "  Leading space. Yes.",
    ],
)
def test_sentence_offsets_slice_back_to_the_sentence(text: str):
    for sentence, start, end in split_sentences(text):
        assert sentence
        assert text[start:end] == sentence


def test_sentence_offsets_are_increasing_and_contiguous():
    text = "First one. Second one! Third?"
    spans = split_sentences(text)
    assert [s for s, _, _ in spans] == [
        "First one.",
        "Second one!",
        "Third?",
    ]
    # Reading order, no overlap.
    for (_, _, end), (_, start, _) in zip(spans, spans[1:], strict=False):
        assert end <= start


def test_empty_and_whitespace_only_text_yields_no_sentences():
    assert split_sentences("") == []
    assert split_sentences("  \n\n  ") == []


def test_token_char_offsets_cover_the_answer():
    offsets = token_char_offsets(_tokenizer(), IDS)
    assert len(offsets) == len(IDS)
    # Each token's span decodes to exactly the piece it adds; the
    # skipped end token adds nothing.
    for (start, end), token_id in zip(offsets, IDS, strict=False):
        if token_id == 99:
            assert start == end
            continue
        assert ANSWER[start:end] == PIECES[token_id]
    # The final span ends at the end of the answer.
    assert offsets[-1][1] == len(ANSWER)


def test_skipped_special_tokens_get_an_empty_span():
    offsets = token_char_offsets(_tokenizer(), IDS)
    assert offsets[3] == (8, 8)  # the end token in the middle


def test_answer_spans_align_sentences_to_tokens():
    spans = answer_spans(ANSWER, IDS, _tokenizer())
    assert spans == [
        Span(text="The cat.", start=0, end=8, token_start=0, token_end=3),
        Span(text="It ran.", start=9, end=16, token_start=4, token_end=7),
    ]
    assert spans[0].n_tokens == 3
    assert spans[1].n_tokens == 3


def test_answer_spans_token_ranges_increase():
    spans = answer_spans(ANSWER, IDS, _tokenizer())
    for first, second in zip(spans, spans[1:], strict=False):
        assert first.token_end <= second.token_start


def test_answer_spans_rejects_an_answer_without_sentences():
    with pytest.raises(ValueError, match="no sentences"):
        answer_spans("   ", IDS, _tokenizer())
