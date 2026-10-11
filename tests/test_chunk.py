"""Tests for the chunker.

The word-count fallback tokenizer is one token per whitespace-separated
word, so token budgets below are word counts and the assertions stay
exact rather than approximate.
"""

import pytest

from halo.chunk import chunk_paragraphs, chunk_sentences
from halo.data import Paragraph


def words(n: int) -> str:
    """Return a string of exactly ``n`` whitespace-separated tokens."""
    return " ".join(f"w{i}" for i in range(n))


def test_empty_input_produces_no_chunks():
    assert chunk_sentences([]) == []
    assert chunk_sentences(["", "   ", "\n"]) == []


def test_blank_sentences_are_skipped_not_counted():
    chunks = chunk_sentences(["", words(4), " ", words(6)])
    assert len(chunks) == 1
    assert chunks[0].sentences == [words(4), words(6)]


def test_no_chunk_exceeds_the_token_ceiling():
    sentences = [words(10)] * 7
    chunks = chunk_sentences(sentences, min_tokens=10, max_tokens=30)
    assert max(c.token_count for c in chunks) <= 30
    assert sum(len(c.sentences) for c in chunks) == 7


def test_short_tail_is_folded_into_the_previous_chunk():
    # 7 sentences of 10 words pack as 30 + 30 + 10; the 10-word
    # remainder is below min_tokens, so it joins the chunk before it.
    chunks = chunk_sentences([words(10)] * 7, min_tokens=15, max_tokens=30)
    assert len(chunks) == 2
    assert chunks[0].token_count == 30
    assert chunks[1].token_count == 40
    assert len(chunks[1].sentences) == 4


def test_no_merge_happens_when_the_tail_already_meets_the_floor():
    chunks = chunk_sentences([words(10)] * 7, min_tokens=5, max_tokens=30)
    assert [len(c.sentences) for c in chunks] == [3, 3, 1]


def test_a_single_oversized_sentence_is_emitted_alone():
    chunks = chunk_sentences([words(100)], min_tokens=10, max_tokens=30)
    assert len(chunks) == 1
    assert chunks[0].token_count == 100
    assert chunks[0].sentences == [words(100)]


def _word_splitter(text: str, max_tokens: int) -> list[str]:
    """Cut on word boundaries; stands in for the tokenizer's splitter."""
    tokens = text.split()
    return [
        " ".join(tokens[i : i + max_tokens])
        for i in range(0, len(tokens), max_tokens)
    ]


def test_a_splitter_keeps_every_chunk_under_the_ceiling():
    # Without a splitter this 40-word sentence becomes one 40-token
    # chunk, which an embedder with a 15-token window would truncate.
    chunks = chunk_sentences(
        [words(40)], splitter=_word_splitter, min_tokens=5, max_tokens=15
    )
    assert len(chunks) == 3
    assert [c.token_count for c in chunks] == [15, 15, 10]
    assert all(c.token_count <= 15 for c in chunks)


def test_split_pieces_are_still_packed_with_neighbours():
    # Splitting runs before packing, so a short final piece is folded
    # into a neighbour instead of being left as its own fragment.
    chunks = chunk_sentences(
        [words(40), words(4)],
        splitter=_word_splitter,
        min_tokens=5,
        max_tokens=20,
    )
    # The 40-word sentence splits to 20 + 20, which pack as two full
    # chunks; the 4-token remainder then folds back, and may push that
    # chunk past max_tokens by design.
    assert [c.token_count for c in chunks] == [20, 24]
    assert sum(len(c.sentences) for c in chunks) == 3


def test_the_tail_fold_stops_at_the_hard_ceiling():
    # 20 + 20 + 4 would fold to 20 + 24, but a hard ceiling of 22
    # forbids it, so the short chunk survives on its own rather than
    # being folded into something the embedder would truncate.
    chunks = chunk_sentences(
        [words(40), words(4)],
        splitter=_word_splitter,
        min_tokens=5,
        max_tokens=20,
        hard_max_tokens=22,
    )
    assert [c.token_count for c in chunks] == [20, 20, 4]


def test_a_hard_ceiling_below_max_tokens_is_rejected():
    with pytest.raises(ValueError, match="hard_max_tokens"):
        chunk_sentences(
            [words(5)], min_tokens=1, max_tokens=20, hard_max_tokens=10
        )


def test_sentences_stay_whole_and_in_order():
    sentences = ["first sentence", "second sentence", "third sentence"]
    chunks = chunk_sentences(sentences, min_tokens=1, max_tokens=100)
    assert [s for c in chunks for s in c.sentences] == sentences


def test_source_title_is_recorded_on_every_chunk():
    chunks = chunk_sentences([words(5)], title="Some Article")
    assert all(c.source_title == "Some Article" for c in chunks)


def test_chunk_paragraphs_keeps_titles_and_order():
    paragraphs = [
        Paragraph(title="A", sentences=[words(4), words(4)]),
        Paragraph(title="B", sentences=[words(4)]),
    ]
    chunks = chunk_paragraphs(paragraphs, min_tokens=2, max_tokens=100)
    assert [c.source_title for c in chunks] == ["A", "B"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_tokens": 0, "max_tokens": 10},
        {"min_tokens": 10, "max_tokens": -1},
        {"min_tokens": 20, "max_tokens": 10},
    ],
)
def test_invalid_token_bounds_raise(kwargs):
    with pytest.raises(ValueError):
        chunk_sentences([words(5)], **kwargs)
