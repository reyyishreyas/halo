"""Sentence spans of an answer, aligned to token log-probabilities.

Phase 3 scores sentences, Phase 2 stored tokens: this module is the
contract between the two. A span is one sentence of the answer text
plus the range of generated token indices that produced it, so the
confidence average is exact rather than a re-tokenization guess.

Alignment never re-encodes the answer. Token offsets come from
decoding progressively longer prefixes of the stored token ids -- the
same call that produced the answer text in the first place -- so the
final prefix *is* the answer by construction, whatever the tokenizer.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

# A boundary is sentence-final punctuation followed by whitespace, or
# a newline. Deliberately simple: see split_sentences.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Split ``text`` into sentences with their character offsets.

    The splitter is deliberately simple: a sentence ends after
    ``.``, ``!`` or ``?`` followed by whitespace, or at a newline.
    Abbreviations such as "U.S." therefore end a sentence early. RAG
    answers in this project are short declarative sentences, and a
    predictable rule beats a heavier dependency.

    Parameters
    ----------
    text : str
        The answer text to split.

    Returns
    -------
    list of tuple[str, int, int]
        ``(sentence, start, end)`` in reading order, with
        ``text[start:end] == sentence``. Empty pieces are dropped.
    """
    pieces: list[tuple[int, int]] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        pieces.append((start, match.start()))
        start = match.end()
    pieces.append((start, len(text)))

    sentences: list[tuple[str, int, int]] = []
    for piece_start, piece_end in pieces:
        raw = text[piece_start:piece_end]
        sentence = raw.strip()
        if not sentence:
            continue
        offset = piece_start + (len(raw) - len(raw.lstrip()))
        sentences.append((sentence, offset, offset + len(sentence)))
    return sentences


def token_char_offsets(
    tokenizer: Any, token_ids: Sequence[int]
) -> list[tuple[int, int]]:
    """Character span of each token in the decoded answer.

    Decodes progressively longer prefixes of ``token_ids`` and records
    how much text each token added. Special tokens the decoder skips
    add nothing and receive an empty span; that is how the end token
    stays out of every sentence without special-casing it.

    Parameters
    ----------
    tokenizer : Any
        Hugging Face tokenizer exposing ``decode(ids,
        skip_special_tokens=...)``.
    token_ids : Sequence[int]
        Generated token ids, as stored in a Phase 2 record.

    Returns
    -------
    list of tuple[int, int]
        Half-open ``(start, end)`` per token, same length and order as
        ``token_ids``. The last non-empty end equals the length of the
        decoded answer.
    """
    offsets: list[tuple[int, int]] = []
    previous = ""
    for index in range(len(token_ids)):
        current = tokenizer.decode(
            list(token_ids[: index + 1]), skip_special_tokens=True
        )
        begin = len(previous)
        offsets.append((begin, max(len(current), begin)))
        previous = current
    return offsets


@dataclass(frozen=True)
class Span:
    """One sentence of an answer and the generated tokens behind it.

    Attributes
    ----------
    text : str
        The sentence as a reader sees it.
    start : int
        Character offset of the sentence in the answer.
    end : int
        End character offset (exclusive).
    token_start : int
        Index of the first token of the span (inclusive), into
        ``token_ids`` and ``token_logprobs``.
    token_end : int
        Index after the last token of the span (exclusive).
    """

    text: str
    start: int
    end: int
    token_start: int
    token_end: int

    @property
    def n_tokens(self) -> int:
        """Number of generated tokens covered by this span."""
        return self.token_end - self.token_start


def answer_spans(
    answer: str, token_ids: Sequence[int], tokenizer: Any
) -> list[Span]:
    """Cut ``answer`` into sentence spans aligned to generated tokens.

    Each sentence claims the tokens whose character span overlaps it.
    Tokens between sentences (whitespace-only pieces) and skipped
    specials belong to no sentence, so the confidence average covers
    exactly the words a reader sees.

    Parameters
    ----------
    answer : str
        The generated answer text (decode of ``token_ids`` with
        special tokens skipped).
    token_ids : Sequence[int]
        Generated token ids, as stored in a Phase 2 record.
    tokenizer : Any
        The generating tokenizer, used to locate each token's
        characters.

    Returns
    -------
    list[Span]
        Sentence spans in reading order; token ranges are increasing
        and never include the end token.

    Raises
    ------
    ValueError
        If ``answer`` contains no sentences.
    """
    sentences = split_sentences(answer)
    if not sentences:
        raise ValueError("answer contains no sentences")
    offsets = token_char_offsets(tokenizer, token_ids)
    nonempty = [i for i, (_, end) in enumerate(offsets) if end > offsets[i][0]]

    spans: list[Span] = []
    for text, char_start, char_end in sentences:
        members = [
            i
            for i in nonempty
            if offsets[i][0] < char_end and offsets[i][1] > char_start
        ]
        spans.append(
            Span(
                text=text,
                start=char_start,
                end=char_end,
                token_start=members[0],
                token_end=members[-1] + 1,
            )
        )
    return spans
