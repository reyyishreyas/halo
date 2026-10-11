"""Sentence-preserving chunking.

The proposal asks for chunks of roughly 200-500 tokens. Sentences are
kept whole and packed greedily rather than cut at a fixed token
boundary, for one concrete reason: Phase 3 scores hallucination at the
*sentence* level, and the grounding signal matches a generated
sentence against retrieved chunks. If a chunk boundary fell in the
middle of a sentence, that sentence's evidence would be split across
two chunks and the similarity score would be systematically low.

The same reasoning forbids the opposite failure. A single sentence
longer than the ceiling is normally emitted whole, but that produces a
chunk the embedding model silently truncates to its context window --
measured at 643 tokens against bge-small's 512 -- so the evidence past
the cut is never embedded and never retrievable. Pass a ``splitter``
to hard-cut such sentences instead; the pipeline always does.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

DEFAULT_MIN_TOKENS = 200
DEFAULT_MAX_TOKENS = 500

Tokenizer = Callable[[str], list[int]]
Splitter = Callable[[str, int], list[str]]


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of text.

    Attributes
    ----------
    text : str
        The chunk as stored in the index.
    sentences : list[str]
        The sentences packed into this chunk, in order.
    source_title : str
        Title of the paragraph the chunk came from. Empty when the
        caller passes bare sentences.
    token_count : int
        Tokens counted with the caller's tokenizer.
    """

    text: str
    sentences: list[str]
    source_title: str
    token_count: int


def _approx_tokens(text: str) -> list[int]:
    """Stand-in tokenizer: one token per whitespace-separated word.

    English runs at roughly 1.3 subword tokens per word, so this
    slightly under-counts and chunks come out a little short of the
    requested range. It exists so the packer can be tested without
    downloading a model; the pipeline passes the real tokenizer.
    """
    return list(range(len(text.split())))


def _units(
    sentences: list[str],
    count: Tokenizer,
    max_tokens: int,
    splitter: Splitter | None,
) -> list[str]:
    """Flatten sentences into packing units, cutting over-long ones.

    With no splitter, an oversized sentence stays whole and is emitted
    as its own chunk, which the embedder will then truncate.
    """
    out: list[str] = []
    for sentence in sentences:
        text = sentence.strip()
        if not text:
            continue
        if splitter is not None and len(count(text)) > max_tokens:
            out.extend(
                piece.strip()
                for piece in splitter(text, max_tokens)
                if piece.strip()
            )
        else:
            out.append(text)
    return out


def _pack(
    units: list[str],
    title: str,
    count: Tokenizer,
    min_tokens: int,
    max_tokens: int,
) -> list[Chunk]:
    """Greedily pack units, cutting only once past the floor.

    A chunk is cut only when it already holds ``min_tokens`` and the
    next unit will not fit. Without that condition a paragraph could
    be split into one full chunk plus a two-sentence remnant.
    """
    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0

    for unit in units:
        size = len(count(unit))
        if (
            current
            and current_len + size > max_tokens
            and current_len >= min_tokens
        ):
            chunks.append(_make(current, title, count))
            current, current_len = [], 0
        current.append(unit)
        current_len += size

    if current:
        chunks.append(_make(current, title, count))
    return chunks


def _make(sentences: list[str], title: str, count: Tokenizer) -> Chunk:
    """Build a ``Chunk`` from the sentences packed into it."""
    text = " ".join(sentences)
    return Chunk(
        text=text,
        sentences=list(sentences),
        source_title=title,
        token_count=len(count(text)),
    )


def chunk_sentences(
    sentences: list[str],
    *,
    title: str = "",
    tokenizer: Tokenizer | None = None,
    splitter: Splitter | None = None,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    hard_max_tokens: int | None = None,
) -> list[Chunk]:
    """Pack sentences into chunks inside the requested token window.

    Parameters
    ----------
    sentences : list[str]
        Sentences to pack, in reading order.
    title : str
        Source title recorded on every chunk produced.
    tokenizer : Callable[[str], list[int]], optional
        Tokenizer used for counting. Defaults to a word-count
        approximation; the pipeline passes the embedding model's.
    splitter : Callable[[str, int], list[str]], optional
        Splits a sentence that exceeds ``max_tokens`` into pieces of
        at most that size. Without one, such a sentence is emitted
        whole and the embedding model truncates it. The pipeline passes
        the tokenizer's splitter so no chunk outgrows the context
        window.
    min_tokens : int
        Floor. A chunk is only cut once it reaches this size, and a
        short final chunk is folded into its predecessor.
    max_tokens : int
        Upper bound. A single unit may exceed it only when no
        ``splitter`` was supplied.
    hard_max_tokens : int, optional
        Ceiling the short-tail fold must never cross, normally the
        embedder's context window. The fold may push a chunk past
        ``max_tokens``, which is tolerable; pushing it past the window
        is not, because the embedder would truncate it and drop the
        tail. Set this whenever ``splitter`` is used.

    Returns
    -------
    list[Chunk]
        Chunks in order. Empty when ``sentences`` holds no text.

    Raises
    ------
    ValueError
        If ``min_tokens`` exceeds ``max_tokens``, if either is not
        positive, or if ``hard_max_tokens`` is below ``max_tokens``.
    """
    if min_tokens <= 0 or max_tokens <= 0:
        raise ValueError("token bounds must be positive")
    if min_tokens > max_tokens:
        raise ValueError(
            f"min_tokens ({min_tokens}) exceeds max_tokens ({max_tokens})"
        )
    if hard_max_tokens is not None and hard_max_tokens < max_tokens:
        raise ValueError(
            f"hard_max_tokens ({hard_max_tokens}) is below "
            f"max_tokens ({max_tokens})"
        )

    count: Tokenizer = tokenizer or _approx_tokens
    chunks = _pack(
        _units(sentences, count, max_tokens, splitter),
        title,
        count,
        min_tokens,
        max_tokens,
    )

    # A paragraph's remainder can sit far below the floor. Leaving it
    # as its own chunk spends a FAISS slot on text too short to
    # retrieve usefully, so it is folded into the predecessor. That
    # can push the result past max_tokens, which is the lesser evil
    # next to a chunk nobody will ever match against.
    if len(chunks) > 1 and chunks[-1].token_count < min_tokens:
        prev, last = chunks[-2], chunks[-1]
        merged = _make(
            prev.sentences + last.sentences,
            title or prev.source_title,
            count,
        )
        # Crossing max_tokens is a tolerable compromise; crossing
        # hard_max_tokens is not, because the embedder would truncate
        # the chunk and silently drop the evidence being folded in.
        if hard_max_tokens is None or merged.token_count <= hard_max_tokens:
            chunks = chunks[:-2] + [merged]

    return chunks


def chunk_paragraphs(
    paragraphs: list, tokenizer: Tokenizer | None = None, **kwargs
) -> list[Chunk]:
    """Chunk every paragraph of a record, keeping its title.

    Parameters
    ----------
    paragraphs : list
        ``halo.data.Paragraph`` objects.
    tokenizer : Callable[[str], list[int]], optional
        Forwarded to :func:`chunk_sentences`.
    **kwargs
        Any further :func:`chunk_sentences` keyword arguments,
        including ``splitter``.

    Returns
    -------
    list[Chunk]
        All chunks across all paragraphs, in paragraph order.
    """
    out: list[Chunk] = []
    for para in paragraphs:
        out.extend(
            chunk_sentences(
                para.sentences, title=para.title, tokenizer=tokenizer, **kwargs
            )
        )
    return out
