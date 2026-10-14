"""The two detection signals: token confidence and grounding.

Signal 1 (confidence) is the mean log-probability the generation step
already assigned to a span's tokens. It costs nothing extra: no second
pass, no extra model. Models can be confidently wrong, so it is not
enough alone.

Signal 2 (grounding) is the maximum cosine similarity between the span
and any retrieved chunk. High means the sentence sits close to the
evidence; low means it may not be supported by the retrieved context.

Both are plain numbers. The detector in :mod:`halo.detector` combines
them after standardization; nothing here loads a neural model at
scoring time -- the embedder is the same one retrieval already uses.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def mean_token_logprob(
    token_logprobs: Sequence[float], token_start: int, token_end: int
) -> float:
    """Average log-probability over one span's tokens.

    Parameters
    ----------
    token_logprobs : Sequence[float]
        Per-token log-probabilities from a Phase 2 record, aligned
        with ``token_ids``.
    token_start : int
        First token of the span (inclusive), as in :class:`Span`.
    token_end : int
        Token after the span (exclusive).

    Returns
    -------
    float
        The mean; closer to zero means more confident.

    Raises
    ------
    ValueError
        If the range is empty or outside the log-probabilities.
    """
    if not 0 <= token_start < token_end <= len(token_logprobs):
        raise ValueError(
            f"span [{token_start}, {token_end}) does not fit "
            f"{len(token_logprobs)} token log-probabilities"
        )
    window = token_logprobs[token_start:token_end]
    return float(np.mean(window))


def _normalise_rows(vectors: np.ndarray) -> np.ndarray:
    """Return unit-length rows, rejecting any zero-length row."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("cannot embed with a zero-length vector")
    return vectors / norms


def max_cosine(span_vector: np.ndarray, chunk_vectors: np.ndarray) -> float:
    """Maximum cosine similarity between one span and many chunks.

    Parameters
    ----------
    span_vector : numpy.ndarray
        Embedding of the span, shape ``(d,)``.
    chunk_vectors : numpy.ndarray
        Embeddings of the retrieved chunks, shape ``(n, d)`` with
        ``n >= 1``.

    Returns
    -------
    float
        The largest cosine similarity over all chunks, in ``[-1, 1]``.

    Raises
    ------
    ValueError
        If there are no chunks or a vector has zero length.
    """
    span = np.atleast_2d(np.asarray(span_vector, dtype=np.float64))
    chunks = np.atleast_2d(np.asarray(chunk_vectors, dtype=np.float64))
    if chunks.shape[0] == 0:
        raise ValueError("grounding needs at least one retrieved chunk")
    if span.shape[1] != chunks.shape[1]:
        raise ValueError(
            f"span dim {span.shape[1]} != chunk dim {chunks.shape[1]}"
        )
    span_unit = _normalise_rows(span)
    chunk_units = _normalise_rows(chunks)
    return float((chunk_units @ span_unit[0]).max())


def grounding_scores(
    encoder: Any, span_texts: Sequence[str], chunk_texts: Sequence[str]
) -> np.ndarray:
    """Maximum cosine similarity of each span to the retrieved chunks.

    Both text sets are embedded with ``encoder`` -- the same model
    retrieval used -- so similarities are comparable to the retrieval
    scores in the same record.

    Parameters
    ----------
    encoder : Any
        Sentence-transformers model with ``encode(...)``
        (duck-typed so tests can stub it).
    span_texts : Sequence[str]
        Sentence spans to score.
    chunk_texts : Sequence[str]
        Retrieved chunk texts for the same question.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(span_texts),)``; empty if there are no spans.

    Raises
    ------
    ValueError
        If ``chunk_texts`` is empty.
    """
    if not chunk_texts:
        raise ValueError("grounding needs at least one retrieved chunk")
    if not span_texts:
        return np.empty(0, dtype=np.float64)

    def encode(texts: Sequence[str]) -> np.ndarray:
        vectors = np.asarray(
            encoder.encode(
                list(texts),
                normalize_embeddings=True,
                convert_to_numpy=True,
            ),
            dtype=np.float64,
        )
        return _normalise_rows(vectors)

    span_units = encode(span_texts)
    chunk_units = encode(chunk_texts)
    similarities = span_units @ chunk_units.T
    return similarities.max(axis=1)


def features(
    confidences: Sequence[float], groundings: Sequence[float]
) -> np.ndarray:
    """Assemble the detector's feature matrix from both signals.

    Parameters
    ----------
    confidences : Sequence[float]
        Mean token log-probability per span.
    groundings : Sequence[float]
        Maximum cosine similarity per span.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, 2)``: column 0 is confidence, column 1 grounding.

    Raises
    ------
    ValueError
        If the two signal lists differ in length.
    """
    if len(confidences) != len(groundings):
        raise ValueError(
            f"{len(confidences)} confidences but {len(groundings)} groundings"
        )
    return np.column_stack(
        [
            np.asarray(confidences, dtype=np.float64),
            np.asarray(groundings, dtype=np.float64),
        ]
    )
