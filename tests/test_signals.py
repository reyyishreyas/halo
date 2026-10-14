"""Unit tests for the two detection signals."""

from __future__ import annotations

import numpy as np
import pytest

from halo.signals import (
    features,
    grounding_scores,
    max_cosine,
    mean_token_logprob,
)

LOGPROBS = [-0.1, -0.5, -2.0, -0.4]


def test_mean_token_logprob_averages_the_window():
    assert mean_token_logprob(LOGPROBS, 0, 3) == pytest.approx(-2.6 / 3)
    assert mean_token_logprob(LOGPROBS, 1, 3) == pytest.approx(-1.25)
    assert mean_token_logprob(LOGPROBS, 3, 4) == pytest.approx(-0.4)


@pytest.mark.parametrize(
    "start, end",
    [(0, 0), (0, 5), (-1, 2), (2, 1)],
)
def test_mean_token_logprob_rejects_bad_ranges(start: int, end: int):
    with pytest.raises(ValueError, match="does not fit"):
        mean_token_logprob(LOGPROBS, start, end)


def test_max_cosine_prefers_the_closest_chunk():
    span = np.array([1.0, 0.0])
    chunks = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    assert max_cosine(span, chunks) == pytest.approx(1.0)


def test_max_cosine_ignores_chunk_scale():
    span = np.array([2.0, 0.0])
    chunks = np.array([[0.5, 0.0]])  # same direction, shorter
    assert max_cosine(span, chunks) == pytest.approx(1.0)


def test_max_cosine_of_orthogonal_vectors_is_zero():
    assert max_cosine(np.array([0.0, 1.0]), np.array([[1.0, 0.0]])) == (
        pytest.approx(0.0)
    )


def test_max_cosine_rejects_no_chunks_and_bad_dimensions():
    with pytest.raises(ValueError, match="at least one"):
        max_cosine(np.array([1.0, 0.0]), np.empty((0, 2)))
    with pytest.raises(ValueError, match="chunk dim"):
        max_cosine(np.array([1.0, 0.0]), np.array([[1.0, 0.0, 0.0]]))


class _StubEncoder:
    """Returns fixed vectors per text; records what was encoded."""

    def __init__(self, table: dict):
        self._table = table
        self.calls: list[list[str]] = []

    def encode(self, texts, normalize_embeddings, convert_to_numpy):
        assert normalize_embeddings is True
        assert convert_to_numpy is True
        self.calls.append(list(texts))
        return np.array([self._table[t] for t in texts], dtype=float)


def test_grounding_scores_takes_the_best_matching_chunk():
    encoder = _StubEncoder(
        {
            "span one": [1.0, 0.0],
            "span two": [0.0, 1.0],
            "chunk a": [1.0, 0.0],
            "chunk b": [0.0, 1.0],
        }
    )
    scores = grounding_scores(
        encoder, ["span one", "span two"], ["chunk a", "chunk b"]
    )
    # Each span matches one chunk exactly.
    assert scores == pytest.approx([1.0, 1.0])
    # Both text sets were embedded once.
    assert encoder.calls == [["span one", "span two"], ["chunk a", "chunk b"]]


def test_grounding_scores_of_no_spans_is_empty():
    encoder = _StubEncoder({"chunk a": [1.0, 0.0]})
    scores = grounding_scores(encoder, [], ["chunk a"])
    assert scores.size == 0


def test_grounding_scores_needs_retrieved_chunks():
    encoder = _StubEncoder({})
    with pytest.raises(ValueError, match="at least one"):
        grounding_scores(encoder, ["span one"], [])


def test_features_stacks_both_signals():
    matrix = features([-1.0, -0.2], [0.9, 0.1])
    assert matrix.shape == (2, 2)
    assert matrix[:, 0] == pytest.approx([-1.0, -0.2])
    assert matrix[:, 1] == pytest.approx([0.9, 0.1])


def test_features_rejects_length_mismatch():
    with pytest.raises(ValueError, match="confidences"):
        features([-1.0], [0.9, 0.1])
