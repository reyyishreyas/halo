"""Unit tests for the Phase 4 baselines (models stubbed)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from halo.baselines import (
    ConfidenceThreshold,
    GroundingThreshold,
    NLIBaseline,
    NoVerification,
    SecondVerifier,
    SpanBaseline,
)
from halo.spans import Span

RECORD = {
    "token_logprobs": [-0.1, -0.5, -2.5, -0.2],
    "retrieved": [
        {"rank": 1, "title": "Cats", "score": 0.9, "text": "Cats purr."},
        {"rank": 2, "title": "Sleep", "score": 0.8, "text": "Cats sleep."},
    ],
}
SPANS = [
    Span(text="Cats purr.", start=0, end=10, token_start=0, token_end=2),
    Span(text="Cats fly.", start=11, end=20, token_start=2, token_end=4),
]
PREMISE = "Cats: Cats purr.\n\nSleep: Cats sleep."


class _StubEncoder:
    """Maps fixed texts to fixed vectors."""

    def __init__(self, table: dict):
        self._table = table

    def encode(self, texts, normalize_embeddings, convert_to_numpy):
        return np.array([self._table[t] for t in texts], dtype=float)


class _StubNLI:
    """Cross-encoder stand-in with fixed probabilities per span."""

    def __init__(self, probabilities: list):
        self.config = SimpleNamespace(
            label2id={"contradiction": 0, "entailment": 1, "neutral": 2}
        )
        self._probabilities = np.asarray(probabilities, dtype=float)
        self.pairs = None

    def predict(self, pairs):
        self.pairs = list(pairs)
        return self._probabilities


def _grounding_encoder() -> _StubEncoder:
    # "Cats purr." doubles as a chunk; "Cats fly." is closest to the
    # second chunk at 0.8.
    return _StubEncoder(
        {
            "Cats purr.": [1.0, 0.0],
            "Cats fly.": [0.6, 0.8],
            "Cats sleep.": [0.0, 1.0],
        }
    )


def test_base_class_refuses_to_predict():
    with pytest.raises(NotImplementedError):
        SpanBaseline().predict(RECORD, SPANS)


def test_no_verification_never_flags_anything():
    labels = NoVerification().predict(RECORD, SPANS)
    assert labels.tolist() == [0, 0]
    assert NoVerification().predict(RECORD, []).size == 0


def test_confidence_threshold_flags_only_the_low_confidence_span():
    labels = ConfidenceThreshold(-1.0).predict(RECORD, SPANS)
    # Span means: -0.30 (kept) and -1.35 (flagged).
    assert labels.tolist() == [0, 1]


def test_confidence_threshold_reports_the_cut():
    baseline = ConfidenceThreshold(-1.0)
    assert baseline.name == "confidence_only"
    assert baseline.threshold == -1.0


def test_confidence_threshold_needs_token_logprobs():
    with pytest.raises(ValueError, match="token_logprobs"):
        ConfidenceThreshold(-1.0).predict({"retrieved": []}, SPANS)


def test_grounding_threshold_flags_the_poorly_grounded_span():
    baseline = GroundingThreshold(0.9, _grounding_encoder())
    labels = baseline.predict(RECORD, SPANS)
    # Best cosines: 1.0 (kept) and 0.8 (flagged).
    assert labels.tolist() == [0, 1]


def test_grounding_threshold_needs_retrieved_chunks():
    baseline = GroundingThreshold(0.9, _grounding_encoder())
    with pytest.raises(ValueError, match="retrieved chunks"):
        baseline.predict({"retrieved": []}, SPANS)


def test_nli_baseline_reuses_the_silver_labeler():
    model = _StubNLI(
        [
            [0.05, 0.90, 0.05],  # entailment -> grounded
            [0.80, 0.10, 0.10],  # contradiction -> hallucinated
        ]
    )
    labels = NLIBaseline(model).predict(RECORD, SPANS)
    assert labels.tolist() == [0, 1]
    # The premise is the titled chunks; hypotheses are span texts.
    assert model.pairs == [
        (PREMISE, "Cats purr."),
        (PREMISE, "Cats fly."),
    ]


def test_second_verifier_maps_support_to_labels():
    calls = []

    def verify(premise: str, text: str) -> bool:
        calls.append((premise, text))
        return "fly" not in text

    labels = SecondVerifier(verify).predict(RECORD, SPANS)
    assert labels.tolist() == [0, 1]
    assert calls == [(PREMISE, "Cats purr."), (PREMISE, "Cats fly.")]
