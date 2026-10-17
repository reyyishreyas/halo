"""Unit tests for NLI silver labels (frozen model stubbed)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from halo.labels import (
    premise_from_record,
    premises_from_record,
    silver_labels,
)


class _StubNLI:
    """Cross-encoder stand-in with fixed three-way probabilities."""

    def __init__(self, label2id: dict, probabilities: list):
        # Config directly on the model, like some wrappers expose it.
        self.config = SimpleNamespace(label2id=label2id)
        self._probabilities = np.asarray(probabilities, dtype=float)
        self.pairs = None

    def predict(self, pairs):
        self.pairs = list(pairs)
        return self._probabilities


class _NestedStubNLI(_StubNLI):
    """Same stub, but the config only exists under ``.model``."""

    def __init__(self, label2id: dict, probabilities: list):
        super().__init__(label2id, probabilities)
        nested = SimpleNamespace(config=SimpleNamespace(label2id=label2id))
        del self.config
        self.model = nested


def test_entailment_wins_means_grounded():
    model = _StubNLI(
        {"contradiction": 0, "entailment": 1, "neutral": 2},
        [
            [0.05, 0.90, 0.05],  # entailment -> grounded
            [0.80, 0.10, 0.10],  # contradiction -> hallucinated
        ],
    )
    labels = silver_labels(
        "Cats purr. Cats sleep.", ["Cats purr.", "Cats fly."], model
    )
    assert labels.tolist() == [0, 1]
    # Each hypothesis was paired with the premise.
    assert model.pairs[0][0] == "Cats purr. Cats sleep."
    assert model.pairs[1] == ("Cats purr. Cats sleep.", "Cats fly.")


def test_label_order_is_read_from_the_model():
    # Entailment sits in a different column here.
    model = _StubNLI(
        {"entailment": 0, "neutral": 1, "contradiction": 2},
        [
            [0.95, 0.03, 0.02],  # entailment in column 0
            [0.05, 0.15, 0.80],  # contradiction in column 2
        ],
    )
    labels = silver_labels("premise", ["yes", "no"], model)
    assert labels.tolist() == [0, 1]


def test_nested_model_config_is_supported():
    model = _NestedStubNLI(
        {"CONTRADICTION": 2, "ENTAILMENT": 0, "NEUTRAL": 1},
        [[0.70, 0.20, 0.10]],  # entailment lives in column 0 here
    )
    assert silver_labels("premise", ["claim"], model).tolist() == [0]


def test_wrong_number_of_model_rows_is_rejected():
    model = _StubNLI(
        {"contradiction": 0, "entailment": 1, "neutral": 2},
        [[0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],  # two rows, one pair
    )
    with pytest.raises(ValueError, match="rows for 1 premise-hypothesis"):
        silver_labels("premise", ["claim"], model)


def test_model_without_an_entailment_label_is_rejected():
    model = _StubNLI({"positive": 0, "negative": 1}, [[0.5, 0.5]])
    with pytest.raises(ValueError, match="entailment"):
        silver_labels("premise", ["claim"], model)


def test_no_hypotheses_means_no_labels_and_no_model_call():
    model = _StubNLI({"contradiction": 0, "entailment": 1, "neutral": 2}, [])
    labels = silver_labels("premise", [], model)
    assert labels.size == 0
    assert model.pairs is None


def test_premise_from_record_joins_titled_chunks():
    record = {
        "retrieved": [
            {"rank": 1, "title": "Cats", "score": 0.9, "text": "Cats purr."},
            {"rank": 2, "title": "Sleep", "score": 0.8, "text": "Cats sleep."},
        ]
    }
    assert premise_from_record(record) == (
        "Cats: Cats purr.\n\nSleep: Cats sleep."
    )


def test_premise_from_record_without_chunks_is_rejected():
    with pytest.raises(ValueError, match="no retrieved chunks"):
        premise_from_record({"retrieved": []})
    with pytest.raises(ValueError, match="no retrieved chunks"):
        premise_from_record({})


def test_multiple_premises_pool_by_max_entailment():
    # Two chunks, one hypothesis. Chunk order in the pairs is
    # (hypothesis, chunk), so the stub returns 1 hypothesis x 2 chunks.
    model = _StubNLI(
        {"contradiction": 0, "entailment": 1, "neutral": 2},
        [
            [0.85, 0.05, 0.10],  # chunk A: contradiction
            [0.05, 0.90, 0.05],  # chunk B: entailment -> grounded
        ],
    )
    labels = silver_labels(["A context", "B context"], ["Cats purr."], model)
    # Grounded because the *second* chunk entails it, even though the
    # first contradicts. This is the truncation fix: evidence in a
    # later chunk still counts.
    assert labels.tolist() == [0]
    assert model.pairs == [
        ("A context", "Cats purr."),
        ("B context", "Cats purr."),
    ]


def test_no_chunk_entails_means_hallucinated():
    model = _StubNLI(
        {"contradiction": 0, "entailment": 1, "neutral": 2},
        [
            [0.70, 0.20, 0.10],  # contradiction
            [0.20, 0.30, 0.50],  # neutral
        ],
    )
    labels = silver_labels(["A", "B"], ["Cats fly."], model)
    assert labels.tolist() == [1]


def test_list_of_premises_is_empty_rejected():
    model = _StubNLI({"contradiction": 0, "entailment": 1, "neutral": 2}, [])
    with pytest.raises(ValueError, match="at least one premise"):
        silver_labels([], ["claim"], model)


def test_premises_from_record_returns_titled_list():
    record = {
        "retrieved": [
            {"rank": 1, "title": "Cats", "score": 0.9, "text": "Cats purr."},
            {"rank": 2, "title": "Sleep", "score": 0.8, "text": "Cats sleep."},
        ]
    }
    # Two per-chunk premises plus one joint top-2 premise (multi-hop).
    assert premises_from_record(record) == [
        "Cats: Cats purr.",
        "Sleep: Cats sleep.",
        "Cats: Cats purr.\n\nSleep: Cats sleep.",
    ]


def test_premises_from_record_joint_can_be_disabled():
    record = {
        "retrieved": [
            {"rank": 1, "title": "Cats", "score": 0.9, "text": "Cats purr."},
            {"rank": 2, "title": "Sleep", "score": 0.8, "text": "Cats sleep."},
        ]
    }
    assert premises_from_record(record, joint_top=0) == [
        "Cats: Cats purr.",
        "Sleep: Cats sleep.",
    ]


def test_premises_from_record_needs_two_chunks_for_joint():
    record = {
        "retrieved": [
            {"rank": 1, "title": "Cats", "score": 0.9, "text": "Hi."}
        ]
    }
    # Only one chunk, so no joint premise is added.
    assert premises_from_record(record) == ["Cats: Hi."]


def test_premises_from_record_without_chunks_is_rejected():
    with pytest.raises(ValueError, match="no retrieved chunks"):
        premises_from_record({"retrieved": []})
