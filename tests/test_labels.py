"""Unit tests for NLI silver labels (frozen model stubbed)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from halo.labels import premise_from_record, silver_labels


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
        [[0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],  # two rows, one hypothesis
    )
    with pytest.raises(ValueError, match="rows for 1 hypotheses"):
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
