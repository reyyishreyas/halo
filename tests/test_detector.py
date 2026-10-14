"""Unit tests for the standardized logistic-regression detector."""

from __future__ import annotations

import numpy as np
import pytest

from halo.detector import SpanDetector

# Two clusters: confident + grounded, unconfident + ungrounded.
X = np.array(
    [
        [-0.9, 0.95],
        [-1.1, 0.88],
        [-0.8, 0.92],
        [0.9, 0.30],
        [1.1, 0.20],
        [0.8, 0.25],
    ]
)
Y = np.array([0, 0, 0, 1, 1, 1])


def test_fit_separates_the_training_clusters():
    detector = SpanDetector().fit(X, Y)
    predictions = detector.predict(X)
    assert predictions.tolist() == Y.tolist()

    probabilities = detector.predict_proba(X)
    assert probabilities.shape == (6, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    # Column 1 is P(hallucinated): low for grounded, high for the rest.
    assert probabilities[:3, 1].max() < 0.5
    assert probabilities[3:, 1].min() > 0.5


def test_fit_returns_self_for_chaining():
    detector = SpanDetector()
    assert detector.fit(X, Y) is detector


def test_standardization_uses_the_training_statistics():
    from_dict = SpanDetector().fit(X, Y).to_dict()
    assert from_dict["scaler"]["mean"] == pytest.approx(
        X.mean(axis=0).tolist()
    )
    assert from_dict["features"] == ["mean_token_logprob", "max_cosine"]
    assert from_dict["classes"] == [0, 1]


def test_single_class_training_is_rejected():
    with pytest.raises(ValueError, match="both grounded and"):
        SpanDetector().fit(X, np.zeros(len(X), dtype=int))


def test_wrong_feature_width_and_length_mismatch_are_rejected():
    with pytest.raises(ValueError, match="\\(n_spans, 2\\)"):
        SpanDetector().fit(np.ones((4, 3)), np.array([0, 0, 1, 1]))
    with pytest.raises(ValueError, match="spans but"):
        SpanDetector().fit(X, np.array([0, 1]))


def test_unfitted_detector_explains_itself():
    with pytest.raises(RuntimeError, match="not fitted"):
        SpanDetector().predict(X)


def test_threshold_must_be_a_probability():
    for threshold in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="threshold"):
            SpanDetector(threshold=threshold)


def test_threshold_moves_the_decision_boundary():
    # Every probability lands between 0 and 1, so an extreme
    # threshold forces one class.
    always_one = SpanDetector(threshold=0.001).fit(X, Y)
    always_zero = SpanDetector(threshold=0.999).fit(X, Y)
    assert always_one.predict(X).tolist() == [1] * len(X)
    assert always_zero.predict(X).tolist() == [0] * len(X)


def test_to_dict_from_dict_round_trip_preserves_predictions():
    original = SpanDetector().fit(X, Y)
    rebuilt = SpanDetector.from_dict(original.to_dict())
    assert rebuilt.predict(X).tolist() == original.predict(X).tolist()
    assert np.allclose(rebuilt.predict_proba(X), original.predict_proba(X))


def test_save_load_round_trip_preserves_predictions(tmp_path):
    original = SpanDetector().fit(X, Y)
    path = tmp_path / "detector.json"
    original.save(path)
    # Readable JSON, not a pickle.
    assert '"intercept"' in path.read_text()

    rebuilt = SpanDetector.load(path)
    assert rebuilt.predict(X).tolist() == original.predict(X).tolist()
    assert np.allclose(rebuilt.predict_proba(X), original.predict_proba(X))
    assert rebuilt.to_dict() == original.to_dict()


def test_load_rejects_a_file_missing_the_model(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"threshold": 0.5}')
    with pytest.raises(KeyError):
        SpanDetector.load(path)
