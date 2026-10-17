"""Unit tests for span-level detection metrics."""

from __future__ import annotations

import pytest

from halo.metrics import (
    DetectionMetrics,
    best_threshold,
    detection_metrics,
    macro_average,
)


def test_hand_computed_scores():
    metrics = detection_metrics([1, 1, 0, 0, 1], [1, 0, 1, 0, 1])
    # tp=2, fp=1, fn=1, tn=1.
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(2 / 3)
    assert metrics.accuracy == pytest.approx(3 / 5)
    assert metrics.hallucination_rate == pytest.approx(3 / 5)
    assert metrics.n_spans == 5
    assert metrics.n_hallucinated == 3
    assert metrics.n_flagged == 3


def test_flagging_nothing_scores_zero_but_reports_the_rate():
    metrics = detection_metrics([1, 0], [0, 0])
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.hallucination_rate == 0.0


def test_perfect_predictions_score_one():
    metrics = detection_metrics([1, 0, 1], [1, 0, 1])
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.accuracy == 1.0
    assert metrics.hallucination_rate == pytest.approx(2 / 3)


def test_flagging_everything_still_scores_zero_accuracy():
    metrics = detection_metrics([0, 0], [1, 1])
    assert metrics.accuracy == 0.0
    assert metrics.recall == 0.0  # no hallucinated spans to find
    assert metrics.hallucination_rate == 1.0


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="shape"):
        detection_metrics([1, 0], [1])


def test_empty_scoring_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        detection_metrics([], [])


def test_non_binary_labels_are_rejected():
    with pytest.raises(ValueError, match="0 or 1"):
        detection_metrics([0, 2], [0, 1])
    with pytest.raises(ValueError, match="y_pred"):
        detection_metrics([0, 1], [0, -1])


def test_to_dict_exposes_every_field():
    metrics = detection_metrics([1], [1])
    data = metrics.to_dict()
    assert set(data) == {
        "precision",
        "recall",
        "f1",
        "accuracy",
        "hallucination_rate",
        "n_spans",
        "n_hallucinated",
        "n_flagged",
    }


def test_macro_average_averages_rates_and_sums_counts():
    easy = detection_metrics([1], [1])  # every rate 1.0
    silent = detection_metrics([0, 0], [0, 0])  # accuracy 1.0, rest 0
    averaged = macro_average([easy, silent])
    assert averaged.precision == pytest.approx(0.5)
    assert averaged.recall == pytest.approx(0.5)
    assert averaged.f1 == pytest.approx(0.5)
    assert averaged.accuracy == pytest.approx(1.0)
    assert averaged.hallucination_rate == pytest.approx(0.5)
    assert averaged.n_spans == 3
    assert averaged.n_hallucinated == 1
    assert averaged.n_flagged == 1


def test_macro_average_of_no_questions_is_rejected():
    with pytest.raises(ValueError, match="zero questions"):
        macro_average([])


def test_metrics_are_immutable_and_storable():
    metrics = detection_metrics([1, 0], [1, 0])
    with pytest.raises(AttributeError):
        metrics.f1 = 0.0  # frozen dataclass
    assert isinstance(metrics, DetectionMetrics)


def test_best_threshold_separates_two_clear_clusters():
    # Low scores are hallucinated (1), high scores grounded (0).
    scores = [0.1, 0.2, 0.3, 0.8, 0.9, 1.0]
    labels = [1, 1, 1, 0, 0, 0]
    cut = best_threshold(scores, labels)
    # Any cut in (0.3, 0.8] separates perfectly; the chosen one must.
    predicted = [int(s < cut) for s in scores]
    assert predicted == [1, 1, 1, 0, 0, 0]


def test_best_threshold_picks_the_cut_maximising_f1():
    scores = [0.1, 0.4, 0.6, 0.9]
    labels = [1, 0, 1, 0]
    cut = best_threshold(scores, labels)

    def f1_at(thr: float) -> float:
        predicted = [int(s < thr) for s in scores]
        tp = sum(
            p == 1 and y == 1 for p, y in zip(predicted, labels, strict=True)
        )
        fp = sum(
            p == 1 and y == 0 for p, y in zip(predicted, labels, strict=True)
        )
        fn = sum(
            p == 0 and y == 1 for p, y in zip(predicted, labels, strict=True)
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

    # Labels are not monotone in score, so no cut separates perfectly;
    # the returned cut must still beat every other candidate.
    assert f1_at(cut) == max(f1_at(t) for t in set(scores))


def test_best_threshold_never_returns_an_empty_flag_set():
    # Labels do not correlate with scores at all, so no cut beats F1 0;
    # the fallback (median) still flags something rather than nothing.
    scores = [0.1, 0.2, 0.3, 0.4]
    labels = [0, 1, 0, 1]
    cut = best_threshold(scores, labels)
    assert any(s < cut for s in scores)


def test_best_threshold_rejects_empty_and_mismatched_inputs():
    with pytest.raises(ValueError, match="non-empty"):
        best_threshold([], [])
    with pytest.raises(ValueError, match="non-empty and equal"):
        best_threshold([0.1, 0.2], [1])
