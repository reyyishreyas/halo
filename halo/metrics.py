"""Span-level detection metrics: precision, recall, F1, hallucination rate.

The positive class is the hallucinated span (label 1), matching
:mod:`halo.labels` and :mod:`halo.detector`. Everything here is
label-source agnostic: ``y_true`` may come from silver labels, an LLM
verifier, or manual annotation -- this module does not care and does
not decide.

One design rule the numbers depend on, stated once here because it is
easy to violate silently: **training labels and evaluation labels must
not come from the same source.** The detector trains on silver labels
from the frozen NLI model, so scoring it against those same NLI
judgments would be circular -- the ``nli`` baseline would score
perfectly by construction, hiding every real difference. Evaluation
labels must come from elsewhere (an LLM verifier or human
annotation), and the report says which source produced them.

Definitions, for a set of scored spans:

- precision -- of the spans a method flagged, the fraction that were
  really hallucinated. Zero when nothing was flagged.
- recall -- of the really hallucinated spans, the fraction flagged.
- F1 -- the harmonic mean of precision and recall; zero when either
  is zero, so a method that never guesses cannot be rewarded for
  caution.
- hallucination rate -- the fraction of all spans flagged. The
  deployment-facing number: a method that flags every sentence may
  score decent recall and still be useless.
- accuracy -- the fraction of spans labelled correctly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionMetrics:
    """Span-level scores for one set of predictions.

    Attributes
    ----------
    precision : float
        Fraction of flagged spans that were hallucinated.
    recall : float
        Fraction of hallucinated spans that were flagged.
    f1 : float
        Harmonic mean of precision and recall.
    accuracy : float
        Fraction of spans labelled correctly.
    hallucination_rate : float
        Fraction of all spans flagged.
    n_spans : int
        Number of spans scored.
    n_hallucinated : int
        Number of spans labelled hallucinated.
    n_flagged : int
        Number of spans flagged by the method.
    """

    precision: float
    recall: float
    f1: float
    accuracy: float
    hallucination_rate: float
    n_spans: int
    n_hallucinated: int
    n_flagged: int

    def to_dict(self) -> dict:
        """Return the metrics as JSON-ready data.

        Returns
        -------
        dict
            Every field, keyed by name.
        """
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "hallucination_rate": self.hallucination_rate,
            "n_spans": self.n_spans,
            "n_hallucinated": self.n_hallucinated,
            "n_flagged": self.n_flagged,
        }


def _validate_binary(name: str, values: np.ndarray) -> None:
    """Reject non-binary label arrays with a readable error.

    Parameters
    ----------
    name : str
        The array's role, for the error message.
    values : numpy.ndarray
        The label array.

    Raises
    ------
    ValueError
        If any value is not 0 or 1.
    """
    if not np.isin(values, (0, 1)).all():
        bad = sorted({int(v) for v in values if v not in (0, 1)})
        raise ValueError(f"{name} must be 0 or 1; found {bad}")


def detection_metrics(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> DetectionMetrics:
    """Score predictions over one set of spans (micro average).

    Parameters
    ----------
    y_true : Sequence[int]
        True labels: 0 grounded, 1 hallucinated.
    y_pred : Sequence[int]
        Predicted labels, same length.

    Returns
    -------
    DetectionMetrics
        The scores; undefined ratios count as zero (a method that
        flags nothing has precision 0 by this convention, stated
        rather than hidden).

    Raises
    ------
    ValueError
        If the arrays differ in length, are empty, or hold values
        other than 0 and 1.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError(
            f"y_true has shape {y_true.shape} but y_pred {y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("cannot score an empty set of spans")
    _validate_binary("y_true", y_true)
    _validate_binary("y_pred", y_pred)

    positive = y_true == 1
    flagged = y_pred == 1
    true_positives = int((positive & flagged).sum())
    false_positives = int((~positive & flagged).sum())
    false_negatives = int((positive & ~flagged).sum())
    true_negatives = int((~positive & ~flagged).sum())
    n_spans = int(y_true.size)

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return DetectionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=(true_positives + true_negatives) / n_spans,
        hallucination_rate=int(flagged.sum()) / n_spans,
        n_spans=n_spans,
        n_hallucinated=int(positive.sum()),
        n_flagged=int(flagged.sum()),
    )


def best_threshold(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Pick the score cut-off that best separates two classes.

    Used to choose an ablation's cut-off on *training* labels so it is
    never fitted on the evaluation labels (which would leak the answer).
    A span is flagged when ``score < threshold`` -- low confidence or low
    grounding means hallucinated. Candidates are the observed scores, and
    the cut-off maximises F1 against the given labels.

    Parameters
    ----------
    scores : Sequence[float]
        One score per span; lower means more hallucinated.
    labels : Sequence[int]
        Binary labels per span: 0 grounded, 1 hallucinated.

    Returns
    -------
    float
        The cut-off with the best F1; the median score when no cut-off
        beats F1 0, so the flagged set is never accidentally empty.

    Raises
    ------
    ValueError
        If the inputs are empty or differ in length.
    """
    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=int)
    if score_array.shape != label_array.shape or score_array.size == 0:
        raise ValueError(
            f"scores {score_array.shape} and labels "
            f"{label_array.shape} must be non-empty and equal"
        )
    best_f1 = -1.0
    best_cut = float(np.median(score_array))
    for cut in np.unique(score_array):
        predicted = (score_array < cut).astype(int)
        tp = int(((predicted == 1) & (label_array == 1)).sum())
        fp = int(((predicted == 1) & (label_array == 0)).sum())
        fn = int(((predicted == 0) & (label_array == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_cut = float(cut)
    return best_cut


def macro_average(
    per_question: Sequence[DetectionMetrics],
) -> DetectionMetrics:
    """Average metrics across questions (macro average).

    Every question counts once, so one long answer cannot dominate the
    score. Counts sum (they describe all spans); rates average.

    Parameters
    ----------
    per_question : Sequence[DetectionMetrics]
        One entry per question.

    Returns
    -------
    DetectionMetrics
        The averaged scores.

    Raises
    ------
    ValueError
        If no question metrics are given.
    """
    if not per_question:
        raise ValueError("cannot average zero questions")
    n_questions = len(per_question)
    return DetectionMetrics(
        precision=sum(m.precision for m in per_question) / n_questions,
        recall=sum(m.recall for m in per_question) / n_questions,
        f1=sum(m.f1 for m in per_question) / n_questions,
        accuracy=sum(m.accuracy for m in per_question) / n_questions,
        hallucination_rate=(
            sum(m.hallucination_rate for m in per_question) / n_questions
        ),
        n_spans=sum(m.n_spans for m in per_question),
        n_hallucinated=sum(m.n_hallucinated for m in per_question),
        n_flagged=sum(m.n_flagged for m in per_question),
    )
