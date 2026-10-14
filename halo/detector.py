"""The detector: standardized logistic regression over two signals.

A span is described by two numbers (see :mod:`halo.signals`): its mean
token log-probability and its maximum cosine similarity to the
retrieved chunks. Two things must happen before a regression can weigh
them fairly:

- **Standardization** (``StandardScaler``). Log-probabilities and
  cosines live on different scales with different spreads; scaling
  both to zero mean and unit variance makes the learned coefficients
  comparable, so a larger absolute weight really does mean a more
  influential signal.
- **Balanced class handling** (``class_weight="balanced"``). The
  proposal requires it, and the silver labels make it necessary:
  grounded spans usually outnumber hallucinated ones, and without
  balancing the regression could buy accuracy by predicting
  "grounded" everywhere. Balanced weights price each class by its
  inverse frequency instead.

The result maps the two standardized values to P(hallucinated). It is
deliberately simple: fast, interpretable, and it loads no neural model
at scoring time. Models save as readable JSON, not pickles.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class SpanDetector:
    """Logistic-regression detector over confidence and grounding.

    Fitting twice replaces the previous fit; the object is a model,
    not a growing cache.

    Attributes
    ----------
    threshold : float
        Probability at or above which a span is called hallucinated.
    scaler_ : sklearn.preprocessing.StandardScaler
        The fitted standardizer (available after ``fit`` or
        ``from_dict``).
    model_ : sklearn.linear_model.LogisticRegression
        The fitted regression (available after ``fit`` or
        ``from_dict``).

    Parameters
    ----------
    threshold : float
        Decision threshold for :meth:`predict`, default 0.5.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        self.threshold = threshold
        self.scaler_: StandardScaler | None = None
        self.model_: LogisticRegression | None = None

    def fit(self, X: np.ndarray, y: Sequence[int]) -> SpanDetector:
        """Standardize the features and fit the regression.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix ``(n_spans, 2)``: confidence, grounding.
        y : Sequence[int]
            Silver labels per span: 0 grounded, 1 hallucinated.

        Returns
        -------
        SpanDetector
            ``self``, fitted.

        Raises
        ------
        ValueError
            If ``X`` is not two-column, the lengths disagree, or the
            labels contain a single class -- a detector that has only
            ever seen one answer cannot learn a boundary.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError(f"X must have shape (n_spans, 2), got {X.shape}")
        if len(X) != len(y):
            raise ValueError(f"{len(X)} spans but {len(y)} labels")
        if np.unique(y).size < 2:
            raise ValueError(
                "detector training needs both grounded and "
                f"hallucinated spans, got classes {np.unique(y).tolist()}"
            )
        self.scaler_ = StandardScaler()
        standardized = self.scaler_.fit_transform(X)
        self.model_ = LogisticRegression(
            class_weight="balanced", max_iter=1000
        ).fit(standardized, y)
        return self

    def _require_fit(self) -> tuple[StandardScaler, LogisticRegression]:
        """Return the fitted parts, or explain that fit is missing.

        Raises
        ------
        RuntimeError
            If the detector has not been fitted or loaded.
        """
        if self.scaler_ is None or self.model_ is None:
            raise RuntimeError(
                "detector is not fitted; call fit() or load() first"
            )
        return self.scaler_, self.model_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Hallucination probability per span.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix ``(n_spans, 2)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_spans, 2)``; column 1 is P(hallucinated).
        """
        scaler, model = self._require_fit()
        return model.predict_proba(scaler.transform(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Call each span grounded (0) or hallucinated (1).

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix ``(n_spans, 2)``.

        Returns
        -------
        numpy.ndarray
            Integer labels, shape ``(n_spans,)``.
        """
        probability = self.predict_proba(X)[:, 1]
        return (probability >= self.threshold).astype(int)

    def to_dict(self) -> dict[str, Any]:
        """Export the fitted model as JSON-ready data.

        Returns
        -------
        dict[str, Any]
            Threshold, the scaler's mean and scale, and the
            regression's coefficients and intercept.

        Raises
        ------
        RuntimeError
            If the detector has not been fitted or loaded.
        """
        scaler, model = self._require_fit()
        return {
            "threshold": self.threshold,
            "features": ["mean_token_logprob", "max_cosine"],
            "classes": [int(c) for c in model.classes_],
            "scaler": {
                "mean": [float(v) for v in scaler.mean_],
                "scale": [float(v) for v in scaler.scale_],
            },
            "coefficients": [[float(v) for v in row] for row in model.coef_],
            "intercept": [float(v) for v in model.intercept_],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpanDetector:
        """Rebuild a fitted detector from :meth:`to_dict` output.

        No pickle is read: the scaler's two vectors and the
        regression's coefficients fully determine the prediction.

        Parameters
        ----------
        data : dict[str, Any]
            Previously exported model data.

        Returns
        -------
        SpanDetector
            Fitted, with predictions identical to the original.
        """
        detector = cls(threshold=data["threshold"])
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(data["scaler"]["mean"])
        scaler.scale_ = np.asarray(data["scaler"]["scale"])
        scaler.n_features_in_ = len(data["scaler"]["mean"])

        model = LogisticRegression()
        model.coef_ = np.asarray(data["coefficients"])
        model.intercept_ = np.asarray(data["intercept"])
        model.classes_ = np.asarray(data["classes"])
        model.n_features_in_ = model.coef_.shape[1]
        model.n_iter_ = np.array([0])

        detector.scaler_ = scaler
        detector.model_ = model
        return detector

    def save(self, path: str | Path) -> None:
        """Write the fitted model to a JSON file.

        Parameters
        ----------
        path : str or Path
            Destination; parent directories are created.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> SpanDetector:
        """Read a fitted model written by :meth:`save`.

        Parameters
        ----------
        path : str or Path
            The JSON file.

        Returns
        -------
        SpanDetector
            Fitted, with predictions identical to the saved model.
        """
        return cls.from_dict(json.loads(Path(path).read_text("utf-8")))
