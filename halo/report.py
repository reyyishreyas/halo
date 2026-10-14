"""Render the Phase 4 method comparison as a Markdown report.

A report row is one method's scores plus the latency of using it --
the proposal compares methods on accuracy *and* cost, and latency is
the cost the project can measure without extra infrastructure: wall
time per ``predict`` call, averaged over the scored questions, or
``None`` until such a run exists.

Tables render from whatever numbers they are handed; nothing here
computes scores (that is :mod:`halo.metrics`) and nothing here invents
values. A ``None`` latency renders as an em dash rather than a zero,
because zero would be a measured claim.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from halo.metrics import DetectionMetrics


@dataclass(frozen=True)
class ReportRow:
    """One method's line in the comparison table.

    Attributes
    ----------
    method : str
        The method's name, e.g. ``"detector"`` or ``"nli"``.
    metrics : DetectionMetrics
        Scores on the evaluation label set.
    mean_latency_ms : float or None
        Mean wall time of one ``predict`` call over the scored
        questions; ``None`` when it has not been measured yet.
    """

    method: str
    metrics: DetectionMetrics
    mean_latency_ms: float | None = None


def _format(value: float) -> str:
    """Render a rate to three decimals.

    Parameters
    ----------
    value : float
        A rate or ratio.

    Returns
    -------
    str
        The value rounded to three decimal places.
    """
    return f"{value:.3f}"


def comparison_table(rows: Sequence[ReportRow]) -> str:
    """Render the method comparison as a Markdown table.

    Row order is the caller's, so a report can list methods cheapest
    to most expensive without this module deciding the story.

    Parameters
    ----------
    rows : Sequence[ReportRow]
        One row per method.

    Returns
    -------
    str
        Markdown table with per-method precision, recall, F1,
        hallucination rate, span counts and mean latency.

    Raises
    ------
    ValueError
        If no rows are given.
    """
    if not rows:
        raise ValueError("cannot render an empty comparison table")
    lines = [
        "| Method | Precision | Recall | F1 | Hallucination rate "
        "| Spans | Latency (ms) |",
        "| ------ | --------- | ------ | -- | ----------------- "
        "| ----- | ------------ |",
    ]
    for row in rows:
        latency = (
            f"{row.mean_latency_ms:.1f}"
            if row.mean_latency_ms is not None
            else "–"
        )
        metrics = row.metrics
        lines.append(
            f"| {row.method} "
            f"| {_format(metrics.precision)} "
            f"| {_format(metrics.recall)} "
            f"| {_format(metrics.f1)} "
            f"| {_format(metrics.hallucination_rate)} "
            f"| {metrics.n_spans} "
            f"| {latency} |"
        )
    return "\n".join(lines)


def save_report(markdown: str, path: str | Path) -> None:
    """Write a Markdown report to a file.

    Parameters
    ----------
    markdown : str
        The rendered report, e.g. from :func:`comparison_table`.
    path : str or Path
        Destination; parent directories are created.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown + "\n", encoding="utf-8")
