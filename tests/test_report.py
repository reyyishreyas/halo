"""Unit tests for the Markdown comparison report."""

from __future__ import annotations

import pytest

from halo.metrics import DetectionMetrics
from halo.report import ReportRow, comparison_table, save_report


def _metrics(**overrides) -> DetectionMetrics:
    values = {
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "accuracy": 0.75,
        "hallucination_rate": 0.375,
        "n_spans": 8,
        "n_hallucinated": 3,
        "n_flagged": 3,
    }
    values.update(overrides)
    return DetectionMetrics(**values)


def test_table_renders_every_row():
    rows = [
        ReportRow("detector", _metrics(), mean_latency_ms=12.34),
        ReportRow("nli", _metrics(precision=0.9), mean_latency_ms=None),
    ]
    table = comparison_table(rows)
    lines = table.splitlines()
    assert lines[0].startswith("| Method |")
    assert lines[1].startswith("| ------ |")
    assert lines[2] == (
        "| detector | 0.500 | 0.500 | 0.500 | 0.375 | 8 | 12.3 |"
    )
    assert lines[3] == ("| nli | 0.900 | 0.500 | 0.500 | 0.375 | 8 | – |")


def test_unmeasured_latency_renders_as_a_dash_not_a_zero():
    table = comparison_table([ReportRow("detector", _metrics())])
    assert "–" in table
    assert "| 0.0 |" not in table


def test_rates_render_with_three_decimals():
    metrics = _metrics(precision=2 / 3, recall=1 / 3)
    table = comparison_table([ReportRow("x", metrics)])
    assert "| 0.667 |" in table
    assert "| 0.333 |" in table


def test_empty_table_is_rejected():
    with pytest.raises(ValueError, match="empty comparison table"):
        comparison_table([])


def test_save_report_writes_markdown(tmp_path):
    table = comparison_table([ReportRow("detector", _metrics())])
    path = tmp_path / "reports" / "phase4.md"
    save_report(table, path)
    assert path.read_text() == table + "\n"
