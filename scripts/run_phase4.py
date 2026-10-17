"""Phase 4: score the detector and baselines against evaluation labels.

The evaluation label source is the ``second_verifier`` -- Phi-3 prompted
"does the context support this sentence?" once per span. That is a
*different* model than the frozen NLI cross-encoder the detector trained
on, so the comparison is not circular (see :mod:`halo.metrics`). Because
the verifier defines the labels, it is not itself scored: its row would
be perfect by construction. The report says which source produced the
labels.

Every scored method is measured the same way (span-level precision,
recall, F1, hallucination rate via :func:`halo.metrics.detection_metrics`)
and timed per ``predict`` call, so accuracy and cost sit side by side.

This runs on CPU or MPS. The verifier is the only generative model
loaded; it reuses the fp16 generator at the default device.

Usage
-----
    python scripts/run_phase4.py                    # fp16.jsonl + weights
    python scripts/run_phase4.py --input run.jsonl --questions 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    from transformers import AutoTokenizer

    from halo.baselines import (
        ConfidenceThreshold,
        GroundingThreshold,
        NLIBaseline,
        NoVerification,
        SecondVerifier,
    )
    from halo.detector import SpanDetector
    from halo.generate import DEFAULT_MODEL, GenerationSettings, load_generator
    from halo.index import DEFAULT_EMBEDDER
    from halo.labels import load_nli, premises_from_record, silver_labels
    from halo.metrics import (
        DetectionMetrics,
        best_threshold,
        detection_metrics,
    )
    from halo.report import ReportRow, comparison_table, save_report
    from halo.signals import features, grounding_scores, mean_token_logprob
    from halo.spans import answer_spans
except ModuleNotFoundError as exc:  # wrong interpreter, not a code bug
    sys.exit(
        f"missing dependency ({exc.name}); run with the halo env's python:\n"
        "  conda activate halo\n"
        "or: conda run -n halo python scripts/run_phase4.py"
    )

# Ablation cut-offs are fitted on the silver training labels at run time
# (see tune_thresholds) rather than hardcoded here: a fixed guess such as
# confidence < -1.0 never fires, since mean token log-probs sit near -0.13,
# which would silently reduce the confidence ablation to the floor.


def load_run(path: str | Path) -> list[dict]:
    """Read a Phase 2 JSONL file into a list of records.

    Parameters
    ----------
    path : str or Path
        Path to a JSONL file such as ``artifacts/generations/fp16.jsonl``.

    Returns
    -------
    list of dict
        One record per line.

    Raises
    ------
    SystemExit
        If the file is missing or empty.
    """
    if not Path(path).exists():
        sys.exit(f"{path} not found. Run Phase 2 first.")
    records = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        sys.exit(f"{path} is empty")
    return records


def build_verifier_prompt(premise: str, span_text: str) -> str:
    """Render one support-check prompt for the verifier.

    Parameters
    ----------
    premise : str
        The retrieved context (all chunks joined).
    span_text : str
        The answer sentence being checked.

    Returns
    -------
    str
        A question asking whether the context supports the sentence.
    """
    return (
        "Read the context and the sentence. Does the context support "
        "the sentence? Answer yes or no.\n\n"
        f"Context:\n{premise}\n\n"
        f"Sentence: {span_text}\nAnswer:"
    )


def make_verifier(model: Any, tokenizer: Any, max_new_tokens: int = 4):
    """Build a ``verify(premise, span_text) -> bool`` callable.

    The callable reads the first generated token: ``yes`` means the
    context supports the span. It is injected into
    :class:`halo.baselines.SecondVerifier`, which owns the per-span
    loop and latency accounting.

    Parameters
    ----------
    model : Any
        Causal LM (the fp16 generator).
    tokenizer : Any
        Its tokenizer.
    max_new_tokens : int
        Token budget for the yes/no answer; 4 is plenty.

    Returns
    -------
    Callable[[str, str], bool]
        True when the verifier answers ``yes``.
    """
    import torch

    def verify(premise: str, span_text: str) -> bool:
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": build_verifier_prompt(premise, span_text),
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(
            output[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip().lower()
        return text.startswith("yes")

    return verify


def tune_thresholds(
    records: list[dict], codec: Any, encoder: Any, nli: Any
) -> tuple[float, float]:
    """Choose the two ablation cut-offs on the silver training labels.

    The ablations need a cut-off, and choosing it on the evaluation
    labels would leak the answer. Instead each cut-off is fitted here
    to the silver labels -- the very source the detector trains on -- so
    every method tunes on the same labels and none of them sees the
    verifier's judgments. The chosen values are reported, not hidden.

    Parameters
    ----------
    records : list of dict
        Phase 2 records to fit the cut-offs on.
    codec : Any
        The generating tokenizer.
    encoder : Any
        Sentence-transformers model for grounding scores.
    nli : Any
        Frozen NLI cross-encoder producing the silver labels.

    Returns
    -------
    tuple of float
        ``(conf_threshold, ground_threshold)``.
    """
    confidences: list[float] = []
    groundings: list[float] = []
    silver: list[int] = []
    for record in records:
        try:
            spans = answer_spans(
                record["answer"], record["token_ids"], codec
            )
            premises = premises_from_record(record)
        except ValueError:
            continue
        if not spans:
            continue
        span_texts = [s.text for s in spans]
        confidences.extend(
            mean_token_logprob(
                record["token_logprobs"], s.token_start, s.token_end
            )
            for s in spans
        )
        chunk_texts = [c["text"] for c in record["retrieved"]]
        groundings.extend(
            grounding_scores(encoder, span_texts, chunk_texts)
        )
        silver.extend(
            int(v) for v in silver_labels(premises, span_texts, nli)
        )
    conf = np.asarray(confidences, dtype=np.float64)
    ground = np.asarray(groundings, dtype=np.float64)
    labels = np.asarray(silver, dtype=int)
    return best_threshold(conf, labels), best_threshold(ground, labels)


def build_methods(
    encoder: Any,
    nli: Any,
    verifier: Any,
    conf_threshold: float,
    ground_threshold: float,
) -> list:
    """Construct every method to score, verifier excluded.

    Parameters
    ----------
    encoder : Any
        Sentence-transformers model for grounding.
    nli : Any
        Frozen NLI cross-encoder for the ``nli`` baseline.
    verifier : Any
        ``verify(premise, span) -> bool`` callable.
    conf_threshold : float
        Cut-off for ``confidence_only``.
    ground_threshold : float
        Cut-off for ``grounding_only``.

    Returns
    -------
    list
        One :class:`SpanBaseline` (or the detector wrapper) per method.
    """
    return [
        NoVerification(),
        ConfidenceThreshold(conf_threshold),
        GroundingThreshold(ground_threshold, encoder),
        NLIBaseline(nli),
        SecondVerifier(verifier),
    ]


def score_records(
    records: list[dict],
    codec: Any,
    encoder: Any,
    detector: SpanDetector,
    methods: list,
) -> tuple[dict[str, DetectionMetrics], dict[str, float]]:
    """Score every method against the verifier labels.

    Parameters
    ----------
    records : list of dict
        Phase 2 records to evaluate.
    codec : Any
        The generating tokenizer (aligns spans to tokens).
    encoder : Any
        Sentence-transformers model for grounding features.
    detector : SpanDetector
        The fitted HALO detector.
    methods : list
        Methods to score; the last is assumed to be the
        ``SecondVerifier`` that defines the evaluation labels.

    Returns
    -------
    tuple of dict
        ``(metrics_by_method, latency_ms_by_method)`` keyed by method
        name; the verifier appears only in latency (it defines labels).
    """
    verifier = methods[-1]
    scored = methods[:-1]
    preds: dict[str, list[int]] = {m.name: [] for m in scored}
    preds[detector_name()] = []
    truths: list[int] = []
    # The verifier is not scored but its latency is still measured, so
    # it gets its own bucket alongside the scored methods.
    latency: dict[str, list[float]] = {name: [] for name in preds}
    latency[verifier.name] = []

    for record in records:
        try:
            spans = answer_spans(
                record["answer"], record["token_ids"], codec
            )
        except ValueError as exc:
            print(f"  skipping {record['id']}: {exc}")
            continue
        if not spans:
            continue

        # Evaluation labels from the verifier (timed as the reference).
        start = time.perf_counter()
        y_true = verifier.predict(record, spans)
        latency[verifier.name].append(
            (time.perf_counter() - start) * 1000.0 / max(len(spans), 1)
        )
        truths.extend(int(v) for v in y_true)

        # Detector features (confidence + grounding) and its labels.
        span_texts = [s.text for s in spans]
        confidences = [
            mean_token_logprob(
                record["token_logprobs"], s.token_start, s.token_end
            )
            for s in spans
        ]
        chunk_texts = [c["text"] for c in record["retrieved"]]
        groundings = grounding_scores(encoder, span_texts, chunk_texts)
        X = features(confidences, groundings)

        for method in scored:
            start = time.perf_counter()
            out = method.predict(record, spans)
            latency[method.name].append(
                (time.perf_counter() - start) * 1000.0 / max(len(spans), 1)
            )
            preds[method.name].extend(int(v) for v in out)

        start = time.perf_counter()
        preds[detector_name()].extend(int(v) for v in detector.predict(X))
        latency[detector_name()].append(
            (time.perf_counter() - start) * 1000.0 / max(len(spans), 1)
        )

    metrics = {
        name: detection_metrics(truths, values)
        for name, values in preds.items()
        if values
    }
    mean_latency = {
        name: float(np.mean(times)) for name, times in latency.items() if times
    }
    return metrics, mean_latency


def detector_name() -> str:
    """Return the detector's row label.

    Returns
    -------
    str
        The name used in the report for the HALO detector.
    """
    return "detector (HALO)"


def main(
    input_path: str,
    weights_path: str,
    questions: int | None,
    conf_threshold: float | None,
    ground_threshold: float | None,
    output: str,
) -> None:
    """Run the full Phase 4 comparison and write the report.

    Parameters
    ----------
    input_path : str
        Phase 2 JSONL to evaluate.
    weights_path : str
        Trained detector weights JSON.
    questions : int or None
        Cap the number of questions (None evaluates all).
    conf_threshold : float or None
        Cut-off for the confidence ablation. None tunes it on the
        silver training labels via :func:`tune_thresholds`.
    ground_threshold : float or None
        Cut-off for the grounding ablation. None tunes it likewise.
    output : str
        Destination for the Markdown report.
    """
    from sentence_transformers import SentenceTransformer

    from halo.device import default_device

    records = load_run(input_path)
    if questions is not None:
        records = records[:questions]
    print(f"evaluating {len(records)} questions ({input_path})")

    if not Path(weights_path).exists():
        sys.exit(
            f"{weights_path} not found. Train first:\n"
            "  python scripts/train_detector.py"
        )
    detector = SpanDetector.load(weights_path)

    codec = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    encoder = SentenceTransformer(DEFAULT_EMBEDDER)
    nli = load_nli()

    # Fit the ablation cut-offs on the silver training labels so they
    # are never chosen on the evaluation (verifier) labels.
    if conf_threshold is None or ground_threshold is None:
        tuned_conf, tuned_ground = tune_thresholds(
            records, codec, encoder, nli
        )
        conf_threshold = (
            conf_threshold if conf_threshold is not None else tuned_conf
        )
        ground_threshold = (
            ground_threshold if ground_threshold is not None else tuned_ground
        )
        print(
            f"cut-offs tuned on silver labels: "
            f"confidence < {conf_threshold:.3f}, "
            f"grounding < {ground_threshold:.3f}"
        )

    # The verifier reuses the fp16 generator on the default device.
    # Only the model + tokenizer matter here; the run-config fields
    # (top_k, questions, seed, output) are required by the dataclass
    # but unused by the verifier.
    settings = GenerationSettings(
        precision="fp16",
        model_id=DEFAULT_MODEL,
        top_k=5,
        max_new_tokens=4,
        questions=1,
        seed=0,
        output="artifacts/generations/_verifier.jsonl",
    )
    model, tokenizer = load_generator(settings, device=default_device())
    verifier = make_verifier(model, tokenizer)

    methods = build_methods(
        encoder, nli, verifier, conf_threshold, ground_threshold
    )
    metrics, latency = score_records(
        records, codec, encoder, detector, methods
    )

    # The verifier defines the labels, so it is a latency-only row and
    # is not scored (it would be perfect by construction).
    rows = [
        ReportRow(
            method=name,
            metrics=m,
            mean_latency_ms=latency.get(name),
        )
        for name, m in metrics.items()
    ]
    table = comparison_table(rows)
    report = _report_header(conf_threshold, ground_threshold) + "\n\n" + table

    # The verifier defines the labels (so it is not scored), but its
    # cost is the reference the detector is trying to avoid, so report
    # it explicitly under the table.
    verifier_ms = latency.get("second_verifier")
    if verifier_ms is not None:
        report += (
            "\n\nLabel source cost: `second_verifier` "
            f"{verifier_ms:.1f} ms/span (not scored — it defines the "
            "labels)."
        )
    save_report(report, output)
    print("\n" + report)
    print(f"\nreport written to {output}")


def _report_header(conf_threshold: float, ground_threshold: float) -> str:
    """Build the report preamble naming the label source and cut-offs.

    Parameters
    ----------
    conf_threshold : float
        Cut-off used for the confidence ablation.
    ground_threshold : float
        Cut-off used for the grounding ablation.

    Returns
    -------
    str
        Markdown stating the label source and the ablation cut-offs.
    """
    return (
        "## Phase 4 — method comparison\n\n"
        "Evaluation labels come from the **second verifier** (Phi-3 asked "
        "\"does the context support this sentence?\" once per span), a "
        "different model than the NLI cross-encoder the detector trained "
        "on, so the comparison is not circular. The verifier defines the "
        "labels and is therefore not scored; its latency is reported "
        "separately below.\n\n"
        "Ablation cut-offs were fitted on the **silver training labels** "
        "(the same source the detector fits on), never on the evaluation "
        f"labels: confidence < {conf_threshold:.3f}, "
        f"grounding < {ground_threshold:.3f}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0]
    )
    parser.add_argument(
        "--input", default="artifacts/generations/fp16.jsonl"
    )
    parser.add_argument(
        "--weights", default="artifacts/detector/weights.json"
    )
    parser.add_argument(
        "--questions", type=int, default=None,
        help="cap the number of questions evaluated",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=None,
        help="override the confidence cut-off (default: tune on silver)",
    )
    parser.add_argument(
        "--ground-threshold",
        type=float,
        default=None,
        help="override the grounding cut-off (default: tune on silver)",
    )
    parser.add_argument(
        "--output", default="artifacts/reports/phase4.md"
    )
    args = parser.parse_args()
    main(
        args.input,
        args.weights,
        args.questions,
        args.conf_threshold,
        args.ground_threshold,
        args.output,
    )
