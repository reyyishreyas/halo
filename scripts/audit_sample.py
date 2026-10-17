"""Audit detector calls with an independent AI judgement.

The Phase 4 table reports one F1 over every span, which says nothing
about *which* calls are right or whether the verifier labels deserve to
be trusted. This tool prints sampled spans as readable cards -- question,
retrieved context, generated answer, each sentence -- so any AI (or a
human) can answer "does the context support this sentence?" before
seeing what any model said, then compare its verdicts against theirs.

Two passes, deliberately separate:

- ``--blind``  cards only. No detector, NLI, or verifier verdicts and no
  large models loaded, just the tokenizer to cut the answer into
  sentences. Answer these first: showing a model's call alongside the
  question anchors the judge on it.
- default      the same cards (sampling is a fixed stride over the file,
  so both passes cover identical spans) with every verdict and an
  agreement tally.

The reveal computes verdicts through the same classes Phase 4 scores
with -- :class:`halo.baselines.NLIBaseline` and
:class:`halo.baselines.SecondVerifier` over ``features`` and the fitted
detector -- so an audit card reproduces exactly what the report counts.

Usage
-----
    # 1. blind cards for the judging AI
    python scripts/audit_sample.py --blind --n 6

    # 2. verdicts for the same spans, to compare against its answers
    python scripts/audit_sample.py --n 6

    # 3. same, as JSON for another script
    python scripts/audit_sample.py --n 6 --json /tmp/audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from run_phase4 import load_run, make_verifier

    from halo.baselines import NLIBaseline, SecondVerifier
    from halo.detector import SpanDetector
    from halo.generate import DEFAULT_MODEL, GenerationSettings, load_generator
    from halo.index import DEFAULT_EMBEDDER
    from halo.signals import features, grounding_scores, mean_token_logprob
    from halo.spans import Span, answer_spans
except ModuleNotFoundError as exc:  # wrong interpreter, not a code bug
    sys.exit(
        f"missing dependency ({exc.name}); run with the halo env's python:\n"
        "  conda activate halo\n"
        "or: conda run -n halo python scripts/audit_sample.py"
    )


def select_records(records: list[dict], n: int) -> list[dict]:
    """Pick ``n`` records spread across the file by fixed stride.

    Deterministic on purpose: a random sample could not be matched
    between the blind pass and the reveal, so the two passes would
    cover different spans and no answer could be compared.

    Parameters
    ----------
    records : list of dict
        All Phase 2 records, in file order.
    n : int
        How many to select.

    Returns
    -------
    list of dict
        Up to ``n`` records, evenly spaced from the first.
    """
    if n <= 0 or not records:
        return []
    step = max(1, len(records) // n)
    return records[::step][:n]


def plan_spans(
    records: list[dict], codec: Any
) -> list[tuple[dict, list[Span]]]:
    """Cut every sampled answer into sentence spans.

    Parameters
    ----------
    records : list of dict
        Sampled Phase 2 records.
    codec : Any
        The generating tokenizer, used to align tokens to characters.

    Returns
    -------
    list of tuple
        ``(record, spans)`` for records whose answer has at least one
        sentence; answers with none are skipped as Phase 4 skips them.
    """
    planned: list[tuple[dict, list[Span]]] = []
    for record in records:
        try:
            spans = answer_spans(record["answer"], record["token_ids"], codec)
        except ValueError:
            continue
        if spans:
            planned.append((record, spans))
    return planned


def format_card(
    index: int, total: int, record: dict, spans: list[Span]
) -> str:
    """Render one blind audit card, with no model verdict on it.

    Parameters
    ----------
    index : int
        0-based position of this card in the sample.
    total : int
        How many cards the sample holds.
    record : dict
        Phase 2 record: ``id``, ``question``, ``retrieved``, ``answer``.
    spans : list of Span
        Sentences of the generated answer.

    Returns
    -------
    str
        Markdown card ending in a yes/no prompt per sentence.
    """
    lines = [
        f"### Card {index + 1} / {total}  (id: {record.get('id', '?')})",
        "",
        f"**Question:** {record.get('question', '?')}",
        "",
        "**Retrieved context:**",
    ]
    for rank, chunk in enumerate(record.get("retrieved") or [], 1):
        score = chunk.get("score", float("nan"))
        lines.append(f"{rank}. *{chunk.get('title', '')}* (score {score:.3f})")
        lines.append(f"   {chunk.get('text', '')}")
    lines += ["", f"**Generated answer:** {record.get('answer', '')}", ""]
    for position, span in enumerate(spans, 1):
        lines += [
            f'**Sentence {position}:** "{span.text}"',
            "",
            "Judgement (answer before the reveal): does the retrieved "
            "context support this sentence? **yes / no**",
            "",
        ]
    return "\n".join(lines)


def verdicts_for(
    record: dict,
    spans: list[Span],
    detector: SpanDetector,
    encoder: Any,
    nli: Any,
    verifier: Any,
) -> list[dict]:
    """Recompute every model's verdict for one record's spans.

    Runs the same arithmetic as ``run_phase4.score_records``: signals
    into ``features``, the detector's probabilities from them, and the
    silver / evaluation labels from the very baselines Phase 4 scores.

    Parameters
    ----------
    record : dict
        Phase 2 record.
    spans : list of Span
        Sentences of its generated answer.
    detector : SpanDetector
        Fitted HALO detector.
    encoder : Any
        Sentence-transformers model for grounding scores.
    nli : Any
        Frozen NLI cross-encoder (the silver training labels).
    verifier : Any
        ``verify(premise, span) -> bool`` callable; the evaluation
        label source.

    Returns
    -------
    list of dict
        One dict per span: text, confidence, grounding, silver label,
        verifier label, detector probability and flag.
    """
    span_texts = [span.text for span in spans]
    confidences = [
        mean_token_logprob(
            record["token_logprobs"], span.token_start, span.token_end
        )
        for span in spans
    ]
    chunk_texts = [chunk["text"] for chunk in record["retrieved"]]
    groundings = list(grounding_scores(encoder, span_texts, chunk_texts))
    X = features(confidences, groundings)
    probabilities = detector.predict_proba(X)[:, 1]
    flags = detector.predict(X)

    silver = NLIBaseline(nli).predict(record, spans)
    truth = SecondVerifier(verifier).predict(record, spans)

    return [
        {
            "id": record.get("id"),
            "question": record.get("question"),
            "text": span.text,
            "confidence": confidences[position],
            "grounding": groundings[position],
            # 1 = hallucinated, matching Phase 4's positive class.
            "silver_hallucinated": int(silver[position]),
            "verifier_hallucinated": int(truth[position]),
            "detector_prob": float(probabilities[position]),
            "detector_flag": int(flags[position]),
        }
        for position, span in enumerate(spans)
    ]


def print_verdicts(index: int, rows: list[dict]) -> None:
    """Print the reveal table for one card.

    Parameters
    ----------
    index : int
        0-based card position, shown 1-based in the heading.
    rows : list of dict
        Per-span verdicts from :func:`verdicts_for`.
    """
    print(f"### Verdicts — card {index + 1}\n")
    print("| # | Sentence | conf | ground | silver | verifier | detector |")
    print("| - | -------- | ---- | ------ | ------ | -------- | -------- |")
    for position, row in enumerate(rows, 1):
        verdict = "**HALL**" if row["verifier_hallucinated"] else "ok"
        flag = (
            f"**FLAG** {row['detector_prob']:.2f}"
            if row["detector_flag"]
            else f"ok {row['detector_prob']:.2f}"
        )
        print(
            f'| {position} | "{row["text"][:70]}" '
            f"| {row['confidence']:.3f} | {row['grounding']:.3f} "
            f"| {row['silver_hallucinated']} | {verdict} | {flag} |"
        )
    print()


def print_tally(rows: list[dict]) -> None:
    """Print how often each label source agrees with the verifier.

    Parameters
    ----------
    rows : list of dict
        All per-span verdicts in the sample.
    """
    if not rows:
        return
    pairs = [
        ("detector vs verifier", "detector_flag", "verifier_hallucinated"),
        (
            "silver (NLI) vs verifier",
            "silver_hallucinated",
            "verifier_hallucinated",
        ),
        ("detector vs silver", "detector_flag", "silver_hallucinated"),
    ]
    print(
        "\n**Agreement on this sample** (read the cards, not just "
        "these numbers):\n"
    )
    for label, left, right in pairs:
        hits = sum(row[left] == row[right] for row in rows)
        print(f"- {label}: {hits}/{len(rows)} = {hits / len(rows):.3f}")
    positives = sum(row["verifier_hallucinated"] for row in rows)
    print(
        f"- verifier positives: {positives}/{len(rows)} "
        f"= {positives / len(rows):.3f}"
    )


def main(
    input_path: str,
    weights_path: str,
    n: int,
    blind: bool,
    json_path: str | None,
) -> None:
    """Sample spans, print blind cards, then reveal the verdicts.

    Parameters
    ----------
    input_path : str
        Phase 2 JSONL to sample from.
    weights_path : str
        Trained detector weights JSON (reveal pass only).
    n : int
        How many records to sample.
    blind : bool
        True prints cards only and loads no large model.
    json_path : str or None
        If given, write the reveal verdicts as JSON.
    """
    from transformers import AutoTokenizer

    records = load_run(input_path)
    sample = select_records(records, n)
    if not sample:
        sys.exit("nothing sampled; give --n a positive value")

    codec = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    planned = plan_spans(sample, codec)
    if not planned:
        sys.exit("no sampled answer contained a sentence")

    print(
        f"# Audit cards — {len(planned)} spans from {len(sample)} "
        f"records of {input_path}\n"
        "Sampling is a fixed stride, so a `--blind` run and a reveal "
        "run cover identical spans.\n"
    )
    for position, (record, spans) in enumerate(planned):
        print(format_card(position, len(planned), record, spans))
        print("---\n")
    if blind:
        print(
            "\n*Blind mode: verdicts withheld. Record judgements, then "
            "rerun without `--blind` to reveal and tally.*"
        )
        return

    # Reveal: load every model and reproduce Phase 4's verdicts.
    from sentence_transformers import SentenceTransformer

    from halo.device import default_device
    from halo.labels import load_nli

    print("\n# Reveal — model verdicts\n")
    detector = SpanDetector.load(weights_path)
    encoder = SentenceTransformer(DEFAULT_EMBEDDER)
    nli = load_nli()
    # Only model + tokenizer matter to the verifier; the run-config
    # fields are required by the dataclass but unused (as in Phase 4).
    settings = GenerationSettings(
        precision="fp16",
        model_id=DEFAULT_MODEL,
        top_k=5,
        max_new_tokens=4,
        questions=1,
        seed=0,
        output="artifacts/generations/_audit.jsonl",
    )
    model, tokenizer = load_generator(settings, device=default_device())
    verifier = make_verifier(model, tokenizer)

    all_rows: list[dict] = []
    for position, (record, spans) in enumerate(planned):
        rows = verdicts_for(record, spans, detector, encoder, nli, verifier)
        print_verdicts(position, rows)
        all_rows.extend(rows)

    print_tally(all_rows)
    if json_path:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).write_text(
            json.dumps(all_rows, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nverdicts written to {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", default="artifacts/generations/fp16.jsonl")
    parser.add_argument("--weights", default="artifacts/detector/weights.json")
    parser.add_argument(
        "--n", type=int, default=6, help="records to sample (default 6)"
    )
    parser.add_argument(
        "--blind",
        action="store_true",
        help="print cards only, withholding every model verdict",
    )
    parser.add_argument("--json", default=None, help="dump verdicts to path")
    args = parser.parse_args()
    main(args.input, args.weights, args.n, args.blind, args.json)
