"""Phase 3: fit the hallucination detector from Phase 2 generations.

Reads the fp16 JSONL (the reference run), labels each answer sentence
against its retrieved context with a frozen NLI model (silver labels),
computes the two signals HALO is built on -- mean token log-probability
and max cosine grounding -- and fits the standardized logistic
regression. The fitted model is written as JSON (no pickle), so the
weights are inspectable and loadable anywhere.

This runs on CPU. The NLI cross-encoder (nli-deberta-v3-base) is the
only model loaded; the generating model is not needed here.

Usage
-----
    python scripts/train_detector.py                       # fp16.jsonl
    python scripts/train_detector.py --input path/to/run.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    from transformers import AutoTokenizer

    from halo.detector import SpanDetector
    from halo.generate import DEFAULT_MODEL
    from halo.index import DEFAULT_EMBEDDER
    from halo.labels import load_nli, premise_from_record, silver_labels
    from halo.signals import features, grounding_scores, mean_token_logprob
    from halo.spans import answer_spans
except ModuleNotFoundError as exc:  # wrong interpreter, not a code bug
    sys.exit(
        f"missing dependency ({exc.name}); run with the halo env's python:\n"
        "  conda activate halo\n"
        "or: conda run -n halo python scripts/train_detector.py"
    )


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
        sys.exit(
            f"{path} not found. Run Phase 2 first:\n"
            "  python scripts/run_precision_local.py --questions 3"
        )
    records = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        sys.exit(f"{path} is empty")
    return records


def main(input_path: str, output: str) -> None:
    """Fit the detector and write its weights.

    Parameters
    ----------
    input_path : str
        Phase 2 JSONL to train from.
    output : str
        Destination for the weights JSON.
    """
    from sentence_transformers import SentenceTransformer

    records = load_run(input_path)
    print(f"training from {len(records)} generations ({input_path})")

    # The generating tokenizer (Phi-3), NOT the retrieval tokenizer:
    # the token_ids in the JSONL are Phi-3 tokens, and span->token
    # alignment only makes sense with the tokenizer that produced them.
    codec = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    nli = load_nli()
    encoder = SentenceTransformer(DEFAULT_EMBEDDER)

    all_features: list[np.ndarray] = []
    all_labels: list[int] = []
    n_spans = 0

    for record in records:
        try:
            premise = premise_from_record(record)
            spans = answer_spans(
                record["answer"], record["token_ids"], codec
            )
        except ValueError as exc:
            print(f"  skipping {record['id']}: {exc}")
            continue
        if not spans:
            continue

        span_texts = [span.text for span in spans]
        confidences = [
            mean_token_logprob(
                record["token_logprobs"], span.token_start, span.token_end
            )
            for span in spans
        ]
        chunk_texts = [c["text"] for c in record["retrieved"]]
        groundings = grounding_scores(encoder, span_texts, chunk_texts)
        labels = silver_labels(premise, span_texts, nli)

        all_features.append(features(confidences, groundings))
        all_labels.extend(int(label) for label in labels)
        n_spans += len(spans)

    if not all_features:
        sys.exit("no usable spans were produced; cannot train")

    X = np.vstack(all_features)
    y = np.asarray(all_labels, dtype=int)
    hallucinated_rate = float((y == 1).mean())
    print(f"\n{n_spans} spans total")
    print(f"silver hallucination rate: {hallucinated_rate:.3f}")

    detector = SpanDetector().fit(X, y)

    # Agreement with silver labels on the training data, reported as
    # exactly that -- not as accuracy against ground truth.
    predicted = detector.predict(X)
    agreement = float((predicted == y).mean())
    print(f"agreement with silver labels (train, in-sample): {agreement:.3f}")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    detector.save(output)
    print(f"\nweights written to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--input",
        default="artifacts/generations/fp16.jsonl",
        help="Phase 2 JSONL to train from",
    )
    parser.add_argument(
        "--output",
        default="artifacts/detector/weights.json",
        help="destination for the weights JSON",
    )
    args = parser.parse_args()
    main(args.input, args.output)
