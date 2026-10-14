"""Silver span labels from a frozen NLI model.

The detector needs span labels; hand-annotating is out of budget for a
mini project, and the proposal allows a small labeling set used only
for training. A frozen Natural Language Inference model reads each
sentence span as a *hypothesis* against the retrieved context as the
*premise*:

- entailment wins the model's three-way judgment -> grounded, label 0
- otherwise (neutral or contradiction)          -> likely
  hallucinated, label 1

The model is ``cross-encoder/nli-deberta-v3-base``: public, ungated,
about 740 MB of weights (measured from the repository file sizes), and
it doubles as the Phase 4 NLI baseline, so the
training labels never come from anything stronger than a baseline the
comparison already reports.

Two honest limitations, both by design:

- Labels are noisy. A neutral-but-reasonable inference becomes
  positive, which is why the detector trains with balanced class
  weights rather than trusting the label distribution.
- The premise is truncated to the NLI model's window (512 tokens), so
  for long contexts only the first retrieved chunks are actually read.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

# Labels follow the proposal: 0 = grounded, 1 = likely hallucinated.
GROUNDED = 0
HALLUCINATED = 1


def load_nli(model_id: str = DEFAULT_NLI_MODEL) -> Any:
    """Load the frozen NLI cross-encoder.

    Parameters
    ----------
    model_id : str
        Hugging Face model to load.

    Returns
    -------
    sentence_transformers.CrossEncoder
        The model, ready for :func:`silver_labels`.
    """
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_id)


def _label_index(model: Any, name: str) -> int:
    """Resolve a class index from the model's label names.

    Parameters
    ----------
    model : Any
        Cross-encoder exposing ``model.config.label2id`` (or
        ``config.label2id`` directly).
    name : str
        Label to find, case-insensitively.

    Returns
    -------
    int
        The column index of that label in ``predict`` output.

    Raises
    ------
    ValueError
        If the model does not publish that label.
    """
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "label2id"):
        config = model.model.config
    label2id = {
        str(label).lower(): int(index)
        for label, index in config.label2id.items()
    }
    if name not in label2id:
        raise ValueError(
            f"NLI model has no {name!r} label; found {sorted(label2id)}"
        )
    return label2id[name]


def silver_labels(
    premise: str, hypotheses: Sequence[str], model: Any
) -> np.ndarray:
    """Label each hypothesis against one premise with frozen NLI.

    Parameters
    ----------
    premise : str
        The retrieved context, e.g. the chunk texts joined with blank
        lines. The model's tokenizer truncates it to its window.
    hypotheses : Sequence[str]
        Sentence spans of the generated answer.
    model : Any
        Frozen NLI cross-encoder, from :func:`load_nli` or a test
        stub with ``predict(pairs) -> (n, n_labels)`` probabilities.

    Returns
    -------
    numpy.ndarray
        Integer labels of shape ``(len(hypotheses),)``: 0 grounded,
        1 likely hallucinated. Empty if there are no hypotheses.
    """
    if not hypotheses:
        return np.empty(0, dtype=int)
    entailment = _label_index(model, "entailment")
    probabilities = np.atleast_2d(
        np.asarray(model.predict([(premise, h) for h in hypotheses]))
    )
    if probabilities.shape[0] != len(hypotheses):
        raise ValueError(
            f"NLI model returned {probabilities.shape[0]} rows for "
            f"{len(hypotheses)} hypotheses"
        )
    judgment = probabilities.argmax(axis=1)
    # Entailment -> grounded (0); any other judgment -> 1.
    return (judgment != entailment).astype(int)


def premise_from_record(record: dict) -> str:
    """Build the NLI premise from a Phase 2 record's retrieved chunks.

    Titles are prefixed to the text because many answers refer to
    entities by name, and the title is evidence too.

    Parameters
    ----------
    record : dict
        A Phase 2 JSONL record with a ``retrieved`` list.

    Returns
    -------
    str
        Retrieved chunks joined with blank lines.

    Raises
    ------
    ValueError
        If the record holds no retrieved chunks.
    """
    retrieved = record.get("retrieved") or []
    if not retrieved:
        raise ValueError("record has no retrieved chunks to use as premise")
    return "\n\n".join(
        f"{chunk['title']}: {chunk['text']}" for chunk in retrieved
    )
