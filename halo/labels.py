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

Each sentence is judged against the retrieved chunks one at a time,
not against all of them concatenated. The cross-encoder reads at most
512 tokens per pair; five HotpotQA chunks joined run well past that,
so a single joined premise would be silently truncated to its first
chunk or two and the rest of the evidence would never be seen. One
chunk per pair keeps every comparison inside the window, and a span is
called grounded when *any* chunk entails it -- the same max-over-chunks
rule the grounding signal already uses.

One honest limitation, by design: labels are noisy. A
neutral-but-reasonable inference becomes positive, which is why the
detector trains with balanced class weights rather than trusting the
label distribution.
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
    premise: str | Sequence[str],
    hypotheses: Sequence[str],
    model: Any,
) -> np.ndarray:
    """Label each hypothesis against the context with frozen NLI.

    Parameters
    ----------
    premise : str or Sequence[str]
        The evidence. A single string is used as one premise. A list
        is treated as separate chunk premises: each hypothesis is
        judged against every chunk, and grounded if any chunk entails
        it. Passing one premise per chunk keeps every pair inside the
        model's 512-token window (see the module docstring).
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
    premises = [premise] if isinstance(premise, str) else list(premise)
    if not premises:
        raise ValueError("silver_labels needs at least one premise")
    entailment = _label_index(model, "entailment")
    # (hypothesis, chunk) order so a single premise yields one row per
    # hypothesis, matching the original single-premise layout.
    pairs = [
        (chunk, hypothesis)
        for hypothesis in hypotheses
        for chunk in premises
    ]
    probabilities = np.atleast_2d(np.asarray(model.predict(pairs)))
    expected = len(hypotheses) * len(premises)
    if probabilities.shape[0] != expected:
        raise ValueError(
            f"NLI model returned {probabilities.shape[0]} rows for "
            f"{expected} premise-hypothesis pairs"
        )
    judgment = probabilities.argmax(axis=1)
    # Grounded when any premise's judgment is entailment (0); else 1.
    entails = (judgment == entailment).reshape(len(hypotheses), len(premises))
    return (~entails.any(axis=1)).astype(int)


def _titled_chunks(record: dict) -> list[str]:
    """Build one title-prefixed string per retrieved chunk.

    Titles are prefixed to the text because many answers refer to
    entities by name, and the title is evidence too.

    Parameters
    ----------
    record : dict
        A Phase 2 JSONL record with a ``retrieved`` list.

    Returns
    -------
    list of str
        One titled string per retrieved chunk.

    Raises
    ------
    ValueError
        If the record holds no retrieved chunks.
    """
    retrieved = record.get("retrieved") or []
    if not retrieved:
        raise ValueError("record has no retrieved chunks to use as premise")
    return [f"{chunk['title']}: {chunk['text']}" for chunk in retrieved]


def premises_from_record(record: dict, joint_top: int = 2) -> list[str]:
    """Build NLI premises from a record's retrieved chunks.

    Returns one premise per chunk, so each comparison stays inside the
    model's window (see the module docstring). When ``joint_top`` is
    at least 2 and that many chunks exist, one extra premise joins the
    top ``joint_top`` chunks. Multi-hop questions (HotpotQA's whole
    design) are entailed by two paragraphs *together* but by neither
    alone, so the joint premise gives such claims a premise that can
    actually entail them. A span stays grounded when any premise
    entails it, so the joint premise can only recover grounded spans,
    never invent hallucinations.

    Parameters
    ----------
    record : dict
        A Phase 2 JSONL record with a ``retrieved`` list.
    joint_top : int
        How many leading chunks to also judge jointly. 0 disables the
        joint premise; default 2 matches HotpotQA's two gold
        paragraphs.

    Returns
    -------
    list of str
        One titled premise per chunk, plus the joint premise if any.

    Raises
    ------
    ValueError
        If the record holds no retrieved chunks.
    """
    premises = _titled_chunks(record)
    if joint_top >= 2 and len(premises) >= joint_top:
        premises = premises + ["\n\n".join(premises[:joint_top])]
    return premises


def premise_from_record(record: dict) -> str:
    """Build a single joined NLI premise from a record's chunks.

    Kept for callers that need one string (for example the
    second-verifier baseline, which passes a whole context to an
    injected callable). Prefer :func:`premises_from_record` for NLI
    judging, which avoids the model's context-window truncation.

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
    return "\n\n".join(_titled_chunks(record))
