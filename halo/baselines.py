"""The Phase 4 comparison methods, behind one interface.

Every baseline answers the detector's question -- which sentence spans
are hallucinated? -- and returns the same labels (0 grounded,
1 hallucinated), so :mod:`halo.metrics` can score every method
identically. The comparison is about accuracy and cost, not about
giving methods different tasks.

The proposal compares four approaches, plus two single-signal
ablations that show what each signal contributes on its own:

1. ``no_verification`` -- never flags anything. The floor: its
   hallucination rate is 0 by construction, so anything that cannot
   beat it adds nothing.
2. ``confidence_only`` -- flag spans whose mean token log-probability
   is low. Uses only what generation already produced; no extra
   model. The single-signal ablation for token confidence.
3. ``grounding_only`` -- flag spans with low maximum cosine similarity
   to the retrieved chunks. One embedding call per answer. The
   single-signal ablation for grounding.
4. ``nli`` -- a frozen NLI model judges each span directly
   (HALT-RAG-style). Deliberately the *same* model that created the
   training labels in :mod:`halo.labels`: the proposal allows an
   NLI-style baseline, and reusing one model means the comparison
   never introduces something stronger than what is already reported.
5. ``second_verifier`` -- a second LLM reads the context and each
   span, one call per sentence. The accurate-but-expensive method
   HALO exists to replace; its per-call latency is the price the
   report must show.

A LettuceDetect-style token classifier is intentionally absent: it
needs a fine-tuned transformer and labelled token data, which the
proposal allows only "if resources allow". The two ablations answer
the question it would otherwise answer -- how much does each signal
contribute alone.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from halo.labels import premise_from_record, silver_labels
from halo.signals import grounding_scores, mean_token_logprob
from halo.spans import Span


class SpanBaseline:
    """One comparison method, answering the detector's question.

    Attributes
    ----------
    name : str
        Short identifier used in reports; subclasses set their own.
    """

    name: str = "baseline"

    def predict(self, record: dict, spans: Sequence[Span]) -> np.ndarray:
        """Label each span hallucinated (1) or grounded (0).

        Parameters
        ----------
        record : dict
            A Phase 2 JSONL record for the question that produced
            ``spans``.
        spans : Sequence[Span]
            Sentence spans of the generated answer.

        Returns
        -------
        numpy.ndarray
            Integer labels, shape ``(len(spans),)``; empty if there
            are no spans.
        """
        raise NotImplementedError


class NoVerification(SpanBaseline):
    """Never flags anything: the cheapest method and the floor.

    Recall is 0 on any answer containing a hallucination, and the
    hallucination rate is 0 by construction -- the report shows both
    so the floor is visible, not hidden.
    """

    name = "no_verification"

    def predict(self, record: dict, spans: Sequence[Span]) -> np.ndarray:
        """Return zero flags, one label per span.

        Parameters
        ----------
        record : dict
            Ignored; kept for the common interface.
        spans : Sequence[Span]
            The sentence spans.

        Returns
        -------
        numpy.ndarray
            Zeros, shape ``(len(spans),)``.
        """
        return np.zeros(len(spans), dtype=int)


class ConfidenceThreshold(SpanBaseline):
    """Flag spans whose mean token log-probability is below a cut.

    No extra model and no extra call: the signal is whatever
    generation already stored. The threshold is required rather than
    defaulted because it is a deployment choice, made on validation
    data -- a hidden default would make the comparison irreproducible.

    Parameters
    ----------
    threshold : float
        Mean log-probability at or below which a span is flagged.
    """

    name = "confidence_only"

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def predict(self, record: dict, spans: Sequence[Span]) -> np.ndarray:
        """Flag low-confidence spans.

        Parameters
        ----------
        record : dict
            Must hold ``token_logprobs`` for the generated answer.
        spans : Sequence[Span]
            Sentence spans of that answer.

        Returns
        -------
        numpy.ndarray
            1 where the span's mean log-probability is below
            ``threshold``.

        Raises
        ------
        ValueError
            If the record has no token log-probabilities.
        """
        if "token_logprobs" not in record:
            raise ValueError("record has no token_logprobs")
        logprobs = record["token_logprobs"]
        flags = [
            int(
                mean_token_logprob(logprobs, span.token_start, span.token_end)
                < self.threshold
            )
            for span in spans
        ]
        return np.asarray(flags, dtype=int)


class GroundingThreshold(SpanBaseline):
    """Flag spans poorly grounded in the retrieved chunks.

    Uses the embedding model retrieval already used, so the cosine
    scores are comparable to the retrieval scores in the same record.

    Parameters
    ----------
    threshold : float
        Maximum cosine similarity at or below which a span is flagged.
    encoder : Any
        Sentence-transformers model with ``encode(...)``; the same
        kind of object retrieval uses (duck-typed for tests).
    """

    name = "grounding_only"

    def __init__(self, threshold: float, encoder: Any) -> None:
        self.threshold = threshold
        self._encoder = encoder

    def predict(self, record: dict, spans: Sequence[Span]) -> np.ndarray:
        """Flag spans whose best chunk similarity is below the cut.

        Parameters
        ----------
        record : dict
            Must hold a non-empty ``retrieved`` list with chunk
            ``text`` fields.
        spans : Sequence[Span]
            Sentence spans of the generated answer.

        Returns
        -------
        numpy.ndarray
            1 where the span's maximum cosine to any chunk is below
            ``threshold``.

        Raises
        ------
        ValueError
            If the record has no retrieved chunks.
        """
        chunks = [chunk["text"] for chunk in record.get("retrieved") or []]
        if not chunks:
            raise ValueError("record has no retrieved chunks")
        scores = grounding_scores(
            self._encoder, [span.text for span in spans], chunks
        )
        return (scores < self.threshold).astype(int)


class NLIBaseline(SpanBaseline):
    """Frozen NLI judges each span against the context (HALT-RAG-style).

    Same model as the training labels in :mod:`halo.labels`, on
    purpose: as an evaluation method it is the accuracy ceiling the
    cheap detector aims to approach, and as a training source it kept
    the pipeline to one small extra model. Its two roles must not meet
    on the same data -- training labels and evaluation labels have to
    come from different sources, which is documented in
    :mod:`halo.metrics`.

    Parameters
    ----------
    model : Any
        Frozen NLI cross-encoder, from ``halo.load_nli`` or a test
        stub.
    """

    name = "nli"

    def __init__(self, model: Any) -> None:
        self._model = model

    def predict(self, record: dict, spans: Sequence[Span]) -> np.ndarray:
        """Return the NLI silver judgment for each span.

        Parameters
        ----------
        record : dict
            A Phase 2 record; its retrieved chunks become the NLI
            premise.
        spans : Sequence[Span]
            Sentence spans of the generated answer.

        Returns
        -------
        numpy.ndarray
            0 where NLI found entailment, 1 otherwise.
        """
        premise = premise_from_record(record)
        return silver_labels(
            premise, [span.text for span in spans], self._model
        )


class SecondVerifier(SpanBaseline):
    """A second LLM reads the context and each span: one call per span.

    The verifier is injected as a callable so the comparison stays
    model-agnostic: scripts pass e.g. a Phi-3 prompt ("Does the
    context entail this sentence? Answer yes or no."), tests pass a
    stub. This is the accurate-but-expensive method HALO exists to
    replace, and its per-call latency is what the report must make
    visible.

    Parameters
    ----------
    verify : Callable[[str, str], bool]
        ``verify(premise, span_text)`` returns True when the context
        supports the span.
    """

    name = "second_verifier"

    def __init__(self, verify: Callable[[str, str], bool]) -> None:
        self._verify = verify

    def predict(self, record: dict, spans: Sequence[Span]) -> np.ndarray:
        """Ask the verifier about each span, in order.

        Parameters
        ----------
        record : dict
            A Phase 2 record; its retrieved chunks become the
            premise.
        spans : Sequence[Span]
            Sentence spans of the generated answer.

        Returns
        -------
        numpy.ndarray
            0 where the verifier found support, 1 otherwise.
        """
        premise = premise_from_record(record)
        return np.asarray(
            [0 if self._verify(premise, span.text) else 1 for span in spans],
            dtype=int,
        )
