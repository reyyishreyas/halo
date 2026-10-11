"""Dataset loading and splitting for HALO.

Two datasets are used, for different jobs.

``hotpot_qa`` (distractor setting) is the grounding dataset. Each
question ships with 10 Wikipedia paragraphs, exactly 2 of which contain
the answer, and those gold sentences are named in ``supporting_facts``
as ``(title, sentence index)``. That sentence-level pointer is what
makes span labels derivable in Phase 3, so it is preserved verbatim
here rather than flattened away.

``truthful_qa`` has no corpus of its own. It is kept as a truthfulness
probe instead: 817 questions with paired correct and incorrect
reference answers, used to measure how often a model repeats a
plausible falsehood even when retrieval gave it nothing useful.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# Newer `datasets` rejects bare dataset ids, so the namespace is
# required even though the repository and its default config happen to
# share a name.
HOTPOT_SOURCE = "hotpotqa/hotpot_qa"
HOTPOT_CONFIG = "distractor"
TRUTHFUL_SOURCE = "truthfulqa/truthful_qa"
TRUTHFUL_CONFIG = "generation"

HotpotSplit = Literal["train", "validation"]


@dataclass(frozen=True)
class Paragraph:
    """One candidate Wikipedia paragraph offered to the retriever."""

    title: str
    sentences: list[str]

    @property
    def text(self) -> str:
        """The paragraph joined into a single string."""
        return " ".join(self.sentences)


@dataclass(frozen=True)
class HotpotRecord:
    """A HotpotQA question with its candidate corpus and gold evidence.

    Attributes
    ----------
    id : str
        HotpotQA question id.
    question : str
        The question asked of the RAG pipeline.
    answer : str
        The short gold answer, or ``yes`` / ``no``.
    paragraphs : list[Paragraph]
        The 10 candidate paragraphs. 2 are gold, 8 are distractors.
    gold : list[tuple[str, int]]
        Gold supporting facts as ``(paragraph title, sentence index)``.
    """

    id: str
    question: str
    answer: str
    paragraphs: list[Paragraph]
    gold: list[tuple[str, int]]

    @property
    def gold_titles(self) -> list[str]:
        """Distinct titles of the gold supporting paragraphs."""
        return sorted({title for title, _ in self.gold})

    @property
    def gold_sentences(self) -> list[tuple[str, int, str]]:
        """Gold evidence as ``(title, sentence index, text)`` triples.

        Out-of-range indices are dropped rather than raising, because a
        few HotpotQA rows point past the end of their paragraph and a
        bad row should not abort a whole run.
        """
        by_title = {p.title: p for p in self.paragraphs}
        out: list[tuple[str, int, str]] = []
        for title, sent_id in self.gold:
            para = by_title.get(title)
            if para is not None and 0 <= sent_id < len(para.sentences):
                out.append((title, sent_id, para.sentences[sent_id]))
        return out


@dataclass(frozen=True)
class TruthfulRecord:
    """One TruthfulQA question and its paired reference answers.

    Attributes
    ----------
    id : str
        ``{index}:{question}``; TruthfulQA ships no stable id column.
    question : str
        The question asked of the model.
    category : str
        TruthfulQA category, e.g. ``Misconceptions``.
    best_answer : str
        The single best reference answer.
    correct_answers : list[str]
        All reference answers judged true.
    incorrect_answers : list[str]
        All reference answers judged false. These are the plausibly
        worded falsehoods the project is trying to catch.
    """

    id: str
    question: str
    category: str
    best_answer: str
    correct_answers: list[str]
    incorrect_answers: list[str]


def load_hotpot(
    split: HotpotSplit, limit: int | None = None
) -> list[HotpotRecord]:
    """Load one HotpotQA distractor split as HALO records.

    Parameters
    ----------
    split : {"train", "validation"}
        Which official split to load. HotpotQA's test labels are not
        public, so only these two exist.
    limit : int, optional
        Keep only the first ``limit`` rows. Used for smoke tests.

    Returns
    -------
    list[HotpotRecord]
        One record per question, in dataset order.
    """
    from datasets import load_dataset

    ds = load_dataset(HOTPOT_SOURCE, HOTPOT_CONFIG, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return [_to_record(ex) for ex in ds]


def _to_record(example: dict) -> HotpotRecord:
    """Convert one raw HotpotQA row into a ``HotpotRecord``."""
    # context arrives as {"title": [...], "sentences": [[...], ...]},
    # not as a list of dicts, so zip the two parallel lists.
    titles = example["context"]["title"]
    texts = example["context"]["sentences"]
    paragraphs = [
        Paragraph(title=str(t), sentences=[str(s) for s in sents])
        for t, sents in zip(titles, texts, strict=True)
    ]
    gold = [
        (str(t), int(i))
        for t, i in zip(
            example["supporting_facts"]["title"],
            example["supporting_facts"]["sent_id"],
            strict=True,
        )
    ]
    return HotpotRecord(
        id=str(example["id"]),
        question=str(example["question"]),
        answer=str(example["answer"]),
        paragraphs=paragraphs,
        gold=gold,
    )


def make_splits(
    test_fraction: float = 0.5,
    seed: int = 0,
    limit: int | None = None,
) -> dict[str, list[HotpotRecord]]:
    """Build deterministic train, validation and test splits.

    HotpotQA publishes no answer labels for its test set, so the
    official validation split is halved instead: one part tunes the
    logistic-regression detector, the other is held back for the
    reported numbers. The split is seeded, because the proposal asks
    for fixed seeds and a reproducible test set.

    Parameters
    ----------
    test_fraction : float
        Fraction of the official validation split reserved for test.
    seed : int
        Seed for the shuffle. Same seed gives the same split.
    limit : int, optional
        Cap applied to each official split before shuffling.

    Returns
    -------
    dict[str, list[HotpotRecord]]
        Keys ``train``, ``validation``, ``test``.

    Raises
    ------
    ValueError
        If ``test_fraction`` is outside ``(0, 1)``.
    """
    if not 0 < test_fraction < 1:
        raise ValueError(
            f"test_fraction must be in (0, 1), got {test_fraction}"
        )

    rng = np.random.default_rng(seed)
    train = load_hotpot("train", limit=limit)
    val = load_hotpot("validation", limit=limit)

    order = rng.permutation(len(val))
    cut = int(len(val) * (1 - test_fraction))
    return {
        "train": train,
        "validation": [val[i] for i in order[:cut]],
        "test": [val[i] for i in order[cut:]],
    }


def load_truthful(limit: int | None = None) -> list[TruthfulRecord]:
    """Load TruthfulQA as HALO records.

    TruthfulQA only publishes a ``validation`` split, so that is what
    is loaded regardless of the name used elsewhere.

    Parameters
    ----------
    limit : int, optional
        Keep only the first ``limit`` questions.

    Returns
    -------
    list[TruthfulRecord]
        One record per question.
    """
    from datasets import load_dataset

    ds = load_dataset(TRUTHFUL_SOURCE, TRUTHFUL_CONFIG, split="validation")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return [
        TruthfulRecord(
            id=f"{i}:{ex['question']}",
            question=str(ex["question"]),
            category=str(ex["category"]),
            best_answer=str(ex["best_answer"]),
            correct_answers=[str(a) for a in ex["correct_answers"]],
            incorrect_answers=[str(a) for a in ex["incorrect_answers"]],
        )
        for i, ex in enumerate(ds)
    ]
