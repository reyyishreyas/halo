"""Quantized answer generation with per-token log-probabilities.

Phase 2 produces what Phase 3 scores: for every question, one record
holds the answer text, the log-probability the model assigned to each
generated token, the retrieved chunks the model saw, and what the
answer cost (latency and peak GPU memory).

The three precisions -- fp16, 8-bit and 4-bit -- differ only in how
the weights are loaded. Prompt, decoding and limits come from the same
settings shape, so any difference between runs is attributable to
quantization, which is the question the project asks.

Quantized loading needs bitsandbytes, which is CUDA-only, so loading
is guarded with :func:`halo.device.cuda_required` instead of failing
deep inside a library call. The per-question logic (prompt building,
score aggregation, record assembly) is importable and testable
everywhere, CUDA or not.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from halo.device import cuda_required, default_device
from halo.index import ScoredChunk

PRECISIONS: tuple[str, ...] = ("fp16", "int8", "nf4")
DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"

# Every field of GenerationSettings except precision and output, in
# the order they appear in the YAML files.
_SETTING_KEYS = (
    "precision",
    "model_id",
    "top_k",
    "max_new_tokens",
    "questions",
    "seed",
    "output",
)


@dataclass(frozen=True)
class GenerationSettings:
    """One precision run's fixed configuration.

    Every field except ``precision`` and ``output`` is identical
    across the three shipped YAML files; a test pins that, so the
    proposal's fairness condition -- the only intended change between
    runs is the quantization setting -- holds by construction.

    Attributes
    ----------
    precision : str
        One of :data:`PRECISIONS`.
    model_id : str
        Hugging Face model to load.
    top_k : int
        Retrieved chunks placed in the prompt.
    max_new_tokens : int
        Greedy decoding budget for the answer.
    questions : int
        How many held-out test questions this run answers.
    seed : int
        Torch seed, fixed for reproducibility. Decoding is greedy,
        so the seed governs nothing stochastic today; it is set
        anyway so a later sampling ablation stays controlled.
    output : str
        JSONL path the records are written to.
    """

    precision: str
    model_id: str
    top_k: int
    max_new_tokens: int
    questions: int
    seed: int
    output: str

    def __post_init__(self) -> None:
        """Validate the precision name and the positive integer fields.

        Raises
        ------
        ValueError
            If ``precision`` is not one of :data:`PRECISIONS` or a
            count is not positive.
        """
        if self.precision not in PRECISIONS:
            raise ValueError(
                f"precision must be one of {PRECISIONS}, "
                f"got {self.precision!r}"
            )
        for name in ("top_k", "max_new_tokens", "questions"):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} must be positive, got {getattr(self, name)}"
                )


def load_settings(path: str | Path) -> GenerationSettings:
    """Read one run's settings from YAML.

    Parameters
    ----------
    path : str or Path
        Path to a file such as ``configs/fp16.yaml``.

    Returns
    -------
    GenerationSettings
        The validated settings.

    Raises
    ------
    ValueError
        If the file is not a mapping, a required key is missing, or a
        value is invalid.
    """
    raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    missing = [key for key in _SETTING_KEYS if key not in raw]
    if missing:
        raise ValueError(f"{path} is missing keys: {missing}")
    return GenerationSettings(**{key: raw[key] for key in _SETTING_KEYS})


def build_quantization_config(precision: str) -> BitsAndBytesConfig | None:
    """Return the bitsandbytes config for a precision, or None for fp16.

    The proposal fixes these settings: 8-bit uses LLM.int8 with the
    standard outlier threshold (bitsandbytes' default, 6.0), and 4-bit
    uses NF4 with fp16 compute and double quantization -- the QLoRA
    recipe. fp16 loads unquantized and is the reference version.

    Parameters
    ----------
    precision : str
        One of :data:`PRECISIONS`.

    Returns
    -------
    BitsAndBytesConfig or None
        The quantization config, or None for fp16.

    Raises
    ------
    ValueError
        If ``precision`` is not one of :data:`PRECISIONS`.
    """
    if precision == "fp16":
        return None
    if precision == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if precision == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    raise ValueError(
        f"precision must be one of {PRECISIONS}, got {precision!r}"
    )


def build_messages(
    question: str, chunks: Sequence[str]
) -> list[dict[str, str]]:
    """Build the single fixed user message every precision obeys.

    The instruction is strict on purpose: the proposal asks the model
    to answer only from the given context, and to admit when the
    context does not contain the answer. That admission is also the
    detector's easiest grounded case.

    Parameters
    ----------
    question : str
        The user question.
    chunks : Sequence[str]
        Retrieved chunk texts, best match first.

    Returns
    -------
    list[dict[str, str]]
        One user message, ready for ``apply_chat_template``.
    """
    context = "\n".join(f"[{i}] {text}" for i, text in enumerate(chunks, 1))
    body = (
        "Answer the question using only the context below. "
        "If the context does not contain the answer, say that you "
        "do not know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )
    return [{"role": "user", "content": body}]


def render_prompt(tokenizer: Any, question: str, chunks: Sequence[str]) -> str:
    """Render the fixed messages through the model's chat template.

    Instruct models expect their own template's special tokens, so
    the template is part of the prompt. It stays identical across
    precisions because all three load the same model and tokenizer.

    Parameters
    ----------
    tokenizer : Any
        Hugging Face tokenizer exposing ``apply_chat_template``.
    question : str
        The user question.
    chunks : Sequence[str]
        Retrieved chunk texts, best match first.

    Returns
    -------
    str
        The rendered prompt.
    """
    return tokenizer.apply_chat_template(
        build_messages(question, chunks),
        tokenize=False,
        add_generation_prompt=True,
    )


def load_generator(
    settings: GenerationSettings,
    device: torch.device | str | None = None,
) -> tuple[Any, Any]:
    """Load the model and tokenizer for one precision run.

    All three conditions use fp16 weights or fp16 compute, so the
    dtype is never a second variable: the runs differ only in
    quantization.

    The reference run (fp16) loads on any torch device, so it can
    also be produced locally on a Mac (MPS). The 8-bit and 4-bit
    runs load through bitsandbytes, which has no reliable MPS or CPU
    backend; attempting them off-CUDA would silently change the
    quantizer and break the comparison, so they are still guarded by
    :func:`halo.device.cuda_required`.

    Parameters
    ----------
    settings : GenerationSettings
        Run configuration.
    device : torch.device or str or None
        Target device for the fp16 run. ``None`` uses
        :func:`halo.device.default_device` (CUDA, else MPS, else
        CPU). Ignored for the quantized runs, which are placed by
        bitsandbytes itself.

    Returns
    -------
    tuple
        ``(model, tokenizer)``. The model is in eval mode, and the
        tokenizer's pad token is set to its EOS token so batched
        generation never pads with a meaningful token.

    Raises
    ------
    RuntimeError
        If an 8-bit or 4-bit run is requested without CUDA.
    """
    if settings.precision != "fp16":
        cuda_required(f"generation ({settings.precision})")
    tokenizer = AutoTokenizer.from_pretrained(settings.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if settings.precision == "fp16":
        target = (
            torch.device(device)
            if device is not None
            else default_device()
        )
        # float16 is unusable on CPU; the fp16-named run keeps its
        # name on CPU by computing in float32 rather than lying
        # about precision.
        dtype = torch.float16 if target.type != "cpu" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            settings.model_id, dtype=dtype
        )
        model.to(target)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            settings.model_id,
            dtype=torch.float16,
            quantization_config=build_quantization_config(settings.precision),
            device_map="auto",
        )
    model.eval()
    return model, tokenizer


@dataclass(frozen=True)
class GeneratedAnswer:
    """One answer plus everything Phase 3 needs to score it.

    Attributes
    ----------
    answer : str
        The generated text, special tokens removed.
    token_ids : list[int]
        Generated token ids, including the end token.
    token_logprobs : list[float]
        Log-probability of each generated token, aligned with
        ``token_ids``. Negative; closer to zero is more confident.
        The end token is included, which is harmless because span
        scoring drops special tokens.
    latency_s : float
        Seconds spent inside ``generate``.
    peak_gpu_mb : float or None
        Peak CUDA memory during generation, in MiB. None when the run
        did not happen on CUDA.
    """

    answer: str
    token_ids: list[int]
    token_logprobs: list[float]
    latency_s: float
    peak_gpu_mb: float | None


def generate_answer(
    model: Any,
    tokenizer: Any,
    question: str,
    chunks: Sequence[str],
    settings: GenerationSettings,
) -> GeneratedAnswer:
    """Answer one question greedily and keep the per-token log-probs.

    Decoding is greedy and the settings are shared, so the three
    precisions differ only in their weights. The log-probabilities
    come from the generation scores themselves -- no second pass, no
    extra model. That is the cost argument the project makes.

    Parameters
    ----------
    model : Any
        Causal LM, loaded by :func:`load_generator` on a GPU.
    tokenizer : Any
        Its tokenizer.
    question : str
        The user question.
    chunks : Sequence[str]
        Retrieved chunk texts, best match first.
    settings : GenerationSettings
        Run configuration (seed and token budget).

    Returns
    -------
    GeneratedAnswer
        Answer text, aligned token log-probabilities, and cost.
    """
    torch.manual_seed(settings.seed)

    prompt = render_prompt(tokenizer, question, chunks)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=settings.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        scores = model.compute_transition_scores(
            output.sequences, output.scores, normalize_logits=True
        )
    latency_s = time.perf_counter() - start
    peak: float | None = None
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**2

    token_ids = output.sequences[0, input_length:].tolist()
    logprobs = [float(value) for value in scores[0]]
    answer = tokenizer.decode(token_ids, skip_special_tokens=True)
    return GeneratedAnswer(
        answer=answer,
        token_ids=token_ids,
        token_logprobs=logprobs,
        latency_s=round(latency_s, 4),
        peak_gpu_mb=round(peak, 1) if peak is not None else None,
    )


def answer_question(
    model: Any,
    tokenizer: Any,
    record: Any,
    hits: Sequence[ScoredChunk],
    settings: GenerationSettings,
) -> dict[str, Any]:
    """Answer one dataset question and package the full record.

    Retrieval happens before this call and is cached by the caller:
    the same hits feed all three precisions, so quantization cannot
    quietly change what the model saw.

    Parameters
    ----------
    model : Any
        Causal LM, loaded by :func:`load_generator`.
    tokenizer : Any
        Its tokenizer.
    record : Any
        Dataset record with ``id``, ``question`` and ``answer``
        (``halo.data.HotpotRecord``).
    hits : Sequence[ScoredChunk]
        Retrieved chunks for this question, best match first.
    settings : GenerationSettings
        Run configuration.

    Returns
    -------
    dict[str, Any]
        JSON-ready record: identity, answer, aligned token
        log-probabilities, retrieved chunks, latency and memory.
    """
    result = generate_answer(
        model,
        tokenizer,
        record.question,
        [hit.chunk.text for hit in hits],
        settings,
    )
    return {
        "id": record.id,
        "question": record.question,
        "gold_answer": record.answer,
        "precision": settings.precision,
        "answer": result.answer,
        "token_ids": result.token_ids,
        "token_logprobs": result.token_logprobs,
        "retrieved": [
            {
                "rank": rank,
                "title": hit.chunk.source_title,
                "score": round(float(hit.score), 4),
                "text": hit.chunk.text,
            }
            for rank, hit in enumerate(hits, start=1)
        ],
        "latency_s": result.latency_s,
        "peak_gpu_mb": result.peak_gpu_mb,
    }


def write_jsonl(path: str | Path, records: Sequence[dict[str, Any]]) -> None:
    """Write records to JSON Lines, creating parent directories.

    Parameters
    ----------
    path : str or Path
        Destination file.
    records : Sequence[dict[str, Any]]
        Records as returned by :func:`answer_question`.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one precision run for the comparison table.

    Parameters
    ----------
    records : Sequence[dict[str, Any]]
        Records as returned by :func:`answer_question`.

    Returns
    -------
    dict[str, Any]
        Question count, mean latency, peak GPU memory, and the mean
        of each answer's mean token log-probability -- the number
        quantization is expected to disturb most.

    Raises
    ------
    ValueError
        If ``records`` is empty.
    """
    if not records:
        raise ValueError("cannot summarize an empty run")
    latencies = [record["latency_s"] for record in records]
    peaks = [
        record["peak_gpu_mb"]
        for record in records
        if record["peak_gpu_mb"] is not None
    ]
    per_answer = [
        sum(record["token_logprobs"]) / len(record["token_logprobs"])
        for record in records
        if record["token_logprobs"]
    ]
    return {
        "questions": len(records),
        "mean_latency_s": round(sum(latencies) / len(latencies), 3),
        "peak_gpu_mb": round(max(peaks), 1) if peaks else None,
        "mean_token_logprob": round(sum(per_answer) / len(per_answer), 4),
    }
