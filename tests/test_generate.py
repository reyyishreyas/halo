"""Unit tests for Phase 2 generation, runnable without a GPU.

Two things matter most and are pinned here: the three shipped config
files differ only in `precision` and `output` (the fairness condition
the proposal states), and decoding stays greedy with per-token
log-probabilities aligned to the generated tokens.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers.tokenization_utils_base import BatchEncoding

from halo.chunk import Chunk
from halo.generate import (
    PRECISIONS,
    GenerationSettings,
    answer_question,
    build_messages,
    build_quantization_config,
    generate_answer,
    load_settings,
    render_prompt,
    summarize,
    write_jsonl,
)
from halo.index import ScoredChunk

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

_SETTINGS = dict(
    precision="fp16",
    model_id="m",
    top_k=5,
    max_new_tokens=128,
    questions=100,
    seed=0,
    output="out.jsonl",
)


class _StubTokenizer:
    """Fixed prompt, fixed ids, fixed decode; no downloads."""

    pad_token_id = 0

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.prompts.append(messages[0]["content"])
        return "PROMPT"

    def __call__(self, text, return_tensors):
        assert return_tensors == "pt"
        self.last_text = text
        return BatchEncoding({"input_ids": torch.tensor([[1, 2, 3, 4]])})

    def decode(self, ids, skip_special_tokens):
        assert skip_special_tokens
        return "the answer ."


class _StubModel:
    """Canned three-token continuation with canned log-probabilities."""

    device = torch.device("cpu")

    def __init__(self) -> None:
        self.kwargs: dict = {}
        self.sequences = torch.tensor([[1, 2, 3, 4, 9, 8, 7]])
        self.transition = torch.tensor([[-0.1, -0.5, -2.0]])

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(sequences=self.sequences, scores=(None,))

    def compute_transition_scores(self, sequences, scores, normalize_logits):
        assert normalize_logits is True
        assert torch.equal(sequences, self.sequences)
        return self.transition


# ------------------------------------------------------------------
# Quantization configs
# ------------------------------------------------------------------


def test_fp16_has_no_quantization_config():
    assert build_quantization_config("fp16") is None


def test_int8_uses_the_standard_outlier_threshold():
    config = build_quantization_config("int8")
    assert config.load_in_8bit
    assert config.llm_int8_threshold == 6.0


def test_nf4_uses_the_qlora_recipe():
    config = build_quantization_config("nf4")
    assert config.load_in_4bit
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_compute_dtype == torch.float16
    assert config.bnb_4bit_use_double_quant


def test_unknown_precision_is_rejected():
    with pytest.raises(ValueError, match="precision"):
        build_quantization_config("int4")


# ------------------------------------------------------------------
# The fixed prompt
# ------------------------------------------------------------------


def test_the_prompt_is_strict_about_context():
    messages = build_messages(
        "Who founded it?", ["Alpha was founded by X.", "Beta came later."]
    )
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert "only the context" in content
    assert "do not know" in content
    assert "[1] Alpha was founded by X." in content
    assert "[2] Beta came later." in content
    assert "Question: Who founded it?" in content


def test_render_prompt_uses_the_chat_template():
    tokenizer = _StubTokenizer()
    rendered = render_prompt(tokenizer, "Who?", ["chunk"])
    assert rendered == "PROMPT"
    assert "Question: Who?" in tokenizer.prompts[0]


# ------------------------------------------------------------------
# Shipped configs: the fairness condition, pinned
# ------------------------------------------------------------------


@pytest.mark.parametrize("precision", PRECISIONS)
def test_shipped_configs_load(precision: str):
    settings = load_settings(CONFIG_DIR / f"{precision}.yaml")
    assert settings.precision == precision
    assert settings.output.endswith(f"{precision}.jsonl")


def test_the_three_runs_differ_only_in_precision_and_output():
    """The proposal's fairness condition: only quantization changes."""
    loaded = {p: load_settings(CONFIG_DIR / f"{p}.yaml") for p in PRECISIONS}
    shared = ("model_id", "top_k", "max_new_tokens", "questions", "seed")
    reference = asdict(loaded["fp16"])
    for precision in ("int8", "nf4"):
        for field in shared:
            assert asdict(loaded[precision])[field] == reference[field], field


def test_invalid_precision_is_rejected(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "precision: int4\nmodel_id: m\ntop_k: 1\nmax_new_tokens: 1\n"
        "questions: 1\nseed: 0\noutput: x.jsonl\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="precision"):
        load_settings(bad)


def test_missing_keys_are_reported(tmp_path: Path):
    partial = tmp_path / "partial.yaml"
    partial.write_text("precision: fp16\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing keys"):
        load_settings(partial)


# ------------------------------------------------------------------
# Greedy decoding and aligned log-probabilities
# ------------------------------------------------------------------


def test_greedy_decoding_keeps_aligned_logprobs():
    model, tokenizer = _StubModel(), _StubTokenizer()
    settings = GenerationSettings(**_SETTINGS)
    result = generate_answer(model, tokenizer, "Who?", ["chunk"], settings)

    assert model.kwargs["do_sample"] is False
    assert model.kwargs["output_scores"] is True
    assert model.kwargs["return_dict_in_generate"] is True
    assert model.kwargs["max_new_tokens"] == 128
    # Prompt tokens stripped, generated tokens kept.
    assert result.token_ids == [9, 8, 7]
    # float32 round-trip, so compare with tolerance
    assert result.token_logprobs == pytest.approx([-0.1, -0.5, -2.0])
    assert len(result.token_ids) == len(result.token_logprobs)
    assert result.answer == "the answer ."
    assert result.latency_s >= 0
    if not torch.cuda.is_available():
        assert result.peak_gpu_mb is None


# ------------------------------------------------------------------
# Record assembly, JSONL round-trip, summary
# ------------------------------------------------------------------


def _hits() -> list[ScoredChunk]:
    return [
        ScoredChunk(
            Chunk(
                text="X founded it.",
                sentences=["X founded it."],
                source_title="Alpha",
                token_count=4,
            ),
            0.9,
        ),
        ScoredChunk(
            Chunk(
                text="distractor text",
                sentences=["distractor text"],
                source_title="Beta",
                token_count=2,
            ),
            0.2,
        ),
    ]


def test_records_carry_retrieval_and_costs(tmp_path: Path):
    model, tokenizer = _StubModel(), _StubTokenizer()
    settings = GenerationSettings(**{**_SETTINGS, "precision": "nf4"})
    record = SimpleNamespace(id="q1", question="Who?", answer="X")

    entry = answer_question(model, tokenizer, record, _hits(), settings)

    assert entry["id"] == "q1"
    assert entry["precision"] == "nf4"
    assert entry["gold_answer"] == "X"
    assert entry["retrieved"][0]["title"] == "Alpha"
    assert entry["retrieved"][0]["score"] == 0.9
    assert entry["retrieved"][1]["rank"] == 2
    assert len(entry["token_logprobs"]) == len(entry["token_ids"])

    path = tmp_path / "run.jsonl"
    write_jsonl(path, [entry])
    back = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert back == entry


def test_summarize_reports_the_comparison_table_fields():
    model, tokenizer = _StubModel(), _StubTokenizer()
    settings = GenerationSettings(**_SETTINGS)
    record = SimpleNamespace(id="q1", question="Who?", answer="X")
    entry = answer_question(model, tokenizer, record, _hits(), settings)

    stats = summarize([entry])
    assert stats["questions"] == 1
    assert stats["mean_latency_s"] >= 0
    # mean of [-0.1, -0.5, -2.0]
    assert stats["mean_token_logprob"] == pytest.approx(-0.8667, abs=1e-3)
    if not torch.cuda.is_available():
        assert stats["peak_gpu_mb"] is None


def test_summarize_rejects_an_empty_run():
    with pytest.raises(ValueError, match="empty"):
        summarize([])
