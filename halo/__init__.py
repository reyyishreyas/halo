"""HALO: lightweight span-level hallucination detection for quantized RAG.

Pipeline stages, in build order:

1. :mod:`halo.data` -- load HotpotQA and TruthfulQA, make splits
2. :mod:`halo.chunk` -- pack sentences into 200-500 token chunks
3. :mod:`halo.index` -- embed chunks and search them with FAISS
4. :mod:`halo.generate` -- load fp16 / 8-bit / 4-bit and answer
   questions, keeping the log-probability of every generated token
5. :mod:`halo.spans` -- cut answers into sentence spans, aligned to
   the tokens that produced them
6. :mod:`halo.signals` -- token confidence and grounding per span
7. :mod:`halo.labels` -- silver span labels from a frozen NLI model
8. :mod:`halo.detector` -- standardized logistic-regression detector
9. :mod:`halo.baselines` -- the comparison methods behind one interface
10. :mod:`halo.metrics` -- span-level precision, recall, F1,
    hallucination rate
11. :mod:`halo.report` -- Markdown comparison tables

Phase 1 and Phase 3 run and are tested on this machine. Phase 2 has
only been exercised where CUDA exists, which this machine does not
have; the measured runs happen on a GPU, and Phases 3 and 4 consume
those runs' output. All four phases are built and tested here; the
measured generation and detection numbers await the GPU runs.
"""

__version__ = "0.1.0"

import os

# torch and faiss-cpu each vendor their own copy of LLVM's libomp, so
# importing both normally aborts the process with OpenMP Error #15.
# Tolerating the duplicate runtime is measured to give bit-identical
# top-k results against a numpy brute-force reference; that property is
# pinned by tests/test_index.py::test_faiss_matches_numpy_ground_truth,
# so a library bump cannot silently corrupt retrieval scores.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from halo.baselines import (
    ConfidenceThreshold,
    GroundingThreshold,
    NLIBaseline,
    NoVerification,
    SecondVerifier,
    SpanBaseline,
)
from halo.chunk import Chunk, chunk_paragraphs, chunk_sentences
from halo.data import (
    HotpotRecord,
    Paragraph,
    TruthfulRecord,
    load_hotpot,
    load_truthful,
    make_splits,
)
from halo.detector import SpanDetector
from halo.device import cuda_required, default_device
from halo.generate import (
    PRECISIONS,
    GeneratedAnswer,
    GenerationSettings,
    answer_question,
    build_messages,
    build_quantization_config,
    generate_answer,
    load_generator,
    load_settings,
    render_prompt,
    summarize,
    write_jsonl,
)
from halo.index import (
    RetrievalIndex,
    ScoredChunk,
    TokenCodec,
    build_index,
    load_tokenizer,
)
from halo.labels import (
    DEFAULT_NLI_MODEL,
    GROUNDED,
    HALLUCINATED,
    load_nli,
    premise_from_record,
    silver_labels,
)
from halo.metrics import DetectionMetrics, detection_metrics, macro_average
from halo.report import ReportRow, comparison_table, save_report
from halo.signals import grounding_scores, max_cosine, mean_token_logprob
from halo.spans import Span, answer_spans, split_sentences, token_char_offsets

__all__ = [
    "Chunk",
    "ConfidenceThreshold",
    "DEFAULT_NLI_MODEL",
    "DetectionMetrics",
    "GROUNDED",
    "GeneratedAnswer",
    "GenerationSettings",
    "GroundingThreshold",
    "HALLUCINATED",
    "HotpotRecord",
    "NLIBaseline",
    "NoVerification",
    "PRECISIONS",
    "Paragraph",
    "ReportRow",
    "RetrievalIndex",
    "ScoredChunk",
    "SecondVerifier",
    "Span",
    "SpanBaseline",
    "SpanDetector",
    "TokenCodec",
    "TruthfulRecord",
    "answer_question",
    "answer_spans",
    "build_index",
    "build_messages",
    "build_quantization_config",
    "chunk_paragraphs",
    "chunk_sentences",
    "comparison_table",
    "cuda_required",
    "default_device",
    "detection_metrics",
    "generate_answer",
    "grounding_scores",
    "load_generator",
    "load_hotpot",
    "load_nli",
    "load_settings",
    "load_tokenizer",
    "load_truthful",
    "macro_average",
    "make_splits",
    "max_cosine",
    "mean_token_logprob",
    "premise_from_record",
    "render_prompt",
    "save_report",
    "silver_labels",
    "split_sentences",
    "summarize",
    "token_char_offsets",
    "write_jsonl",
]
