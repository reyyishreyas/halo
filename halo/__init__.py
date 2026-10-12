"""HALO: lightweight span-level hallucination detection for quantized RAG.

Pipeline stages, in build order:

1. :mod:`halo.data` -- load HotpotQA and TruthfulQA, make splits
2. :mod:`halo.chunk` -- pack sentences into 200-500 token chunks
3. :mod:`halo.index` -- embed chunks and search them with FAISS
4. :mod:`halo.generate` -- load fp16 / 8-bit / 4-bit and answer
   questions, keeping the log-probability of every generated token

Phase 2 is built but has only been exercised where CUDA exists, which
this machine does not have; the measured runs happen on a GPU. Phases
3 and 4 -- the span detector and the baseline comparison -- are not
built yet.
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

from halo.chunk import Chunk, chunk_paragraphs, chunk_sentences
from halo.data import (
    HotpotRecord,
    Paragraph,
    TruthfulRecord,
    load_hotpot,
    load_truthful,
    make_splits,
)
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

__all__ = [
    "Chunk",
    "GeneratedAnswer",
    "GenerationSettings",
    "HotpotRecord",
    "PRECISIONS",
    "Paragraph",
    "RetrievalIndex",
    "ScoredChunk",
    "TokenCodec",
    "TruthfulRecord",
    "answer_question",
    "build_index",
    "build_messages",
    "build_quantization_config",
    "chunk_paragraphs",
    "chunk_sentences",
    "cuda_required",
    "default_device",
    "generate_answer",
    "load_generator",
    "load_hotpot",
    "load_settings",
    "load_tokenizer",
    "load_truthful",
    "make_splits",
    "render_prompt",
    "summarize",
    "write_jsonl",
]
