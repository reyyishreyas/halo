# HALO

**Hallucination Assessment in Lightweight Optimized Language Models**

Mini project BAI506 · Department of Artificial Intelligence and Machine Learning ·
BMS Institute of Technology and Management · 2026-27

HALO finds the sentences in a RAG answer that are **not supported by the
retrieved context**, using a detector that needs neither a second LLM call nor a
separate detector model. The same pipeline is run at fp16, 8-bit and 4-bit to
measure how far quantization degrades both generation quality and detection
reliability.

## Why

Strong verification methods are expensive at inference time. Some ask a second
LLM to check every answer; others load a fine-tuned NLI or transformer
detector. On a single T4 — the hardware this project targets — that extra cost
is the difference between deployable and not.

HALO reuses two signals that generation already produces:

1. **Token confidence** — the mean log-probability of the tokens in a sentence.
   Models can be confidently wrong, so this signal alone is not enough.
2. **Grounding** — the maximum cosine similarity between the sentence and the
   retrieved chunks.

A logistic-regression classifier combines them into a per-sentence
hallucination probability. It is deliberately simple: fast, interpretable, and
it loads no extra neural model.

## Status

| Phase | Weeks | Scope | State |
| ----- | ----- | ----- | ----- |
| 1 | 1–2 | Data loading, chunking, FAISS retrieval | **built** |
| 2 | 3–4 | Quantized generation at fp16 / 8-bit / 4-bit | not started |
| 3 | 5–6 | Span segmentation, silver labels, detector | not started |
| 4 | 7–8 | Baselines and evaluation report | not started |

Only Phase 1 exists. Nothing about detection accuracy is claimed yet, because
nothing about it has been measured.

## Phase 1 — measured results

50 HotpotQA validation questions, embedder `BAAI/bge-small-en-v1.5`:

| Measure | Value |
| ------- | ----- |
| Questions evaluated | 50 |
| Candidate paragraphs per question | 10 (2 gold, 8 distractors) |
| Chunks per question | mean 10.0, min 2, max 13 |
| Chunk tokens | min 15, max 497 (bge-small window is 512) |
| **recall@5** — a gold paragraph appears in the top 5 | **50 / 50 = 1.000** |

Retrieval surfaces the gold evidence on every question tested. That is the
useful finding at this stage: if later phases produce wrong answers, the cause
will be generation rather than a retriever that failed to find the evidence.

Reproduce:

```bash
python scripts/smoke_retrieval.py 50 5
```

## Setup

```bash
conda create -n halo python=3.11 -y
conda activate halo
pip install -e ".[dev]"
pytest tests/ -q
```

Datasets and the embedding model download from Hugging Face on first use.

### Phase 2 needs a CUDA GPU

`bitsandbytes` — which implements the 8-bit and 4-bit loading the project is
built on — has no MPS or CPU backend. Phase 2 runs on **Google Colab (free
T4)**. Everything else runs locally, including on a Mac.

Phi-3-mini (3.8B) is used rather than Llama-3-8B because its fp16 weights need
about 7.6 GB, so all three precisions fit a free T4. Llama-3-8B in fp16 needs
roughly 16 GB and does not.

### Note for macOS contributors

`torch` and `faiss-cpu` each bundle their own copy of LLVM's `libomp`, and
loading both normally aborts the process with `OMP: Error #15`. `halo` sets
`KMP_DUPLICATE_LIB_OK` on import to tolerate the duplicate runtime. This is
usually described as an unsafe workaround, so
[`test_faiss_matches_numpy_ground_truth`](tests/test_index.py) pins faiss's
top-k results against a numpy brute-force reference. If a dependency bump ever
perturbs retrieval scores, that test fails loudly rather than letting the
grounding signal drift silently.

## Layout

```
halo/
  data.py     HotpotQA and TruthfulQA loading, deterministic splits
  chunk.py    sentence-preserving packing into 200-500 token chunks
  index.py    embeddings, FAISS index, cosine search, token codec
  device.py   CUDA / MPS / CPU selection
scripts/
  smoke_retrieval.py
tests/
```

## Team

Reyyi Shreyas, Mayank Prakash, Sonali N Gowda, Anubhav
