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
| 1 | 1–2 | Data loading, chunking, FAISS retrieval | **measured** |
| 2 | 3–4 | Quantized generation at fp16 / 8-bit / 4-bit | **measured** (fp16 only; 8-bit / 4-bit await a CUDA GPU) |
| 3 | 5–6 | Span segmentation, silver labels, detector | **measured** |
| 4 | 7–8 | Baselines and evaluation report | **measured** |

All four phases have run on real data. The numbers below come from 100
HotpotQA validation questions at fp16, and three caveats bound what they
mean:

- **fp16 ran on Apple MPS, not the CUDA GPU the project targets.** Latency
  here is not comparable to a T4, and peak GPU memory was not recorded, so
  that column reads "–". The 8-bit and 4-bit runs still need a CUDA GPU.
- **The detector saw these 114 spans while fitting.** Phase 4 scores it on
  the same questions it trained on, with labels from a different source than
  its training labels. It is a consistency check, not a generalization
  number. A held-out split is the next step.
- **The verifier that defines the Phase 4 labels is the same model that
  wrote the answers** (Phi-3-mini). The labels are independent of the NLI
  model the detector trained on, so the comparison is not circular, but a
  model judging its own output is a weak ground truth.

Nothing here is extrapolated. Values that were not measured read "–".

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

## Phase 2 — measured (fp16, Apple MPS)

`halo/generate.py` loads the same model at three precisions and answers
questions greedily, keeping the log-probability of every generated token plus
per-answer latency and peak GPU memory:

| Config | Weights |
| ------ | ------- |
| `configs/fp16.yaml` | fp16, the reference baseline |
| `configs/int8.yaml` | 8-bit (LLM.int8, standard outlier threshold 6.0) |
| `configs/nf4.yaml` | 4-bit NF4, fp16 compute, double quantization (QLoRA recipe) |

The three files share every other field — model, top-k, token budget, question
count, seed — and `tests/test_generate.py` fails if that ever drifts, so the
only intended change between runs is the quantization setting. Retrieval is
computed once and reused by all three precisions, so quantization cannot
change what the model saw.

### fp16 run (measured)

`microsoft/Phi-3-mini-4k-instruct`, 100 questions, greedy decoding, top-k 5,
128 new-token budget, seed 0, on Apple MPS:

| Measure | Value |
| ------- | ----- |
| Questions answered | 100 |
| Mean generated tokens | 22.2 (min 2, max 76) |
| Retrieved chunks reused | mean 4.97 per question |
| Latency per question | mean **7.68 s**, median 7.22 s, min 1.28 s, max 15.42 s |
| Mean token log-probability | **−0.131** |
| Peak GPU memory | **–** (not recorded; see below) |

This machine has no CUDA GPU, so the run used Apple's MPS backend and
`peak_gpu_mb` stayed empty — MPS does not report GPU memory. The 8-bit and
4-bit runs have not happened yet: `bitsandbytes` has no MPS backend, so they
need a Colab T4. Until they run, this table is the fp16 reference only, and
the README claims nothing about how quantization changes generation.

## Phase 3 — measured

Four small modules, all runnable and tested on this machine:

| Module | Job |
| ------ | --- |
| `halo/spans.py` | Cut each answer into sentences and align them to the exact tokens that produced them |
| `halo/signals.py` | Mean token log-probability, and max cosine similarity to any retrieved chunk |
| `halo/labels.py` | Silver training labels from a frozen NLI model (`cross-encoder/nli-deberta-v3-base`) |
| `halo/detector.py` | Standardized logistic regression with balanced class handling, saved as readable JSON |

Alignment never re-tokenizes the answer: offsets come from decoding
progressively longer prefixes of the stored token ids — the same call that
produced the answer in the first place — so the end token and the spaces
between sentences are excluded from every span by construction.

The NLI labeler is deliberately the same model that will appear as a Phase 4
baseline, so the detector never trains on labels from anything stronger than a
method the comparison already reports. Its labels are noisy (a neutral but
reasonable inference becomes a positive), which is exactly why the regression
uses balanced class weights.

### Span and label counts (measured)

On the fp16 run: **114 sentence spans** across the 100 answers (most answers
are one short sentence; mean 22.2 tokens).

| Measure | Value |
| ------- | ----- |
| Spans | 114 |
| Silver-labelled hallucinated | **70 / 114 = 0.614** |
| Detector fitted coefficients | confidence −0.366, grounding −0.681 |
| Detector intercept | 0.115 |
| Decision threshold | 0.5 |

The two coefficients are the one interpretable result worth reading: grounding
carries about twice the weight of confidence, in the direction you would want
— a sentence far from any retrieved chunk is more suspect than one that merely
sounds unsure.

### The 0.614 ceiling is a label-quality bound, not a bug

The detector was refitted after a label-quality fix (see Phase 4 below) and
now agrees with its own silver training labels on **70 / 114 = 0.614** of
spans. That number is not an accuracy; it is the ceiling these two features
can reach against noisy labels, and it moves when the labels move. Earlier
label revisions read 0.877, then 0.719, before the NLI truncation fix — the
same detector, different labels.

Both features and balanced class weights were chosen before these labels
existed, so the number is not tuned to hit a target. It is here to be
replaced, not defended.

## Phase 4 — measured

`halo/baselines.py` implements the comparison methods behind one interface, so
`halo/metrics.py` scores every method the same way:

| Method | Extra cost | Role |
| ------ | ---------- | ---- |
| `no_verification` | none | never flags anything; the floor |
| `confidence_only` | none — uses generation's own log-probs | confidence ablation |
| `grounding_only` | one embedding per answer | grounding ablation |
| `nli` | one frozen NLI call per span | HALT-RAG-style; the *same* model as the training labels, deliberately |
| `second_verifier` | one LLM call per span | the expensive method HALO exists to replace |

A LettuceDetect-style token classifier is intentionally not built: it needs a
fine-tuned transformer and labelled token data, which the proposal allows only
"if resources allow". The two ablations answer the question it would
otherwise — what does each signal contribute on its own?

`halo/metrics.py` reports span-level precision, recall, F1, accuracy and
hallucination rate, micro (all spans) or macro (one vote per question).
`halo/report.py` renders the comparison table including per-method latency,
measured for every method in the run below.

### Which labels, and why they differ from the training labels

One rule the evaluation depends on: **training labels and evaluation labels
must not come from the same source.** The detector trains on silver labels
from the frozen NLI model, so scoring it against those same judgments would
be circular — the `nli` baseline would score perfectly by construction.

Evaluation labels therefore come from the **second verifier**: Phi-3 asked
"does the context support this sentence?" once per span. That is a different
model than the NLI cross-encoder the detector trained on, so the comparison
is not circular. Because the verifier *defines* the labels it is not itself
scored — it would score 1.000 by construction — and only its cost is
reported. Its latency is the reference the detector is trying to beat.

Ablation cut-offs were fitted on the **silver training labels**, never on the
evaluation labels, so no method sees the answer key: `confidence < −0.002`,
`grounding < 0.861`.

### Results (measured)

114 spans from 100 fp16 answers, Apple MPS:

| Method | Precision | Recall | F1 | Flag rate | Latency (ms/span) |
| ------ | --------- | ------ | -- | --------- | ----------------- |
| `no_verification` | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| `confidence_only` | 0.372 | 1.000 | 0.542 | 0.991 | 0.1 |
| `grounding_only` | 0.381 | 0.952 | 0.544 | 0.921 | 71.1 |
| `nli` | 0.457 | 0.762 | 0.571 | 0.614 | 828.2 |
| **detector (HALO)** | **0.620** | **0.738** | **0.674** | 0.439 | **3.6** |

Label source, not scored: `second_verifier` at **6195.7 ms/span**.

Read the cost column alongside F1, because that is the project's actual
claim. The detector reaches F1 0.674 in **3.6 ms/span**, against the NLI
baseline's 0.571 in 828.2 ms — 230× cheaper for a better F1 — and against the
verifier's 6195.7 ms, about **1720× cheaper**. The two ablations explain why:
confidence alone flags essentially everything (rate 0.991, so precision
collapses to 0.372), and grounding alone flags 0.921. Combining them lets the
regression be selective, which is the only reason precision rises to 0.620.

### What these numbers are not

They are **in-sample**. The detector fitted on these same 114 spans, using NLI
labels rather than the verifier labels scored here, so the label source
differs but the sentences do not. A held-out split is the next measurement,
and the honest generalization number is whatever it says. Until then, 0.674
is a consistency check between two label sources on familiar text — not a
detection accuracy to quote.

The verifier is also the same model that wrote the answers (Phi-3-mini). It
is independent of the NLI training labels, which is what breaks circularity,
but a model grading its own output is a weak ground truth. Human annotation
would be stronger, and is not in scope yet.

### Auditing the calls, not just the averages

`scripts/audit_sample.py` samples spans as readable cards — question,
retrieved context, answer, each sentence — for an independent AI or a human
to judge **before** seeing any model verdict. `--blind` prints cards only and
loads no large model; the default pass reveals the detector, NLI and verifier
verdicts through the same classes Phase 4 scores with, then tallies
agreement. It exists because a single F1 cannot show *which* calls are wrong
or whether the labels deserve trust. Sampling is a fixed stride, so both
passes cover identical spans.

```bash
python scripts/audit_sample.py --blind --n 6   # cards, no verdicts
python scripts/audit_sample.py --n 6           # reveal + agreement tally
```

## Setup

```bash
conda create -n halo python=3.11 -y
conda activate halo
pip install -e ".[dev]"
pytest tests/ -q
```

Datasets and the embedding model download from Hugging Face on first use.

### Phase 2 quantization needs a CUDA GPU

`bitsandbytes` — which implements the 8-bit and 4-bit loading the project is
built on — has no MPS or CPU backend, so only the **8-bit and 4-bit** runs
need **Google Colab (free T4)**. The fp16 reference has already run locally on
Apple MPS (see Phase 2), and everything else — retrieval, training, Phase 4
evaluation, the audit script — runs locally, including on a Mac.

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
  generate.py quantized loading, greedy answers, per-token log-probs
  spans.py    sentence spans aligned to generated tokens
  signals.py  confidence and grounding scores per span
  labels.py   silver labels from a frozen NLI model
  detector.py standardized logistic-regression detector
  baselines.py comparison methods: no-verification, thresholds, NLI,
               second verifier
  metrics.py  span-level precision, recall, F1, hallucination rate
  report.py   markdown comparison tables for Phase 4
  device.py   CUDA / MPS / CPU selection
configs/      one YAML per precision run
scripts/
  smoke_retrieval.py  Phase 1 retrieval recall
  run_precision_local.py  Phase 2 fp16 generation (MPS)
  train_detector.py   Phase 3 silver labels + detector fit
  run_phase4.py       Phase 4 method comparison report
  audit_sample.py     blind / reveal AI audit of detector calls
notebooks/
  halo_colab.ipynb   Phase 2 GPU runs
tests/
```

## Team

Reyyi Shreyas, Mayank Prakash, Sonali N Gowda, Anubhav
