"""Phase 1 smoke test on real data.

Downloads from Hugging Face on first run, then measures two things the
unit tests cannot: that both dataset loaders survive real rows, and
that retrieval actually surfaces the gold evidence. The second is the
one that matters. Code that runs proves nothing if the retriever never
returns the paragraphs the answer came from.

Usage
-----
    python scripts/smoke_retrieval.py [n_questions] [k]
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from halo.chunk import chunk_sentences
    from halo.data import load_hotpot, load_truthful
    from halo.device import default_device
    from halo.index import DEFAULT_EMBEDDER, build_index, load_tokenizer
except ModuleNotFoundError as exc:  # wrong interpreter, not a code bug
    sys.exit(
        f"missing dependency ({exc.name}); this script must run with the\n"
        "halo env's python, not another interpreter on your PATH. Check:\n"
        "  conda activate halo\n"
        "  which python   # expect .../envs/halo/bin/python\n"
        "or bypass PATH entirely:\n"
        "  conda run -n halo python scripts/smoke_retrieval.py 50 5"
    )


def main(n_questions: int = 50, k: int = 5) -> None:
    """Load real data, index it, and report measured retrieval recall.

    Parameters
    ----------
    n_questions : int
        HotpotQA questions to evaluate.
    k : int
        Number of chunks retrieved per question.
    """
    from sentence_transformers import SentenceTransformer

    tokenizer = load_tokenizer()

    records = load_hotpot("validation", limit=n_questions)
    print(f"hotpotqa validation loaded : {len(records)} questions")
    print(f"paragraphs per question    : {len(records[0].paragraphs)}")
    print(f"gold supporting sentences  : {len(records[0].gold_sentences)}")

    truthful = load_truthful(limit=20)
    categories = sorted({r.category for r in truthful})
    print(f"truthfulqa loaded          : {len(truthful)} questions")
    print(f"  categories seen          : {categories[:4]}")

    # One encoder for the whole run: reloading per question would
    # dominate the runtime of an already cheap loop.
    encoder = SentenceTransformer(
        DEFAULT_EMBEDDER, device=str(default_device())
    )

    hits = 0
    chunk_counts: list[int] = []
    token_counts: list[int] = []
    for record in records:
        chunks = []
        for para in record.paragraphs:
            chunks.extend(
                chunk_sentences(
                    para.sentences,
                    title=para.title,
                    tokenizer=tokenizer,
                    # Without this, a sentence longer than the embedder's
                    # context window is silently truncated and its tail
                    # is never retrievable.
                    splitter=tokenizer.split,
                    # Likewise the tail fold must not rebuild a chunk
                    # the model would truncate again.
                    hard_max_tokens=tokenizer.context_window or None,
                )
            )
        chunk_counts.append(len(chunks))
        token_counts.extend(c.token_count for c in chunks)

        index = build_index(chunks, encoder=encoder)
        retrieved = index.search(record.question, k=k)

        # A hit means a retrieved chunk came from a paragraph holding a
        # gold supporting sentence. Chunk-level rather than sentence-
        # level, since one chunk may pack several sentences.
        gold_titles = set(record.gold_titles)
        if any(h.chunk.source_title in gold_titles for h in retrieved):
            hits += 1

    print()
    print(
        f"chunks per question        : mean "
        f"{statistics.mean(chunk_counts):.1f}"
        f"  min {min(chunk_counts)}  max {max(chunk_counts)}"
    )
    print(
        f"chunk tokens               : min {min(token_counts)}"
        f"  max {max(token_counts)}  (bge-small window is 512)"
    )
    print(
        f"recall@{k} (gold paragraph retrieved): "
        f"{hits}/{len(records)} = {hits / len(records):.3f}"
    )


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:3]]
    main(*args)
