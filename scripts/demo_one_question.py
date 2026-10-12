"""Trace one HotpotQA question end to end.

The smoke test proves retrieval works on average; this makes it
visible. For a single question it prints the ten candidate paragraphs,
the gold evidence sentences the retriever is supposed to surface, the
top-k chunks with their cosine scores, and a verdict. Use it when
someone asks "but what does it actually do?".

Usage
-----
    python scripts/demo_one_question.py [question_index] [k]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from halo.chunk import chunk_sentences
    from halo.data import load_hotpot
    from halo.device import default_device
    from halo.index import DEFAULT_EMBEDDER, build_index, load_tokenizer
except ModuleNotFoundError as exc:  # wrong interpreter, not a code bug
    sys.exit(
        f"missing dependency ({exc.name}); this script must run with the\n"
        "halo env's python, not another interpreter on your PATH. Check:\n"
        "  conda activate halo\n"
        "  which python   # expect .../envs/halo/bin/python\n"
        "or bypass PATH entirely:\n"
        "  conda run -n halo python scripts/demo_one_question.py"
    )


def main(index: int = 0, k: int = 5) -> None:
    """Retrieve for one question and print the full trace.

    Parameters
    ----------
    index : int
        Which question of the validation split to trace.
    k : int
        Number of chunks to retrieve and display.
    """
    from sentence_transformers import SentenceTransformer

    record = load_hotpot("validation", limit=index + 1)[index]
    tokenizer = load_tokenizer()

    chunks = []
    for para in record.paragraphs:
        chunks.extend(
            chunk_sentences(
                para.sentences,
                title=para.title,
                tokenizer=tokenizer,
                splitter=tokenizer.split,
                hard_max_tokens=tokenizer.context_window or None,
            )
        )

    encoder = SentenceTransformer(
        DEFAULT_EMBEDDER, device=str(default_device())
    )
    index_obj = build_index(chunks, encoder=encoder)
    retrieved = index_obj.search(record.question, k=k)

    gold_titles = set(record.gold_titles)
    print(f"question : {record.question}")
    print(f"answer   : {record.answer}")
    print()
    print("gold evidence the retriever must surface:")
    for title, sent_id, text in record.gold_sentences:
        print(f"  [{title}, sentence {sent_id}] {text}")
    print()
    print(f"corpus   : {len(record.paragraphs)} paragraphs ->")
    print(f"           {len(chunks)} chunks ({index_obj.dim}-d embeddings)")
    print()
    print(f"top {k} retrieved (cosine similarity to the question):")
    for rank, hit in enumerate(retrieved, start=1):
        tag = "GOLD" if hit.chunk.source_title in gold_titles else "    "
        print(
            f"  {rank}. {hit.score:.3f}  [{tag}] {hit.chunk.source_title}"
            f"  ({len(hit.chunk.sentences)} sentences,"
            f" {hit.chunk.token_count} tokens)"
        )

    found = {
        h.chunk.source_title
        for h in retrieved
        if h.chunk.source_title in gold_titles
    }
    print()
    verdict = "yes" if len(found) == len(gold_titles) else "PARTIAL"
    print(
        f"gold paragraphs in top {k}: {verdict}"
        f" ({len(found)}/{len(gold_titles)})"
    )


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:3]]
    main(*args)
