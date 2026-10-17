"""Run Phase 2 generation locally on a Mac (MPS).

This is the local counterpart of ``notebooks/halo_colab.ipynb``. The
reference run (fp16) loads on MPS and produces the generations,
per-token log-probabilities, and retrieved chunks that Phase 3 trains
on, so a Mac is enough to unblock the detector.

The 8-bit and 4-bit runs need bitsandbytes, which has no reliable MPS
or CPU backend; loading them off-CUDA would silently change the
quantizer and break the fp16-vs-quantized comparison. Those runs are
skipped here with an explicit message, not faked, and stay reserved
for the Colab T4.

Retrieval is computed once and shared, exactly as in the notebook, so
no precision ever sees different chunks.

Usage
-----
    python scripts/run_precision_local.py              # all configs
    python scripts/run_precision_local.py --questions 5   # quick smoke
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch

    from halo.chunk import chunk_paragraphs
    from halo.data import make_splits
    from halo.device import default_device
    from halo.generate import (
        GenerationSettings,
        answer_question,
        load_generator,
        load_settings,
        summarize,
        write_jsonl,
    )
    from halo.index import DEFAULT_EMBEDDER, build_index, load_tokenizer
except ModuleNotFoundError as exc:  # wrong interpreter, not a code bug
    sys.exit(
        f"missing dependency ({exc.name}); run with the halo env's python:\n"
        "  conda activate halo\n"
        "or: conda run -n halo python scripts/run_precision_local.py"
    )


def build_retrieval(
    settings: GenerationSettings, n_questions: int, device: torch.device
) -> list[tuple[object, object]]:
    """Retrieve top-k chunks once, shared by every precision run.

    Parameters
    ----------
    settings : GenerationSettings
        Supplies ``questions`` (upper bound) and ``top_k``.
    n_questions : int
        How many test questions to retrieve for.
    device : torch.device
        Device for the sentence encoder.

    Returns
    -------
    list of tuple
        ``(record, hits)`` pairs, one per question with at least one
        chunk.
    """
    from sentence_transformers import SentenceTransformer

    records = make_splits()["test"][:n_questions]
    codec = load_tokenizer()
    encoder = SentenceTransformer(DEFAULT_EMBEDDER, device=str(device))

    retrieval: list[tuple[object, object]] = []
    for record in records:
        chunks = chunk_paragraphs(record.paragraphs, tokenizer=codec)
        if not chunks:
            print(f"skipping {record.id}: no chunks")
            continue
        index = build_index(chunks, encoder=encoder)
        hits = index.search(record.question, settings.top_k)
        retrieval.append((record, hits))
    return retrieval


def main(questions: int | None = None) -> None:
    """Run every precision the current device supports.

    Parameters
    ----------
    questions : int or None
        Override the YAML ``questions`` value (useful for a smoke
        run). ``None`` uses each config's own value.
    """
    device = default_device()
    print(f"device: {device}")
    if device.type != "cuda":
        print(
            "note: no CUDA GPU here. fp16 will run; int8/nf4 need "
            "bitsandbytes and stay on Colab.\n"
        )

    configs = [
        load_settings(f"configs/{p}.yaml") for p in ("fp16", "int8", "nf4")
    ]
    if questions is not None:
        configs = [
            dataclasses.replace(s, questions=questions) for s in configs
        ]

    runnable = [
        s
        for s in configs
        if s.precision == "fp16" or device.type == "cuda"
    ]
    for skipped in set(s.precision for s in configs) - set(
        s.precision for s in runnable
    ):
        print(f"skipping {skipped}: needs a CUDA GPU (bitsandbytes)")

    n = max(s.questions for s in runnable)
    retrieval = build_retrieval(configs[0], n, device)
    print(f"cached retrieval for {len(retrieval)} questions\n")

    for settings in runnable:
        print(f"=== {settings.precision} ===", flush=True)
        model, tokenizer = load_generator(settings, device=device)
        run = [
            answer_question(model, tokenizer, record, hits, settings)
            for record, hits in retrieval
        ]
        write_jsonl(settings.output, run)
        print(summarize(run), flush=True)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    print("\nwritten:")
    for settings in runnable:
        print(f"  {settings.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--questions",
        type=int,
        default=None,
        help="override the per-config question count (smoke runs)",
    )
    args = parser.parse_args()
    main(args.questions)
