"""Dense retrieval over a FAISS index.

Embeddings are L2-normalised and searched with an inner-product index,
which makes the returned score a cosine similarity. That matters
downstream: Phase 3's grounding signal *is* the max cosine similarity
between a generated sentence and the retrieved chunks, so retrieval
and detection agree on the scale by construction.

The default embedder is ``BAAI/bge-small-en-v1.5`` (33M parameters),
chosen because the proposal asks for a lightweight pipeline and this
model is small enough to run on the same GPU as a quantized generator.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import faiss
import numpy as np

from halo.chunk import Chunk
from halo.device import default_device

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from transformers import PreTrainedTokenizerBase

DEFAULT_EMBEDDER = "BAAI/bge-small-en-v1.5"

# BGE v1.5 is trained to expect this instruction on queries only.
# Omitting it costs a few points of retrieval recall.
BGE_QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)


def query_prefix(model_name: str) -> str:
    """Return the query-side instruction a model was trained to expect.

    Parameters
    ----------
    model_name : str
        Hugging Face model id.

    Returns
    -------
    str
        Prefix to prepend to queries, or ``""`` for models that want
        none.
    """
    if model_name.startswith("BAAI/bge-"):
        return BGE_QUERY_INSTRUCTION
    return ""


@dataclass
class TokenCodec:
    """A tokenizer that can also cut text it would otherwise truncate.

    ``encode`` and ``__call__`` count tokens; :meth:`split` hard-cuts a
    span too long for the embedder's context window. Without this, a
    chunk longer than that window is silently truncated by the model
    and its tail is never embedded.

    Attributes
    ----------
    tokenizer : PreTrainedTokenizerBase
        The wrapped Hugging Face tokenizer.
    """

    tokenizer: PreTrainedTokenizerBase
    context_window: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        # Splitting enforces the context window, so a long input is
        # expected here rather than a mistake. Remember the real window
        # before raising the reported limit, which silences
        # transformers' truncation warning on exactly the texts we cut.
        self.context_window = int(
            getattr(self.tokenizer, "model_max_length", 0) or 0
        )
        if self.context_window > 0:
            self.tokenizer.model_max_length = 1_000_000

    def encode(self, text: str) -> list[int]:
        """Return token ids without special tokens."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def __call__(self, text: str) -> list[int]:
        """Alias for :meth:`encode`, so a codec counts as a ``Tokenizer``."""
        return self.encode(text)

    def split(self, text: str, max_tokens: int) -> list[str]:
        """Cut ``text`` into pieces of at most ``max_tokens`` tokens.

        The cut falls on token boundaries, so pieces can break inside a
        word. That is the standard trade: an ungrammatical boundary
        still embeds and retrieves, while a truncated chunk loses its
        evidence entirely.

        Parameters
        ----------
        text : str
            Text to cut.
        max_tokens : int
            Ceiling applied to each piece.

        Returns
        -------
        list[str]
            Non-empty pieces in order, or a single piece when the text
            already fits.
        """
        ids = self.encode(text)
        if len(ids) <= max_tokens:
            return [text]
        return [
            self.tokenizer.decode(
                ids[i : i + max_tokens], skip_special_tokens=True
            ).strip()
            for i in range(0, len(ids), max_tokens)
        ]


def load_tokenizer(model_name: str = DEFAULT_EMBEDDER) -> TokenCodec:
    """Load the embedder's tokenizer as a count-and-split codec.

    Parameters
    ----------
    model_name : str
        Model whose tokenizer to load. Using the embedder's own keeps
        chunk sizes, splitting, and embedding input in one token space.

    Returns
    -------
    TokenCodec
        Callable as ``str -> list[int]``, and able to split over-long
        spans.
    """
    from transformers import AutoTokenizer

    return TokenCodec(AutoTokenizer.from_pretrained(model_name))


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk with its cosine similarity to the query.

    Attributes
    ----------
    chunk : Chunk
        The retrieved chunk.
    score : float
        Cosine similarity, in ``[-1, 1]``.
    """

    chunk: Chunk
    score: float


class RetrievalIndex:
    """Cosine-similarity search over chunk embeddings.

    Attributes
    ----------
    embedder_name : str
        Model that produced the vectors.
    size : int
        Number of indexed chunks.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
        embedder_name: str = DEFAULT_EMBEDDER,
        encoder: SentenceTransformer | None = None,
    ) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"{len(chunks)} chunks but vectors have shape {vectors.shape}"
            )
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        self.embedder_name = embedder_name
        self._chunks = list(chunks)
        self._encoder = encoder
        # Keep the dimension even when the index is empty: a (0, d)
        # matrix still reports d, and faiss needs it to accept a later
        # search rather than rejecting the query's column count.
        self._dim = vectors.shape[1] if vectors.ndim == 2 else 0
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(vectors)

    @property
    def size(self) -> int:
        """Number of chunks in the index."""
        return len(self._chunks)

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        return self._dim

    @property
    def chunks(self) -> list[Chunk]:
        """The indexed chunks, in insertion order."""
        return list(self._chunks)

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        """Retrieve the ``k`` chunks most similar to a text query.

        Parameters
        ----------
        query : str
            The user question.
        k : int
            Number of chunks to return.

        Returns
        -------
        list[ScoredChunk]
            Best match first. Empty if the index is empty.

        Raises
        ------
        RuntimeError
            If the index was built from raw vectors and has no encoder
            attached; use :meth:`search_vector` in that case.
        """
        if self._encoder is None:
            raise RuntimeError(
                "this index has no encoder attached; build it with "
                "build_index(), or call search_vector() directly"
            )
        text = query_prefix(self.embedder_name) + query
        vector = self._encoder.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )
        return self.search_vector(vector, k)

    def search_vector(
        self, vector: np.ndarray, k: int = 5
    ) -> list[ScoredChunk]:
        """Retrieve the ``k`` chunks nearest to a precomputed vector.

        Parameters
        ----------
        vector : np.ndarray
            Shape ``(dim,)`` or ``(1, dim)``. Should already be
            normalised if cosine similarity is wanted.
        k : int
            Number of chunks to return.

        Returns
        -------
        list[ScoredChunk]
            Best match first.
        """
        if self.size == 0:
            return []
        query = np.asarray(vector, dtype=np.float32)
        if query.ndim == 1:
            query = query[None, :]
        scores, ids = self._index.search(query, min(k, self.size))
        return [
            ScoredChunk(chunk=self._chunks[i], score=float(s))
            for s, i in zip(scores[0], ids[0], strict=True)
            if i >= 0
        ]

    def save(self, directory: str | Path) -> None:
        """Write the index and its chunks to a directory.

        Parameters
        ----------
        directory : str or Path
            Destination. Created if missing.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(out / "index.faiss"))
        payload = {
            "embedder_name": self.embedder_name,
            "chunks": [
                {
                    "text": c.text,
                    "sentences": c.sentences,
                    "source_title": c.source_title,
                    "token_count": c.token_count,
                }
                for c in self._chunks
            ],
        }
        (out / "chunks.json").write_text(json.dumps(payload))

    @classmethod
    def load(
        cls, directory: str | Path, encoder: SentenceTransformer | None = None
    ) -> RetrievalIndex:
        """Rebuild an index written by :meth:`save`.

        Parameters
        ----------
        directory : str or Path
            Directory previously passed to :meth:`save`.
        encoder : SentenceTransformer, optional
            Encoder to attach so :meth:`search` works. Without it only
            :meth:`search_vector` is available.

        Returns
        -------
        RetrievalIndex
            The rebuilt index.
        """
        src = Path(directory)
        payload = json.loads((src / "chunks.json").read_text())
        chunks = [Chunk(**c) for c in payload["chunks"]]
        vectors = faiss.read_index(str(src / "index.faiss"))
        # reconstruct_n takes (start, n). Its single-argument form
        # treats that argument as a start offset and returns nothing.
        matrix = (
            vectors.reconstruct_n(0, vectors.ntotal)
            if vectors.ntotal
            else np.zeros((0, vectors.d), dtype=np.float32)
        )
        return cls(
            chunks,
            matrix,
            embedder_name=payload["embedder_name"],
            encoder=encoder,
        )


def build_index(
    chunks: Sequence[Chunk],
    *,
    model_name: str = DEFAULT_EMBEDDER,
    device: str | None = None,
    batch_size: int = 64,
    encoder: SentenceTransformer | None = None,
) -> RetrievalIndex:
    """Embed every chunk and build the FAISS index.

    Parameters
    ----------
    chunks : Sequence[Chunk]
        Chunks to index.
    model_name : str
        Sentence-transformers model to embed with. Ignored when
        ``encoder`` is supplied.
    device : str, optional
        Torch device. Defaults to the fastest available.
    batch_size : int
        Encoding batch size.
    encoder : SentenceTransformer, optional
        Already-loaded model to reuse. The evaluation loop builds one
        index per question, and reloading a model once per question
        costs more than the retrieval it performs.

    Returns
    -------
    RetrievalIndex
        Ready to search, with the encoder attached.

    Raises
    ------
    ValueError
        If ``chunks`` is empty.
    """
    if not chunks:
        raise ValueError("cannot build an index from zero chunks")

    if encoder is None:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(
            model_name, device=device or str(default_device())
        )
    vectors = encoder.encode(
        [c.text for c in chunks],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(chunks) > 1_000,
    )
    return RetrievalIndex(
        chunks,
        np.asarray(vectors, dtype=np.float32),
        model_name,
        encoder=encoder,
    )
