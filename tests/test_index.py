"""Tests for the retrieval index.

These use hand-built vectors rather than a real embedder, so they run
offline in milliseconds and pin the cosine-similarity contract that
Phase 3 depends on.
"""

import numpy as np
import pytest

from halo.chunk import Chunk, chunk_sentences
from halo.index import RetrievalIndex, TokenCodec, query_prefix


def make_chunks() -> list[Chunk]:
    """Three chunks with distinct, readable texts."""
    return [
        Chunk(
            text="alpha", sentences=["alpha"], source_title="A", token_count=1
        ),
        Chunk(
            text="beta", sentences=["beta"], source_title="B", token_count=1
        ),
        Chunk(
            text="gamma delta",
            sentences=["gamma delta"],
            source_title="C",
            token_count=2,
        ),
    ]


def make_vectors() -> np.ndarray:
    """Unit vectors: A on +x, B on +y, C halfway between."""
    return np.array([[1, 0], [0, 1], [0.7071, 0.7071]], dtype=np.float32)


def test_results_are_ordered_by_cosine_similarity():
    index = RetrievalIndex(make_chunks(), make_vectors())
    hits = index.search_vector(np.array([1.0, 0.0]), k=3)
    assert [h.chunk.text for h in hits] == ["alpha", "gamma delta", "beta"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_scores_are_cosine_similarities_in_unit_range():
    index = RetrievalIndex(make_chunks(), make_vectors())
    hits = index.search_vector(np.array([0.0, 1.0]), k=3)
    assert all(-1.0 <= h.score <= 1.0 for h in hits)
    assert hits[0].chunk.text == "beta"


def test_empty_index_returns_no_hits_rather_than_raising():
    index = RetrievalIndex([], np.zeros((0, 2), dtype=np.float32))
    assert index.size == 0
    assert index.search_vector(np.array([1.0, 0.0])) == []


def test_k_is_clamped_to_the_index_size():
    index = RetrievalIndex(make_chunks(), make_vectors())
    assert len(index.search_vector(np.array([1.0, 0.0]), k=99)) == 3


def test_integer_vectors_are_coerced_to_float32():
    index = RetrievalIndex(make_chunks(), make_vectors().astype(np.int64))
    assert index.search_vector(np.array([1, 0]), k=1)[0].chunk.text == "alpha"


def test_mismatched_chunk_and_vector_counts_raise():
    with pytest.raises(ValueError, match="chunks but vectors"):
        RetrievalIndex(make_chunks(), np.zeros((2, 2), dtype=np.float32))


def test_search_without_an_encoder_explains_how_to_fix_it():
    index = RetrievalIndex(make_chunks(), make_vectors())
    with pytest.raises(RuntimeError, match="no encoder"):
        index.search("what is alpha?")


def test_chunks_property_returns_the_indexed_chunks():
    index = RetrievalIndex(make_chunks(), make_vectors())
    assert [c.text for c in index.chunks] == ["alpha", "beta", "gamma delta"]


def test_bge_models_receive_the_retrieval_instruction():
    assert query_prefix("BAAI/bge-small-en-v1.5").startswith("Represent")
    assert query_prefix("BAAI/bge-base-en-v1.5").startswith("Represent")


def test_other_embedders_receive_no_instruction():
    assert query_prefix("sentence-transformers/all-mpnet-base-v2") == ""


def test_save_and_load_roundtrip(tmp_path):
    index = RetrievalIndex(make_chunks(), make_vectors(), embedder_name="fake")
    index.save(tmp_path)

    loaded = RetrievalIndex.load(tmp_path)
    assert loaded.size == 3
    assert loaded.embedder_name == "fake"
    assert [c.text for c in loaded.chunks] == ["alpha", "beta", "gamma delta"]
    assert loaded.chunks[2].source_title == "C"


def test_loaded_index_can_still_search(tmp_path):
    index = RetrievalIndex(make_chunks(), make_vectors())
    index.save(tmp_path)
    loaded = RetrievalIndex.load(tmp_path)
    assert (
        loaded.search_vector(np.array([0.0, 1.0]), k=1)[0].chunk.text == "beta"
    )


def test_faiss_matches_numpy_ground_truth():
    """Pin faiss top-k against a numpy brute-force reference.

    torch and faiss-cpu each bundle LLVM's libomp, and halo tolerates
    the duplicate runtime so both can be imported. This test measures
    that the tolerated conflict stays numerically exact, because a
    library bump that quietly perturbed these scores would corrupt
    every grounding signal in Phase 3 without any test noticing.
    """
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((500, 384)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    chunks = [
        Chunk(
            text=f"c{i}", sentences=[f"c{i}"], source_title="", token_count=1
        )
        for i in range(500)
    ]
    index = RetrievalIndex(chunks, vectors)

    query = rng.standard_normal(384).astype(np.float32)
    query /= np.linalg.norm(query)

    hits = index.search_vector(query, k=10)
    expected_ids = np.argsort(-(vectors @ query))[:10]

    assert [h.chunk.text for h in hits] == [f"c{i}" for i in expected_ids]
    np.testing.assert_allclose(
        [h.score for h in hits], vectors[expected_ids] @ query, atol=1e-6
    )


class _CharTokenizer:
    """Character-level stand-in: one token per character.

    Lets TokenCodec be tested offline, without downloading bge-small.
    """

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(i) for i in ids)


def test_token_codec_counts_tokens_when_called():
    codec = TokenCodec(_CharTokenizer())
    assert codec("abc") == [ord("a"), ord("b"), ord("c")]


def test_token_codec_passes_short_text_through_unchanged():
    codec = TokenCodec(_CharTokenizer())
    assert codec.split("abc", max_tokens=4) == ["abc"]


def test_token_codec_cuts_long_text_to_the_ceiling():
    codec = TokenCodec(_CharTokenizer())
    assert codec.split("abcdefghij", max_tokens=4) == ["abcd", "efgh", "ij"]


def test_token_codec_is_a_drop_in_tokenizer():
    """A codec must satisfy the Tokenizer contract chunk.py expects."""
    codec = TokenCodec(_CharTokenizer())
    chunks = chunk_sentences(
        ["a" * 40],
        tokenizer=codec,
        splitter=codec.split,
        min_tokens=5,
        max_tokens=15,
    )
    assert all(c.token_count <= 15 for c in chunks)


def test_token_codec_reports_the_real_context_window():
    class Windowed(_CharTokenizer):
        model_max_length = 512

    codec = TokenCodec(Windowed())
    assert codec.context_window == 512
    # The reported limit is raised afterwards, so transformers stops
    # warning about the very inputs we are about to split.
    assert codec.tokenizer.model_max_length > 512


def test_token_codec_tolerates_a_tokenizer_without_a_window():
    codec = TokenCodec(_CharTokenizer())
    assert codec.context_window == 0
