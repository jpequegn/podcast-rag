"""Tests for hybrid search (BM25 + vector)."""

from podcast_rag.hybrid import HybridSearch, BM25Index, tokenize, ScoredEpisode


def test_tokenize():
    tokens = tokenize("Hello, World! This is a TEST.")
    assert tokens == ["hello", "world", "this", "is", "a", "test"]


def test_bm25_index():
    idx = BM25Index(
        episode_ids=["1", "2", "3"],
        documents=["AI and machine learning", "cooking recipes pasta", "AI agents and LLMs"],
    )
    results = idx.search("AI machine learning", k=2)
    assert len(results) >= 1
    assert results[0][0] in ("1", "3")


def test_hybrid_search_vector_mode():
    search = HybridSearch()
    results = search.search("AI infrastructure", k=3, mode="vector")
    assert len(results) == 3
    assert all(isinstance(r, ScoredEpisode) for r in results)
    assert all(r.vector_score > 0 for r in results)


def test_hybrid_search_bm25_mode():
    search = HybridSearch()
    results = search.search("Dylan Patel Nvidia", k=3, mode="bm25")
    assert len(results) >= 1
    assert all(r.bm25_score > 0 for r in results)


def test_hybrid_search_hybrid_mode():
    search = HybridSearch()
    results = search.search("AI agents deployment", k=5, mode="hybrid")
    assert len(results) == 5
    # At least some should have both scores
    has_both = any(r.vector_score > 0 and r.bm25_score > 0 for r in results)
    assert has_both, "Hybrid mode should find episodes with both vector and BM25 scores"


def test_named_entity_bm25_advantage():
    """BM25 should find specific named entities better than generic vector search."""
    search = HybridSearch()
    bm25_results = search.search("Dylan Patel Nvidia Blackwell", k=5, mode="bm25")
    # Top BM25 result should contain these terms
    top_doc = bm25_results[0].document.lower()
    assert "nvidia" in top_doc or "patel" in top_doc or "blackwell" in top_doc
