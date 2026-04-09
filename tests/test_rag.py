"""Tests for the RAG pipeline."""

from podcast_rag.rag import retrieve, build_context, query, Episode
from podcast_rag.store import VectorStore


def test_retrieve_returns_episodes():
    store = VectorStore()
    episodes = retrieve(store, "AI infrastructure", k=5)
    assert len(episodes) == 5
    assert all(isinstance(ep, Episode) for ep in episodes)
    assert all(ep.podcast for ep in episodes)


def test_retrieve_distances_sorted():
    store = VectorStore()
    episodes = retrieve(store, "venture capital fundraising", k=10)
    distances = [ep.distance for ep in episodes]
    assert distances == sorted(distances), "Results should be sorted by distance"


def test_build_context_format():
    episodes = [
        Episode(
            episode_id="1", podcast="Test Pod", date="2026-01-01",
            key_topics=["AI"], themes=["future"],
            document="Test document content", distance=0.1,
        ),
    ]
    context = build_context(episodes)
    assert "[Source 1]" in context
    assert "Test Pod" in context
    assert "Test document content" in context


def test_full_rag_pipeline():
    result = query(
        "What has been said about AI infrastructure?",
        k=3,
        use_expansion=False,
        use_rerank=False,
    )
    assert result.answer
    assert len(result.episodes) == 3
    assert result.query == "What has been said about AI infrastructure?"


def test_rag_with_podcast_filter():
    result = query(
        "AI trends",
        k=3,
        where={"podcast": "The AI Breakdown"},
        use_expansion=False,
        use_rerank=False,
    )
    for ep in result.episodes:
        assert ep.podcast == "The AI Breakdown"
