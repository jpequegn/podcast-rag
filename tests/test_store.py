"""Tests for the ChromaDB vector store."""

from podcast_rag.store import VectorStore, build_metadata


def test_store_count():
    store = VectorStore()
    assert store.count() >= 1000, f"Expected >= 1000, got {store.count()}"


def test_search_returns_results():
    store = VectorStore()
    results = store.search(query="world models AGI", n_results=5)
    assert len(results["ids"][0]) == 5
    assert len(results["distances"][0]) == 5
    assert len(results["metadatas"][0]) == 5


def test_search_relevance():
    store = VectorStore()
    results = store.search(query="world models AGI", n_results=3)
    docs = " ".join(results["documents"][0]).lower()
    assert "world" in docs or "model" in docs or "agi" in docs


def test_search_with_podcast_filter():
    store = VectorStore()
    results = store.search(
        query="fundraising",
        n_results=3,
        where={"podcast": "The Twenty Minute VC"},
    )
    for meta in results["metadatas"][0]:
        assert meta["podcast"] == "The Twenty Minute VC"


def test_build_metadata():
    episode = {
        "episode_id": 42,
        "podcast": "Test Pod",
        "date": "2026-01-15T00:00:00",
        "key_topics": ["AI", "ML"],
        "themes": ["future"],
    }
    meta = build_metadata(episode)
    assert meta["episode_id"] == "42"
    assert meta["podcast"] == "Test Pod"
    assert meta["date"] == "2026-01-15T00:00:00"
    assert "AI" in meta["key_topics"]
