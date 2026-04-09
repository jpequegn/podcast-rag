"""Tests for the embedding pipeline."""

import json
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "embeddings"
EPISODES_PATH = Path(__file__).resolve().parents[1] / "data" / "episodes.jsonl"


def test_cache_exists():
    assert (CACHE_DIR / "vectors.npy").exists()
    assert (CACHE_DIR / "episode_ids.json").exists()


def test_embedding_count_matches_episodes():
    with open(EPISODES_PATH) as f:
        episode_count = sum(1 for _ in f)
    ids = json.loads((CACHE_DIR / "episode_ids.json").read_text())
    vectors = np.load(CACHE_DIR / "vectors.npy")
    assert len(ids) == episode_count
    assert vectors.shape[0] == episode_count


def test_embedding_dimensions():
    vectors = np.load(CACHE_DIR / "vectors.npy")
    assert vectors.shape[1] == 768, f"Expected 768 dims, got {vectors.shape[1]}"


def test_no_nan_embeddings():
    vectors = np.load(CACHE_DIR / "vectors.npy")
    assert not np.any(np.isnan(vectors)), "Found NaN values in embeddings"


def test_build_document():
    from podcast_rag.embed import build_document

    episode = {
        "title": "Test Episode",
        "key_topics": ["AI", "ML"],
        "key_takeaways": ["takeaway1"],
        "full_summary": "A summary.",
    }
    doc = build_document(episode)
    assert "Test Episode" in doc
    assert "AI, ML" in doc
    assert "takeaway1" in doc
    assert "A summary." in doc
