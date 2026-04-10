"""Tests for the moat analysis module."""

from pathlib import Path

from podcast_rag.moat import compute_stats, MODEL_CUTOFFS

MOAT_PATH = Path(__file__).resolve().parents[1] / "MOAT.md"


def test_moat_document_exists():
    assert MOAT_PATH.exists()
    content = MOAT_PATH.read_text()
    # Key sections
    assert "## TL;DR" in content
    assert "## The Corpus" in content
    assert "Post-cutoff" in content
    assert "## Where the RAG" in content
    assert "## The Honest Moat Breakdown" in content


def test_compute_stats_basic():
    stats = compute_stats()
    assert stats["total_episodes"] >= 1000
    assert stats["podcast_count"] >= 30
    assert stats["date_range"][0] < stats["date_range"][1]
    assert stats["avg_summary_len"] > 0


def test_post_cutoff_is_100_percent():
    """All episodes should post-date every model's training cutoff."""
    stats = compute_stats()
    for label, (count, pct) in stats["post_cutoff"].items():
        assert pct == 100.0, f"{label}: expected 100%, got {pct:.0f}%"


def test_eval_summary_loaded():
    stats = compute_stats()
    assert stats["eval_summary"] is not None
    e = stats["eval_summary"]
    assert e["wins"] + e["ties"] + e["losses"] == 20


def test_model_cutoffs_defined():
    assert len(MODEL_CUTOFFS) >= 4
    for label, cutoff in MODEL_CUTOFFS.items():
        assert cutoff.year >= 2023
