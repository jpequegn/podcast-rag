"""Tests for the CLI interface."""

from typer.testing import CliRunner
from podcast_rag.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "query" in result.output
    assert "search" in result.output
    assert "stats" in result.output


def test_stats():
    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0
    assert "episodes indexed" in result.output
    assert "podcasts" in result.output


def test_search():
    result = runner.invoke(app, ["search", "AI agents", "--top-k", "3"])
    assert result.exit_code == 0
    assert "AI" in result.output


def test_search_with_podcast_filter():
    result = runner.invoke(app, ["search", "AI", "--podcast", "The AI Breakdown", "--top-k", "3"])
    assert result.exit_code == 0
    assert "The AI Breakdown" in result.output


def test_query_fast():
    result = runner.invoke(app, ["query", "What is AI?", "--no-expansion", "--no-rerank", "--top-k", "2"])
    assert result.exit_code == 0
    assert "Answer" in result.output
    assert "Sources" in result.output
