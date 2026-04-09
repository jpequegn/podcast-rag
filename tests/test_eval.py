"""Tests for the evaluation framework."""

import json
from pathlib import Path

from podcast_rag.eval import EVAL_QUESTIONS, score_answer

RESULTS_PATH = Path(__file__).resolve().parents[1] / "data" / "eval_results.json"


def test_eval_questions_count():
    assert len(EVAL_QUESTIONS) == 20


def test_eval_questions_schema():
    required_keys = {"id", "question", "ground_truth", "category"}
    for q in EVAL_QUESTIONS:
        assert required_keys.issubset(q.keys()), f"Q{q.get('id', '?')} missing keys"


def test_eval_questions_unique_ids():
    ids = [q["id"] for q in EVAL_QUESTIONS]
    assert len(ids) == len(set(ids)), "Duplicate question IDs"


def test_eval_results_exist():
    assert RESULTS_PATH.exists(), "Run eval first: python -m podcast_rag.eval"
    results = json.loads(RESULTS_PATH.read_text())
    assert len(results) == 20


def test_eval_results_schema():
    results = json.loads(RESULTS_PATH.read_text())
    for r in results:
        assert "rag_score" in r
        assert "baseline_score" in r
        assert r["rag_score"] in (0, 1, 2)
        assert r["baseline_score"] in (0, 1, 2)


def test_score_answer_basic():
    score = score_answer(
        "The capital of France is Paris.",
        "Paris is the capital of France.",
    )
    assert score == 2
