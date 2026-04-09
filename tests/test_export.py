"""Tests for the export module."""

import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "episodes.jsonl"
REQUIRED_FIELDS = {"episode_id", "title", "podcast", "date", "key_topics", "themes", "quotes", "key_takeaways", "full_summary"}


def test_jsonl_exists():
    assert DATA_PATH.exists(), f"Expected {DATA_PATH} to exist"


def test_episode_count():
    with open(DATA_PATH) as f:
        count = sum(1 for _ in f)
    assert count >= 1000, f"Expected >= 1000 episodes, got {count}"


def test_record_schema():
    with open(DATA_PATH) as f:
        record = json.loads(f.readline())
    assert REQUIRED_FIELDS == set(record.keys()), f"Schema mismatch: {set(record.keys())}"


def test_no_null_critical_fields():
    nulls = []
    with open(DATA_PATH) as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            for field in ("title", "podcast", "full_summary"):
                if not record.get(field):
                    nulls.append((i, field))
    assert len(nulls) == 0, f"Found {len(nulls)} null critical fields: {nulls[:5]}"


def test_valid_json_per_line():
    with open(DATA_PATH) as f:
        for i, line in enumerate(f):
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                raise AssertionError(f"Invalid JSON on line {i}: {e}")
