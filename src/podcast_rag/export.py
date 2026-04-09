"""Export processed episodes from P3 DuckDB to JSONL for embedding."""

import json
import sys
from pathlib import Path

import duckdb
from rich.console import Console
from rich.table import Table

P3_DB_PATH = Path.home() / "Code" / "parakeet-podcast-processor" / "data" / "p3_snapshot.duckdb"
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "episodes.jsonl"

EXPORT_QUERY = """
SELECT
    e.id AS episode_id,
    e.title,
    p.title AS podcast,
    e.date,
    s.key_topics,
    s.themes,
    s.quotes,
    s.key_takeaways,
    s.full_summary
FROM episodes e
JOIN summaries s ON e.id = s.episode_id
JOIN podcasts p ON e.podcast_id = p.id
WHERE e.status = 'processed'
ORDER BY e.date DESC
"""

STATS_QUERY = """
SELECT
    p.title AS podcast,
    COUNT(*) AS episode_count,
    MIN(e.date) AS earliest,
    MAX(e.date) AS latest,
    ROUND(AVG(LENGTH(s.full_summary)), 0) AS avg_summary_len
FROM episodes e
JOIN summaries s ON e.id = s.episode_id
JOIN podcasts p ON e.podcast_id = p.id
WHERE e.status = 'processed'
GROUP BY p.title
ORDER BY episode_count DESC
"""


def parse_json_field(value):
    """Parse a JSON field, returning it as-is if already parsed by DuckDB."""
    if value is None:
        return []
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def export(db_path: Path = P3_DB_PATH, output_path: Path = OUTPUT_PATH) -> int:
    console = Console()

    if not db_path.exists():
        console.print(f"[red]Database not found: {db_path}[/red]")
        sys.exit(1)

    con = duckdb.connect(str(db_path), read_only=True)

    try:
        rows = con.execute(EXPORT_QUERY).fetchall()
        columns = [desc[0] for desc in con.description]

        output_path.parent.mkdir(parents=True, exist_ok=True)

        null_count = 0
        with open(output_path, "w") as f:
            for row in rows:
                record = dict(zip(columns, row))

                # Convert datetime to ISO string
                if record["date"]:
                    record["date"] = record["date"].isoformat()

                # Parse JSON fields
                for field in ("key_topics", "themes", "quotes", "key_takeaways"):
                    record[field] = parse_json_field(record[field])

                # Track nulls in critical fields
                for field in ("title", "podcast", "full_summary"):
                    if not record.get(field):
                        null_count += 1

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Validation
        exported = len(rows)
        db_count = con.execute(
            "SELECT COUNT(*) FROM episodes e JOIN summaries s ON e.id = s.episode_id WHERE e.status = 'processed'"
        ).fetchone()[0]

        console.print()
        if exported != db_count:
            console.print(f"[red]Count mismatch: exported {exported}, expected {db_count}[/red]")
        else:
            console.print(f"[green]Exported {exported} episodes to {output_path}[/green]")

        if null_count > 0:
            console.print(f"[yellow]Warning: {null_count} null values in critical fields[/yellow]")

        # Stats
        stats = con.execute(STATS_QUERY).fetchall()
        stat_cols = [desc[0] for desc in con.description] if con.description else []

        table = Table(title="Episodes by Podcast")
        for col in stat_cols:
            table.add_column(col)
        for stat_row in stats:
            table.add_row(*[str(v) for v in stat_row])
        console.print(table)

        return exported

    finally:
        con.close()


if __name__ == "__main__":
    export()
