"""Compute moat metrics: post-cutoff coverage, category wins, corpus stats."""

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

EPISODES_PATH = Path(__file__).resolve().parents[2] / "data" / "episodes.jsonl"
EVAL_RESULTS_PATH = Path(__file__).resolve().parents[2] / "data" / "eval_results.json"

MODEL_CUTOFFS = {
    "llama3.2 (Dec 2023)": datetime(2024, 1, 1),
    "Claude Sonnet 3.5 (Apr 2024)": datetime(2024, 5, 1),
    "GPT-4 / gemma3 (Aug 2024)": datetime(2024, 9, 1),
    "Claude 4 / GPT-5 / gemma4 (Jan 2025)": datetime(2025, 2, 1),
}


def compute_stats() -> dict:
    """Compute all moat-relevant stats from episode corpus and eval results."""
    with open(EPISODES_PATH) as f:
        episodes = [json.loads(line) for line in f]

    # Post-cutoff coverage
    post_cutoff = {}
    for label, cutoff in MODEL_CUTOFFS.items():
        post = sum(
            1 for ep in episodes
            if ep.get("date") and datetime.fromisoformat(ep["date"]) >= cutoff
        )
        post_cutoff[label] = (post, post / len(episodes) * 100)

    # Podcast breakdown
    podcasts = Counter(ep["podcast"] for ep in episodes)

    # Date range
    dates = sorted([ep["date"] for ep in episodes if ep.get("date")])
    date_range = (dates[0][:10], dates[-1][:10]) if dates else ("", "")

    # Top topics
    all_topics: list[str] = []
    for ep in episodes:
        all_topics.extend(ep.get("key_topics") or [])
    top_topics = Counter(all_topics).most_common(10)

    # Eval breakdown if available
    eval_summary = None
    if EVAL_RESULTS_PATH.exists():
        eval_results = json.loads(EVAL_RESULTS_PATH.read_text())
        wins = sum(1 for r in eval_results if r["rag_score"] > r["baseline_score"])
        ties = sum(1 for r in eval_results if r["rag_score"] == r["baseline_score"])
        losses = sum(1 for r in eval_results if r["rag_score"] < r["baseline_score"])

        by_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"rag": 0, "base": 0, "n": 0})
        for r in eval_results:
            by_cat[r["category"]]["rag"] += r["rag_score"]
            by_cat[r["category"]]["base"] += r["baseline_score"]
            by_cat[r["category"]]["n"] += 1

        eval_summary = {
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "by_category": dict(by_cat),
        }

    return {
        "total_episodes": len(episodes),
        "podcast_count": len(podcasts),
        "date_range": date_range,
        "avg_summary_len": sum(
            len(ep.get("full_summary") or "") for ep in episodes
        ) / len(episodes),
        "post_cutoff": post_cutoff,
        "top_topics": top_topics,
        "eval_summary": eval_summary,
    }


def print_report():
    console = Console()
    stats = compute_stats()

    console.print("\n[bold]Podcast-RAG Moat Report[/bold]\n")
    console.print(f"Total episodes: [bold]{stats['total_episodes']}[/bold]")
    console.print(f"Podcasts: [bold]{stats['podcast_count']}[/bold]")
    console.print(f"Date range: [green]{stats['date_range'][0]}[/green] → [green]{stats['date_range'][1]}[/green]")
    console.print(f"Avg summary length: {stats['avg_summary_len']:.0f} chars\n")

    # Post-cutoff table
    table = Table(title="Post-Cutoff Coverage")
    table.add_column("Model", style="cyan")
    table.add_column("Episodes", justify="right", style="bold")
    table.add_column("% of Corpus", justify="right", style="green")
    for label, (count, pct) in stats["post_cutoff"].items():
        table.add_row(label, str(count), f"{pct:.0f}%")
    console.print(table)

    # Top topics
    console.print("\n[bold]Top 10 Topics:[/bold]")
    for topic, count in stats["top_topics"]:
        console.print(f"  [yellow]{count:3d}x[/yellow]  {topic}")

    # Eval summary
    if stats["eval_summary"]:
        e = stats["eval_summary"]
        console.print(f"\n[bold]20-Question Eval:[/bold] Wins {e['wins']}, Ties {e['ties']}, Losses {e['losses']}")
        cat_table = Table(title="Eval by Category")
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("RAG", justify="right")
        cat_table.add_column("Baseline", justify="right")
        cat_table.add_column("N", justify="right")
        for cat, s in sorted(e["by_category"].items()):
            cat_table.add_row(cat, f"{s['rag']}/{s['n']*2}", f"{s['base']}/{s['n']*2}", str(s["n"]))
        console.print(cat_table)


if __name__ == "__main__":
    print_report()
