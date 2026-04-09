"""CLI for querying the podcast knowledge base."""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from podcast_rag.store import VectorStore

app = typer.Typer(name="praq", help="Podcast RAG — query your podcast knowledge base.")
console = Console()

EPISODES_PATH = Path(__file__).resolve().parents[2] / "data" / "episodes.jsonl"


def _build_where(podcast: str | None = None) -> dict | None:
    if podcast:
        return {"podcast": podcast}
    return None


@app.command()
def query(
    question: str = typer.Argument(..., help="Your question"),
    podcast: str | None = typer.Option(None, "--podcast", "-p", help="Filter by podcast name"),
    k: int = typer.Option(5, "--top-k", "-k", help="Number of sources to retrieve"),
    no_expansion: bool = typer.Option(False, "--no-expansion", help="Disable query expansion"),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Disable LLM reranking"),
    search_mode: str = typer.Option("vector", "--search-mode", "-m", help="Search mode: vector, bm25, or hybrid"),
):
    """Ask a question and get a cited answer from podcast episodes."""
    from podcast_rag.rag import query as rag_query

    where = _build_where(podcast)

    with console.status("[bold]Thinking..."):
        result = rag_query(
            question,
            k=k,
            where=where,
            use_expansion=not no_expansion,
            use_rerank=not no_rerank,
            search_mode=search_mode,
        )

    # Sources
    console.print()
    sources_table = Table(title="Sources", show_lines=True)
    sources_table.add_column("#", style="bold", width=3)
    sources_table.add_column("Podcast", style="cyan")
    sources_table.add_column("Date", style="green")
    sources_table.add_column("Topics", style="yellow")
    sources_table.add_column("Dist", style="dim", width=6)

    for i, ep in enumerate(result.episodes, 1):
        topics = ", ".join(ep.key_topics[:3]) if ep.key_topics else ""
        sources_table.add_row(str(i), ep.podcast, ep.date, topics, f"{ep.distance:.3f}")

    console.print(sources_table)
    console.print()

    # Answer
    console.print(Panel(Markdown(result.answer), title="Answer", border_style="blue"))

    if result.expanded_queries:
        console.print("\n[dim]Query variants used:[/dim]")
        for q in result.expanded_queries:
            console.print(f"  [dim]- {q}[/dim]")


@app.command()
def search(
    keywords: str = typer.Argument(..., help="Keywords to search for"),
    podcast: str | None = typer.Option(None, "--podcast", "-p", help="Filter by podcast name"),
    k: int = typer.Option(10, "--top-k", "-k", help="Number of results"),
):
    """Search for matching episodes by keywords."""
    from podcast_rag.rag import retrieve

    store = VectorStore()
    where = _build_where(podcast)
    episodes = retrieve(store, keywords, k=k, where=where)

    table = Table(title=f"Search: '{keywords}'", show_lines=True)
    table.add_column("#", style="bold", width=3)
    table.add_column("Podcast", style="cyan", max_width=25)
    table.add_column("Date", style="green", width=10)
    table.add_column("Title / Summary", style="white", max_width=70)
    table.add_column("Dist", style="dim", width=6)

    for i, ep in enumerate(episodes, 1):
        title = ep.document[:100] + "..." if len(ep.document) > 100 else ep.document
        table.add_row(str(i), ep.podcast, ep.date, title, f"{ep.distance:.3f}")

    console.print(table)


@app.command()
def stats():
    """Show corpus stats."""
    store = VectorStore()
    count = store.count()

    # Load episodes for detailed stats
    episodes = []
    if EPISODES_PATH.exists():
        with open(EPISODES_PATH) as f:
            for line in f:
                episodes.append(json.loads(line))

    # Podcast breakdown
    podcasts: dict[str, int] = {}
    dates: list[str] = []
    for ep in episodes:
        pod = ep.get("podcast", "Unknown")
        podcasts[pod] = podcasts.get(pod, 0) + 1
        if ep.get("date"):
            dates.append(ep["date"][:10])

    dates.sort()

    console.print(Panel(
        f"[bold]{count}[/bold] episodes indexed in ChromaDB\n"
        f"[bold]{len(podcasts)}[/bold] podcasts\n"
        f"Date range: [green]{dates[0]}[/green] → [green]{dates[-1]}[/green]" if dates else "",
        title="Corpus Stats",
        border_style="blue",
    ))

    table = Table(title="Episodes by Podcast")
    table.add_column("Podcast", style="cyan")
    table.add_column("Count", style="bold", justify="right")

    for pod, cnt in sorted(podcasts.items(), key=lambda x: -x[1]):
        table.add_row(pod, str(cnt))

    console.print(table)


def main():
    app()
