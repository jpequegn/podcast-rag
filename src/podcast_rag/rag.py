"""RAG pipeline: retrieve → build context → generate answer with citations."""

import json
from dataclasses import dataclass, field

import ollama
from rich.console import Console
from rich.markdown import Markdown

from podcast_rag.store import VectorStore

GENERATION_MODEL = "gemma3"
TOP_K = 5
EXPAND_K = 20
RERANK_K = 5


@dataclass
class Episode:
    episode_id: str
    podcast: str
    date: str
    key_topics: list[str]
    themes: list[str]
    document: str
    distance: float


@dataclass
class RAGResult:
    query: str
    answer: str
    episodes: list[Episode]
    expanded_queries: list[str] = field(default_factory=list)


def retrieve(store: VectorStore, query: str, k: int = TOP_K, where: dict | None = None) -> list[Episode]:
    """Embed query and retrieve top-k similar episodes."""
    results = store.search(query=query, n_results=k, where=where)
    episodes = []
    for i, doc_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        episodes.append(Episode(
            episode_id=doc_id,
            podcast=meta.get("podcast", ""),
            date=meta.get("date", "")[:10],
            key_topics=json.loads(meta.get("key_topics", "[]")),
            themes=json.loads(meta.get("themes", "[]")),
            document=results["documents"][0][i] if results["documents"] else "",
            distance=results["distances"][0][i],
        ))
    return episodes


def expand_query(query: str, n: int = 3) -> list[str]:
    """Generate query variants for diversified retrieval."""
    response = ollama.chat(
        model=GENERATION_MODEL,
        messages=[{
            "role": "user",
            "content": (
                f"Generate {n} alternative search queries for: \"{query}\"\n"
                "Each should capture a different angle or aspect.\n"
                "Return ONLY the queries, one per line, no numbering or explanation."
            ),
        }],
    )
    variants = [line.strip() for line in response.message.content.strip().split("\n") if line.strip()]
    return variants[:n]


def diversified_retrieve(store: VectorStore, query: str, k: int = RERANK_K, where: dict | None = None) -> list[Episode]:
    """Expand query into variants, retrieve from each, deduplicate and merge."""
    variants = expand_query(query)
    all_queries = [query] + variants

    seen: dict[str, Episode] = {}
    for q in all_queries:
        episodes = retrieve(store, q, k=EXPAND_K, where=where)
        for ep in episodes:
            if ep.episode_id not in seen or ep.distance < seen[ep.episode_id].distance:
                seen[ep.episode_id] = ep

    # Sort by best distance and take top-k
    ranked = sorted(seen.values(), key=lambda e: e.distance)
    return ranked[:k]


def rerank(query: str, episodes: list[Episode], k: int = RERANK_K) -> list[Episode]:
    """LLM-based reranking: ask the model to score relevance."""
    if len(episodes) <= k:
        return episodes

    episode_summaries = "\n".join(
        f"[{i}] ({ep.podcast}, {ep.date}) {ep.document[:200]}"
        for i, ep in enumerate(episodes)
    )
    response = ollama.chat(
        model=GENERATION_MODEL,
        messages=[{
            "role": "user",
            "content": (
                f"Query: \"{query}\"\n\n"
                f"Rank these episodes by relevance to the query. Return ONLY the indices "
                f"of the top {k} most relevant, comma-separated (e.g. 3,7,1,5,0):\n\n"
                f"{episode_summaries}"
            ),
        }],
    )

    # Parse indices
    try:
        text = response.message.content.strip().split("\n")[0]
        indices = [int(x.strip()) for x in text.split(",") if x.strip().isdigit()]
        indices = [i for i in indices if 0 <= i < len(episodes)]
        reranked = [episodes[i] for i in indices[:k]]
        # Fill remaining if LLM returned fewer than k
        remaining = [ep for i, ep in enumerate(episodes) if i not in indices]
        reranked.extend(remaining[: k - len(reranked)])
        return reranked
    except (ValueError, IndexError):
        return episodes[:k]


def build_context(episodes: list[Episode]) -> str:
    """Format retrieved episodes into a structured context block."""
    blocks = []
    for i, ep in enumerate(episodes, 1):
        topics = ", ".join(ep.key_topics[:5]) if ep.key_topics else "N/A"
        blocks.append(
            f"[Source {i}] Podcast: {ep.podcast} | Date: {ep.date} | "
            f"Topics: {topics}\n{ep.document}"
        )
    return "\n\n---\n\n".join(blocks)


SYSTEM_PROMPT = """You are a podcast research assistant with access to summaries from tech and business podcasts.
Answer questions using ONLY the provided source episodes. For each claim, cite the source using [Source N] notation.
If the sources don't contain enough information, say so explicitly. Be concise and specific."""


def generate(query: str, context: str) -> str:
    """Generate an answer using Ollama with the retrieved context."""
    response = ollama.chat(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )
    return response.message.content


def query(
    question: str,
    k: int = RERANK_K,
    where: dict | None = None,
    use_expansion: bool = True,
    use_rerank: bool = True,
    search_mode: str = "vector",
) -> RAGResult:
    """Full RAG pipeline: retrieve → rerank → generate with citations."""
    if search_mode in ("bm25", "hybrid"):
        from podcast_rag.hybrid import HybridSearch
        hybrid = HybridSearch(alpha=0.5)
        scored = hybrid.search(question, k=k * 4 if use_rerank else k, mode=search_mode, where=where if search_mode == "hybrid" else None)
        episodes = [
            Episode(
                episode_id=s.episode_id, podcast=s.podcast, date=s.date,
                key_topics=s.key_topics, themes=s.themes,
                document=s.document, distance=1.0 - s.hybrid_score,
            )
            for s in scored
        ]
        expanded = []
    else:
        store = VectorStore()
        if use_expansion:
            episodes = diversified_retrieve(store, question, k=k * 4, where=where)
            expanded = expand_query(question)
        else:
            episodes = retrieve(store, question, k=k * 4 if use_rerank else k, where=where)
            expanded = []

    if use_rerank and len(episodes) > k:
        episodes = rerank(question, episodes, k=k)
    else:
        episodes = episodes[:k]

    context = build_context(episodes)
    answer = generate(question, context)

    return RAGResult(
        query=question,
        answer=answer,
        episodes=episodes,
        expanded_queries=expanded,
    )


def main():
    console = Console()
    question = "What has been said about AI infrastructure bottlenecks in the last 6 months?"

    console.print(f"\n[bold]Query:[/bold] {question}\n")

    result = query(question)

    if result.expanded_queries:
        console.print("[dim]Query variants:[/dim]")
        for q in result.expanded_queries:
            console.print(f"  - {q}")
        console.print()

    console.print("[bold]Sources:[/bold]")
    for i, ep in enumerate(result.episodes, 1):
        console.print(f"  {i}. [{ep.podcast}] {ep.document[:80]}... ({ep.date}, dist={ep.distance:.3f})")

    console.print()
    console.print(Markdown(result.answer))


if __name__ == "__main__":
    main()
