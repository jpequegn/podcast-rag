"""ChromaDB vector store for podcast episodes."""

import json
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from podcast_rag.embed import CACHE_DIR, EPISODES_PATH, build_document, load_episodes

CHROMA_DIR = Path(__file__).resolve().parents[2] / "data" / "chroma"
COLLECTION_NAME = "episodes"
BATCH_SIZE = 256


class VectorStore:
    def __init__(self, chroma_dir: Path = CHROMA_DIR):
        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.collection.count()

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        batch_size: int = BATCH_SIZE,
    ) -> int:
        """Bulk insert episodes. Returns count added."""
        added = 0
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self.collection.upsert(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )
            added += end - i
        return added

    def search(
        self,
        query: str | None = None,
        query_embedding: list[float] | None = None,
        n_results: int = 5,
        where: dict | None = None,
    ) -> dict:
        """Search by text query (requires embedding) or pre-computed embedding."""
        kwargs: dict[str, Any] = {"n_results": n_results}
        if where:
            kwargs["where"] = where
        if query_embedding:
            kwargs["query_embeddings"] = [query_embedding]
        elif query:
            import ollama
            from podcast_rag.embed import MODEL
            result = ollama.embed(model=MODEL, input=[query])
            kwargs["query_embeddings"] = [result.embeddings[0]]
        else:
            raise ValueError("Provide either query or query_embedding")
        return self.collection.query(**kwargs)


def build_metadata(episode: dict) -> dict[str, str]:
    """Build flat metadata dict for ChromaDB (string values only)."""
    topics = episode.get("key_topics") or []
    themes = episode.get("themes") or []
    return {
        "episode_id": str(episode["episode_id"]),
        "podcast": episode.get("podcast", ""),
        "date": episode.get("date", ""),
        "key_topics": json.dumps(topics),
        "themes": json.dumps(themes),
    }


def ingest(
    episodes_path: Path = EPISODES_PATH,
    cache_dir: Path = CACHE_DIR,
    chroma_dir: Path = CHROMA_DIR,
) -> int:
    """Load embeddings cache and ingest into ChromaDB. Returns count."""
    console = Console()

    vectors_path = cache_dir / "vectors.npy"
    ids_path = cache_dir / "episode_ids.json"
    if not vectors_path.exists() or not ids_path.exists():
        console.print("[red]Embeddings cache not found. Run embed.py first.[/red]")
        return 0

    episode_ids = json.loads(ids_path.read_text())
    vectors = np.load(vectors_path)
    episodes = load_episodes(episodes_path)
    ep_map = {ep["episode_id"]: ep for ep in episodes}

    store = VectorStore(chroma_dir)

    if store.count() == len(episode_ids):
        console.print(f"[green]All {store.count()} episodes already in ChromaDB.[/green]")
        return store.count()

    ids: list[str] = []
    embeddings: list[list[float]] = []
    documents: list[str] = []
    metadatas: list[dict[str, str]] = []
    seen: set[int] = set()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Preparing", total=len(episode_ids))
        for i, eid in enumerate(episode_ids):
            progress.advance(task)
            if eid in seen:
                continue
            seen.add(eid)
            ep = ep_map.get(eid)
            if not ep:
                continue
            ids.append(str(eid))
            embeddings.append(vectors[i].tolist())
            documents.append(build_document(ep))
            metadatas.append(build_metadata(ep))

    console.print(f"Inserting {len(ids)} episodes into ChromaDB...")
    added = store.add(ids, embeddings, documents, metadatas)

    console.print(f"[green]ChromaDB: {store.count()} episodes indexed.[/green]")

    # Demo search
    console.print("\n[bold]Demo search: 'world models AGI'[/bold]")
    results = store.search(query="world models AGI", n_results=5)
    for i, (doc_id, dist) in enumerate(zip(results["ids"][0], results["distances"][0])):
        meta = results["metadatas"][0][i]
        doc = results["documents"][0][i][:80] if results["documents"] else ""
        console.print(f"  {i+1}. [{meta['podcast']}] {doc}... (dist={dist:.3f})")

    return added


if __name__ == "__main__":
    ingest()
