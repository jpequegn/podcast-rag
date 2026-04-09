"""Batch embedding pipeline using nomic-embed-text via Ollama."""

import json
import time
from pathlib import Path

import numpy as np
import ollama
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

MODEL = "nomic-embed-text"
BATCH_SIZE = 32
EPISODES_PATH = Path(__file__).resolve().parents[2] / "data" / "episodes.jsonl"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "embeddings"


def build_document(episode: dict) -> str:
    """Build a single text document from an episode for embedding."""
    topics = ", ".join(episode.get("key_topics", []) or [])
    takeaways = " ".join(episode.get("key_takeaways", []) or [])
    summary = episode.get("full_summary", "") or ""
    return f"{episode['title']} | {topics} | {takeaways} | {summary}"


def load_episodes(path: Path = EPISODES_PATH) -> list[dict]:
    episodes = []
    with open(path) as f:
        for line in f:
            episodes.append(json.loads(line))
    return episodes


def load_cache(cache_dir: Path = CACHE_DIR) -> dict[int, bool]:
    """Return set of episode_ids that are already embedded."""
    cached = {}
    ids_path = cache_dir / "episode_ids.json"
    if ids_path.exists():
        cached = {eid: True for eid in json.loads(ids_path.read_text())}
    return cached


def save_cache(episode_ids: list[int], embeddings: np.ndarray, cache_dir: Path = CACHE_DIR):
    """Save embeddings and episode IDs to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "vectors.npy", embeddings)
    (cache_dir / "episode_ids.json").write_text(json.dumps(episode_ids))


def embed(
    episodes_path: Path = EPISODES_PATH,
    cache_dir: Path = CACHE_DIR,
    batch_size: int = BATCH_SIZE,
) -> tuple[list[int], np.ndarray]:
    console = Console()
    episodes = load_episodes(episodes_path)
    cached = load_cache(cache_dir)

    # Split into cached and new
    to_embed = [(ep, build_document(ep)) for ep in episodes if ep["episode_id"] not in cached]
    already_cached = [ep for ep in episodes if ep["episode_id"] in cached]

    if not to_embed and already_cached:
        console.print(f"[green]All {len(already_cached)} episodes already cached.[/green]")
        ids = json.loads((cache_dir / "episode_ids.json").read_text())
        vectors = np.load(cache_dir / "vectors.npy")
        return ids, vectors

    console.print(f"Embedding {len(to_embed)} episodes ({len(already_cached)} cached)...")

    all_ids: list[int] = []
    all_vectors: list[list[float]] = []
    errors: list[tuple[int, str]] = []
    start = time.time()

    # Load existing cached vectors if any
    if already_cached and (cache_dir / "vectors.npy").exists():
        prev_ids = json.loads((cache_dir / "episode_ids.json").read_text())
        prev_vectors = np.load(cache_dir / "vectors.npy").tolist()
        all_ids.extend(prev_ids)
        all_vectors.extend(prev_vectors)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding", total=len(to_embed))

        for i in range(0, len(to_embed), batch_size):
            batch = to_embed[i : i + batch_size]
            batch_texts = [doc for _, doc in batch]
            batch_eps = [ep for ep, _ in batch]

            try:
                result = ollama.embed(model=MODEL, input=batch_texts)
                for ep, vec in zip(batch_eps, result.embeddings):
                    all_ids.append(ep["episode_id"])
                    all_vectors.append(vec)
            except Exception as e:
                for ep, _ in batch:
                    errors.append((ep["episode_id"], str(e)))
                console.print(f"[red]Batch error at {i}: {e}[/red]")

            progress.advance(task, len(batch))

    elapsed = time.time() - start
    embedded_count = len(all_ids) - len(already_cached)
    throughput = embedded_count / elapsed if elapsed > 0 else 0

    # Save cache
    vectors_array = np.array(all_vectors, dtype=np.float32)
    save_cache(all_ids, vectors_array, cache_dir)

    # Report
    console.print()
    console.print(f"[green]Embedded {embedded_count} episodes in {elapsed:.1f}s ({throughput:.0f} docs/s)[/green]")
    console.print(f"Total cached: {len(all_ids)} episodes, vectors shape: {vectors_array.shape}")
    if errors:
        console.print(f"[yellow]Errors: {len(errors)} episodes skipped[/yellow]")
        for eid, err in errors[:5]:
            console.print(f"  episode {eid}: {err}")

    return all_ids, vectors_array


if __name__ == "__main__":
    embed()
