"""Hybrid search: BM25 + vector similarity with tunable alpha."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from rich.console import Console

from podcast_rag.embed import build_document, load_episodes
from podcast_rag.store import VectorStore

HYBRID_CACHE = Path(__file__).resolve().parents[2] / "data" / "bm25_cache"


def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\w+", text.lower())


@dataclass
class ScoredEpisode:
    episode_id: str
    podcast: str
    date: str
    key_topics: list[str]
    themes: list[str]
    document: str
    vector_score: float
    bm25_score: float
    hybrid_score: float


class BM25Index:
    """BM25 index over episode documents."""

    def __init__(self, episode_ids: list[str], documents: list[str]):
        self.episode_ids = episode_ids
        self.documents = documents
        tokenized = [tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        """Return top-k (episode_id, score) pairs."""
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(self.episode_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]


class HybridSearch:
    """Combines BM25 and vector search with configurable alpha."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha  # 0.0=pure BM25, 1.0=pure vector
        self.store = VectorStore()

        # Build BM25 index from episodes
        episodes = load_episodes()
        self.ep_map = {str(ep["episode_id"]): ep for ep in episodes}
        ep_ids = [str(ep["episode_id"]) for ep in episodes]
        docs = [build_document(ep) for ep in episodes]
        self.bm25_index = BM25Index(ep_ids, docs)
        self.doc_map = dict(zip(ep_ids, docs))

    def search(
        self,
        query: str,
        k: int = 5,
        mode: str = "hybrid",
        where: dict | None = None,
    ) -> list[ScoredEpisode]:
        """Search with configurable mode: 'vector', 'bm25', or 'hybrid'."""
        if mode == "bm25":
            return self._bm25_search(query, k)
        elif mode == "vector":
            return self._vector_search(query, k, where)
        else:
            return self._hybrid_search(query, k, where)

    def _vector_search(self, query: str, k: int, where: dict | None = None) -> list[ScoredEpisode]:
        results = self.store.search(query=query, n_results=k, where=where)
        episodes = []
        for i, doc_id in enumerate(results["ids"][0]):
            ep = self.ep_map.get(doc_id, {})
            dist = results["distances"][0][i]
            # Convert cosine distance to similarity score (0-1)
            vec_score = 1.0 - dist
            episodes.append(ScoredEpisode(
                episode_id=doc_id,
                podcast=ep.get("podcast", ""),
                date=str(ep.get("date", ""))[:10],
                key_topics=ep.get("key_topics") or [],
                themes=ep.get("themes") or [],
                document=self.doc_map.get(doc_id, ""),
                vector_score=vec_score,
                bm25_score=0.0,
                hybrid_score=vec_score,
            ))
        return episodes

    def _bm25_search(self, query: str, k: int) -> list[ScoredEpisode]:
        results = self.bm25_index.search(query, k=k)
        if not results:
            return []
        # Normalize BM25 scores to 0-1
        max_score = max(s for _, s in results) if results else 1.0
        episodes = []
        for doc_id, score in results:
            ep = self.ep_map.get(doc_id, {})
            norm_score = score / max_score if max_score > 0 else 0
            episodes.append(ScoredEpisode(
                episode_id=doc_id,
                podcast=ep.get("podcast", ""),
                date=str(ep.get("date", ""))[:10],
                key_topics=ep.get("key_topics") or [],
                themes=ep.get("themes") or [],
                document=self.doc_map.get(doc_id, ""),
                vector_score=0.0,
                bm25_score=norm_score,
                hybrid_score=norm_score,
            ))
        return episodes

    def _hybrid_search(self, query: str, k: int, where: dict | None = None) -> list[ScoredEpisode]:
        # Get wider results from both
        fetch_k = k * 4
        vector_results = self.store.search(query=query, n_results=fetch_k, where=where)
        bm25_results = self.bm25_index.search(query, k=fetch_k)

        # Build score maps
        vec_scores: dict[str, float] = {}
        for i, doc_id in enumerate(vector_results["ids"][0]):
            dist = vector_results["distances"][0][i]
            vec_scores[doc_id] = 1.0 - dist

        bm25_scores: dict[str, float] = {}
        max_bm25 = max((s for _, s in bm25_results), default=1.0)
        for doc_id, score in bm25_results:
            bm25_scores[doc_id] = score / max_bm25 if max_bm25 > 0 else 0

        # Merge all candidate IDs
        all_ids = set(vec_scores.keys()) | set(bm25_scores.keys())

        # Compute hybrid scores
        scored: list[ScoredEpisode] = []
        for doc_id in all_ids:
            vs = vec_scores.get(doc_id, 0.0)
            bs = bm25_scores.get(doc_id, 0.0)
            hybrid = self.alpha * vs + (1 - self.alpha) * bs
            ep = self.ep_map.get(doc_id, {})
            scored.append(ScoredEpisode(
                episode_id=doc_id,
                podcast=ep.get("podcast", ""),
                date=str(ep.get("date", ""))[:10],
                key_topics=ep.get("key_topics") or [],
                themes=ep.get("themes") or [],
                document=self.doc_map.get(doc_id, ""),
                vector_score=vs,
                bm25_score=bs,
                hybrid_score=hybrid,
            ))

        scored.sort(key=lambda e: e.hybrid_score, reverse=True)
        return scored[:k]


def benchmark():
    """Benchmark vector vs BM25 vs hybrid on eval questions."""
    console = Console()
    from podcast_rag.eval import EVAL_QUESTIONS

    search = HybridSearch(alpha=0.5)

    console.print("\n[bold]Hybrid Search Benchmark[/bold]\n")

    for mode in ("vector", "bm25", "hybrid"):
        hits = 0
        for q in EVAL_QUESTIONS:
            results = search.search(q["question"], k=5, mode=mode)
            # Check if any result mentions key terms from ground truth
            gt_terms = set(tokenize(q["ground_truth"]))
            for ep in results:
                doc_terms = set(tokenize(ep.document))
                overlap = len(gt_terms & doc_terms) / len(gt_terms) if gt_terms else 0
                if overlap > 0.15:
                    hits += 1
                    break

        console.print(f"  {mode:8s}: {hits}/{len(EVAL_QUESTIONS)} questions with relevant retrieval")

    # Alpha sweep
    console.print("\n[bold]Alpha sweep (0.0=pure BM25, 1.0=pure vector):[/bold]")
    for alpha in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        search.alpha = alpha
        hits = 0
        for q in EVAL_QUESTIONS:
            results = search.search(q["question"], k=5, mode="hybrid")
            gt_terms = set(tokenize(q["ground_truth"]))
            for ep in results:
                doc_terms = set(tokenize(ep.document))
                overlap = len(gt_terms & doc_terms) / len(gt_terms) if gt_terms else 0
                if overlap > 0.15:
                    hits += 1
                    break
        console.print(f"  alpha={alpha:.1f}: {hits}/{len(EVAL_QUESTIONS)}")


if __name__ == "__main__":
    benchmark()
