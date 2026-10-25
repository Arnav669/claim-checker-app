from __future__ import annotations

from factcheck.embeddings import Embedder
from factcheck.schemas import Hit
from factcheck.vector_store import VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, top_k: int = 15):
        self._embedder = embedder
        self._store = store
        self._top_k = top_k

    def retrieve(self, queries: list[str], top_k: int | None = None) -> list[Hit]:
        k = top_k or self._top_k
        best: dict[str, Hit] = {}
        for query in queries:
            vector = self._embedder.embed_query(query)
            for hit in self._store.query(vector, k):
                current = best.get(hit.chunk_id)
                if current is None or hit.score > current.score:
                    best[hit.chunk_id] = hit
        ranked = sorted(best.values(), key=lambda h: h.score, reverse=True)
        return ranked[:k]
