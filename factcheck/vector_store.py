from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from factcheck.config import Settings, get_settings
from factcheck.schemas import Chunk, Hit


class VectorStore(ABC):
    name: str

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None: ...

    @abstractmethod
    def query(self, embedding: np.ndarray, k: int) -> list[Hit]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def reset(self) -> None: ...


class ChromaStore(VectorStore):
    name = "chroma"

    def __init__(self, path, collection_name: str):
        import chromadb

        Path(path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection_name = collection_name
        self._collection = self._open()

    def _open(self):
        return self._client.get_or_create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], embeddings: np.ndarray, batch_size: int = 1000) -> None:
        for start in range(0, len(chunks), batch_size):
            part = chunks[start : start + batch_size]
            vectors = embeddings[start : start + batch_size]
            self._collection.add(
                ids=[c.chunk_id for c in part],
                embeddings=[v.tolist() for v in vectors],
                documents=[c.text for c in part],
                metadatas=[{"doc_id": c.doc_id, "title": c.title} for c in part],
            )

    def query(self, embedding: np.ndarray, k: int) -> list[Hit]:
        total = self.count()
        if total == 0:
            return []
        result = self._collection.query(
            query_embeddings=[np.asarray(embedding).tolist()],
            n_results=min(k, total),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            hits.append(
                Hit(
                    chunk_id=chunk_id,
                    doc_id=meta["doc_id"],
                    title=meta["title"],
                    text=text,
                    score=1.0 - float(distance),
                )
            )
        return hits

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._collection = self._open()


class FaissStore(VectorStore):
    name = "faiss"

    def __init__(self, path):
        import faiss

        self._faiss = faiss
        self._dir = Path(path)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "index.faiss"
        self._meta_path = self._dir / "meta.json"
        self._index = None
        self._meta: list[dict] = []
        if self._index_path.exists() and self._meta_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            self._meta = json.loads(self._meta_path.read_text(encoding="utf-8"))

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        vectors = np.ascontiguousarray(embeddings, dtype="float32")
        if self._index is None:
            self._index = self._faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)
        self._meta.extend(chunk.model_dump() for chunk in chunks)
        self._save()

    def _save(self) -> None:
        self._faiss.write_index(self._index, str(self._index_path))
        self._meta_path.write_text(json.dumps(self._meta), encoding="utf-8")

    def query(self, embedding: np.ndarray, k: int) -> list[Hit]:
        if self._index is None or self._index.ntotal == 0:
            return []
        k = min(k, self._index.ntotal)
        query = np.ascontiguousarray(np.asarray(embedding).reshape(1, -1), dtype="float32")
        scores, ids = self._index.search(query, k)
        hits = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            meta = self._meta[int(idx)]
            hits.append(
                Hit(
                    chunk_id=meta["chunk_id"],
                    doc_id=meta["doc_id"],
                    title=meta["title"],
                    text=meta["text"],
                    score=float(score),
                )
            )
        return hits

    def count(self) -> int:
        return int(self._index.ntotal) if self._index is not None else 0

    def reset(self) -> None:
        self._index = None
        self._meta = []
        for path in (self._index_path, self._meta_path):
            if path.exists():
                path.unlink()


def build_store(backend: str | None = None, settings: Settings | None = None) -> VectorStore:
    settings = settings or get_settings()
    backend = (backend or settings.vector_backend).lower()
    index_dir = Path(settings.index_dir)
    if backend == "chroma":
        return ChromaStore(index_dir / "chroma", settings.collection_name)
    if backend == "faiss":
        return FaissStore(index_dir / "faiss")
    raise ValueError(f"Unknown vector backend: {backend}")
