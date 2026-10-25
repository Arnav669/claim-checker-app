from __future__ import annotations

from typing import Iterable

from factcheck.chunking import chunk_corpus
from factcheck.embeddings import Embedder
from factcheck.schemas import Document
from factcheck.vector_store import VectorStore


def build_index(
    docs: Iterable[Document],
    embedder: Embedder,
    store: VectorStore,
    chunk_size: int,
    overlap: int,
    batch_size: int = 512,
    reset: bool = True,
) -> int:
    if reset:
        store.reset()
    buffer = []
    total = 0
    for chunk in chunk_corpus(docs, chunk_size, overlap):
        buffer.append(chunk)
        if len(buffer) >= batch_size:
            store.add(buffer, embedder.embed_documents([c.text for c in buffer]))
            total += len(buffer)
            buffer = []
    if buffer:
        store.add(buffer, embedder.embed_documents([c.text for c in buffer]))
        total += len(buffer)
    return total
