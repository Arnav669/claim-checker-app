from __future__ import annotations

from typing import Protocol

import numpy as np

from factcheck.config import Settings, get_settings


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, batch_size: int = 64, query_instruction: str = ""):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        self._query_instruction = query_instruction
        probe = self._model.encode(["dimension probe"], convert_to_numpy=True)
        self.dim = int(probe.shape[1])

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 256,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        vector = self._model.encode(
            [self._query_instruction + text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vector[0], dtype="float32")


def build_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    return SentenceTransformerEmbedder(
        settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        query_instruction=settings.query_instruction,
    )
