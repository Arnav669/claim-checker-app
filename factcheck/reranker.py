from __future__ import annotations

from typing import Protocol

import numpy as np

from factcheck.schemas import Hit


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]: ...


class NoOpReranker:
    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        return list(hits)[:top_n]


def to_probability(raw) -> np.ndarray:
    values = np.asarray(raw, dtype="float64")
    if values.size and values.min() >= 0.0 and values.max() <= 1.0:
        return values
    return 1.0 / (1.0 + np.exp(-values))


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        if not hits:
            return []
        raw = self._model.predict([(query, hit.text) for hit in hits])
        scores = to_probability(raw)
        order = np.argsort(-scores)[:top_n]
        return [hits[i].model_copy(update={"score": float(scores[i])}) for i in order]
