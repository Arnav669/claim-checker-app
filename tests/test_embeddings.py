import sys
import types

import numpy as np


def test_embedder_dimension_does_not_depend_on_deprecated_api(monkeypatch):
    class FakeSentenceTransformer:
        def __init__(self, name):
            self.name = name

        def encode(self, texts, **kwargs):
            return np.zeros((len(texts), 7), dtype="float32")

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    from factcheck.embeddings import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder("any-model", query_instruction="q: ")
    assert embedder.dim == 7
    assert embedder.embed_query("hello").shape == (7,)
    assert embedder.embed_documents(["a", "b"]).shape == (2, 7)
