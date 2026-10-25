import pytest

from factcheck.chunking import chunk_corpus
from factcheck.vector_store import ChromaStore, FaissStore
from tests.fakes import HashEmbedder, sample_docs


def make_store(kind, tmp_path):
    if kind == "chroma":
        return ChromaStore(tmp_path / "chroma", "test")
    return FaissStore(tmp_path / "faiss")


@pytest.fixture
def chunks():
    return list(chunk_corpus(sample_docs(), 100, 10))


@pytest.mark.parametrize("kind", ["chroma", "faiss"])
def test_add_and_query_returns_best_match_first(kind, tmp_path, chunks):
    embedder = HashEmbedder()
    store = make_store(kind, tmp_path)
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    assert store.count() == len(chunks)
    hits = store.query(embedder.embed_query("statin therapy cholesterol cardiovascular"), 2)
    assert hits[0].doc_id == "2"
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score


@pytest.mark.parametrize("kind", ["chroma", "faiss"])
def test_empty_store_query_returns_nothing(kind, tmp_path):
    store = make_store(kind, tmp_path)
    assert store.query(HashEmbedder().embed_query("anything"), 3) == []


@pytest.mark.parametrize("kind", ["chroma", "faiss"])
def test_reset_clears_store(kind, tmp_path, chunks):
    embedder = HashEmbedder()
    store = make_store(kind, tmp_path)
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    store.reset()
    assert store.count() == 0


@pytest.mark.parametrize("kind", ["chroma", "faiss"])
def test_k_larger_than_store_is_capped(kind, tmp_path, chunks):
    embedder = HashEmbedder()
    store = make_store(kind, tmp_path)
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    assert len(store.query(embedder.embed_query("bone density"), 50)) == len(chunks)


def test_faiss_persists_to_disk(tmp_path, chunks):
    embedder = HashEmbedder()
    store = FaissStore(tmp_path / "faiss")
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    reopened = FaissStore(tmp_path / "faiss")
    assert reopened.count() == len(chunks)
    assert reopened.query(embedder.embed_query("resistance training bone"), 1)[0].doc_id == "3"


def test_chroma_persists_to_disk(tmp_path, chunks):
    embedder = HashEmbedder()
    store = ChromaStore(tmp_path / "chroma", "persist")
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    del store
    reopened = ChromaStore(tmp_path / "chroma", "persist")
    assert reopened.count() == len(chunks)
