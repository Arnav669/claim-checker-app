from factcheck.indexing import build_index
from factcheck.retriever import Retriever
from factcheck.vector_store import FaissStore
from tests.fakes import HashEmbedder, sample_docs


def build(tmp_path):
    embedder = HashEmbedder()
    store = FaissStore(tmp_path / "faiss")
    total = build_index(sample_docs(), embedder, store, chunk_size=100, overlap=10, batch_size=2)
    return embedder, store, total


def test_build_index_counts_chunks(tmp_path):
    _, store, total = build(tmp_path)
    assert total == store.count() == 3


def test_build_index_reset_replaces_content(tmp_path):
    embedder, store, _ = build(tmp_path)
    build_index(sample_docs()[:1], embedder, store, 100, 10, reset=True)
    assert store.count() == 1


def test_retrieve_finds_relevant_doc(tmp_path):
    embedder, store, _ = build(tmp_path)
    retriever = Retriever(embedder, store, top_k=3)
    hits = retriever.retrieve(["vitamin D supplementation respiratory infection"])
    assert hits[0].doc_id == "1"


def test_multi_query_merge_deduplicates_and_keeps_best(tmp_path):
    embedder, store, _ = build(tmp_path)
    retriever = Retriever(embedder, store, top_k=3)
    hits = retriever.retrieve(["bone density training", "statin cholesterol"])
    ids = [h.chunk_id for h in hits]
    assert len(ids) == len(set(ids))
    assert {h.doc_id for h in hits[:2]} == {"2", "3"}
