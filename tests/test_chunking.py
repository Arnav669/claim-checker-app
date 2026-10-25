from factcheck.chunking import chunk_corpus, chunk_document, clean_text
from factcheck.schemas import Document


def make_doc(count: int) -> Document:
    sentences = [f"sentence number {i} " + "word " * 7 for i in range(count)]
    return Document(doc_id="42", title="A Title", sentences=sentences)


def test_clean_text_collapses_whitespace():
    assert clean_text("  a \n\t b   c ") == "a b c"


def test_short_document_is_single_chunk_with_title():
    chunks = chunk_document(make_doc(2), chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("A Title. ")
    assert chunks[0].chunk_id == "42_0"


def test_long_document_splits_with_overlap():
    doc = make_doc(8)
    chunks = chunk_document(doc, chunk_size=30, overlap=10)
    assert len(chunks) > 1
    joined = " ".join(c.text for c in chunks)
    for i in range(8):
        assert f"sentence number {i} " in joined
    for current, following in zip(chunks, chunks[1:]):
        last_sentence = current.text.rsplit("sentence number", 1)[-1]
        assert last_sentence.strip() in following.text


def test_zero_overlap_has_no_repeated_sentences():
    chunks = chunk_document(make_doc(6), chunk_size=30, overlap=0)
    seen = []
    for chunk in chunks:
        for i in range(6):
            if f"sentence number {i} " in chunk.text:
                seen.append(i)
    assert len(seen) == len(set(seen))


def test_empty_document_yields_nothing():
    assert chunk_document(Document(doc_id="1", title="T", sentences=["", "  "]), 50, 10) == []


def test_chunk_ids_are_unique_across_corpus():
    docs = [make_doc(8), Document(doc_id="43", title="B", sentences=["one two three."])]
    ids = [c.chunk_id for c in chunk_corpus(docs, 30, 10)]
    assert len(ids) == len(set(ids))
