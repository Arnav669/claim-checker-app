from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Iterator

from factcheck.schemas import Chunk, Document

_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def _word_count(text: str) -> int:
    return len(text.split())


def _overlap_tail(group: list[str], overlap: int) -> list[str]:
    if overlap <= 0:
        return []
    tail = []
    total = 0
    for sentence in reversed(group):
        words = _word_count(sentence)
        if total + words > overlap:
            break
        tail.append(sentence)
        total += words
    tail.reverse()
    return tail


def chunk_document(doc: Document, chunk_size: int, overlap: int) -> list[Chunk]:
    sentences = [s for s in (clean_text(s) for s in doc.sentences) if s]
    if not sentences:
        return []
    title = clean_text(doc.title)

    groups = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        words = _word_count(sentence)
        if current and current_words + words > chunk_size:
            groups.append(current)
            current = _overlap_tail(current, overlap)
            current_words = sum(_word_count(s) for s in current)
        current.append(sentence)
        current_words += words
    if current:
        groups.append(current)

    chunks = []
    for index, group in enumerate(groups):
        body = " ".join(group)
        text = f"{title}. {body}" if title else body
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}_{index}",
                doc_id=doc.doc_id,
                title=title,
                text=text,
            )
        )
    return chunks


def chunk_corpus(
    docs: Iterable[Document], chunk_size: int, overlap: int
) -> Iterator[Chunk]:
    for doc in docs:
        yield from chunk_document(doc, chunk_size, overlap)
