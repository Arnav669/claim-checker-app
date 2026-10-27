from __future__ import annotations

import io
import re
from pathlib import Path

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}


def _pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF support needs pypdf: pip install pypdf") from exc
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        raw = page.extract_text() or ""
        raw = re.sub(r"(\w)-\n(\w)", r"\1\2", raw)
        paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", raw)]
        pages.append("\n\n".join(p for p in paragraphs if p))
    return "\n\n".join(p for p in pages if p)


def extract_text(name: str, data: bytes) -> str:
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        text = _pdf_text(data)
    elif suffix in TEXT_SUFFIXES:
        text = data.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Use PDF, TXT or MD.")
    text = text.strip()
    if not text:
        raise ValueError("No text could be read from the document.")
    return text


def load_document(path) -> str:
    path = Path(path)
    return extract_text(path.name, path.read_bytes())
