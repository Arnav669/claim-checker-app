from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from factcheck.config import get_settings
from factcheck.data_loader import load_corpus
from factcheck.embeddings import build_embedder
from factcheck.indexing import build_index
from factcheck.vector_store import build_store


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["chroma", "faiss", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    settings = get_settings()
    backends = ["chroma", "faiss"] if args.backend == "both" else [args.backend]
    embedder = build_embedder(settings)

    for backend in backends:
        docs = load_corpus(settings.data_dir)
        if args.limit:
            docs = itertools.islice(docs, args.limit)
        store = build_store(backend, settings)
        start = time.perf_counter()
        total = build_index(
            docs,
            embedder,
            store,
            settings.chunk_size_words,
            settings.chunk_overlap_words,
        )
        elapsed = time.perf_counter() - start
        print(f"{backend}: indexed {total} chunks in {elapsed:.1f}s (store count {store.count()})")


if __name__ == "__main__":
    main()
