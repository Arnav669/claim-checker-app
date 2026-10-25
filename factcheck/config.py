from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name, default, cast=str):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return cast(raw.strip())


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    index_dir: Path
    reports_dir: Path
    cache_path: Path
    chunk_size_words: int
    chunk_overlap_words: int
    embedding_model: str
    embedding_batch_size: int
    query_instruction: str
    vector_backend: str
    collection_name: str
    retrieval_top_k: int
    rerank_model: str
    rerank_top_n: int
    gate_min_score: float
    max_retries: int
    llm_provider: str
    groq_model: str
    gemini_model: str
    anthropic_model: str
    llm_temperature: float
    llm_max_rpm: float
    history_turns: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        data_dir=_env("DATA_DIR", ROOT / "data" / "scifact", Path),
        index_dir=_env("INDEX_DIR", ROOT / "storage", Path),
        reports_dir=_env("REPORTS_DIR", ROOT / "reports", Path),
        cache_path=_env("LLM_CACHE_PATH", ROOT / "storage" / "llm_cache.sqlite", Path),
        chunk_size_words=_env("CHUNK_SIZE_WORDS", 200, int),
        chunk_overlap_words=_env("CHUNK_OVERLAP_WORDS", 40, int),
        embedding_model=_env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        embedding_batch_size=_env("EMBEDDING_BATCH_SIZE", 64, int),
        query_instruction=_env(
            "QUERY_INSTRUCTION",
            "Represent this sentence for searching relevant passages: ",
        ),
        vector_backend=_env("VECTOR_BACKEND", "chroma"),
        collection_name=_env("COLLECTION_NAME", "scifact"),
        retrieval_top_k=_env("RETRIEVAL_TOP_K", 15, int),
        rerank_model=_env("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        rerank_top_n=_env("RERANK_TOP_N", 4, int),
        gate_min_score=_env("GATE_MIN_SCORE", 0.12, float),
        max_retries=_env("MAX_RETRIES", 1, int),
        llm_provider=_env("LLM_PROVIDER", "groq"),
        groq_model=_env("GROQ_MODEL", "openai/gpt-oss-120b"),
        gemini_model=_env("GEMINI_MODEL", "gemini-2.5-flash"),
        anthropic_model=_env("ANTHROPIC_MODEL", "claude-haiku-5-5"),
        llm_temperature=_env("LLM_TEMPERATURE", 0.0, float),
        llm_max_rpm=_env("LLM_MAX_RPM", 25.0, float),
        history_turns=_env("HISTORY_TURNS", 6, int),
    )
