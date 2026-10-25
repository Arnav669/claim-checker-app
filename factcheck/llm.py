from __future__ import annotations

from pathlib import Path

from factcheck.config import Settings, get_settings


def enable_cache(path) -> bool:
    try:
        from langchain_community.cache import SQLiteCache
        from langchain_core.globals import set_llm_cache
    except ImportError:
        return False
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=str(path)))
    return True


def _rate_limiter(max_rpm: float):
    from langchain_core.rate_limiters import InMemoryRateLimiter

    return InMemoryRateLimiter(
        requests_per_second=max(max_rpm, 1.0) / 60.0,
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )


def active_model_name(settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    provider = settings.llm_provider.lower()
    if provider == "gemini":
        return settings.gemini_model
    if provider in ("claude", "anthropic"):
        return settings.anthropic_model
    return settings.groq_model


def build_llm(settings: Settings | None = None, provider: str | None = None, model: str | None = None):
    settings = settings or get_settings()
    provider = (provider or settings.llm_provider).lower()
    enable_cache(settings.cache_path)
    limiter = _rate_limiter(settings.llm_max_rpm)

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model or settings.groq_model,
            temperature=settings.llm_temperature,
            max_retries=3,
            rate_limiter=limiter,
        )
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model or settings.gemini_model,
            temperature=settings.llm_temperature,
            max_retries=3,
            rate_limiter=limiter,
        )
    if provider in ("claude", "anthropic"):
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model or settings.anthropic_model,
            temperature=settings.llm_temperature,
            max_retries=3,
            rate_limiter=limiter,
        )
    raise ValueError(f"Unknown LLM provider: {provider}")
