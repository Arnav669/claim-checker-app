from dataclasses import replace

import pytest
from langchain_core.globals import set_llm_cache

from factcheck.config import get_settings
from factcheck.llm import active_model_name, build_llm, enable_cache


@pytest.fixture(autouse=True)
def reset_cache():
    yield
    set_llm_cache(None)


def test_groq_llm_builds_with_configured_model(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    settings = replace(get_settings(), cache_path=tmp_path / "cache.sqlite", groq_model="some-model")
    llm = build_llm(settings, provider="groq")
    assert llm.model_name == "some-model"
    assert llm.temperature == 0.0 or llm.temperature is not None


def test_unknown_provider_raises(tmp_path):
    settings = replace(get_settings(), cache_path=tmp_path / "cache.sqlite")
    with pytest.raises(ValueError):
        build_llm(settings, provider="nope")


def test_cache_file_is_created(tmp_path):
    assert enable_cache(tmp_path / "nested" / "cache.sqlite") is True
    assert (tmp_path / "nested" / "cache.sqlite").exists()


def test_active_model_name_follows_provider():
    base = get_settings()
    assert active_model_name(replace(base, llm_provider="groq")) == base.groq_model
    assert active_model_name(replace(base, llm_provider="gemini")) == base.gemini_model
    assert active_model_name(replace(base, llm_provider="claude")) == base.anthropic_model
