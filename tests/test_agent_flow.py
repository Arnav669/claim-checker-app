from dataclasses import replace

import pytest

from factcheck.agent import FactCheckAgent
from factcheck.config import get_settings
from factcheck.graph import FactCheckGraph
from factcheck.indexing import build_index
from factcheck.memory import SessionMemory
from factcheck.prompts import OTHER_REPLY
from factcheck.retriever import Retriever
from factcheck.vector_store import FaissStore
from tests.fakes import FakeLLM, HashEmbedder, OverlapReranker, sample_docs

SUPPORTED_REPLY = '{"verdict": "SUPPORTED", "reasoning": "Stated directly.", "citations": ["P1"]}'
IN_DOMAIN_CLAIM = "Vitamin D supplementation reduces the risk of respiratory infection"
OUT_OF_DOMAIN_CLAIM = "Quarterly revenue of the shipping company doubled"


def make_agent(tmp_path, verdict_reply=SUPPORTED_REPLY):
    embedder = HashEmbedder()
    store = FaissStore(tmp_path / "faiss")
    build_index(sample_docs(), embedder, store, 100, 10)
    settings = replace(get_settings(), rerank_top_n=3, gate_min_score=0.3, max_retries=1)
    llm = FakeLLM(verdict_reply)
    graph = FactCheckGraph(Retriever(embedder, store, 5), OverlapReranker(), llm.runnable, settings)
    return FactCheckAgent(graph, SessionMemory(4)), llm


def test_supported_claim_is_grounded_and_timed(tmp_path):
    agent, llm = make_agent(tmp_path)
    result = agent.check_claim(IN_DOMAIN_CLAIM)
    assert result.label == "SUPPORTED"
    assert result.abstained is False
    assert len(result.sources) == 1 and result.sources[0].doc_id == "1"
    assert result.attempts == 1
    assert {"retrieve", "rerank", "judge", "ground"} <= set(result.timings)
    assert llm.calls == ["judge"]


def test_verdict_without_citations_is_discarded(tmp_path):
    reply = '{"verdict": "SUPPORTED", "reasoning": "x", "citations": []}'
    agent, _ = make_agent(tmp_path, reply)
    result = agent.check_claim(IN_DOMAIN_CLAIM)
    assert result.label == "NOT_ENOUGH_INFO"
    assert result.abstain_reason == "ungrounded_verdict"


def test_unknown_citation_ids_are_ignored(tmp_path):
    reply = '{"verdict": "REFUTED", "reasoning": "x", "citations": ["P9"]}'
    agent, _ = make_agent(tmp_path, reply)
    assert agent.check_claim(IN_DOMAIN_CLAIM).abstain_reason == "ungrounded_verdict"


def test_unparseable_model_output_abstains(tmp_path):
    agent, _ = make_agent(tmp_path, "I think it is probably true.")
    result = agent.check_claim(IN_DOMAIN_CLAIM)
    assert result.label == "NOT_ENOUGH_INFO"
    assert result.abstain_reason == "unparseable_model_output"


def test_model_can_judge_insufficient(tmp_path):
    reply = '{"verdict": "NOT_ENOUGH_INFO", "reasoning": "Related only.", "citations": []}'
    agent, _ = make_agent(tmp_path, reply)
    result = agent.check_claim(IN_DOMAIN_CLAIM)
    assert result.abstain_reason == "model_judged_insufficient"
    assert result.sources


def test_out_of_domain_claim_retries_then_abstains_without_judging(tmp_path):
    agent, llm = make_agent(tmp_path)
    result = agent.check_claim(OUT_OF_DOMAIN_CLAIM)
    assert result.label == "NOT_ENOUGH_INFO"
    assert result.abstain_reason == "low_retrieval_confidence"
    assert result.attempts == 2
    assert llm.calls == ["rewrite"]


def test_check_claim_skips_router(tmp_path):
    agent, llm = make_agent(tmp_path)
    agent.check_claim(IN_DOMAIN_CLAIM)
    assert "router" not in llm.calls


def test_conversation_with_followup_uses_session_memory(tmp_path):
    agent, llm = make_agent(tmp_path)
    first = agent.ask("s1", IN_DOMAIN_CLAIM)
    assert first.route == "claim" and first.result.label == "SUPPORTED"
    second = agent.ask("s1", "why?")
    assert second.route == "followup"
    assert second.result is None
    assert "first passage" in second.text
    assert llm.calls.count("followup") == 1


def test_followup_without_prior_claim_falls_back_to_guidance(tmp_path):
    agent, _ = make_agent(tmp_path)
    reply = agent.ask("fresh", "why?")
    assert reply.route == "other"
    assert reply.text == OTHER_REPLY


def test_greeting_routes_to_other(tmp_path):
    agent, _ = make_agent(tmp_path)
    assert agent.ask("s2", "hello there").route == "other"


def test_sessions_are_isolated(tmp_path):
    agent, _ = make_agent(tmp_path)
    agent.ask("a", IN_DOMAIN_CLAIM)
    assert agent.ask("b", "why?").route == "other"


def test_clear_resets_session(tmp_path):
    agent, _ = make_agent(tmp_path)
    agent.ask("a", IN_DOMAIN_CLAIM)
    agent.clear("a")
    assert agent.ask("a", "why?").route == "other"
