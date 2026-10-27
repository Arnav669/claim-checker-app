from __future__ import annotations

from factcheck.agent import FactCheckAgent
from factcheck.auditor import DocumentAuditor
from factcheck.config import Settings, get_settings
from factcheck.embeddings import build_embedder
from factcheck.graph import FactCheckGraph
from factcheck.llm import build_llm
from factcheck.memory import SessionMemory
from factcheck.observability import build_run_config
from factcheck.reranker import CrossEncoderReranker, NoOpReranker
from factcheck.retriever import Retriever
from factcheck.vector_store import build_store


def build_agent(
    settings: Settings | None = None,
    *,
    embedder=None,
    store=None,
    reranker=None,
    llm=None,
    backend: str | None = None,
    use_reranker: bool = True,
) -> FactCheckAgent:
    settings = settings or get_settings()
    embedder = embedder if embedder is not None else build_embedder(settings)
    store = store if store is not None else build_store(backend, settings)
    if reranker is None:
        reranker = (
            CrossEncoderReranker(settings.rerank_model) if use_reranker else NoOpReranker()
        )
    llm = llm if llm is not None else build_llm(settings)
    retriever = Retriever(embedder, store, settings.retrieval_top_k)
    graph = FactCheckGraph(retriever, reranker, llm, settings)
    return FactCheckAgent(graph, SessionMemory(settings.history_turns), build_run_config)


def build_auditor(agent: FactCheckAgent, llm, settings: Settings | None = None) -> DocumentAuditor:
    return DocumentAuditor(agent, llm, settings or get_settings())


def build_system(settings: Settings | None = None, **kwargs):
    settings = settings or get_settings()
    llm = kwargs.pop("llm", None)
    llm = llm if llm is not None else build_llm(settings)
    agent = build_agent(settings, llm=llm, **kwargs)
    return agent, build_auditor(agent, llm, settings)
