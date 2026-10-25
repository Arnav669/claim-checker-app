import json
from dataclasses import replace

from evaluation.runners import (
    calibrate_gate,
    compare_backends,
    evaluate_ood,
    evaluate_retrieval,
    evaluate_verdicts,
    markdown_table,
    sample_claims,
    verdict_markdown,
    write_csv,
    write_json,
)
from factcheck.agent import FactCheckAgent
from factcheck.config import get_settings
from factcheck.graph import FactCheckGraph
from factcheck.indexing import build_index
from factcheck.memory import SessionMemory
from factcheck.retriever import Retriever
from factcheck.schemas import Claim
from factcheck.vector_store import ChromaStore, FaissStore
from tests.fakes import FakeLLM, HashEmbedder, OverlapReranker, sample_docs

CLAIMS = [
    Claim(claim_id="1", text="Vitamin D supplementation reduces respiratory infection risk", label="SUPPORTED", gold_doc_ids=["1"]),
    Claim(claim_id="2", text="Statin therapy lowers LDL cholesterol", label="SUPPORTED", gold_doc_ids=["2"]),
    Claim(claim_id="3", text="Resistance training does not change bone density", label="REFUTED", gold_doc_ids=["3"]),
    Claim(claim_id="4", text="Statins cause muscle problems in most users", label="NOT_ENOUGH_INFO", gold_doc_ids=[]),
]


def build(tmp_path, verdict_reply):
    embedder = HashEmbedder()
    faiss_store = FaissStore(tmp_path / "faiss")
    chroma_store = ChromaStore(tmp_path / "chroma", "eval")
    for store in (faiss_store, chroma_store):
        build_index(sample_docs(), embedder, store, 100, 10)
    retriever = Retriever(embedder, faiss_store, 5)
    settings = replace(get_settings(), rerank_top_n=3, gate_min_score=0.3, max_retries=1)
    llm = FakeLLM(verdict_reply)
    graph = FactCheckGraph(retriever, OverlapReranker(), llm.runnable, settings)
    return embedder, retriever, faiss_store, chroma_store, FactCheckAgent(graph, SessionMemory())


def test_sample_claims_is_stratified_and_reproducible():
    pool = [Claim(claim_id=str(i), text="t", label=l) for i, l in enumerate(["SUPPORTED"] * 5 + ["REFUTED"] * 5 + ["NOT_ENOUGH_INFO"] * 5)]
    first = sample_claims(pool, 2, seed=1)
    second = sample_claims(pool, 2, seed=1)
    assert [c.claim_id for c in first] == [c.claim_id for c in second]
    assert len(first) == 6
    assert {c.label for c in first} == {"SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"}


def test_evaluate_retrieval_reports_both_stages(tmp_path):
    _, retriever, _, _, _ = build(tmp_path, "{}")
    report = evaluate_retrieval(retriever, OverlapReranker(), CLAIMS, ks=(1, 3), pool=5)
    assert report["n_claims"] == 3
    assert report["dense"]["hit@3"] == 1.0
    assert 0.0 <= report["reranked"]["mrr"] <= 1.0
    assert "mean_ms" in report["dense"]


def test_compare_backends_summarises_both(tmp_path):
    embedder, _, faiss_store, chroma_store, _ = build(tmp_path, "{}")
    summary = compare_backends(embedder, {"chroma": chroma_store, "faiss": faiss_store}, CLAIMS, k=3)
    assert summary["chroma"]["chunks"] == summary["faiss"]["chunks"] == 3
    assert summary["faiss"]["hit@3"] == 1.0
    assert 0.0 <= summary["overlap_jaccard"] <= 1.0


def test_evaluate_verdicts_scores_and_writes_reports(tmp_path):
    reply = '{"verdict": "SUPPORTED", "reasoning": "ok", "citations": ["P1"]}'
    _, _, _, _, agent = build(tmp_path, reply)
    rows, summary = evaluate_verdicts(agent, CLAIMS, progress=False)
    assert summary["n_scored"] == 4 and summary["n_errors"] == 0
    assert 0.0 <= summary["accuracy"] <= 1.0
    assert sum(sum(r) for r in summary["confusion_matrix"]) == 4
    write_csv(tmp_path / "rows.csv", rows)
    write_json(tmp_path / "summary.json", summary)
    assert json.loads((tmp_path / "summary.json").read_text())["n_total"] == 4
    assert "Accuracy" in verdict_markdown(summary)


class ExplodingAgent:
    def check_claim(self, text):
        raise RuntimeError("rate limited")


def test_errors_are_recorded_not_raised():
    rows, summary = evaluate_verdicts(ExplodingAgent(), CLAIMS[:2], progress=False)
    assert summary["n_errors"] == 2 and summary["n_scored"] == 0
    assert rows[0]["pred"] == "ERROR"
    _, ood_summary = evaluate_ood(ExplodingAgent(), ["x"])
    assert ood_summary["n_scored"] == 0


def test_ood_abstention_rate(tmp_path):
    reply = '{"verdict": "SUPPORTED", "reasoning": "ok", "citations": ["P1"]}'
    _, _, _, _, agent = build(tmp_path, reply)
    rows, summary = evaluate_ood(agent, ["Quarterly revenue of the shipping company doubled", "Football match results last night"])
    assert summary["abstain_rate"] == 1.0
    assert rows[0]["abstain_reason"] == "low_retrieval_confidence"


def test_calibrate_gate_separates_groups(tmp_path):
    _, retriever, _, _, _ = build(tmp_path, "{}")
    report = calibrate_gate(
        retriever,
        OverlapReranker(),
        CLAIMS,
        ["Quarterly revenue of the shipping company doubled", "Football match results last night"],
        pool=5,
    )
    assert report["separable"] is True
    assert report["groups"]["OUT_OF_DOMAIN"]["p95"] < report["groups"]["SUPPORTED"]["p05"]
    assert report["suggested_gate_min_score"] is not None


def test_markdown_table_shape():
    table = markdown_table(["a", "b"], [[1, 2], [3, 4]])
    assert table.splitlines()[0] == "| a | b |"
    assert len(table.splitlines()) == 4
