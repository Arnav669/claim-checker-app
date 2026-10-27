import json
from dataclasses import replace

import pytest

from factcheck.agent import FactCheckAgent
from factcheck.auditor import DocumentAuditor, split_sections, wording_flags
from factcheck.config import get_settings
from factcheck.documents import extract_text
from factcheck.formatting import format_audit_report
from factcheck.graph import FactCheckGraph
from factcheck.indexing import build_index
from factcheck.memory import SessionMemory
from factcheck.parsing import parse_claims
from factcheck.retriever import Retriever
from factcheck.vector_store import FaissStore
from tests.fakes import FakeLLM, HashEmbedder, OverlapReranker, sample_docs

SUPPORTED_REPLY = '{"verdict": "SUPPORTED", "reasoning": "Stated directly.", "citations": ["P1"]}'
GOOD = "Vitamin D supplementation reduces the risk of respiratory infection"
OFF_TOPIC = "Quarterly revenue of the shipping company doubled"
OVERSTATED_OFF_TOPIC = "Quarterly revenue of the shipping company always doubled"


def make_auditor(tmp_path, claims, max_claims=8, **overrides):
    embedder = HashEmbedder()
    store = FaissStore(tmp_path / "faiss")
    build_index(sample_docs(), embedder, store, 100, 10)
    settings = replace(
        get_settings(),
        rerank_top_n=3,
        gate_min_score=0.3,
        max_retries=1,
        audit_pause_seconds=0.0,
        audit_max_claims=max_claims,
        **overrides,
    )
    llm = FakeLLM(SUPPORTED_REPLY, extract_reply=json.dumps({"claims": claims}))
    graph = FactCheckGraph(Retriever(embedder, store, 5), OverlapReranker(), llm.runnable, settings)
    agent = FactCheckAgent(graph, SessionMemory(4))
    return DocumentAuditor(agent, llm.runnable, settings), agent, llm


def test_parse_claims_cleans_and_limits():
    text = '<think>{"claims": ["ignore me"]}</think>{"claims": ["  A causes   B in mice ", "too short", "C raises D levels", "E lowers F risk"]}'
    assert parse_claims(text, limit=2) == ["A causes B in mice", "C raises D levels"]
    assert parse_claims("no json") == []
    assert parse_claims('{"claims": "oops"}') == []


def test_split_sections_groups_paragraphs_and_splits_long_ones():
    text = "one two three\n\nfour five six\n\n" + "seven eight nine. " * 10
    sections = split_sections(text, 8)
    assert len(sections) >= 3
    assert all(len(s.split()) <= 12 for s in sections)
    assert split_sections("   ", 8) == []


def test_wording_flags():
    assert wording_flags("This always works and proves the point, always") == ["always", "proves"]
    assert wording_flags("The drug may reduce risk") == []


def test_audit_labels_and_counts(tmp_path):
    auditor, _, llm = make_auditor(tmp_path, [GOOD, OFF_TOPIC])
    report = auditor.audit("Some article text about health.")
    counts = report.counts()
    assert counts["SUPPORTED"] == 1 and counts["NOT_ENOUGH_INFO"] == 1
    assert report.n_checked == 2 and report.n_found == 2
    assert not report.claims_truncated
    assert llm.calls.count("extract") == 1
    assert report.items[0].result.sources[0].doc_id == "1"


def test_overstated_wording_is_flagged_only_when_not_supported(tmp_path):
    auditor, _, _ = make_auditor(tmp_path, [GOOD, OVERSTATED_OFF_TOPIC])
    report = auditor.audit("Article.")
    by_claim = {i.claim: i for i in report.items}
    assert by_claim[OVERSTATED_OFF_TOPIC].possibly_overstated is True
    assert by_claim[GOOD].possibly_overstated is False
    assert [i.claim for i in report.flagged()] == [OVERSTATED_OFF_TOPIC]


def test_claim_limit_and_dedupe(tmp_path):
    auditor, _, _ = make_auditor(tmp_path, [GOOD, GOOD.upper(), OFF_TOPIC, OVERSTATED_OFF_TOPIC])
    report = auditor.audit("Article.", max_claims=2)
    assert report.n_found == 3
    assert report.n_checked == 2 and report.claims_truncated is True


def test_claims_are_spread_across_sections(tmp_path):
    claims = [GOOD, OFF_TOPIC]
    auditor, _, _ = make_auditor(tmp_path, claims, audit_section_words=5)
    text = "alpha beta gamma delta epsilon.\n\nzeta eta theta iota kappa.\n\nlambda mu nu xi omicron."
    report = auditor.audit(text, max_claims=2)
    assert report.n_sections == 3
    assert report.n_found == 2 and report.n_checked == 2


def test_check_failure_is_recorded_not_raised(tmp_path):
    auditor, agent, _ = make_auditor(tmp_path, [GOOD, OFF_TOPIC])
    original = agent.check_claim

    def flaky(claim, session_id="eval"):
        if claim == GOOD:
            raise RuntimeError("rate limited")
        return original(claim, session_id)

    agent.check_claim = flaky
    report = auditor.audit("Article.")
    assert report.counts()["ERROR"] == 1
    assert "rate limited" in report.items[0].error
    assert report.items[1].label == "NOT_ENOUGH_INFO"


def test_pause_between_claims_is_applied(tmp_path):
    auditor, _, _ = make_auditor(tmp_path, [GOOD, OFF_TOPIC, OVERSTATED_OFF_TOPIC])
    sleeps = []
    auditor._settings = replace(auditor._settings, audit_pause_seconds=5.0)
    auditor._sleep = sleeps.append
    ticks = iter(range(0, 1000, 1))
    auditor._clock = lambda: float(next(ticks))
    auditor.audit("Article.")
    assert len(sleeps) == 2 and all(0 < s <= 5.0 for s in sleeps)


def test_progress_reports_every_step(tmp_path):
    auditor, _, _ = make_auditor(tmp_path, [GOOD, OFF_TOPIC])
    events = []
    auditor.audit("Article.", progress=lambda d, t, m: events.append((d, t)))
    assert events[0] == (0, 1) and events[-1] == (2, 2)


def test_report_markdown_contains_summary_and_flags(tmp_path):
    auditor, _, _ = make_auditor(tmp_path, [GOOD, OVERSTATED_OFF_TOPIC])
    markdown = format_audit_report(auditor.audit("Article."))
    assert "Checked 2 of 2 claims" in markdown
    assert "Supported: 1" in markdown and "Not enough info: 1" in markdown
    assert "Strong wording: always" in markdown
    assert "| 1 |" in markdown and "does not mean the claim is false" in markdown


def test_extract_text_handles_text_and_rejects_bad_input():
    assert extract_text("notes.txt", b"  hello world  ") == "hello world"
    with pytest.raises(ValueError):
        extract_text("notes.docx", b"x")
    with pytest.raises(ValueError):
        extract_text("empty.md", b"   ")
