from __future__ import annotations

import time
from typing import Annotated, Optional, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph

from factcheck.config import Settings
from factcheck.formatting import format_passages, format_reply
from factcheck.parsing import parse_queries, parse_route, parse_verdict
from factcheck.prompts import (
    FOLLOWUP_PROMPT,
    OTHER_REPLY,
    REWRITE_PROMPT,
    ROUTER_PROMPT,
    VERDICT_PROMPT,
)
from factcheck.reranker import Reranker
from factcheck.retriever import Retriever
from factcheck.schemas import NOT_ENOUGH_INFO, CheckResult, Hit, Verdict


def _merge(left: Optional[dict], right: Optional[dict]) -> dict:
    return {**(left or {}), **(right or {})}


class GraphState(TypedDict, total=False):
    user_input: str
    history_text: str
    last_result: Optional[CheckResult]
    skip_router: bool
    route: str
    claim: str
    queries: list[str]
    hits: list[Hit]
    ranked: list[Hit]
    top_score: float
    attempts: int
    verdict: Optional[Verdict]
    parse_failed: bool
    result: Optional[CheckResult]
    reply: str
    timings: Annotated[dict, _merge]


def _timed(name: str, fn):
    def wrapper(state):
        start = time.perf_counter()
        update = fn(state)
        previous = (state.get("timings") or {}).get(name, 0.0)
        update["timings"] = {name: round(previous + time.perf_counter() - start, 4)}
        return update

    return wrapper


class FactCheckGraph:
    def __init__(self, retriever: Retriever, reranker: Reranker, llm, settings: Settings):
        self._retriever = retriever
        self._reranker = reranker
        self._settings = settings
        parser = StrOutputParser()
        self._router = ROUTER_PROMPT | llm | parser
        self._rewriter = REWRITE_PROMPT | llm | parser
        self._judge = VERDICT_PROMPT | llm | parser
        self._follower = FOLLOWUP_PROMPT | llm | parser
        self.graph = self._build()

    def _route(self, state: GraphState) -> dict:
        text = state["user_input"].strip()
        last = state.get("last_result")
        if state.get("skip_router"):
            route = "claim"
        else:
            raw = self._router.invoke(
                {
                    "last_claim": last.claim if last else "none",
                    "last_verdict": last.label if last else "none",
                    "history": state.get("history_text") or "none",
                    "message": text,
                }
            )
            route = parse_route(raw)
            if route == "followup" and last is None:
                route = "other"
        return {"route": route, "claim": text, "queries": [text], "attempts": 0}

    def _retrieve(self, state: GraphState) -> dict:
        hits = self._retriever.retrieve(state["queries"])
        return {"hits": hits, "attempts": state.get("attempts", 0) + 1}

    def _rerank(self, state: GraphState) -> dict:
        ranked = self._reranker.rerank(
            state["claim"], state["hits"], self._settings.rerank_top_n
        )
        top_score = float(ranked[0].score) if ranked else 0.0
        return {"ranked": ranked, "top_score": top_score}

    def _gate(self, state: GraphState) -> str:
        if state.get("top_score", 0.0) >= self._settings.gate_min_score:
            return "judge"
        if state.get("attempts", 1) <= self._settings.max_retries:
            return "rewrite"
        return "abstain"

    def _rewrite(self, state: GraphState) -> dict:
        raw = self._rewriter.invoke(
            {"claim": state["claim"], "failed": " | ".join(state["queries"])}
        )
        return {"queries": parse_queries(raw, state["claim"])}

    def _judge_node(self, state: GraphState) -> dict:
        raw = self._judge.invoke(
            {"claim": state["claim"], "passages": format_passages(state["ranked"])}
        )
        verdict = parse_verdict(raw)
        if verdict is None:
            return {
                "verdict": Verdict(
                    label=NOT_ENOUGH_INFO,
                    reasoning="The model reply could not be parsed, so no verdict is given.",
                ),
                "parse_failed": True,
            }
        return {"verdict": verdict, "parse_failed": False}

    def _ground(self, state: GraphState) -> dict:
        verdict = state["verdict"]
        ranked = state["ranked"]
        by_id = {f"P{i}": hit for i, hit in enumerate(ranked, start=1)}
        cited = [by_id[c] for c in dict.fromkeys(verdict.citations) if c in by_id]

        label = verdict.label
        reasoning = verdict.reasoning
        reason = None
        if state.get("parse_failed"):
            reason = "unparseable_model_output"
        elif label != NOT_ENOUGH_INFO and not cited:
            label = NOT_ENOUGH_INFO
            reasoning = (
                "The model gave a verdict without citing any retrieved passage, "
                "so the verdict was discarded."
            )
            reason = "ungrounded_verdict"
        elif label == NOT_ENOUGH_INFO:
            reason = "model_judged_insufficient"

        abstained = label == NOT_ENOUGH_INFO
        result = CheckResult(
            claim=state["claim"],
            label=label,
            reasoning=reasoning,
            sources=ranked if abstained else cited,
            top_score=state.get("top_score", 0.0),
            abstained=abstained,
            abstain_reason=reason if abstained else None,
            queries=state.get("queries", []),
            attempts=state.get("attempts", 1),
        )
        return {"result": result, "reply": format_reply(result)}

    def _abstain(self, state: GraphState) -> dict:
        result = CheckResult(
            claim=state["claim"],
            label=NOT_ENOUGH_INFO,
            reasoning="No retrieved passage was relevant enough to judge this claim.",
            sources=state.get("ranked", []),
            top_score=state.get("top_score", 0.0),
            abstained=True,
            abstain_reason="low_retrieval_confidence",
            queries=state.get("queries", []),
            attempts=state.get("attempts", 1),
        )
        return {"result": result, "reply": format_reply(result)}

    def _followup(self, state: GraphState) -> dict:
        last = state["last_result"]
        raw = self._follower.invoke(
            {
                "claim": last.claim,
                "verdict": last.label,
                "reasoning": last.reasoning,
                "passages": format_passages(last.sources),
                "history": state.get("history_text") or "none",
                "message": state["user_input"],
            }
        )
        return {"reply": raw.strip()}

    def _other(self, state: GraphState) -> dict:
        return {"reply": OTHER_REPLY}

    def _build(self):
        builder = StateGraph(GraphState)
        builder.add_node("router", _timed("router", self._route))
        builder.add_node("retrieve", _timed("retrieve", self._retrieve))
        builder.add_node("rerank", _timed("rerank", self._rerank))
        builder.add_node("rewrite", _timed("rewrite", self._rewrite))
        builder.add_node("judge", _timed("judge", self._judge_node))
        builder.add_node("ground", _timed("ground", self._ground))
        builder.add_node("abstain", _timed("abstain", self._abstain))
        builder.add_node("followup", _timed("followup", self._followup))
        builder.add_node("other", _timed("other", self._other))

        builder.add_edge(START, "router")
        builder.add_conditional_edges(
            "router",
            lambda state: state["route"],
            {"claim": "retrieve", "followup": "followup", "other": "other"},
        )
        builder.add_edge("retrieve", "rerank")
        builder.add_conditional_edges(
            "rerank",
            self._gate,
            {"judge": "judge", "rewrite": "rewrite", "abstain": "abstain"},
        )
        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("judge", "ground")
        for node in ("ground", "abstain", "followup", "other"):
            builder.add_edge(node, END)
        return builder.compile()
