from __future__ import annotations

from typing import Callable, Optional

from factcheck.graph import FactCheckGraph
from factcheck.memory import SessionMemory
from factcheck.schemas import AgentReply, CheckResult


class FactCheckAgent:
    def __init__(
        self,
        graph: FactCheckGraph,
        memory: SessionMemory,
        run_config: Optional[Callable[[str], dict]] = None,
    ):
        self._graph = graph
        self._memory = memory
        self._run_config = run_config

    def _invoke(self, state: dict, session_id: str) -> dict:
        config = self._run_config(session_id) if self._run_config else {}
        return self._graph.graph.invoke(state, config=config)

    @staticmethod
    def _finalize(final: dict) -> Optional[CheckResult]:
        result = final.get("result")
        if result is None:
            return None
        return result.model_copy(update={"timings": final.get("timings", {})})

    def ask(self, session_id: str, text: str) -> AgentReply:
        final = self._invoke(
            {
                "user_input": text,
                "history_text": self._memory.history_text(session_id),
                "last_result": self._memory.last_result(session_id),
                "skip_router": False,
            },
            session_id,
        )
        result = self._finalize(final)
        reply = final["reply"]
        self._memory.add_turn(session_id, text, reply, result)
        return AgentReply(text=reply, route=final.get("route", "claim"), result=result)

    def check_claim(self, claim: str, session_id: str = "eval") -> CheckResult:
        final = self._invoke(
            {
                "user_input": claim,
                "history_text": "",
                "last_result": None,
                "skip_router": True,
            },
            session_id,
        )
        return self._finalize(final)

    def clear(self, session_id: str) -> None:
        self._memory.clear(session_id)
