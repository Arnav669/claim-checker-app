from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from factcheck.schemas import CheckResult


@dataclass
class _Session:
    turns: deque
    last_result: Optional[CheckResult] = None


class SessionMemory:
    def __init__(self, max_turns: int = 6):
        self._max_turns = max_turns
        self._sessions: dict[str, _Session] = {}

    def _get(self, session_id: str) -> _Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = _Session(turns=deque(maxlen=self._max_turns))
        return self._sessions[session_id]

    def history_text(self, session_id: str) -> str:
        session = self._get(session_id)
        if not session.turns:
            return ""
        lines = []
        for user_text, reply_text in session.turns:
            lines.append(f"User: {user_text}")
            lines.append(f"Assistant: {reply_text[:400]}")
        return "\n".join(lines)

    def last_result(self, session_id: str) -> Optional[CheckResult]:
        return self._get(session_id).last_result

    def add_turn(
        self,
        session_id: str,
        user_text: str,
        reply_text: str,
        result: Optional[CheckResult] = None,
    ) -> None:
        session = self._get(session_id)
        session.turns.append((user_text, reply_text))
        if result is not None:
            session.last_result = result

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
