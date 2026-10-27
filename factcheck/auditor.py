from __future__ import annotations

import re
import time
from typing import Callable, Optional

from langchain_core.output_parsers import StrOutputParser

from factcheck.agent import FactCheckAgent
from factcheck.config import Settings
from factcheck.parsing import parse_claims
from factcheck.prompts import EXTRACT_PROMPT
from factcheck.schemas import AuditedClaim, AuditReport

CLAIMS_PER_SECTION = 4

_STRONG_WORDING = re.compile(
    r"\b(proves?|proven|proof|definitively|conclusively|cures?|cured|guarantees?|always|never|"
    r"completely|totally|eliminates?|100\s?%|all (?:patients|people|cases)|everyone|no one|nobody)\b",
    re.I,
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

ProgressFn = Callable[[int, int, str], None]


def wording_flags(claim: str) -> list[str]:
    seen: list[str] = []
    for match in _STRONG_WORDING.finditer(claim):
        word = " ".join(match.group(0).lower().split())
        if word not in seen:
            seen.append(word)
    return seen


def _units(text: str, limit: int) -> list[str]:
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        if len(paragraph.split()) <= limit:
            units.append(paragraph)
        else:
            units.extend(s for s in _SENTENCE_END.split(paragraph) if s.strip())
    return units


def split_sections(text: str, section_words: int) -> list[str]:
    sections: list[str] = []
    current: list[str] = []
    count = 0
    for unit in _units(text, section_words):
        words = len(unit.split())
        if current and count + words > section_words:
            sections.append("\n\n".join(current))
            current, count = [], 0
        current.append(unit)
        count += words
    if current:
        sections.append("\n\n".join(current))
    return sections


def _key(claim: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", claim.lower()).strip()


def _spread(found: list[tuple[int, str]], limit: int) -> list[tuple[int, str]]:
    if len(found) <= limit:
        return found
    groups: dict[int, list[tuple[int, str]]] = {}
    for item in found:
        groups.setdefault(item[0], []).append(item)
    chosen: list[tuple[int, str]] = []
    queues = [list(v) for _, v in sorted(groups.items())]
    while len(chosen) < limit and any(queues):
        for queue in queues:
            if queue and len(chosen) < limit:
                chosen.append(queue.pop(0))
    order = {id(item): i for i, item in enumerate(found)}
    return sorted(chosen, key=lambda item: order[id(item)])


class DocumentAuditor:
    def __init__(
        self,
        agent: FactCheckAgent,
        llm,
        settings: Settings,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._agent = agent
        self._extract = EXTRACT_PROMPT | llm | StrOutputParser()
        self._settings = settings
        self._sleep = sleep
        self._clock = clock

    def extract_claims(self, text: str) -> tuple[list[tuple[int, str]], int, bool]:
        sections = split_sections(text, self._settings.audit_section_words)
        truncated = len(sections) > self._settings.audit_max_sections
        sections = sections[: self._settings.audit_max_sections]
        found: list[tuple[int, str]] = []
        seen: set[str] = set()
        for number, section in enumerate(sections, start=1):
            try:
                raw = self._extract.invoke(
                    {"passage": section, "max_claims": CLAIMS_PER_SECTION}
                )
            except Exception:
                continue
            for claim in parse_claims(raw, CLAIMS_PER_SECTION):
                key = _key(claim)
                if key and key not in seen:
                    seen.add(key)
                    found.append((number, claim))
        return found, len(sections), truncated

    def audit(
        self,
        text: str,
        max_claims: Optional[int] = None,
        progress: Optional[ProgressFn] = None,
    ) -> AuditReport:
        limit = max_claims or self._settings.audit_max_claims
        if progress:
            progress(0, 1, "Reading the document and extracting claims...")
        found, n_sections, sections_truncated = self.extract_claims(text)
        chosen = _spread(found, limit)
        report = AuditReport(
            n_sections=n_sections,
            n_found=len(found),
            sections_truncated=sections_truncated,
            claims_truncated=len(found) > len(chosen),
        )
        total = len(chosen)
        pause = self._settings.audit_pause_seconds
        for position, (section, claim) in enumerate(chosen, start=1):
            if progress:
                progress(position - 1, total, f"Checking claim {position} of {total}")
            started = self._clock()
            item = AuditedClaim(
                index=position, claim=claim, section=section, wording_flags=wording_flags(claim)
            )
            try:
                item.result = self._agent.check_claim(claim, session_id="audit")
            except Exception as exc:
                item.error = f"{type(exc).__name__}: {exc}"
            report.items.append(item)
            remaining = pause - (self._clock() - started)
            if position < total and remaining > 0:
                self._sleep(remaining)
        report.n_checked = total
        if progress:
            progress(total, max(total, 1), "Done")
        return report
