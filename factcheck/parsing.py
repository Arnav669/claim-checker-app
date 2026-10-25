from __future__ import annotations

import json
import re
from typing import Optional

from factcheck.schemas import LABELS, NOT_ENOUGH_INFO, REFUTED, SUPPORTED, Verdict

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)

_LABEL_ALIASES = {
    "SUPPORT": SUPPORTED,
    "SUPPORTS": SUPPORTED,
    "CONTRADICT": REFUTED,
    "CONTRADICTED": REFUTED,
    "REFUTE": REFUTED,
    "REFUTES": REFUTED,
    "NEI": NOT_ENOUGH_INFO,
    "NOT_ENOUGH_EVIDENCE": NOT_ENOUGH_INFO,
    "INSUFFICIENT_INFO": NOT_ENOUGH_INFO,
}


def extract_json(text: str) -> Optional[dict]:
    candidates = []
    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_verdict(text: str) -> Optional[Verdict]:
    data = extract_json(text)
    if not data:
        return None
    raw = str(data.get("verdict", data.get("label", ""))).strip().upper()
    raw = raw.replace(" ", "_").replace("-", "_")
    label = _LABEL_ALIASES.get(raw, raw)
    if label not in LABELS:
        return None
    citations = data.get("citations") or []
    if isinstance(citations, str):
        citations = [citations]
    citations = [str(c).strip().upper() for c in citations if str(c).strip()]
    return Verdict(
        label=label,
        reasoning=str(data.get("reasoning", "")).strip(),
        citations=citations,
    )


def parse_queries(text: str, fallback: str) -> list[str]:
    data = extract_json(text)
    queries: list[str] = []
    if data and isinstance(data.get("queries"), list):
        queries = [str(q).strip() for q in data["queries"] if str(q).strip()]
    return queries[:3] or [fallback]


def parse_route(text: str) -> str:
    word = text.strip().upper()
    if word.startswith("FOLLOW"):
        return "followup"
    if word.startswith("OTHER"):
        return "other"
    return "claim"
