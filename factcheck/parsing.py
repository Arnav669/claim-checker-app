from __future__ import annotations

import json
import re
from typing import Optional

from factcheck.schemas import LABELS, NOT_ENOUGH_INFO, REFUTED, SUPPORTED, Verdict

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)
_THINK_OPEN_TAIL = re.compile(r"<think>.*\Z", re.S | re.I)
_THINK_CLOSE = re.compile(r"</think>", re.I)
_DECODER = json.JSONDecoder()

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


def strip_reasoning(text: str) -> str:
    text = _THINK_BLOCK.sub("", text)
    text = _THINK_OPEN_TAIL.sub("", text)
    parts = _THINK_CLOSE.split(text)
    return parts[-1].strip()


def extract_json(text: str) -> Optional[dict]:
    cleaned = strip_reasoning(text)
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    found = None
    index = 0
    while True:
        start = cleaned.find("{", index)
        if start == -1:
            break
        try:
            value, end = _DECODER.raw_decode(cleaned, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(value, dict):
            found = value
        index = end
    return found


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
    word = strip_reasoning(text).upper()
    if word.startswith("FOLLOW"):
        return "followup"
    if word.startswith("OTHER"):
        return "other"
    return "claim"
