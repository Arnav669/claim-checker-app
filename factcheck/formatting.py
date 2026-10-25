from __future__ import annotations

from factcheck.schemas import CheckResult, Hit


def format_passages(hits: list[Hit]) -> str:
    if not hits:
        return "none"
    return "\n\n".join(f"[P{i}] {hit.text}" for i, hit in enumerate(hits, start=1))


def format_reply(result: CheckResult) -> str:
    lines = [f"**Verdict: {result.label.replace('_', ' ')}**", "", result.reasoning]
    if result.sources:
        heading = "Closest passages found:" if result.abstained else "Sources:"
        lines += ["", heading]
        seen = set()
        for hit in result.sources:
            if hit.doc_id in seen:
                continue
            seen.add(hit.doc_id)
            lines.append(f"- {hit.title or 'Untitled'} (doc {hit.doc_id})")
    return "\n".join(lines)
