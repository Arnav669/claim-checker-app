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


_LABEL_TEXT = {
    "SUPPORTED": "Supported",
    "REFUTED": "Refuted",
    "NOT_ENOUGH_INFO": "Not enough info",
}


def label_text(label) -> str:
    return _LABEL_TEXT.get(label, "Error")


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "/")


def _titles(item) -> str:
    if not item.result:
        return ""
    seen, titles = set(), []
    for hit in item.result.sources:
        if hit.doc_id not in seen:
            seen.add(hit.doc_id)
            titles.append(f"{hit.title or 'Untitled'} (doc {hit.doc_id})")
    return "; ".join(titles)


def format_audit_report(report) -> str:
    counts = report.counts()
    lines = [
        "# Document audit report",
        "",
        f"Checked {report.n_checked} of {report.n_found} claims found in {report.n_sections} section(s).",
        "",
        f"- Supported: {counts['SUPPORTED']}",
        f"- Refuted: {counts['REFUTED']}",
        f"- Not enough info: {counts['NOT_ENOUGH_INFO']}",
    ]
    if counts["ERROR"]:
        lines.append(f"- Could not be checked: {counts['ERROR']}")
    if report.claims_truncated:
        lines += ["", "More claims were found than the limit allows, so a sample spread across the document was checked."]
    if report.sections_truncated:
        lines += ["", "The document was longer than the section limit, so only the first part was read."]
    flagged = report.flagged()
    lines += ["", "## Claims needing attention", ""]
    if not flagged:
        lines.append("None. No claim was refuted or flagged for overstated wording.")
    for item in flagged:
        lines.append(f"{item.index}. [{label_text(item.label)}] {item.claim}")
        if item.result and item.result.reasoning:
            lines.append(f"   - Why: {_cell(item.result.reasoning)}")
        if item.wording_flags:
            lines.append(
                "   - Strong wording: " + ", ".join(item.wording_flags)
                + " (check that the evidence is this strong)"
            )
        titles = _titles(item)
        if titles:
            lines.append(f"   - Sources: {titles}")
    lines += ["", "## All claims", "", "| # | Claim | Verdict | Evidence |", "| --- | --- | --- | --- |"]
    for item in report.items:
        evidence = _titles(item) if item.result else _cell(item.error or "")
        lines.append(f"| {item.index} | {_cell(item.claim)} | {label_text(item.label)} | {_cell(evidence)} |")
    lines += [
        "",
        "Not enough info means the research abstracts available do not settle the claim. "
        "It does not mean the claim is false. Research tool only, not medical advice.",
    ]
    return "\n".join(lines)
