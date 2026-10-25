from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from factcheck.schemas import NOT_ENOUGH_INFO, REFUTED, SUPPORTED, Claim, Document

LABEL_MAP = {"SUPPORT": SUPPORTED, "CONTRADICT": REFUTED}


def _read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_corpus(data_dir) -> Iterator[Document]:
    path = Path(data_dir) / "corpus.jsonl"
    for row in _read_jsonl(path):
        yield Document(
            doc_id=str(row["doc_id"]),
            title=row.get("title") or "",
            sentences=list(row.get("abstract") or []),
        )


def _claim_label(evidence: dict):
    labels = {
        LABEL_MAP[item["label"]]
        for items in evidence.values()
        for item in items
        if item.get("label") in LABEL_MAP
    }
    if not labels:
        return NOT_ENOUGH_INFO
    if len(labels) == 1:
        return next(iter(labels))
    return None


def load_claims(data_dir, split: str = "dev") -> list[Claim]:
    path = Path(data_dir) / f"claims_{split}.jsonl"
    claims = []
    for row in _read_jsonl(path):
        has_labels = "evidence" in row
        label = None
        gold = []
        if has_labels:
            evidence = row["evidence"] or {}
            label = _claim_label(evidence)
            if label is None:
                continue
            gold = [str(doc_id) for doc_id in evidence.keys()]
        claims.append(
            Claim(
                claim_id=str(row["id"]),
                text=row["claim"],
                label=label,
                gold_doc_ids=gold,
            )
        )
    return claims
