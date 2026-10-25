from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from evaluation.metrics import (
    accuracy,
    confusion_matrix,
    doc_hit_at_k,
    doc_recall_at_k,
    macro_f1,
    per_label_prf,
    percentile,
    reciprocal_rank,
    unique_in_order,
)
from factcheck.schemas import LABELS, NOT_ENOUGH_INFO, REFUTED, SUPPORTED, Claim


def sample_claims(claims: Sequence[Claim], per_label: int, seed: int = 42) -> list[Claim]:
    rng = random.Random(seed)
    sampled = []
    for label in LABELS:
        pool = [c for c in claims if c.label == label]
        rng.shuffle(pool)
        sampled.extend(pool[:per_label])
    rng.shuffle(sampled)
    return sampled


def evaluate_retrieval(retriever, reranker, claims: Sequence[Claim], ks=(1, 3, 5), pool: int = 15) -> dict:
    relevant = [c for c in claims if c.gold_doc_ids]
    dense_scores = {f"{m}@{k}": [] for m in ("hit", "recall") for k in ks}
    rerank_scores = {f"{m}@{k}": [] for m in ("hit", "recall") for k in ks}
    dense_rr, rerank_rr = [], []
    dense_ms, rerank_ms = [], []

    for claim in relevant:
        start = time.perf_counter()
        hits = retriever.retrieve([claim.text], top_k=pool)
        dense_ms.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        reranked = reranker.rerank(claim.text, hits, pool)
        rerank_ms.append((time.perf_counter() - start) * 1000)

        dense_docs = [h.doc_id for h in hits]
        rerank_docs = [h.doc_id for h in reranked]
        for k in ks:
            dense_scores[f"hit@{k}"].append(doc_hit_at_k(claim.gold_doc_ids, dense_docs, k))
            dense_scores[f"recall@{k}"].append(doc_recall_at_k(claim.gold_doc_ids, dense_docs, k))
            rerank_scores[f"hit@{k}"].append(doc_hit_at_k(claim.gold_doc_ids, rerank_docs, k))
            rerank_scores[f"recall@{k}"].append(doc_recall_at_k(claim.gold_doc_ids, rerank_docs, k))
        dense_rr.append(reciprocal_rank(claim.gold_doc_ids, dense_docs))
        rerank_rr.append(reciprocal_rank(claim.gold_doc_ids, rerank_docs))

    def mean(values):
        return round(float(np.mean(values)), 4) if values else 0.0

    return {
        "n_claims": len(relevant),
        "dense": {**{k: mean(v) for k, v in dense_scores.items()}, "mrr": mean(dense_rr), "mean_ms": mean(dense_ms)},
        "reranked": {**{k: mean(v) for k, v in rerank_scores.items()}, "mrr": mean(rerank_rr), "mean_ms": mean(rerank_ms)},
    }


def compare_backends(embedder, stores: dict, claims: Sequence[Claim], k: int = 5) -> dict:
    relevant = [c for c in claims if c.gold_doc_ids]
    vectors = [embedder.embed_query(c.text) for c in relevant]
    summary = {}
    retrieved = {}
    for name, store in stores.items():
        latencies, hits_at_k, ids = [], [], []
        for claim, vector in zip(relevant, vectors):
            start = time.perf_counter()
            hits = store.query(vector, k)
            latencies.append((time.perf_counter() - start) * 1000)
            hits_at_k.append(doc_hit_at_k(claim.gold_doc_ids, [h.doc_id for h in hits], k))
            ids.append({h.chunk_id for h in hits})
        retrieved[name] = ids
        summary[name] = {
            "chunks": store.count(),
            f"hit@{k}": round(float(np.mean(hits_at_k)), 4) if hits_at_k else 0.0,
            "mean_ms": round(float(np.mean(latencies)), 3) if latencies else 0.0,
            "p95_ms": round(percentile(latencies, 95), 3),
        }
    names = list(stores)
    if len(names) >= 2:
        a, b = retrieved[names[0]], retrieved[names[1]]
        overlaps = [len(x & y) / len(x | y) for x, y in zip(a, b) if x | y]
        summary["overlap_jaccard"] = round(float(np.mean(overlaps)), 4) if overlaps else 0.0
    return summary


def evaluate_verdicts(agent, claims: Sequence[Claim], progress: bool = True) -> tuple[list[dict], dict]:
    iterator: Iterable[Claim] = claims
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(claims, desc="verdicts")
        except ImportError:
            pass

    rows = []
    for claim in iterator:
        try:
            result = agent.check_claim(claim.text)
        except Exception as error:
            rows.append(
                {
                    "claim_id": claim.claim_id,
                    "claim": claim.text,
                    "gold": claim.label,
                    "pred": "ERROR",
                    "correct": False,
                    "top_score": 0.0,
                    "abstain_reason": "",
                    "attempts": 0,
                    "cited_docs": "",
                    "latency_s": 0.0,
                    "error": f"{type(error).__name__}: {error}"[:200],
                }
            )
            continue
        rows.append(
            {
                "claim_id": claim.claim_id,
                "claim": claim.text,
                "gold": claim.label,
                "pred": result.label,
                "correct": result.label == claim.label,
                "top_score": round(result.top_score, 4),
                "abstain_reason": result.abstain_reason or "",
                "attempts": result.attempts,
                "cited_docs": ";".join(unique_in_order([s.doc_id for s in result.sources])),
                "latency_s": round(sum(result.timings.values()), 3),
                "error": "",
            }
        )

    scored = [r for r in rows if r["pred"] != "ERROR"]
    y_true = [r["gold"] for r in scored]
    y_pred = [r["pred"] for r in scored]
    summary = {
        "n_total": len(rows),
        "n_scored": len(scored),
        "n_errors": len(rows) - len(scored),
        "accuracy": round(accuracy(y_true, y_pred), 4),
        "macro_f1": round(macro_f1(y_true, y_pred, LABELS), 4),
        "per_label": per_label_prf(y_true, y_pred, LABELS),
        "labels": list(LABELS),
        "confusion_matrix": confusion_matrix(y_true, y_pred, LABELS),
        "mean_latency_s": round(float(np.mean([r["latency_s"] for r in scored])), 3) if scored else 0.0,
        "retry_rate": round(float(np.mean([r["attempts"] > 1 for r in scored])), 4) if scored else 0.0,
    }
    return rows, summary


def evaluate_ood(agent, texts: Sequence[str]) -> tuple[list[dict], dict]:
    rows = []
    for text in texts:
        try:
            result = agent.check_claim(text)
            rows.append(
                {
                    "claim": text,
                    "pred": result.label,
                    "abstained": result.label == NOT_ENOUGH_INFO,
                    "top_score": round(result.top_score, 4),
                    "abstain_reason": result.abstain_reason or "",
                    "error": "",
                }
            )
        except Exception as error:
            rows.append(
                {
                    "claim": text,
                    "pred": "ERROR",
                    "abstained": False,
                    "top_score": 0.0,
                    "abstain_reason": "",
                    "error": f"{type(error).__name__}: {error}"[:200],
                }
            )
    scored = [r for r in rows if r["pred"] != "ERROR"]
    summary = {
        "n_total": len(rows),
        "n_scored": len(scored),
        "abstain_rate": round(float(np.mean([r["abstained"] for r in scored])), 4) if scored else 0.0,
    }
    return rows, summary


def calibrate_gate(retriever, reranker, claims: Sequence[Claim], ood_texts: Sequence[str], pool: int = 15) -> dict:
    def top_score(text: str) -> float:
        hits = retriever.retrieve([text], top_k=pool)
        ranked = reranker.rerank(text, hits, 1)
        return float(ranked[0].score) if ranked else 0.0

    groups = {
        SUPPORTED: [top_score(c.text) for c in claims if c.label == SUPPORTED],
        REFUTED: [top_score(c.text) for c in claims if c.label == REFUTED],
        NOT_ENOUGH_INFO: [top_score(c.text) for c in claims if c.label == NOT_ENOUGH_INFO],
        "OUT_OF_DOMAIN": [top_score(t) for t in ood_texts],
    }
    stats = {
        name: {
            "n": len(values),
            "p05": round(percentile(values, 5), 4),
            "p50": round(percentile(values, 50), 4),
            "p95": round(percentile(values, 95), 4),
        }
        for name, values in groups.items()
    }
    relevant = groups[SUPPORTED] + groups[REFUTED]
    ood = groups["OUT_OF_DOMAIN"]
    relevant_p05 = percentile(relevant, 5)
    ood_p95 = percentile(ood, 95)
    return {
        "groups": stats,
        "separable": bool(relevant and ood and ood_p95 < relevant_p05),
        "suggested_gate_min_score": round((ood_p95 + relevant_p05) / 2, 4) if relevant and ood else None,
    }


def write_csv(path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def verdict_markdown(summary: dict) -> str:
    labels = summary["labels"]
    per_label_rows = [
        [label, s["precision"], s["recall"], s["f1"], s["support"]]
        for label, s in summary["per_label"].items()
    ]
    matrix_rows = [[labels[i]] + row for i, row in enumerate(summary["confusion_matrix"])]
    parts = [
        f"Claims scored: {summary['n_scored']} of {summary['n_total']} (errors: {summary['n_errors']})",
        f"Accuracy: {summary['accuracy']}  |  Macro-F1: {summary['macro_f1']}",
        f"Mean latency per claim: {summary['mean_latency_s']} s  |  Retry rate: {summary['retry_rate']}",
        "",
        markdown_table(["Label", "Precision", "Recall", "F1", "Support"], per_label_rows),
        "",
        "Confusion matrix (rows = gold, columns = predicted)",
        markdown_table(["gold \\ pred"] + labels, matrix_rows),
    ]
    return "\n".join(parts)
