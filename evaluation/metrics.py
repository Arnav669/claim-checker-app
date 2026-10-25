from __future__ import annotations

from typing import Sequence

import numpy as np


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> list[list[int]]:
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t in index and p in index:
            matrix[index[t]][index[p]] += 1
    return matrix


def per_label_prf(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> dict[str, dict[str, float]]:
    scores = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for t in y_true if t == label),
        }
    return scores


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    scores = per_label_prf(y_true, y_pred, labels)
    present = [scores[l]["f1"] for l in labels if scores[l]["support"] > 0]
    return float(np.mean(present)) if present else 0.0


def unique_in_order(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(items))


def doc_hit_at_k(gold: Sequence[str], ranked_doc_ids: Sequence[str], k: int) -> float:
    top = set(unique_in_order(ranked_doc_ids)[:k])
    return 1.0 if any(g in top for g in gold) else 0.0


def doc_recall_at_k(gold: Sequence[str], ranked_doc_ids: Sequence[str], k: int) -> float:
    if not gold:
        return 0.0
    top = set(unique_in_order(ranked_doc_ids)[:k])
    return sum(1 for g in set(gold) if g in top) / len(set(gold))


def reciprocal_rank(gold: Sequence[str], ranked_doc_ids: Sequence[str]) -> float:
    gold_set = set(gold)
    for rank, doc_id in enumerate(unique_in_order(ranked_doc_ids), start=1):
        if doc_id in gold_set:
            return 1.0 / rank
    return 0.0


def percentile(values: Sequence[float], q: float) -> float:
    if not len(values):
        return 0.0
    return float(np.percentile(np.asarray(values, dtype="float64"), q))
