import pytest

from evaluation.metrics import (
    accuracy,
    confusion_matrix,
    doc_hit_at_k,
    doc_recall_at_k,
    macro_f1,
    per_label_prf,
    percentile,
    reciprocal_rank,
)
from factcheck.reranker import NoOpReranker, to_probability
from factcheck.schemas import Hit

LABELS = ["A", "B", "C"]


def test_accuracy_and_confusion():
    y_true = ["A", "A", "B", "C"]
    y_pred = ["A", "B", "B", "C"]
    assert accuracy(y_true, y_pred) == 0.75
    assert confusion_matrix(y_true, y_pred, LABELS) == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]


def test_macro_f1_perfect_and_partial():
    assert macro_f1(["A", "B"], ["A", "B"], LABELS) == 1.0
    scores = per_label_prf(["A", "A", "B"], ["A", "B", "B"], LABELS)
    assert scores["A"]["precision"] == 1.0
    assert scores["A"]["recall"] == 0.5
    assert scores["C"]["support"] == 0


def test_doc_level_retrieval_metrics():
    ranked = ["1", "1", "2", "3"]
    assert doc_hit_at_k(["2"], ranked, 1) == 0.0
    assert doc_hit_at_k(["2"], ranked, 2) == 1.0
    assert doc_recall_at_k(["2", "9"], ranked, 3) == 0.5
    assert reciprocal_rank(["3"], ranked) == pytest.approx(1 / 3)
    assert reciprocal_rank(["9"], ranked) == 0.0


def test_percentile_handles_empty():
    assert percentile([], 95) == 0.0
    assert percentile([1, 2, 3], 50) == 2.0


def test_to_probability_handles_logits_and_probabilities():
    probabilities = to_probability([0.1, 0.9])
    assert list(probabilities) == [0.1, 0.9]
    converted = to_probability([-10.0, 0.0, 10.0])
    assert converted[0] < 0.01 and converted[1] == pytest.approx(0.5) and converted[2] > 0.99


def test_noop_reranker_truncates():
    hits = [Hit(chunk_id=str(i), doc_id="d", title="t", text="x", score=1.0 - i / 10) for i in range(5)]
    assert [h.chunk_id for h in NoOpReranker().rerank("q", hits, 2)] == ["0", "1"]
