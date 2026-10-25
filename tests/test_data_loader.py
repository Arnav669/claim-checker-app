import json

from factcheck.data_loader import load_claims, load_corpus


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_load_corpus(tmp_path):
    write_jsonl(
        tmp_path / "corpus.jsonl",
        [{"doc_id": 5, "title": "T", "abstract": ["a.", "b."], "structured": False}],
    )
    docs = list(load_corpus(tmp_path))
    assert docs[0].doc_id == "5"
    assert docs[0].sentences == ["a.", "b."]


def test_load_claims_labels(tmp_path):
    rows = [
        {"id": 1, "claim": "c1", "evidence": {"10": [{"sentences": [0], "label": "SUPPORT"}]}, "cited_doc_ids": [10]},
        {"id": 2, "claim": "c2", "evidence": {"11": [{"sentences": [1], "label": "CONTRADICT"}]}, "cited_doc_ids": [11]},
        {"id": 3, "claim": "c3", "evidence": {}, "cited_doc_ids": [12]},
        {
            "id": 4,
            "claim": "c4",
            "evidence": {
                "13": [{"sentences": [0], "label": "SUPPORT"}],
                "14": [{"sentences": [0], "label": "CONTRADICT"}],
            },
        },
    ]
    write_jsonl(tmp_path / "claims_dev.jsonl", rows)
    claims = load_claims(tmp_path, "dev")
    by_id = {c.claim_id: c for c in claims}
    assert by_id["1"].label == "SUPPORTED" and by_id["1"].gold_doc_ids == ["10"]
    assert by_id["2"].label == "REFUTED"
    assert by_id["3"].label == "NOT_ENOUGH_INFO" and by_id["3"].gold_doc_ids == []
    assert "4" not in by_id


def test_test_split_has_no_labels(tmp_path):
    write_jsonl(tmp_path / "claims_test.jsonl", [{"id": 9, "claim": "x", "cited_doc_ids": [1]}])
    claims = load_claims(tmp_path, "test")
    assert claims[0].label is None
