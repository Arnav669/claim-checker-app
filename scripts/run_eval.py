from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.runners import (
    calibrate_gate,
    compare_backends,
    evaluate_ood,
    evaluate_retrieval,
    evaluate_verdicts,
    markdown_table,
    sample_claims,
    verdict_markdown,
    write_csv,
    write_json,
)
from factcheck.config import get_settings
from factcheck.data_loader import load_claims
from factcheck.embeddings import build_embedder
from factcheck.factory import build_agent
from factcheck.observability import flush
from factcheck.reranker import CrossEncoderReranker
from factcheck.retriever import Retriever
from factcheck.vector_store import build_store

OOD_PATH = ROOT / "data" / "ood_claims.txt"


def load_ood() -> list[str]:
    return [line.strip() for line in OOD_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["calibrate", "retrieval", "backends", "verdict", "ood"])
    parser.add_argument("--split", default="dev")
    parser.add_argument("--per-label", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    reports = Path(settings.reports_dir)
    claims = load_claims(settings.data_dir, args.split)

    if args.command in ("calibrate", "retrieval", "backends"):
        embedder = build_embedder(settings)

    if args.command == "calibrate":
        store = build_store(args.backend, settings)
        retriever = Retriever(embedder, store, settings.retrieval_top_k)
        reranker = CrossEncoderReranker(settings.rerank_model)
        report = calibrate_gate(retriever, reranker, claims, load_ood(), settings.retrieval_top_k)
        write_json(reports / "gate_calibration.json", report)
        print(markdown_table(
            ["Group", "n", "p05", "p50", "p95"],
            [[name, s["n"], s["p05"], s["p50"], s["p95"]] for name, s in report["groups"].items()],
        ))
        print(f"Separable: {report['separable']}  Suggested GATE_MIN_SCORE: {report['suggested_gate_min_score']}")

    elif args.command == "retrieval":
        store = build_store(args.backend, settings)
        retriever = Retriever(embedder, store, settings.retrieval_top_k)
        reranker = CrossEncoderReranker(settings.rerank_model)
        report = evaluate_retrieval(retriever, reranker, claims, pool=settings.retrieval_top_k)
        write_json(reports / "retrieval_eval.json", report)
        keys = ["hit@1", "hit@3", "hit@5", "recall@5", "mrr", "mean_ms"]
        print(f"Claims with gold evidence: {report['n_claims']}")
        print(markdown_table(
            ["Stage"] + keys,
            [["dense (bi-encoder)"] + [report["dense"][k] for k in keys],
             ["reranked (cross-encoder)"] + [report["reranked"][k] for k in keys]],
        ))

    elif args.command == "backends":
        stores = {"chroma": build_store("chroma", settings), "faiss": build_store("faiss", settings)}
        report = compare_backends(embedder, stores, claims)
        write_json(reports / "backend_comparison.json", report)
        print(report)

    elif args.command == "verdict":
        agent = build_agent(settings, backend=args.backend, use_reranker=not args.no_rerank)
        sample = sample_claims(claims, args.per_label, args.seed)
        rows, summary = evaluate_verdicts(agent, sample)
        suffix = "no_rerank" if args.no_rerank else "full"
        write_csv(reports / f"verdict_results_{suffix}.csv", rows)
        write_json(reports / f"verdict_summary_{suffix}.json", summary)
        text = verdict_markdown(summary)
        (reports / f"verdict_summary_{suffix}.md").write_text(text, encoding="utf-8")
        print(text)
        flush()

    elif args.command == "ood":
        agent = build_agent(settings, backend=args.backend, use_reranker=not args.no_rerank)
        rows, summary = evaluate_ood(agent, load_ood())
        write_csv(reports / "ood_results.csv", rows)
        write_json(reports / "ood_summary.json", summary)
        print(summary)
        flush()


if __name__ == "__main__":
    main()
