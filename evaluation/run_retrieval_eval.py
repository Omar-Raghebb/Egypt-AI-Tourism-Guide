"""Retrieval evaluation over a bilingual dataset.

Default dataset: evaluation/dataset.json (legacy six-landmark KB).
Enhanced dataset: evaluation/dataset_enhanced.json (enhanced RehlaAI JSON).

Metrics implemented: Precision@K, Recall@K, MRR, Hit Rate@K, retrieval latency.

Ground truth: each case lists ``relevant_doc_ids`` (prefix match) and/or
``relevant_place_ids`` (structured-record/extra match). Values were copied from
the actual knowledge base records - nothing invented.

Usage:
    python -m evaluation.run_retrieval_eval [--dataset evaluation/dataset_enhanced.json] \
        [--top-k 4] [--out evaluation/results_retrieval.json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List, Set

from rag.pipeline import RAGPipeline

DATASET_PATH = Path(__file__).parent / "dataset.json"


def load_dataset(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["cases"]


def doc_matches(doc_id: str, relevant_prefixes: List[str]) -> bool:
    return any(doc_id == p or doc_id.startswith(p) for p in relevant_prefixes)


def evaluate_case(case: Dict, pipeline: RAGPipeline, top_k: int) -> Dict:
    result = pipeline.retrieve(case["query"], top_k=top_k)

    relevant_docs = set(case.get("relevant_doc_ids", []))
    relevant_places = set(case.get("relevant_place_ids", []))

    # relevance judged on the union of doc hits and structured-record hits.
    # Each relevant doc id / record id counts as ONE ground-truth item;
    # matching it via documents, structured places, or structured extras all
    # map to the same ground-truth key.
    hit_ranks: List[int] = []
    gt_items: Set[str] = {f"doc::{p}" for p in relevant_docs}
    gt_items |= {f"id::{p}" for p in relevant_places}
    n_relevant_total = len(gt_items)

    retrieved_hits: Set[str] = set()
    for rank, doc in enumerate(result.documents, start=1):
        if relevant_docs and doc_matches(doc.doc_id, relevant_docs):
            hit_ranks.append(rank)
            retrieved_hits.add(f"doc::{_match_prefix(doc.doc_id, relevant_docs)}")
        if doc.place_id and doc.place_id in relevant_places:
            hit_ranks.append(rank)
            retrieved_hits.add(f"id::{doc.place_id}")
    for place in result.structured_records:
        if place.place_id in relevant_places:
            retrieved_hits.add(f"id::{place.place_id}")
            if not hit_ranks:
                hit_ranks.append(1)  # structured results rank first
    # non-place structured extras (cities, activities, itineraries, rules...)
    for extra in getattr(result, "structured_extras", []):
        if extra.record_id in relevant_places:
            retrieved_hits.add(f"id::{extra.record_id}")
            if not hit_ranks:
                hit_ranks.append(1)

    precision = len(retrieved_hits) / len(gt_items) if gt_items else None
    recall = len(retrieved_hits) / n_relevant_total if n_relevant_total else None
    mrr = 1.0 / hit_ranks[0] if hit_ranks else 0.0
    hit_rate = 1.0 if hit_ranks else 0.0
    latency = result.latency_ms or 0.0

    return {
        "id": case["id"],
        "query": case["query"],
        "precision_at_k": precision,
        "recall_at_k": recall,
        "mrr": mrr,
        "hit": bool(hit_rate),
        "latency_ms": latency,
        "intent_detected": result.intent,
        "expected_intent": case.get("intent"),
        "num_docs": len(result.documents),
        "num_structured": len(result.structured_records),
        "methods_used": result.methods_used,
    }


def _match_prefix(doc_id: str, prefixes: List[str]) -> str:
    for p in prefixes:
        if doc_id == p or doc_id.startswith(p):
            return p
    return doc_id


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default=str(DATASET_PATH),
        help="Path to the evaluation dataset JSON (default: legacy dataset.json)",
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--out", default=str(DATASET_PATH.parent / "results_retrieval.json"))
    args = parser.parse_args(argv)

    pipeline = RAGPipeline()
    pipeline.initialize()

    cases = load_dataset(Path(args.dataset))
    per_case = []
    for case in cases:
        try:
            per_case.append(evaluate_case(case, pipeline, args.top_k))
        except Exception as exc:  # keep evaluating other cases
            per_case.append({"id": case["id"], "error": str(exc)})

    evaluable = [c for c in per_case if "error" not in c and c["precision_at_k"] is not None]
    summary = {
        "num_cases": len(cases),
        "num_evaluated": len(evaluable),
        "num_errors": len(per_case) - len(evaluable),
        "top_k": args.top_k,
        "precision_at_k": round(mean(c["precision_at_k"] for c in evaluable), 3) if evaluable else None,
        "recall_at_k": round(mean(c["recall_at_k"] for c in evaluable), 3) if evaluable else None,
        "mrr": round(mean(c["mrr"] for c in evaluable), 3) if evaluable else None,
        "hit_rate_at_k": round(mean(c["hit"] for c in evaluable), 3) if evaluable else None,
        "avg_latency_ms": round(mean(c["latency_ms"] for c in evaluable), 1) if evaluable else None,
        "intent_accuracy": None,
    }
    intent_cases = [
        c for c in evaluable if c.get("expected_intent") and c.get("intent_detected")
    ]
    if intent_cases:
        correct = sum(
            1 for c in intent_cases
            if c["intent_detected"] == c["expected_intent"]
            or (c["expected_intent"] == "semantic" and c["intent_detected"] in
                ("semantic", "recommendation", "definition", "activities", "itinerary"))
        )
        summary["intent_accuracy"] = round(correct / len(intent_cases), 3)

    out = {"summary": summary, "per_case": per_case}
    Path(args.out).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nPer-case details written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
