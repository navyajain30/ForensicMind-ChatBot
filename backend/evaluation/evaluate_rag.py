"""
evaluate_rag.py — RAG Retrieval Quality Evaluation
====================================================
Computes Precision, Recall, and MRR (Mean Reciprocal Rank)
against a golden evaluation dataset.

Run from the backend/ directory with venv active:
    python -m evaluation.evaluate_rag

Or via the FastAPI endpoint:
    GET http://localhost:8000/evaluate
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.vector_store import basic_search

GOLDEN_DATASET = [
    {"query": "What time did the robbery occur?",              "relevant_sources": ["FIR.txt"]},
    {"query": "Who witnessed the crime?",                      "relevant_sources": ["witness.txt"]},
    {"query": "What was stolen from the victim?",             "relevant_sources": ["FIR.txt", "witness.txt"]},
    {"query": "Describe the suspect and their escape route.", "relevant_sources": ["witness.txt", "FIR.txt"]},
    {"query": "Where did the crime take place?",              "relevant_sources": ["FIR.txt"]},
    {"query": "Which IPC section applies to robbery?",        "relevant_sources": ["FIR.txt", "witness.txt"]},
    {"query": "What injuries did the victim suffer?",         "relevant_sources": ["FIR.txt"]},
    {"query": "What was the victim's name?",                  "relevant_sources": ["FIR.txt"]},
]

N_RESULTS = 3  # chunks retrieved per query

def _retrieved_sources(query: str) -> list:
    try:
        results = basic_search(query=query, n_results=N_RESULTS)
        if not results or not results.get("metadatas") or not results["metadatas"][0]:
            return []
        return [m.get("source", "").lower() for m in results["metadatas"][0] if m.get("source")]
    except Exception as e:
        print(f"  ⚠️  Search failed for '{query}': {e}")
        return []


def _precision(retrieved: list, relevant: list) -> float:
    if not retrieved:
        return 0.0
    rel_lower = [r.lower() for r in relevant]
    return sum(1 for s in retrieved if s in rel_lower) / len(retrieved)


def _recall(retrieved: list, relevant: list) -> float:
    if not relevant:
        return 1.0
    rel_lower = [r.lower() for r in relevant]
    return sum(1 for r in rel_lower if r in retrieved) / len(rel_lower)


def _reciprocal_rank(retrieved: list, relevant: list) -> float:
    rel_lower = [r.lower() for r in relevant]
    for rank, src in enumerate(retrieved, start=1):
        if src in rel_lower:
            return 1.0 / rank
    return 0.0

def run_evaluation() -> dict:
    """Run evaluation and return dict with precision, recall, mrr."""
    precisions, recalls, rr_scores = [], [], []

    for entry in GOLDEN_DATASET:
        retrieved = _retrieved_sources(entry["query"])
        precisions.append(_precision(retrieved, entry["relevant_sources"]))
        recalls.append(_recall(retrieved, entry["relevant_sources"]))
        rr_scores.append(_reciprocal_rank(retrieved, entry["relevant_sources"]))

    return {
        "precision": round(sum(precisions) / len(precisions), 4),
        "recall":    round(sum(recalls)    / len(recalls),    4),
        "mrr":       round(sum(rr_scores)  / len(rr_scores),  4),
    }


def print_results(metrics: dict):
    print("\n" + "=" * 40)
    print("   RAG Retrieval Evaluation")
    print("=" * 40)
    print(f"   Precision  : {metrics['precision']}")
    print(f"   Recall     : {metrics['recall']}")
    print(f"   MRR        : {metrics['mrr']}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    print(f"\nRunning evaluation over {len(GOLDEN_DATASET)} queries...")
    metrics = run_evaluation()
    print_results(metrics)
