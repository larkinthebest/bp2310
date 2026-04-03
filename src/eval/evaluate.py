"""
DEMO evaluation runner.

⚠️  This is a development-time demo with hardcoded synthetic questions.
    It is NOT a real evaluation dataset and should not be cited as such.

    To run real evaluation, create a proper test set with ground truth
    answers and load them from a JSON/CSV file.

Usage:
    python -m src.eval.evaluate
"""

from __future__ import annotations

import json
import logging
import os

from src.config import cfg
from src.rag.pipeline import RAGPipeline
from src.rag.vector_store import VectorStore
from src.eval.metrics import check_ragas_metrics

logger = logging.getLogger(__name__)


def run_demo_evaluation():
    """Run a lightweight demo evaluation with synthetic questions."""
    print("=== DEMO Evaluation (synthetic questions) ===")
    print("⚠️  These are placeholder questions — results are illustrative only.\n")

    vs = VectorStore()
    pipeline = RAGPipeline(vs)

    # Synthetic questions — replace with a real test set for production evaluation
    test_cases = [
        {"question": "What is the main topic of the video?", "ground_truth": None},
        {"question": "Describe the action happening in the footage.", "ground_truth": None},
    ]

    results = []
    for i, case in enumerate(test_cases, 1):
        q = case["question"]
        gt = case.get("ground_truth")
        print(f"[{i}/{len(test_cases)}] Q: {q}")

        result = pipeline.query(q)
        print(f"  A: {result.answer}")
        print(f"  Sources: {result.sources}")

        scores = check_ragas_metrics(q, result.answer, result.contexts, ground_truth=gt)
        print(f"  Metrics: {scores}\n")

        results.append({
            "question": q,
            "answer": result.answer,
            "sources": result.sources,
            "metrics": scores,
        })

    # Save results
    out_path = "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    if not cfg.openai_api_key:
        print("Please set OPENAI_API_KEY")
    else:
        run_demo_evaluation()
