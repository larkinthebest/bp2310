"""
RAGAS evaluation metrics.

IMPORTANT DESIGN NOTE:
  The original code set `ground_truth = answer` which is misleading —
  it evaluates how well the answer matches *itself*. That has been removed.

  Metrics that require ground truth (context_precision, context_recall) are only
  evaluated when real ground truth is explicitly provided.

  Without ground truth, only `faithfulness` and `answer_relevancy` are computed.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.config import cfg

logger = logging.getLogger(__name__)


def check_ragas_metrics(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str | None = None,
) -> dict[str, Any]:
    """
    Compute RAGAS metrics for a single QA pair.

    Args:
        question: The user's question.
        answer: The generated answer.
        contexts: Retrieved context strings.
        ground_truth: Optional real ground truth. If not provided,
                      metrics that need it are skipped.

    Returns:
        Dict of metric name → score, or {"error": "..."} on failure.
    """
    if not cfg.openai_api_key:
        return {"error": "OPENAI_API_KEY not set"}

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from datasets import Dataset
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    except ImportError:
        return {"error": "ragas not installed"}

    # Select metrics based on whether ground truth is available
    metrics = [faithfulness, answer_relevancy]
    data: dict[str, list] = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }

    if ground_truth:
        metrics.extend([context_precision, context_recall])
        data["ground_truth"] = [ground_truth]
        data["reference"] = [ground_truth]
    else:
        logger.info("No ground truth provided — skipping context_precision and context_recall")

    dataset = Dataset.from_dict(data)
    eval_model = cfg.eval_model
    evaluator_llm = ChatOpenAI(model=eval_model)
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=cfg.embedding_model)
    )

    try:
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )
        df = results.to_pandas()
        row = df.iloc[0].to_dict()
        wanted = {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
        return {k: v for k, v in row.items() if k in wanted}
    except Exception as e:
        logger.error("RAGAS evaluation failed: %s", e)
        return {"error": str(e)}
