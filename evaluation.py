"""
evaluation.py - Evaluation metrics and benchmark pipeline.
"""

import logging
import time
from typing import Dict, List

from config import EVAL_SET, EMBEDDING_COST_PER_QUERY, RERANK_COST_PER_QUERY
from search.hybrid import HybridSearchEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def recall_at_k(predicted: List[str], relevant: List[str], k: int = 5) -> int:
    """
    Binary Recall@K — 1 if any relevant doc appears in the top-k results.

    Args:
        predicted: Ranked list of retrieved documents.
        relevant:  Ground-truth relevant documents.
        k:         Cutoff rank.

    Returns:
        1 if a relevant document is in the top-k, else 0.
    """
    return int(any(doc in predicted[:k] for doc in relevant))


def mean_reciprocal_rank(predicted: List[str], relevant: List[str]) -> float:
    """
    Mean Reciprocal Rank (MRR) for a single query.

    Args:
        predicted: Ranked list of retrieved documents.
        relevant:  Ground-truth relevant documents.

    Returns:
        Reciprocal rank of the first relevant document, or 0.0.
    """
    for rank, doc in enumerate(predicted, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def estimate_cost(num_queries: int) -> float:
    """
    Estimate the total search cost for a given number of queries.

    Args:
        num_queries: Number of queries to estimate cost for.

    Returns:
        Estimated cost in USD based on config pricing.
    """
    return num_queries * (EMBEDDING_COST_PER_QUERY + RERANK_COST_PER_QUERY)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_evaluation(engine: HybridSearchEngine) -> Dict[str, float]:
    """
    Run the full evaluation benchmark against the ground-truth eval set.

    Args:
        engine: Initialised HybridSearchEngine instance.

    Returns:
        Dict with avg_recall, avg_mrr, avg_latency_ms.
    """
    logger.info("Starting evaluation on %d queries.", len(EVAL_SET))
    print("\n=== EVALUATION START ===")

    total_recall = 0.0
    total_mrr = 0.0
    latencies: List[float] = []

    for item in EVAL_SET:
        start = time.perf_counter()
        results = engine.search(item["query"])
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

        r = recall_at_k(results, item["relevant"])
        m = mean_reciprocal_rank(results, item["relevant"])
        total_recall += r
        total_mrr += m

        print(f"\nQuery    : {item['query']}")
        print(f"Results  : {results}")
        print(f"Recall@5 : {r}")
        print(f"MRR      : {round(m, 3)}")
        print(f"Latency  : {round(latency_ms, 2)} ms")

    n = len(EVAL_SET)
    avg_recall = total_recall / n
    avg_mrr = total_mrr / n
    avg_latency = sum(latencies) / n

    print("\n=== FINAL METRICS ===")
    print(f"Avg Recall@5 : {round(avg_recall, 3)}")
    print(f"Avg MRR      : {round(avg_mrr, 3)}")
    print(f"Avg Latency  : {round(avg_latency, 2)} ms")
    print(f"Est. cost / 1000 queries : ${round(estimate_cost(1000), 4)}")

    logger.info(
        "Evaluation complete — Recall@5=%.3f, MRR=%.3f, Latency=%.2fms",
        avg_recall,
        avg_mrr,
        avg_latency,
    )

    return {"avg_recall": avg_recall, "avg_mrr": avg_mrr, "avg_latency_ms": avg_latency}
