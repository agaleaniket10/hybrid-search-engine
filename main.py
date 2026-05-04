"""
main.py - Entry point for the hybrid search engine.

Usage:
    python main.py              # interactive CLI mode
    python main.py --eval       # run evaluation benchmark
"""

import argparse
import logging
import time

from config import EMBED_MODEL, RERANKER_MODEL, SAMPLE_DOCS, TOP_K
from evaluation import run_evaluation
from search.hybrid import HybridSearchEngine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------


def interactive(engine: HybridSearchEngine) -> None:
    """
    Run an interactive query loop in the terminal.

    Args:
        engine: Initialised HybridSearchEngine instance.
    """
    print("\nHybrid Search Engine — type 'eval' to benchmark, 'exit' to quit.\n")

    while True:
        try:
            query = input("Enter query: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() == "exit":
            break
        if query.lower() == "eval":
            run_evaluation(engine)
            continue

        start = time.perf_counter()
        results = engine.search(query)
        latency_ms = (time.perf_counter() - start) * 1000

        print("\nTop results:")
        for i, doc in enumerate(results, start=1):
            print(f"  {i}. {doc}")
        print(f"Latency: {round(latency_ms, 2)} ms\n")


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid Search Engine")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation benchmark and exit.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging verbosity (default: INFO).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(args.log_level)

    logger.info("Loading models — this may take a moment on first run...")
    engine = HybridSearchEngine(
        docs=SAMPLE_DOCS,
        embed_model_name=EMBED_MODEL,
        reranker_model_name=RERANKER_MODEL,
        top_k=TOP_K,
    )

    if args.eval:
        run_evaluation(engine)
    else:
        interactive(engine)


if __name__ == "__main__":
    main()
