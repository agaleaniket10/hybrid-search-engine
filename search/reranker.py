"""
search/reranker.py - Neural reranking using a CrossEncoder model.
"""

import logging
from typing import List

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class Reranker:
    """Scores query-document pairs and returns results sorted by relevance."""

    def __init__(self, model_name: str) -> None:
        """
        Load the CrossEncoder reranking model.

        Args:
            model_name: HuggingFace model name for the cross-encoder.
        """
        self.model = CrossEncoder(model_name)
        logger.debug("Reranker loaded: %s", model_name)

    def rerank(self, query: str, docs: List[str]) -> List[str]:
        """
        Rerank a list of documents by relevance to the query.

        Args:
            query: The search query.
            docs:  Candidate documents to rerank.

        Returns:
            Documents sorted from most to least relevant.
        """
        if not docs:
            return []
        try:
            pairs = [(query, doc) for doc in docs]
            scores = self.model.predict(pairs)
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            logger.debug("Reranked %d documents for query '%s'.", len(docs), query)
            return [doc for doc, _ in ranked]
        except Exception as exc:
            logger.error("Reranking failed: %s", exc)
            return docs  # fall back to original order
