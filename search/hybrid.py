"""
search/hybrid.py - Combines BM25 + vector search with neural reranking.
"""

import logging
from typing import List

from search.bm25 import BM25Search
from search.reranker import Reranker
from search.vector import VectorSearch

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """
    End-to-end hybrid search pipeline:
      1. BM25 keyword retrieval
      2. Dense vector retrieval
      3. Result merging + deduplication
      4. Neural reranking
    """

    def __init__(
        self,
        docs: List[str],
        embed_model_name: str,
        reranker_model_name: str,
        top_k: int = 5,
    ) -> None:
        """
        Initialise all search components.

        Args:
            docs:                 Documents to search over.
            embed_model_name:     Embedding model for vector search.
            reranker_model_name:  CrossEncoder model for reranking.
            top_k:                Number of results to return.
        """
        self.top_k = top_k
        self.bm25 = BM25Search(docs)
        self.vector = VectorSearch(docs, embed_model_name)
        self.reranker = Reranker(reranker_model_name)
        logger.info("HybridSearchEngine initialised with %d documents.", len(docs))

    def search(self, query: str) -> List[str]:
        """
        Run the full hybrid search pipeline.

        Args:
            query: User search query.

        Returns:
            Reranked list of the most relevant documents.
        """
        bm25_results = self.bm25.search(query, k=self.top_k)
        vector_results = self.vector.search(query, k=self.top_k)

        # Merge and deduplicate while preserving order
        seen: set = set()
        merged: List[str] = []
        for doc in bm25_results + vector_results:
            if doc not in seen:
                seen.add(doc)
                merged.append(doc)

        logger.debug(
            "Merged %d unique candidates (bm25=%d, vector=%d).",
            len(merged),
            len(bm25_results),
            len(vector_results),
        )
        return self.reranker.rerank(query, merged)
