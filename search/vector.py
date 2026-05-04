"""
search/vector.py - Dense vector search using FAISS + SentenceTransformers.
"""

import logging
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class VectorSearch:
    """Semantic search backed by a FAISS flat L2 index."""

    def __init__(self, docs: List[str], model_name: str) -> None:
        """
        Encode documents and build the FAISS index.

        Args:
            docs:       List of document strings to index.
            model_name: HuggingFace model name for sentence embeddings.
        """
        self.docs = docs
        self.model = SentenceTransformer(model_name)
        self.index = self._build_index()

    def _build_index(self) -> faiss.IndexFlatL2:
        """
        Encode all documents and add them to a FAISS flat L2 index.

        Returns:
            A populated faiss.IndexFlatL2 instance.
        """
        embeddings = self.model.encode(self.docs, show_progress_bar=False)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings, dtype="float32"))
        logger.debug("FAISS index built: %d vectors, dim=%d.", len(self.docs), dim)
        return index

    def search(self, query: str, k: int = 5) -> List[str]:
        """
        Run a nearest-neighbour vector search.

        Args:
            query: Search query string.
            k:     Maximum number of results to return.

        Returns:
            List of matching document strings (up to k).
        """
        try:
            q_vec = self.model.encode([query], show_progress_bar=False)
            q_vec = np.array(q_vec, dtype="float32")
            _, indices = self.index.search(q_vec, k)
            results = [self.docs[i] for i in indices[0] if i < len(self.docs)]
            logger.debug(
                "Vector search returned %d results for query '%s'.", len(results), query
            )
            return results
        except Exception as exc:
            logger.error("Vector search failed: %s", exc)
            return []
