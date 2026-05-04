"""
search/bm25.py - BM25 keyword search using SQLite FTS5.
"""

import logging
import sqlite3
from typing import List

logger = logging.getLogger(__name__)


class BM25Search:
    """Lexical search backed by SQLite FTS5 (approximates BM25 ranking)."""

    def __init__(self, docs: List[str]) -> None:
        """
        Initialise the in-memory FTS5 index and insert documents.

        Args:
            docs: List of document strings to index.
        """
        self.docs = docs
        self.conn = sqlite3.connect(":memory:")
        self._build_index()

    def _build_index(self) -> None:
        """Create the FTS5 virtual table and populate it."""
        c = self.conn.cursor()
        c.execute("CREATE VIRTUAL TABLE docs USING fts5(content)")
        c.executemany("INSERT INTO docs(content) VALUES (?)", [(d,) for d in self.docs])
        self.conn.commit()
        logger.debug("BM25 index built with %d documents.", len(self.docs))

    @staticmethod
    def _sanitize(query: str) -> str:
        """
        Strip characters that are special in FTS5 syntax.

        Args:
            query: Raw user query string.

        Returns:
            A safe query string containing only alphanumeric tokens.
        """
        return " ".join(word for word in query.split() if word.isalnum())

    def search(self, query: str, k: int = 5) -> List[str]:
        """
        Run a BM25 keyword search.

        Args:
            query: Search query.
            k:     Maximum number of results to return.

        Returns:
            List of matching document strings (up to k).
        """
        safe_query = self._sanitize(query)
        if not safe_query:
            logger.warning("BM25 received an empty query after sanitisation.")
            return []
        try:
            cursor = self.conn.execute(
                "SELECT content FROM docs WHERE docs MATCH ?", (safe_query,)
            )
            results = [row[0] for row in cursor.fetchall()][:k]
            logger.debug(
                "BM25 returned %d results for query '%s'.", len(results), query
            )
            return results
        except sqlite3.OperationalError as exc:
            logger.error("BM25 search failed: %s", exc)
            return []
