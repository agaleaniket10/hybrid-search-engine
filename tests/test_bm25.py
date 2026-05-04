"""Tests for BM25 keyword search."""

import pytest
from search.bm25 import BM25Search

DOCS = [
    "Invoice 12345 payment pending",
    "Refund policy allows 30 days return",
    "Customer account locked due to suspicious activity",
    "FAISS is a vector similarity search library",
    "Hybrid search combines lexical and semantic retrieval",
]


@pytest.fixture
def bm25() -> BM25Search:
    return BM25Search(DOCS)


def test_basic_match(bm25: BM25Search) -> None:
    results = bm25.search("invoice")
    assert any("Invoice" in r for r in results)


def test_empty_query_returns_empty(bm25: BM25Search) -> None:
    assert bm25.search("") == []


def test_special_chars_sanitized(bm25: BM25Search) -> None:
    # Should not raise; special chars are stripped
    results = bm25.search('"invoice"')
    assert isinstance(results, list)


def test_top_k_limit(bm25: BM25Search) -> None:
    results = bm25.search("search", k=2)
    assert len(results) <= 2


def test_no_match_returns_empty(bm25: BM25Search) -> None:
    results = bm25.search("xyznonexistent")
    assert results == []
