"""
config.py - Central configuration for the hybrid search engine.
All tuneable parameters live here; override via environment variables.
"""

import os

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
RERANKER_MODEL: str = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
TOP_K: int = int(os.getenv("TOP_K", "5"))

# ---------------------------------------------------------------------------
# Cost estimation ($ per query — adjust to your actual provider pricing)
# ---------------------------------------------------------------------------
EMBEDDING_COST_PER_QUERY: float = float(os.getenv("EMBEDDING_COST_PER_QUERY", "0.0001"))
RERANK_COST_PER_QUERY: float = float(os.getenv("RERANK_COST_PER_QUERY", "0.0003"))

# ---------------------------------------------------------------------------
# Sample dataset (replace or extend with your own documents)
# ---------------------------------------------------------------------------
SAMPLE_DOCS: list[str] = [
    "Invoice 12345 payment pending",
    "Refund policy allows 30 days return",
    "Customer account locked due to suspicious activity",
    "FAISS is a vector similarity search library",
    "Hybrid search combines lexical and semantic retrieval",
]

# Ground-truth evaluation set
EVAL_SET: list[dict] = [
    {"query": "invoice payment", "relevant": ["Invoice 12345 payment pending"]},
    {"query": "refund policy", "relevant": ["Refund policy allows 30 days return"]},
    {
        "query": "vector search library",
        "relevant": ["FAISS is a vector similarity search library"],
    },
]
