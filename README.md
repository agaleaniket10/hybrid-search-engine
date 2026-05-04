# 🔍 Hybrid Search Engine (BM25 + FAISS + Reranker)

[![CI](https://github.com/agaleaniket10/hybrid-search-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/agaleaniket10/hybrid-search-engine/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)


A production-style hybrid search system combining lexical and semantic retrieval with neural reranking.

---

## 🚀 Features

- **BM25 keyword search** via SQLite FTS5
- **Dense vector search** via FAISS + SentenceTransformers
- **Neural reranking** via CrossEncoder
- **Evaluation metrics** — Recall@K, MRR, latency
- **Cost estimation** based on configurable per-query pricing
- **Modular architecture** — each component is independently testable
- **Fully configurable** via environment variables or `config.py`
- **Interactive CLI** + benchmark mode

---

## 🧱 Architecture

```
         User Query
              │
     ┌────────┴────────┐
     ▼                 ▼
 BM25 Search     Vector Search
 (SQLite FTS5)   (FAISS + ST)
     └────────┬────────┘
              ▼
       Merge + Dedupe
              ▼
      Neural Reranker
       (CrossEncoder)
              ▼
      Final Ranked Results
```

---

## 📁 Project Structure

```
├── main.py              # Entry point (CLI + eval mode)
├── config.py            # All config — models, costs, dataset
├── evaluation.py        # Metrics: Recall@K, MRR, cost estimator
├── search/
│   ├── bm25.py          # BM25 keyword search
│   ├── vector.py        # FAISS vector search
│   ├── reranker.py      # CrossEncoder reranker
│   └── hybrid.py        # Full hybrid pipeline
├── tests/
│   ├── test_bm25.py
│   └── test_evaluation.py
├── requirements.txt
└── requirements-dev.txt
```

---

## ⚙️ Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🏃 Usage

**Interactive mode:**
```bash
python main.py
```

**Evaluation benchmark:**
```bash
python main.py --eval
```

**Debug logging:**
```bash
python main.py --log-level DEBUG
```

---

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## ⚙️ Configuration

All settings are in `config.py` and can be overridden with environment variables:

| Variable | Default | Description |
|---|---|---|
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `TOP_K` | `5` | Number of results to return |
| `EMBEDDING_COST_PER_QUERY` | `0.0001` | Cost per embedding query ($) |
| `RERANK_COST_PER_QUERY` | `0.0003` | Cost per rerank query ($) |

---

## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| Recall@K | Was a relevant doc in the top-K results? |
| MRR | Mean Reciprocal Rank of first relevant result |
| Latency | End-to-end query time in milliseconds |

---

## 📄 License

MIT