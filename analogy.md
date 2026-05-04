Imagine you are a librarian in a massive library, and someone walks in asking for a specific piece of information. This project is like a "Super Librarian" system that uses three different strategies to find the perfect answer.

Here is the simple breakdown of how it works:

1. The Three Search Strategies
Instead of just looking at the book titles, the system uses three layers of "intelligence":

The Keyword Search (BM25): This is like using an index. If you search for "Invoice," it looks for every document that contains the exact word "Invoice." It’s fast but "dumb"—if you type "bill" instead of "invoice," it might miss it.

The "Vibe" Search (Vector Search): This is much smarter. It understands meaning. If you search for "refund," it knows that "money back" or "return policy" are related, even if those exact words aren't there. It looks for the "vibe" or the concept of your question.

The Expert Judge (Reranker): After the first two searches find a bunch of possible answers, the Expert Judge takes a close look at the top results. It carefully compares your question to each result to make sure the very best one is at the top of the list.

2. The "Report Card" (Evaluation)
A big part of this project isn't just searching; it's checking the homework. The system runs a test to see how well it's doing.

It asks: "Did I find the right answer in my top 5 results?"

It also asks: "How fast was I?" (Nobody likes a slow librarian).

By doing this, the system can prove it is getting better over time.

3. The Price Tag (Cost Estimation)
Advanced AI "brains" cost money every time you ask them a question. This project includes a calculator that predicts how much it will cost to run 1,000 searches. This helps a business decide if the system is worth the investment before they spend a cent.



Tech explanation:

This project is a **Hybrid RAG (Retrieval-Augmented Generation) Pipeline** designed to solve the "Precision vs. Recall" trade-off. 

Using the **Super Librarian** analogy, here is how the code implements those layers:

### 1. The "Index" (Lexical Search via BM25)
**The Analogy:** The librarian checks the literal back-of-the-book index for exact matches.
**The Tech:** We use **SQLite FTS5** (Full-Text Search) to handle keyword matching. This ensures that if a user searches for a specific ID like "Invoice 12345," we find it instantly via token matching.

```python
# Creating the virtual table for fast keyword lookups
c.execute("CREATE VIRTUAL TABLE docs USING fts5(content)")

# Performing the literal match
res = c.execute("SELECT content FROM docs WHERE docs MATCH ?", (safe_query,))
```

### 2. The "Vibe" (Semantic Search via Bi-Encoders)
**The Analogy:** The librarian understands the *concept* of the request (e.g., "refund" means "return policy").
**The Tech:** We use a **Bi-Encoder** (`all-MiniLM-L6-v2`) to project text into a 384-dimensional vector space. We then use **FAISS** to perform an $L2$ distance search to find conceptually similar documents.

```python
# Turning text into "vibes" (vectors)
doc_embeddings = embed_model.encode(docs)

# Using FAISS for "Approximate Nearest Neighbor" search
index = faiss.IndexFlatL2(dim)
index.add(np.array(doc_embeddings).astype("float32"))
```

### 3. The "Expert Judge" (Cross-Encoder Reranking)
**The Analogy:** The librarian takes the 10 best books found so far and reads them closely to pick the #1 winner.
**The Tech:** We take the union of the Keyword and Vector results and pass them through a **Cross-Encoder**. Unlike the Bi-Encoder, this model processes the Query and Document *simultaneously*, allowing for deep attention-based scoring.

```python
def rerank(query, docs):
    # The "Expert" looks at both query and doc at the same time
    pairs = [(query, d) for d in docs]
    scores = reranker.predict(pairs)
    
    # Sort results by the expert's confidence score
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [r[0] for r in ranked]
```

### 4. The "Report Card" (Evaluation Metrics)
**The Analogy:** We track if the librarian found the right book and how high it was on the pile.
**The Tech:** We measure **MRR (Mean Reciprocal Rank)** and **Recall@K**. 
*   **Recall@5:** "Is the correct answer in the top 5?"
*   **MRR:** "Is the correct answer at the very top (1.0) or buried at the bottom (0.2)?"

```python
def mrr(predicted, relevant):
    for i, p in enumerate(predicted):
        if p in relevant:
            return 1 / (i + 1) # Higher is better (1.0 = first place)
    return 0
```

### 5. The "Price Tag" (Cost Estimation)
**The Analogy:** Calculating the "salary" of the AI brains used for the search.
**The Tech:** Since Cross-Encoders are computationally expensive ($O(N \cdot M)$ complexity), we track the "cost" of the inference to ensure the business logic remains viable at scale.

```python
def estimate_cost(num_queries):
    # Inference is more expensive than standard DB lookups
    embedding_cost = num_queries * 0.0001
    rerank_cost = num_queries * 0.0003
    return embedding_cost + rerank_cost
```

---

### Why this is a "Senior" approach:
By combining these, you prevent the two biggest failures in AI search:
1.  **Vector-only failure:** Searching for "User_99" but getting a bio of "User_100" because they "sound similar." (Keyword search fixes this).
2.  **Keyword-only failure:** Searching for "I'm unhappy with my purchase" and finding nothing because the document says "Customer complaint regarding item." (Vector search fixes this).