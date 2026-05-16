# Vector Search, Embeddings, and Semantic Similarity — From Scratch

## Core Concept

Given a dataset, find semantically similar content to any query. This works by converting text into
vectors (lists of numbers) that capture meaning, then finding the closest vectors to a query vector.

---

## Embeddings

- An embedding model takes text and outputs a vector (e.g. 384 numbers for `all-MiniLM-L6-v2`)
- Similar meaning = similar numbers. "cat" and "dog" end up with similar vectors; "cat" and "apple" do not
- The model learned these relationships by training on billions of sentences — words appearing in similar
  contexts end up with similar vectors (distributional hypothesis)
- **Different models produce completely incompatible vectors.** All documents and queries must use the
  same model. Switching models requires re-embedding everything from scratch

> In this repo: `embedder.py` → `load_model()` and `embed()` handle this.

---

## Cosine Similarity

- Measures the **angle** between two vectors, not the distance
- Score ranges -1 to 1 (in practice 0–1 for natural language):
  - `0.9+` = near identical
  - `0.7–0.9` = very similar
  - `0.5–0.7` = related
  - `0.3–0.5` = loosely related
- Ignores vector magnitude (length), only cares about direction — so a short and long document about
  the same topic score high
- Most embedding models output normalised vectors (magnitude = 1), reducing cosine similarity to a
  simple dot product

> **Note:** Our `similarity.py` returns raw cosine scores (-1 to 1). In practice all IMDB scores are
> positive, but don't assume the range is 0–1 in code.
>
> In this repo: `similarity.py` → `cosine_similarity()` and `top_k_indices()` handle this.

---

## The Search Pipeline

```
Phase 1 (once):       embed all documents → store vectors
Phase 2 (per query):  embed query (same model) → find nearest vectors → return docs
```

Key advantage over keyword search: finds **meaning**, not just exact words.
"fluffy pets" finds documents about cats and dogs even if those words never appear.

> In this repo: `imdb_search.py` runs both phases on every invocation (no persistence yet).

---

## The numpy Search Function (line by line)

```python
model.encode([query])[0]                          # embed the query → shape (384,)
np.linalg.norm(doc_embeddings, axis=1)            # magnitude of each document vector → shape (N,)
np.dot(doc_embeddings, query) / (norms * q_norm)  # cosine similarity for all docs at once → shape (N,)
np.argsort()[::-1][:top_k]                        # sort descending, take top k
```

This brute-force approach works for hundreds of docs but **not millions**.

> In this repo: `similarity.py` implements exactly this. Original version inline in `main.py`.

---

## How Models Learn (Gradient Descent)

1. All weights start as random numbers
2. Model makes a prediction → measures how wrong it was (loss)
3. Nudges all weights slightly in the direction that reduces loss
4. Repeats billions of times

"Learning" = weights settling into a configuration that captures patterns from training data.

The downloaded `pytorch_model.bin` file **literally is the knowledge** — 22 million numbers encoding
relationships learned from billions of sentences.

---

## `all-MiniLM-L6-v2` Specifics

- Downloaded from Hugging Face (~90 MB), cached locally after first run
- 6-layer distilled model, **384-dimensional** output vectors
- The HF warning about unauthenticated requests is harmless
  - Set `TRANSFORMERS_OFFLINE=1` to suppress it once the model is cached
- Hugging Face is essentially GitHub for AI models

---

## Scaling: How Vector DBs Handle Large Datasets

| Approach | How it works | When to use |
|---|---|---|
| Brute force (numpy) | Compare query to every vector | < ~10k docs |
| HNSW | Layered graph, searches sparse top layer first, drills down for precision. Checks ~log(n) nodes instead of n | Millions of docs |
| Product Quantization (PQ) | Compresses vectors to save memory, often combined with HNSW | When RAM is a constraint |

Compare to **Elasticsearch's reverse index** (exact word lookup, O(1)): vector search is approximate
but semantic. Many modern systems combine both.

---

## Practical Notes

- **Trust ranking more than absolute scores** — a score of 0.4 for a one-word query against a full
  sentence is normal and correct
- **Richer queries produce higher scores** — query style should match document style
- **Year / metadata not embedded = not searchable** — if a field isn't in the text passed to `embed()`,
  queries mentioning it won't match. See `csv_loader.py → to_searchable_text()`.
- **HyDE** (Hypothetical Document Embeddings) — advanced technique: use an LLM to expand a short query
  into a hypothetical answer before embedding, dramatically improving scores

---

## Topics Not Yet Covered

- **Chunking** — splitting long documents before embedding
- **RAG** (Retrieval Augmented Generation) — using retrieved docs as context for an LLM
- **Persistent vector store** — saving embeddings to disk so Phase 1 doesn't re-run every time

---

## Codebase Cross-Reference

| Concept | File |
|---|---|
| Model loading + encoding | `learn_vector_db/embedder.py` |
| Cosine similarity + top-k | `learn_vector_db/similarity.py` |
| CSV parsing + searchable text | `learn_vector_db/csv_loader.py` |
| Full pipeline + CLI | `learn_vector_db/imdb_search.py` |
| Original minimal example | `learn_vector_db/main.py` |
