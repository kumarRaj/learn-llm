# Learn Vector DB

A hands-on project for learning how semantic/vector search works from scratch — no vector database required, just numpy and a pre-trained embedding model.

## What this does

Given a dataset of IMDB's top 1000 movies, you can search using natural language:

```
Search: dark psychological thriller with a twist ending
Search: feel good movie about friendship
Search: space adventure with great visuals
```

Results are ranked by **semantic similarity**, not keyword matching — so "fluffy pets" finds movies about cats and dogs even if those words never appear in the overview.

## How it works

```
CSV (movie overviews)
      ↓
embedder.py       — converts text → 384-dimensional vectors (once at startup)
      ↓
similarity.py     — cosine similarity between query vector and all movie vectors
      ↓
imdb_search.py    — orchestrates everything, interactive CLI
```

Each file has a single responsibility. Read them in order to understand the full pipeline.

## Setup

**Requirements:** Python 3.13+, [Poetry](https://python-poetry.org/)

```bash
poetry install
```

Download `imdb_top_1000.csv` from [Kaggle](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows) and place it in the repo root.

## Usage

```bash
# Interactive search (recommended)
poetry run python learn_vector_db/imdb_search.py

# Show more results
poetry run python learn_vector_db/imdb_search.py --top 10

# Minimal example (no CSV needed)
poetry run python learn_vector_db/main.py "loyal pets"
```

The embedding model (`all-MiniLM-L6-v2`, ~90MB) is downloaded from Hugging Face automatically on first run and cached locally. Subsequent runs are instant.

Type `quit` or press `Ctrl+C` to exit the interactive loop.

## Key concepts covered

- **Embeddings** — text → vectors that capture meaning
- **Cosine similarity** — measuring angle between vectors, not distance
- **Semantic search** — finding meaning, not exact words
- **Why this is deterministic** — frozen model weights, pure math, no sampling
- **Why LLMs are not** — temperature-controlled random sampling during generation
- **Hallucination** — what happens when a model has no signal to draw from
- **Guardrails / RLHF** — how models are fine-tuned to behave predictably

See [`learning.md`](./learning.md) for detailed notes on all of the above.

## File overview

| File | Responsibility |
|---|---|
| `learn_vector_db/main.py` | Minimal self-contained example — start here |
| `learn_vector_db/embedder.py` | Load model, encode text → vectors |
| `learn_vector_db/similarity.py` | Cosine similarity and top-k ranking |
| `learn_vector_db/csv_loader.py` | Parse CSV into `Movie` dataclass |
| `learn_vector_db/imdb_search.py` | Orchestrate pipeline + interactive CLI |
| `learning.md` | Detailed notes on embeddings, similarity, LLMs, scaling |
