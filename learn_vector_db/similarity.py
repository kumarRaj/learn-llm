"""
similarity.py — Single Responsibility: compute how similar two vectors are.

Uses cosine similarity: a score between -1 and 1.
  1.0  → identical meaning
  0.0  → unrelated
 -1.0  → opposite meaning

Nothing else lives here. No model, no CSV, no I/O.
"""

import numpy as np


def cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between one query vector and every row in a matrix.

    Args:
        matrix: shape (N, D) — e.g. embeddings for 1000 movies
        vector: shape (D,)   — e.g. embedding for the search query

    Returns:
        scores: shape (N,)   — one similarity score per row, range [-1, 1]

    How it works:
        cosine_similarity = dot(a, b) / (||a|| * ||b||)
        Dividing by the magnitudes makes the score independent of vector length,
        so it measures *direction* (meaning) not *magnitude* (intensity).
    """
    # Magnitude (length) of each document vector
    doc_norms = np.linalg.norm(matrix, axis=1)       # shape (N,)

    # Magnitude of the query vector
    query_norm = np.linalg.norm(vector)               # scalar

    # Dot product of every document with the query
    dot_products = np.dot(matrix, vector)             # shape (N,)

    return dot_products / (doc_norms * query_norm)


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return the indices of the k highest scores, best first."""
    return np.argsort(scores)[::-1][:k]
