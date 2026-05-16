"""
embedder.py — Single Responsibility: load the model and turn text into vectors.

Nothing else lives here. No CSV logic, no similarity math.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# The model is downloaded once (~90 MB) and cached by sentence-transformers.
MODEL_NAME = "all-MiniLM-L6-v2"


def load_model() -> SentenceTransformer:
    """Load (or reuse from cache) the embedding model."""
    return SentenceTransformer(MODEL_NAME)


def embed(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """
    Convert a list of strings into a 2-D array of vectors.

    Input:  ["hello world", "foo bar"]
    Output: numpy array of shape (2, 384)
              — 2 texts, each represented by 384 numbers
    """
    return model.encode(texts)
