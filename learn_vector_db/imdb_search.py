"""
imdb_search.py — Single Responsibility: wire everything together and provide an interactive CLI.

This file does NOT contain embedding logic, similarity math, or CSV parsing.
It only orchestrates the other modules.

Usage:
    poetry run python learn_vector_db/imdb_search.py
    poetry run python learn_vector_db/imdb_search.py --top 10

Movies and embeddings are loaded ONCE at startup, then the loop accepts queries instantly.
Type 'quit' or press Ctrl+C to exit.
"""

import argparse
import numpy as np

from embedder import load_model, embed
from csv_loader import load_movies
from similarity import cosine_similarity, top_k_indices


def search(query: str, model, movie_embeddings: np.ndarray, movies: list, top_k: int = 5) -> list[tuple]:
    """
    Search without reloading — model and embeddings are passed in from the startup phase.

    Steps:
      1. Embed the query (fast, single text)
      2. Compute cosine similarity against all pre-embedded movies
      3. Return top-k results
    """
    query_embedding: np.ndarray = embed(model, [query])[0]  # shape (384,)
    scores = cosine_similarity(movie_embeddings, query_embedding)  # shape (N,)
    best_indices = top_k_indices(scores, k=top_k)
    return [(movies[i], round(float(scores[i]), 3)) for i in best_indices]


def print_results(results: list[tuple]) -> None:
    """Pretty-print search results."""
    for rank, (movie, score) in enumerate(results, start=1):
        print(f"{rank}. [{score}] {movie.title} ({movie.year})")
        print(f"   Genre:    {movie.genre}")
        print(f"   Director: {movie.director}  |  IMDB: {movie.rating}")
        print(f"   Overview: {movie.overview}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic search over IMDB Top 1000")
    parser.add_argument("--top", type=int, default=5, help="Number of results to show (default: 5)")
    args = parser.parse_args()

    # --- Startup: load everything once ---
    print("Loading movies...")
    movies = load_movies()

    print(f"Loading model and embedding {len(movies)} movies (one-time setup)...")
    model = load_model()
    texts = [m.to_searchable_text() for m in movies]
    movie_embeddings: np.ndarray = embed(model, texts)  # shape (N, 384)

    print("\nReady. Type a search query and press Enter. Type 'quit' to exit.\n")

    # --- Interactive loop: only the query is re-embedded each time ---
    while True:
        try:
            query = input("Search: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        results = search(query, model, movie_embeddings, movies, top_k=args.top)
        print()
        print_results(results)
