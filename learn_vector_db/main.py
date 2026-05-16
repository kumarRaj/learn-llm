from sentence_transformers import SentenceTransformer
import numpy as np

# Load a free embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # small, fast, free

# Your "dataset" — this is what you want to search
documents = [
    "Cats are fluffy and independent pets",
    "Dogs are loyal and love to play fetch",
    "Apples and mangoes are tropical fruits",
    "Python is a great programming language",
    "Machine learning uses lots of math",
]

# Embed every document (do this once, save results)
doc_embeddings = model.encode(documents)
# doc_embeddings is now shape (5, 384) — 5 docs, 384 numbers each


def search(query, top_k=3):
    # Embed the query using the SAME model
    query_embedding = model.encode([query])[0]

    # Compute cosine similarity to every document
    # (dot product of normalized vectors = cosine similarity)
    norms = np.linalg.norm(doc_embeddings, axis=1)
    similarities = np.dot(doc_embeddings, query_embedding) / (norms * np.linalg.norm(query_embedding))

    # Sort by most similar
    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [(documents[i], round(float(similarities[i]), 3)) for i in top_indices]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python main.py \"<query>\"")
        sys.exit(1)

    query = sys.argv[1]
    results = search(query)
    for doc, score in results:
        print(f"Score {score}: {doc}")
