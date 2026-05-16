"""
csv_loader.py — Single Responsibility: read imdb_top_1000.csv and return structured data.

Nothing else lives here. No embeddings, no similarity, no CLI.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

# Path to the CSV relative to this file:  learn_vector_db/../imdb_top_1000.csv
CSV_PATH = Path(__file__).parent.parent / "imdb_top_1000.csv"


@dataclass
class Movie:
    """One row from the CSV, keeping only the fields we care about."""
    title: str
    year: str
    genre: str
    overview: str
    director: str
    rating: str

    def to_searchable_text(self) -> str:
        """
        Combine fields into a single string that will be embedded.

        We include title, genre, and overview because:
          - title  → lets queries like "Dark Knight" match directly
          - genre  → lets queries like "psychological thriller" match
          - overview → the richest semantic signal (plot description)
        """
        return f"Title: {self.title}. Year: {self.year}. Genre: {self.genre}. {self.overview}"


def load_movies(csv_path: Path = CSV_PATH) -> list[Movie]:
    """
    Read the CSV and return a list of Movie objects.

    Skips rows where the overview is empty.
    """
    movies = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            overview = row["Overview"].strip()
            if not overview:
                continue
            movies.append(Movie(
                title=row["Series_Title"].strip(),
                year=row["Released_Year"].strip(),
                genre=row["Genre"].strip(),
                overview=overview,
                director=row["Director"].strip(),
                rating=row["IMDB_Rating"].strip(),
            ))
    return movies
