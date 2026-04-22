"""Build the movie→genre_ids cache from every movie in Qdrant.
Run once after indexing new movies:
    python -m Scripts.build_genre_cache

Idempotent: skips movies already in cache. Safe to re-run as you add more films.
"""
import os
import sys
import json
import time
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

load_dotenv(os.path.join(ROOT, "backend/.env"))

TMDB_API_KEY = os.environ["TMDB_API_KEY"]
CACHE_PATH = os.path.join(ROOT, "backend/cache/genre_cache.json")
BASE_URL = "https://api.themoviedb.org/3"


def fetch_genres(title: str, year: int) -> list[int]:
    for params in [
        {"api_key": TMDB_API_KEY, "query": title, "year": year},
        {"api_key": TMDB_API_KEY, "query": title},  # fallback if year mismatch
    ]:
        r = requests.get(f"{BASE_URL}/search/movie", params=params)
        if r.status_code != 200:
            continue
        results = r.json().get("results", [])
        if results:
            return results[0].get("genre_ids", [])
    return []


def main():
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    cache: dict[str, list[int]] = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        print(f"Loaded existing cache: {len(cache)} movies")

    client = QdrantClient(host="localhost", port=6333)
    hits = client.scroll(
        collection_name="accio_subtitles",
        limit=100000,
        with_payload=["movie", "year"],
        with_vectors=False,
    )[0]

    movies = {}
    for h in hits:
        m, y = h.payload["movie"], h.payload["year"]
        if m not in movies:
            movies[m] = y

    todo = [(m, y) for m, y in movies.items() if m not in cache]
    print(f"DB has {len(movies)} movies. {len(todo)} need lookup.")

    for i, (title, year) in enumerate(todo, 1):
        genres = fetch_genres(title, year)
        cache[title] = genres
        print(f"  [{i}/{len(todo)}] {title} ({year}) → {genres}")
        if i % 20 == 0:
            with open(CACHE_PATH, "w") as f:
                json.dump(cache, f, indent=2)
        time.sleep(0.05)  # TMDB rate limit courtesy

    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"Saved {len(cache)} entries to {CACHE_PATH}")


if __name__ == "__main__":
    main()
