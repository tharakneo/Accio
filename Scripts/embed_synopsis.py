import os
import requests
import hashlib
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

load_dotenv(os.path.join(os.path.dirname(__file__), "../backend/.env"))

TMDB_API_KEY = os.environ["TMDB_API_KEY"]
TMDB_BASE = "https://api.themoviedb.org/3"
COLLECTION_NAME = "accio_synopsis"
VECTOR_SIZE = 768

client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer("BAAI/bge-base-en-v1.5")


def get_all_movies() -> list[tuple[str, int]]:
    result = client.scroll(
        collection_name="accio_subtitles",
        limit=10000,
        with_payload=["movie", "year"],
        with_vectors=False,
    )
    seen = set()
    movies = []
    for p in result[0]:
        key = (p.payload["movie"], p.payload["year"])
        if key not in seen:
            seen.add(key)
            movies.append(key)
    return movies


def fetch_synopsis(title: str, year: int) -> str | None:
    r = requests.get(f"{TMDB_BASE}/search/movie", params={
        "api_key": TMDB_API_KEY,
        "query": title,
        "year": year,
        "include_adult": False,
    })
    results = r.json().get("results", [])
    if not results:
        r = requests.get(f"{TMDB_BASE}/search/movie", params={
            "api_key": TMDB_API_KEY,
            "query": title,
            "include_adult": False,
        })
        results = r.json().get("results", [])
    if not results:
        return None
    overview = results[0].get("overview", "").strip()
    return overview if overview else None


def make_id(movie: str, year: int) -> int:
    key = f"synopsis|{movie}|{year}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**63)


def main():
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted old collection '{COLLECTION_NAME}'")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"Created collection '{COLLECTION_NAME}' (768-dim, bge-base)")

    movies = get_all_movies()
    print(f"Found {len(movies)} unique movies in accio_subtitles\n")

    points = []
    skipped = 0
    for movie, year in movies:
        synopsis = fetch_synopsis(movie, year)
        if not synopsis:
            print(f"  SKIP: {movie} ({year})")
            skipped += 1
            continue
        vector = model.encode(synopsis, normalize_embeddings=True).tolist()
        points.append(PointStruct(
            id=make_id(movie, year),
            vector=vector,
            payload={"movie": movie, "year": year, "synopsis": synopsis},
        ))
        print(f"  OK: {movie} ({year}) — {synopsis[:80]}...")

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"\nDone — {len(points)} synopses embedded, {skipped} skipped")


if __name__ == "__main__":
    main()
