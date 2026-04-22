"""Re-embed a single movie: delete old vectors, chunk SRT, insert new vectors."""
import sys
import hashlib
import re
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"
COLLECTION = "accio_subtitles"
CHUNK_SIZE = 4
OVERLAP = 1


def make_id(movie: str, year: int, chunk_index: int) -> int:
    key = f"{movie}|{year}|{chunk_index}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**63)


def clean_srt(content: str) -> list[str]:
    content = content.lstrip("\ufeff")
    content = re.sub(r"\d+:\d+:\d+,\d+\s*-->\s*\d+:\d+:\d+,\d+", "", content)
    content = re.sub(r"^\d+\s*$", "", content, flags=re.MULTILINE)
    content = re.sub(r"<[^>]+>", "", content)
    return [l.strip() for l in content.splitlines() if l.strip()]


def chunk_lines(lines: list[str]) -> list[str]:
    chunks, step = [], CHUNK_SIZE - OVERLAP
    for i in range(0, len(lines), step):
        seg = lines[i:i + CHUNK_SIZE]
        if seg:
            chunks.append(" ".join(seg))
    return chunks


def main(srt_filename: str):
    srt_path = DATA_DIR / srt_filename
    if not srt_path.exists():
        print(f"File not found: {srt_path}")
        sys.exit(1)

    stem = srt_path.stem
    match = re.match(r"^(.+?)_(\d{4})$", stem)
    if not match:
        print("Filename must be like Movie_Name_2008.srt")
        sys.exit(1)

    movie = match.group(1).replace("_", " ")
    year = int(match.group(2))
    print(f"Movie: {movie} ({year})")

    client = QdrantClient(host="localhost", port=6333)

    # Delete old vectors for this movie
    deleted = client.delete(
        collection_name=COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="movie", match=MatchValue(value=movie))]
        ),
    )
    print(f"Deleted old vectors for '{movie}'")

    # Chunk the new SRT
    content = srt_path.read_text(encoding="utf-8", errors="replace")
    chunks = chunk_lines(clean_srt(content))
    print(f"Chunked into {len(chunks)} chunks")

    # Embed and insert
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    texts = [c for c in chunks]
    vectors = model.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True)

    points = [
        PointStruct(
            id=make_id(movie, year, i),
            vector=v.tolist(),
            payload={"movie": movie, "year": year, "chunk_index": i, "text": t},
        )
        for i, (t, v) in enumerate(zip(texts, vectors))
    ]

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Done — {len(points)} chunks inserted for '{movie} ({year})'")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Scripts/reembed_movie.py Iron_Man_2008.srt")
        sys.exit(1)
    main(sys.argv[1])
