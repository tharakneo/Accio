# Embed dialogue chunks and store in Qdrant

import json
from pathlib import Path

from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path(__file__).parent / "data" / "chunks.json"
COLLECTION_NAME = "accio_subtitles"
VECTOR_SIZE = 768
BATCH_SIZE = 256


def make_id(movie: str, year: int, chunk_index: int) -> int:
    """Stable numeric ID from movie+year+chunk so re-runs are safe."""
    import hashlib

    key = f"{movie}|{year}|{chunk_index}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**63)


def main():
    chunks = json.loads(CHUNKS_FILE.read_text())
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    client = QdrantClient(host="localhost", port=6333)

    existing_collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing_collections:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted old collection '{COLLECTION_NAME}'")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"Created collection '{COLLECTION_NAME}' (768-dim, bge-base)")

    new_chunks = chunks
    print(f"Embedding {len(new_chunks)} chunks...")

    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    texts = [c["text"] for c in new_chunks]

    points = []
    for chunk, vector in zip(
        new_chunks,
        tqdm(
            model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True),
            total=len(texts),
            desc="Embedding",
        ),
    ):
        points.append(
            PointStruct(
                id=make_id(chunk["movie"], chunk["year"], chunk["chunk_index"]),
                vector=vector.tolist(),
                payload={
                    "movie": chunk["movie"],
                    "year": chunk["year"],
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                },
            )
        )

    for start in tqdm(range(0, len(points), BATCH_SIZE), desc="Upserting"):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[start : start + BATCH_SIZE],
        )

    print(f"\nDone — {len(points)} new chunks embedded into '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()