import re
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from backend.retrieval.embedder import embed

client = QdrantClient(host="localhost", port=6333)

_db_movies_cache: list[tuple[str, int]] | None = None


def get_all_db_movies() -> list[tuple[str, int]]:
    global _db_movies_cache
    if _db_movies_cache is None:
        hits = client.scroll(
            collection_name="accio_subtitles",
            limit=100000,
            with_payload=["movie", "year"],
            with_vectors=False,
        )[0]
        seen: dict[str, int] = {}
        for h in hits:
            m, y = h.payload["movie"], h.payload["year"]
            if m not in seen:
                seen[m] = y
        _db_movies_cache = list(seen.items())
    return _db_movies_cache


def _normalize(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s).lower().strip()


def resolve_titles_to_db(titles: list[str]) -> list[str]:
    """Map arbitrary (TMDB or free-form) titles to exact DB titles via normalized match."""
    db = get_all_db_movies()
    db_norm = {_normalize(t): t for t, _ in db}
    out = []
    for t in titles:
        hit = db_norm.get(_normalize(t))
        if hit:
            out.append(hit)
    return out


def search_subtitle_chunks(query_text: str, allowed_titles: list[str] | None = None, limit: int = 100) -> list[dict]:
    """Raw subtitle chunks (NOT aggregated per movie). Multiple chunks per movie allowed.
    Used when downstream reranking needs to pick the right scene within a movie."""
    vector = embed(query_text)
    query_filter = None
    if allowed_titles is not None:
        if not allowed_titles:
            return []
        query_filter = Filter(must=[FieldCondition(key="movie", match=MatchAny(any=allowed_titles))])

    hits = client.query_points(
        collection_name="accio_subtitles",
        query=vector,
        query_filter=query_filter,
        limit=limit,
    ).points
    return [
        {"movie": h.payload["movie"], "year": h.payload["year"],
         "context": h.payload["text"], "similarity": h.score}
        for h in hits
    ]


def get_all_chunks_for_movies(movies: list[str]) -> list[dict]:
    """Pull EVERY chunk for the given movies — no cosine cap.
    Use when the whitelist is narrow and you want the reranker to see everything."""
    if not movies:
        return []
    hits = client.scroll(
        collection_name="accio_subtitles",
        scroll_filter=Filter(must=[FieldCondition(key="movie", match=MatchAny(any=movies))]),
        limit=100000,
        with_payload=True,
        with_vectors=False,
    )[0]
    return [
        {"movie": h.payload["movie"], "year": h.payload["year"],
         "context": h.payload["text"], "similarity": 0.0}
        for h in hits
    ]


def aggregate_chunks(chunks: list[dict]) -> list[dict]:
    """Collapse per-chunk results into one row per movie — best chunk + signal boost from strong chunks."""
    grouped: dict[tuple, list] = {}
    for c in chunks:
        grouped.setdefault((c["movie"], c["year"]), []).append(c)
    out = []
    for (movie, year), group in grouped.items():
        best = max(group, key=lambda x: x["similarity"])
        strong = [c for c in group if c["similarity"] >= best["similarity"] - 0.02]
        score = best["similarity"] + (len(strong) - 1) * 0.005
        out.append({"movie": movie, "year": year, "context": best["context"], "similarity": score})
    return sorted(out, key=lambda x: x["similarity"], reverse=True)


def search_subtitles(query_text: str, allowed_titles: list[str] | None = None, limit: int = 100) -> list[dict]:
    """Semantic search over subtitles, aggregated per movie."""
    return aggregate_chunks(search_subtitle_chunks(query_text, allowed_titles, limit))


def search_synopsis(query_text: str, allowed_titles: list[str] | None = None, limit: int = 20) -> list[dict]:
    """Semantic search over movie synopses."""
    vector = embed(query_text)
    query_filter = None
    if allowed_titles is not None:
        if not allowed_titles:
            return []
        query_filter = Filter(must=[FieldCondition(key="movie", match=MatchAny(any=allowed_titles))])

    hits = client.query_points(
        collection_name="accio_synopsis",
        query=vector,
        query_filter=query_filter,
        limit=limit,
    ).points

    return [
        {"movie": h.payload["movie"], "year": h.payload["year"], "context": h.payload["synopsis"], "similarity": h.score}
        for h in hits
    ]


def find_movies_by_characters(characters: list[str], threshold: float = 0.45) -> list[str]:
    """Find movies whose subtitles mention the given characters. Returns DB titles."""
    scores: dict[str, float] = {}
    for c in characters:
        vector = embed(c)
        hits = client.query_points(
            collection_name="accio_subtitles", query=vector, limit=30,
        ).points
        for h in hits:
            if h.score >= threshold:
                m = h.payload["movie"]
                scores[m] = max(scores.get(m, 0), h.score)
    return [m for m, _ in sorted(scores.items(), key=lambda x: -x[1])]
