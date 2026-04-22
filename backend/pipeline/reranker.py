import re
import io
import contextlib
from sentence_transformers import CrossEncoder


_model = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        with contextlib.redirect_stdout(io.StringIO()):
            _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _model


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", "", text)).strip()


def rerank(query: str, candidates: list[dict], max_pairs: int = 800) -> list[dict]:
    if not candidates:
        return []

    # Too many chunks = slow cross-encoder. Cap per-movie by cosine so every movie stays represented.
    if len(candidates) > max_pairs:
        from collections import defaultdict
        per_movie: dict[tuple, list] = defaultdict(list)
        for c in candidates:
            per_movie[(c["movie"], c["year"])].append(c)
        per_movie_cap = max(max_pairs // len(per_movie), 20)
        capped = []
        for chunks in per_movie.values():
            chunks.sort(key=lambda x: x.get("similarity", 0), reverse=True)
            capped.extend(chunks[:per_movie_cap])
        candidates = capped

    model = _get_model()
    pairs = [
        (query, f"{c['movie']}. {_clean(c['context'])}")
        for c in candidates
    ]
    scores = model.predict(pairs)

    raw = [float(s) for s in scores]

    if len(raw) == 1:
        candidates[0]["score"] = 10.0
    else:
        lo, hi = min(raw), max(raw)
        span = hi - lo
        for c, s in zip(candidates, raw):
            c["score"] = round(((s - lo) / span) * 9 + 1, 1)

    # Deduplicate by (movie, year) keeping the highest scoring chunk
    seen: dict[tuple, dict] = {}
    for c in sorted(candidates, key=lambda x: x["score"], reverse=True):
        key = (c["movie"], c["year"])
        if key not in seen:
            seen[key] = c

    return list(seen.values())[:10]
