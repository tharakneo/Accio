"""Accio search pipeline — signal-driven, 4-node LangGraph.

Flow:
    parse → filter → search → (rerank if ambiguous) → finalize → END

No mode-based routing. The LLM extracts structured signals (actors, characters,
genres, scene_text, theme_text) and the presence of those signals deterministically
drives what the pipeline does. This removes LLM mode-hallucination as a failure mode.
"""
from langgraph.graph import StateGraph, END

from backend.models.state import AccioState
from backend.models.schemas import ParsedQuery
from backend.pipeline.intent_parser import parse_query
from backend.pipeline.tmdb_filter import get_filmography, filter_by_genres
from backend.pipeline.retriever import (
    search_subtitles, search_subtitle_chunks, aggregate_chunks,
    search_synopsis, find_movies_by_characters,
    resolve_titles_to_db, get_all_db_movies,
)
from backend.pipeline.reranker import rerank


SIM_GAP = 0.05           # gap below top score to include in final results
SIM_FLOOR = 0.55         # absolute minimum similarity
RERANK_AMBIGUITY = 0.03  # if top-2 within this, rerank


# ─────────────────────────── nodes ───────────────────────────

def node_parse(state: AccioState) -> dict:
    parsed = parse_query(state["query"])
    print(f"\n{'='*60}")
    print(f"QUERY: {state['query']}")
    print(f"SIGNALS: actors={parsed.actors} directors={parsed.directors} "
          f"characters={parsed.characters} genres={parsed.genres} "
          f"year={parsed.year_range} list_only={parsed.list_only}")
    print(f"  scene_text={parsed.scene_text!r}")
    print(f"  theme_text={parsed.theme_text!r}")
    return {"parsed": parsed}


def node_filter(state: AccioState) -> dict:
    """Build the whitelist of allowed DB titles by intersecting every active filter signal.

    Returns:
        allowed_titles = None  →  no filter (open search)
        allowed_titles = [...] →  restrict search to these
        allowed_titles = []    →  intentionally empty (every filter eliminated everything)
    """
    p: ParsedQuery = state["parsed"]
    db_titles = [t for t, _ in get_all_db_movies()]
    whitelist: set[str] | None = None

    def intersect(new: list[str], label: str):
        nonlocal whitelist
        new_set = set(new)
        whitelist = new_set if whitelist is None else (whitelist & new_set)
        print(f"  after {label}: {len(whitelist)}")

    # actors/directors → TMDB filmography
    if p.actors or p.directors:
        tmdb_titles = get_filmography(actors=p.actors, directors=p.directors, year_range=p.year_range)
        resolved = resolve_titles_to_db(tmdb_titles)
        print(f"FILMOGRAPHY: {len(tmdb_titles)} TMDB → {len(resolved)} in DB")
        intersect(resolved, "people")

    # characters → movies that mention them in dialogue
    if p.characters:
        found = find_movies_by_characters(p.characters)
        print(f"CHARACTERS {p.characters}: {len(found)} DB movies → {found[:5]}")
        intersect(found, "characters")

    # genres → local cache (zero API calls)
    if p.genres:
        pool = list(whitelist) if whitelist is not None else db_titles
        filtered = filter_by_genres(pool, p.genres)
        print(f"GENRES {p.genres}: {len(pool)} → {len(filtered)}")
        whitelist = set(filtered)

    if whitelist is None:
        return {"allowed_titles": None}
    return {"allowed_titles": list(whitelist)}


def node_search(state: AccioState) -> dict:
    """Rank within the whitelist. Uses scene_text if present, else theme_text, else raw query."""
    p: ParsedQuery = state["parsed"]
    allowed = state["allowed_titles"]

    if p.list_only and allowed is not None:
        results = [
            {"movie": t, "year": y, "context": "", "similarity": 1.0}
            for t, y in get_all_db_movies() if t in set(allowed)
        ]
        print(f"LIST ONLY: {len(results)} movies")
        return {"results": results, "raw_chunks": []}

    search_text = p.scene_text or p.theme_text or state["query"]

    # Small whitelist (≤15 movies): 30 chunks/movie for fair representation
    # Larger/open: cap at 200 — cosine already surfaces the most relevant
    if allowed is not None and len(allowed) <= 15:
        limit = len(allowed) * 30
    else:
        limit = 200
    raw = search_subtitle_chunks(search_text, allowed_titles=allowed, limit=limit)
    print(f"SUBTITLES ({search_text!r}): {len(raw)} chunks (limit={limit})")

    results = aggregate_chunks(raw)
    print(f"  aggregated → {[(r['movie'], round(r['similarity'], 3)) for r in results[:5]]}")
    return {"results": results, "raw_chunks": raw}


def node_rerank(state: AccioState) -> dict:
    """Cross-encoder rerank — feeds it raw chunks (multiple per movie) so it can pick the right scene,
    then dedupes by movie keeping the highest-scoring chunk."""
    raw = state.get("raw_chunks") or state["results"]
    reranked = rerank(state["query"], raw)
    print(f"RERANKED: {[(r['movie'], round(r.get('score', 0), 2)) for r in reranked[:5]]}")
    return {"results": reranked}


def node_finalize(state: AccioState) -> dict:
    """One gap-filter rule for every query type."""
    p: ParsedQuery = state["parsed"]
    results = state["results"]
    if not results:
        print("FINAL: []")
        return {"final_results": []}

    # list_only results are unranked — return all of them
    if p.list_only:
        final = sorted(results, key=lambda x: x["movie"])
        print(f"FINAL ({len(final)}): {[r['movie'] for r in final[:10]]}")
        return {"final_results": final}

    # Use cross-encoder score if present, else cosine similarity
    key = "score" if results[0].get("score") is not None else "similarity"
    sorted_results = sorted(results, key=lambda x: x.get(key, 0), reverse=True)
    top = sorted_results[0][key]

    if key == "score":
        final = [r for r in sorted_results if top - r[key] <= 1.5][:5]
    else:
        # Narrow whitelist = tightly-clustered scores → tight gap to pick the winner.
        # Wide/open search = wider spread → loose gap so related results survive.
        allowed = state.get("allowed_titles")
        gap = 0.008 if (allowed is not None and len(allowed) <= 30) else SIM_GAP
        min_score = max(top - gap, SIM_FLOOR)
        final = [r for r in sorted_results if r[key] >= min_score][:5]

    print(f"FINAL: {[(r['movie'], round(r.get(key, 0), 3)) for r in final]}")
    return {"final_results": final}


# ─────────────────────────── conditional edge ───────────────────────────

def should_rerank(state: AccioState) -> str:
    """Rerank only when cosine genuinely can't distinguish top-2 (gap ≤ 0.015).
    The cross-encoder is noisy on narrative dialogue — trust cosine when it has a clear winner."""
    p: ParsedQuery = state["parsed"]
    results = state["results"]
    if p.list_only or len(results) < 2:
        return "finalize"
    top, second = results[0]["similarity"], results[1]["similarity"]
    return "rerank" if (top - second) <= 0.005 else "finalize"


# ─────────────────────────── graph ───────────────────────────

builder = StateGraph(AccioState)
builder.add_node("parse", node_parse)
builder.add_node("filter", node_filter)
builder.add_node("search", node_search)
builder.add_node("rerank", node_rerank)
builder.add_node("finalize", node_finalize)

builder.set_entry_point("parse")
builder.add_edge("parse", "filter")
builder.add_edge("filter", "search")
builder.add_conditional_edges("search", should_rerank, {"rerank": "rerank", "finalize": "finalize"})
builder.add_edge("rerank", "finalize")
builder.add_edge("finalize", END)

app = builder.compile()
