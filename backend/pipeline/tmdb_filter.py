import os
import json
import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../backend/.env"))

TMDB_API_KEY = os.environ["TMDB_API_KEY"]
BASE_URL = "https://api.themoviedb.org/3"

CACHE_PATH = os.path.join(os.path.dirname(__file__), "../cache/genre_cache.json")

GENRE_MAP = {
    "Action": 28, "Adventure": 12, "Animation": 16, "Comedy": 35,
    "Crime": 80, "Documentary": 99, "Drama": 18, "Fantasy": 14,
    "Horror": 27, "Mystery": 9648, "Romance": 10749,
    "Science Fiction": 878, "Thriller": 53, "War": 10752, "Western": 37,
}

_genre_cache: dict[str, list[int]] | None = None


def _load_genre_cache() -> dict[str, list[int]]:
    """Load movie→genre_ids cache built once from all DB movies."""
    global _genre_cache
    if _genre_cache is None:
        if not os.path.exists(CACHE_PATH):
            print(f"[WARN] Genre cache missing at {CACHE_PATH}. Genre filtering disabled. Run: python -m Scripts.build_genre_cache")
            _genre_cache = {}
        else:
            with open(CACHE_PATH) as f:
                _genre_cache = json.load(f)
    return _genre_cache


def filter_by_genres(db_titles: list[str], genre_names: list[str]) -> list[str]:
    """Filter DB titles to those matching any of the requested genres (from local cache). Zero API calls."""
    if not genre_names:
        return db_titles
    wanted = {GENRE_MAP[g] for g in genre_names if g in GENRE_MAP}
    if not wanted:
        return db_titles
    cache = _load_genre_cache()
    if not cache:
        return db_titles  # cache missing — don't lose all results, skip filter
    return [t for t in db_titles if wanted.issubset(set(cache.get(t, [])))]


def _get_person_id(name: str, department: str | None = None) -> int | None:
    r = requests.get(f"{BASE_URL}/search/person", params={"api_key": TMDB_API_KEY, "query": name})
    results = r.json().get("results", [])
    if not results:
        return None
    if department:
        for p in results:
            if p.get("known_for_department", "").lower() == department.lower():
                return p["id"]
    return results[0]["id"]


def _discover_titles(
    actor_ids: list[int] = [],
    director_ids: list[int] = [],
    year_range: tuple[int, int] | None = None,
    max_pages: int = 5,
) -> list[str]:
    params = {"api_key": TMDB_API_KEY, "sort_by": "popularity.desc"}
    if actor_ids:
        params["with_cast"] = ",".join(str(a) for a in actor_ids)
    if director_ids:
        params["with_crew"] = ",".join(str(d) for d in director_ids)
    if year_range:
        params["primary_release_date.gte"] = f"{year_range[0]}-01-01"
        params["primary_release_date.lte"] = f"{year_range[1]}-12-31"

    titles = []
    for page in range(1, max_pages + 1):
        params["page"] = page
        data = requests.get(f"{BASE_URL}/discover/movie", params=params).json()
        results = data.get("results", [])
        if not results:
            break
        titles.extend(m["title"] for m in results)
        if page >= data.get("total_pages", 1):
            break
    return titles


def get_filmography(
    actors: list[str] = [],
    directors: list[str] = [],
    year_range: tuple[int, int] | None = None,
) -> list[str]:
    """Filmography for the given actors/directors via TMDB. Returns TMDB titles (not yet resolved to DB)."""
    actor_ids = [pid for a in actors if (pid := _get_person_id(a, "Acting")) is not None]
    director_ids = [pid for d in directors if (pid := _get_person_id(d, "Directing")) is not None]
    if not actor_ids and not director_ids:
        return []
    return _discover_titles(actor_ids, director_ids, year_range)
