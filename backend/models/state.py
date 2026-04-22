from typing import Any, Optional
from typing_extensions import TypedDict
from backend.models.schemas import ParsedQuery


class AccioState(TypedDict):
    query: str
    parsed: Optional[ParsedQuery]
    allowed_titles: Optional[list[str]]
    results: list[dict[str, Any]]        # aggregated (one row per movie)
    raw_chunks: list[dict[str, Any]]     # multiple chunks per movie — feeds reranker
    final_results: list[dict[str, Any]]
