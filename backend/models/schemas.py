from typing import Optional
from pydantic import BaseModel


class ParsedQuery(BaseModel):
    # Filter signals — narrow the movie universe
    actors: list[str] = []
    directors: list[str] = []
    characters: list[str] = []  # fictional characters (Batman, Joker, Frodo)
    genres: list[str] = []
    year_range: Optional[tuple[int, int]] = None

    # Search signals — rank within the filtered universe
    scene_text: Optional[str] = None   # vivid description of a specific scene/action/location
    theme_text: Optional[str] = None   # abstract theme/mood (used when no scene specified)
    list_only: bool = False            # query is just "all movies by X" — no ranking needed
