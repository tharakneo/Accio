import json
import os
from dotenv import load_dotenv
from groq import Groq
from backend.models.schemas import ParsedQuery

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../backend/.env"))
client = Groq(api_key=os.environ["GROQ_API_KEY"])

SYSTEM_PROMPT = """Extract structured search signals from a movie query. Output JSON only.

EXTRACT (any subset can be non-empty):
- actors: real actor/actress names (Megan Fox, Keanu Reeves, Scarlett Johansson). Use full proper names.
- directors: real director names (Christopher Nolan, Quentin Tarantino). Use full proper names.
- characters: fictional character or franchise names (Batman, Joker, Frodo, Harry Potter, Optimus Prime). NEVER real people. NEVER concepts like "hero" or "female lead".
- genres: TMDB genres mentioned. Allowed values ONLY: Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Fantasy, Horror, Mystery, Romance, Science Fiction, Thriller, War, Western
- year_range: [start, end] if a decade/year is mentioned (e.g., "90s movies" → [1990, 1999])
- scene_text: the scene/event/location the user described. Keep it CLOSE to the user's original words — do NOT invent specific details, character names, or plot points you weren't told. Only strip actor names (replace with "woman"/"man") and expand obvious slang ("cap" → "Captain America"). If the user said "girlfriend dies", write "girlfriend dies" — do NOT invent "dies in arms" or "warehouse explosion". Hallucinated details hurt retrieval.
- theme_text: abstract theme/mood/premise when query has NO specific scene (e.g., "feel good movie", "movies about AI"). Leave null if scene_text is set.
- list_only: true if query asks for ALL movies by an actor/director with no scene, theme, or detail ("megan fox movies", "chris nolan films"). Otherwise false.

RULES:
- scene_text and theme_text are mutually exclusive. Scene beats theme when both seem present.
- A query with a specific location ("in NYC", "on Mars") or action ("repairing a car", "bomb blast") → scene_text.
- A query with only vibes/genre ("feel good", "underdog story") → theme_text.
- Fictional character name → characters (NOT actors). Real human name → actors.
- "cap vs ironman" / "batman vs joker" (two named characters fighting) → leave characters empty, put the whole phrase in scene_text.
- Always expand slang: "cap" → "Captain America", "chris nolan" → "Christopher Nolan".

EXAMPLES:

Query: "joker bomb blast movie"
{"actors":[],"directors":[],"characters":["Joker"],"genres":[],"year_range":null,"scene_text":"bomb explosion detonation chaos city","theme_text":null,"list_only":false}

Query: "megan fox movies where she repairs a car"
{"actors":["Megan Fox"],"directors":[],"characters":[],"genres":[],"year_range":null,"scene_text":"young woman leaning over car engine fixing it garage","theme_text":null,"list_only":false}

Query: "megan fox movies"
{"actors":["Megan Fox"],"directors":[],"characters":[],"genres":[],"year_range":null,"scene_text":null,"theme_text":null,"list_only":true}

Query: "rom coms in NYC"
{"actors":[],"directors":[],"characters":[],"genres":["Romance","Comedy"],"year_range":null,"scene_text":"New York City Manhattan streets apartment taxi love","theme_text":null,"list_only":false}

Query: "feel good underdog sports movie"
{"actors":[],"directors":[],"characters":[],"genres":[],"year_range":null,"scene_text":null,"theme_text":"uplifting underdog sports triumph perseverance heartwarming","list_only":false}

Query: "chris nolan movie about dreams"
{"actors":[],"directors":["Christopher Nolan"],"characters":[],"genres":[],"year_range":null,"scene_text":"dreams within dreams subconscious layered reality","theme_text":null,"list_only":false}

Query: "captain america fighting iron man"
{"actors":[],"directors":[],"characters":[],"genres":[],"year_range":null,"scene_text":"Captain America fighting Iron Man superhero battle airport clash","theme_text":null,"list_only":false}

Query: "movies about AI"
{"actors":[],"directors":[],"characters":[],"genres":[],"year_range":null,"scene_text":null,"theme_text":"artificial intelligence sentient machine robot dystopia future","list_only":false}

Query: "batman rachel death"
{"actors":[],"directors":[],"characters":["Batman"],"genres":[],"year_range":null,"scene_text":"Rachel dies death","theme_text":null,"list_only":false}

Query: "batman movies where his girlfriend dies"
{"actors":[],"directors":[],"characters":["Batman"],"genres":[],"year_range":null,"scene_text":"girlfriend dies death killed","theme_text":null,"list_only":false}
"""


def parse_query(query: str) -> ParsedQuery:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {query!r}"},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)
    return ParsedQuery(**data)
