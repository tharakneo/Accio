import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../backend/.env"))
client = Groq(api_key=os.environ["GROQ_API_KEY"])

SYSTEM_PROMPT = """You are a movie assistant. A user searched for a movie and got a result.
You are given the movie title, the user's search query, and a subtitle/synopsis excerpt as evidence.

Write ONE short confident sentence describing what happens in this movie related to the search.
Use your knowledge of the movie — the evidence is just a hint to ground you.
Be specific: name characters, describe the actual scene or plot point.
Never say "does not match" or "no mention of" — if it's in the results, find the connection.
Output only the one sentence. Nothing else."""


def generate_explanations(query: str, results: list[dict]) -> dict[str, str]:
    """Returns a dict of movie -> one-line explanation."""
    if not results:
        return {}

    explanations = {}
    for r in results[:5]:
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Search: \"{query}\"\nMovie: {r['movie']} ({r['year']})\nEvidence: \"{r['context']}\""},
                ],
                temperature=0,
                max_tokens=80,
            )
            explanations[r['movie']] = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Generator error for {r['movie']}: {e}")

    return explanations
