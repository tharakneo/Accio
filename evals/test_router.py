from backend.pipeline.intent_parser import parse_query

test_queries = [
    "NYC romcoms",
    "zombie movies",
    "90s romcoms",
    "joker bomb blast batman movie",
    "Megan Fox car repair",
    "based on true story",
    "movies with sad ending",
    "prison break movie",
    "3 kids joining a magical school",
    "natural disaster movies",
    "horror movies",
    "Christopher Nolan thriller",
    "movies where MC is so smart",
    "police comedy movie",
]

for query in test_queries:
    parsed = parse_query(query)
    print(f"Query: {query}")
    print(parsed.model_dump_json(indent=2))
