"""
RAGAS evaluation for Accio pipeline.
Run: python evals/run_ragas.py

Metrics:
- context_precision: are the retrieved chunks relevant to the query?
- context_recall: do the chunks cover what the query is asking about?
"""
import os
import sys
import json
import requests
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper

load_dotenv(os.path.join(Path(__file__).parent.parent, "backend/.env"))

sys.path.insert(0, str(Path(__file__).parent.parent))

from ragas import evaluate
from ragas.metrics import context_precision, context_recall

groq_llm = LangchainLLMWrapper(ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
))

for metric in [context_precision, context_recall]:
    metric.llm = groq_llm

QUERIES_FILE = Path(__file__).parent / "test_queries.json"
API_URL = "http://localhost:8000/search"


def run_query(query: str) -> dict:
    r = requests.post(API_URL, json={"query": query}, timeout=60)
    return r.json()


def build_dataset(queries: list[dict]) -> Dataset:
    rows = []
    for item in queries:
        query = item["query"]
        expected = item["expected"]

        print(f"  querying: {query}")
        response = run_query(query)
        results = response.get("results", [])

        # Retrieved contexts = the subtitle/synopsis chunks Accio found
        contexts = [r.get("context", "") for r in results if r.get("context")]

        rows.append({
            "user_input": query,
            "retrieved_contexts": contexts,
            "reference": f"The correct movie is {expected}. Return chunks relevant to the query from that movie.",
        })

    return Dataset.from_list(rows)


def main():
    queries = json.loads(QUERIES_FILE.read_text())
    print(f"Running RAGAS eval on {len(queries)} queries...\n")

    dataset = build_dataset(queries)

    print("\nScoring with RAGAS...\n")
    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall],
    )

    print("\n" + "=" * 50)
    print(f"Context Precision: {result['context_precision']:.3f}")
    print(f"Context Recall:    {result['context_recall']:.3f}")

    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(result.to_pandas().to_dict(orient="records"), indent=2))
    print(f"\nDetailed results saved to {out}")


if __name__ == "__main__":
    main()
