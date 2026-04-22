import os
import logging
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

# Must be set before any sentence-transformers / huggingface imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.pipeline.graph import app as pipeline
from backend.pipeline.generator import generate_explanations

app = FastAPI(title="Accio Movie Search")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str


@app.post("/search")
def search(request: SearchRequest):
    result = pipeline.invoke({
        "query": request.query,
        "parsed": None,
        "allowed_titles": None,
        "results": [],
        "raw_chunks": [],
        "final_results": [],
    })
    results = result["final_results"]
    explanations = generate_explanations(request.query, results)
    for r in results:
        r["explanation"] = explanations.get(r["movie"], "")
    return {"query": request.query, "results": results}


@app.get("/health")
def health():
    return {"status": "ok"}

