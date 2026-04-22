import io
import contextlib
from sentence_transformers import SentenceTransformer

_model = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with contextlib.redirect_stdout(io.StringIO()):
            _model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    return _model

def embed(text: str) -> list[float]:
    return _get_model().encode(text, normalize_embeddings=True).tolist()