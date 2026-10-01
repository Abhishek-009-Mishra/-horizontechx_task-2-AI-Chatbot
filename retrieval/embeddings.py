"""Ollama embeddings: health check + embeddings factory."""

from __future__ import annotations

import ollama
from langchain_ollama import OllamaEmbeddings

from config.errors import PadhoError, friendly_error
from config.settings import EMBEDDING_MODEL, OLLAMA_BASE_URL


def _installed_model_names(client: ollama.Client) -> list[str]:
    """Return installed model names (works across ollama-python versions)."""
    response = client.list()
    models = response["models"] if isinstance(response, dict) else response.models
    names = []
    for m in models:
        name = m.get("model") or m.get("name") if isinstance(m, dict) else (
            getattr(m, "model", None) or getattr(m, "name", None)
        )
        if name:
            names.append(name)
    return names


def ensure_ollama_ready() -> None:
    """Raise a friendly PadhoError if Ollama is down or the embedding model is missing."""
    try:
        names = _installed_model_names(ollama.Client(host=OLLAMA_BASE_URL))
    except Exception as exc:  # connection refused, timeout, ...
        raise PadhoError(
            "Ollama is not running (or not reachable at "
            f"{OLLAMA_BASE_URL}).\nStart the Ollama app or run: ollama serve"
        ) from exc

    base = EMBEDDING_MODEL.split(":")[0]
    if not any(n == EMBEDDING_MODEL or n.split(":")[0] == base for n in names):
        raise PadhoError(
            f"The embedding model '{EMBEDDING_MODEL}' is not installed in Ollama.\n"
            f"Run: ollama pull {base}"
        )


def get_embeddings() -> OllamaEmbeddings:
    """Return the embeddings object after verifying Ollama is ready."""
    ensure_ollama_ready()
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)


def embed_texts(embeddings: OllamaEmbeddings, texts: list[str], batch_size: int, on_progress=None) -> list[list[float]]:
    """Embed texts in batches so progress can be reported and failures are friendly."""
    vectors: list[list[float]] = []
    try:
        for start in range(0, len(texts), batch_size):
            vectors.extend(embeddings.embed_documents(texts[start:start + batch_size]))
            if on_progress:
                on_progress(min(start + batch_size, len(texts)), len(texts))
    except Exception as exc:
        raise PadhoError(f"Embedding failed: {friendly_error(exc)}") from exc
    return vectors
