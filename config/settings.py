"""Central configuration: paths, book definitions, tunables and .env handling."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from config.errors import PadhoError

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class BookConfig:
    code: str       # short label shown in metadata and sources
    source: str     # full book title
    pdf_path: Path  # where the user places the PDF


BOOKS: list[BookConfig] = [
    BookConfig(
        code="CLF-C02",
        source="AWS Certified Cloud Practitioner Foundational (CLF-C02) Exam Study Guide",
        pdf_path=BASE_DIR / "data" / "clf_c02" / "clf_c02.pdf",
    ),
    BookConfig(
        code="SAA-C02",
        source="AWS Certified Solutions Architect Study Guide, 3rd Edition (SAA-C02)",
        pdf_path=BASE_DIR / "data" / "saa_c02" / "saa_c02.pdf",
    ),
]

VECTOR_DB_DIR = BASE_DIR / "vector_db"

# Embeddings (Ollama)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_BATCH_SIZE = 32

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Retrieval / agent loop
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))
MAX_RETRIEVAL_ATTEMPTS = int(os.getenv("MAX_RETRIEVAL_ATTEMPTS", "3"))

# Groq
GROQ_TEMPERATURE = 0.1  # low: this is a factual Q&A assistant
GROQ_TIMEOUT_SECONDS = 60

_PLACEHOLDERS = {"", "your_groq_api_key_here", "your_groq_model_here"}


def get_groq_api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if key in _PLACEHOLDERS:
        raise PadhoError(
            "GROQ_API_KEY is missing.\n"
            "Copy .env.example to .env and put your real key in it "
            "(create one at https://console.groq.com/keys)."
        )
    return key


def get_groq_model() -> str:
    model = os.getenv("GROQ_MODEL", "").strip()
    if model in _PLACEHOLDERS:
        raise PadhoError(
            "GROQ_MODEL is missing.\n"
            "Set GROQ_MODEL in your .env file to a model name from "
            "https://console.groq.com/docs/models"
        )
    return model
