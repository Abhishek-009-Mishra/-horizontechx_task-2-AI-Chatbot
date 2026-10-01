"""FAISS save/load, similarity search and context/source formatting."""

from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.errors import PadhoError, friendly_error
from config.settings import VECTOR_DB_DIR

NO_PAGE_SOURCE = "AWS Certification Study Material"


def build_vectorstore(chunks: list[Document], vectors: list[list[float]], embeddings: Embeddings) -> FAISS:
    """Create a FAISS index from precomputed embeddings."""
    return FAISS.from_embeddings(
        text_embeddings=list(zip([c.page_content for c in chunks], vectors)),
        embedding=embeddings,
        metadatas=[c.metadata for c in chunks],
    )


def save_vectorstore(store: FAISS) -> None:
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(VECTOR_DB_DIR))


def load_vectorstore(embeddings: Embeddings) -> FAISS:
    """Load the saved FAISS index (index.faiss + index.pkl)."""
    if not (VECTOR_DB_DIR / "index.faiss").exists() or not (VECTOR_DB_DIR / "index.pkl").exists():
        raise PadhoError(
            "The vector database was not found.\n"
            "Place the PDFs in data/clf_c02/ and data/saa_c02/, then run: python ingest.py"
        )
    try:
        # index.pkl is a pickle file created by our own ingest.py, so deserialization is safe here.
        return FAISS.load_local(str(VECTOR_DB_DIR), embeddings, allow_dangerous_deserialization=True)
    except Exception as exc:
        raise PadhoError(
            "The vector database looks corrupted or incompatible.\n"
            "Delete the files in vector_db/ and run: python ingest.py"
        ) from exc


def search(store: FAISS, query: str, k: int, book: str | None = None) -> list[Document]:
    """Similarity search, optionally restricted to one book."""
    try:
        if book:
            # Filtering happens after the fetch_k nearest hits, so fetch generously.
            return store.similarity_search(query, k=k, filter={"book": book}, fetch_k=max(k * 10, 50))
        return store.similarity_search(query, k=k)
    except Exception as exc:
        raise PadhoError(f"Retrieval failed: {friendly_error(exc)}") from exc


def source_label(doc: Document) -> str:
    book, page = doc.metadata.get("book"), doc.metadata.get("page")
    if book and page:
        return f"{book} — Page {page}"
    return book or NO_PAGE_SOURCE


def format_context(docs: list[Document]) -> str:
    """Join chunks into one labelled context string for the LLM."""
    return "\n\n".join(f"[Passage {i}: {source_label(d)}]\n{d.page_content}" for i, d in enumerate(docs, 1))


def unique_sources(docs: list[Document]) -> list[str]:
    """Unique, ordered source labels. Never invents pages."""
    labels = list(dict.fromkeys(source_label(d) for d in docs))
    if labels and all(" — Page " not in label for label in labels):
        return [NO_PAGE_SOURCE]
    return labels
