"""Build the FAISS vector database from the two AWS PDFs.

Usage:  python ingest.py
"""

from __future__ import annotations

import sys

from config.errors import PadhoError, friendly_error
from config.settings import BOOKS, EMBEDDING_BATCH_SIZE, VECTOR_DB_DIR
from retrieval.embeddings import embed_texts, get_embeddings
from retrieval.loader import load_book, split_documents
from retrieval.retriever import build_vectorstore, save_vectorstore

LINE = "=" * 50


def print_missing_pdfs_message() -> None:
    print(f"{LINE}\nAWS PADHO - DOCUMENT INGESTION\n{LINE}\n")
    print("PDF files are not available yet.\n")
    print("Please place the books here:\n")
    for book in BOOKS:
        print(f"{book.code}:\n{book.pdf_path.relative_to(book.pdf_path.parents[2]).as_posix()}\n")
    print("After adding the PDFs, run:\n\npython ingest.py\n")


def run() -> int:
    available = [b for b in BOOKS if b.pdf_path.is_file()]
    if not available:
        print_missing_pdfs_message()
        return 1

    print(f"{LINE}\nAWS PADHO - DOCUMENT INGESTION\n{LINE}\n")
    for book in BOOKS:
        if book not in available:
            print(f"WARNING: {book.code} PDF not found at {book.pdf_path.relative_to(book.pdf_path.parents[2]).as_posix()}"
                  " - it will NOT be included.")

    pages = []
    for book in available:
        print(f"Loading {book.code}...")
        book_pages = load_book(book)
        print(f"  {len(book_pages)} pages with text")
        pages.extend(book_pages)

    print("Splitting documents...")
    chunks = split_documents(pages)
    print(f"  {len(chunks)} chunks")

    print("Creating embeddings (this can take several minutes)...")
    embeddings = get_embeddings()  # checks Ollama + model first

    def progress(done: int, total: int) -> None:
        print(f"\r  {done}/{total} chunks embedded", end="", flush=True)

    vectors = embed_texts(embeddings, [c.page_content for c in chunks], EMBEDDING_BATCH_SIZE, progress)
    print()

    print("Building FAISS...")
    store = build_vectorstore(chunks, vectors, embeddings)

    print("Saving vector database...")
    save_vectorstore(store)
    print(f"  saved to {VECTOR_DB_DIR}")
    print("Done.\n\nNow run: python main.py")
    return 0


def main() -> None:
    try:
        sys.exit(run())
    except PadhoError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nIngestion cancelled.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nERROR: {friendly_error(exc)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
