"""PDF loading and chunking. Metadata: book, source, page (1-based PDF page index)."""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config.errors import PadhoError
from config.settings import CHUNK_OVERLAP, CHUNK_SIZE, BookConfig


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_book(book: BookConfig) -> list[Document]:
    """Read one PDF into one Document per page that contains text."""
    try:
        reader = PdfReader(str(book.pdf_path))
        if reader.is_encrypted and not reader.decrypt(""):
            raise PadhoError(f"{book.code}: the PDF is password-protected. Provide an unlocked copy.")
        pages = list(reader.pages)
    except PadhoError:
        raise
    except Exception as exc:
        raise PadhoError(
            f"{book.code}: '{book.pdf_path}' is not a valid, readable PDF ({type(exc).__name__}). "
            "Re-download or re-export the file."
        ) from exc

    docs: list[Document] = []
    for index, page in enumerate(pages):
        try:
            text = _clean(page.extract_text() or "")
        except Exception:
            continue  # skip a single unreadable page
        if text:
            docs.append(Document(
                page_content=text,
                metadata={"book": book.code, "source": book.source, "page": index + 1},
            ))

    if not docs:
        raise PadhoError(
            f"{book.code}: no extractable text found in '{book.pdf_path}'. "
            "The PDF may be empty or scanned images (OCR would be needed)."
        )
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """Split page documents into chunks; each chunk keeps its page metadata."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)
