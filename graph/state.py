"""LangGraph state."""

from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    question: str                       # original user question
    book_filter: str | None             # "CLF-C02" / "SAA-C02" if the user named a book
    rewritten_question: str             # current search query
    retrieved_documents: list[Document]
    context: str                        # formatted context sent to the LLM
    sources: list[str]                  # human-readable source labels
    retrieval_attempts: int
    context_sufficient: bool
    answer: str
