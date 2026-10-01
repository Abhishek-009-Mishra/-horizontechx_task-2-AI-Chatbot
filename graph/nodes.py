"""LangGraph nodes: analyze, retrieve, evaluate, rewrite, generate (streaming)."""

from __future__ import annotations

import re

from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field

from config.settings import RETRIEVAL_K
from graph.state import GraphState
from prompts.prompts import (
    ANSWER_USER_TEMPLATE,
    EVALUATE_JSON_REMINDER,
    EVALUATE_SYSTEM_PROMPT,
    EVALUATE_USER_TEMPLATE,
    INSUFFICIENT_NOTE,
    REWRITE_SYSTEM_PROMPT,
    REWRITE_USER_TEMPLATE,
    SYSTEM_PROMPT,
)
from retrieval.retriever import format_context, search, unique_sources


class ContextGrade(BaseModel):
    """Structured output of the context evaluator."""

    sufficient: bool = Field(description="True if the passages are enough to answer the question.")


def _text(content) -> str:
    return content if isinstance(content, str) else ""


def _parse_sufficient(text: str) -> bool:
    """Fallback parser for when structured output fails. Defaults to True (let the answer step decide)."""
    match = re.search(r'"?sufficient"?\s*[:=]\s*(true|false)', text, re.I)
    if match:
        return match.group(1).lower() == "true"
    match = re.search(r"\b(yes|no)\b", text, re.I)
    return match.group(1).lower() == "yes" if match else True


class AgentNodes:
    """Holds shared dependencies (vector store + LLM) so each node stays a simple function."""

    def __init__(self, vectorstore: FAISS, llm: BaseChatModel) -> None:
        self.vectorstore = vectorstore
        self.llm = llm

    # 1. Analyze: decide whether the user targets a specific book (cheap, no LLM call)
    def analyze_question(self, state: GraphState) -> GraphState:
        question = state["question"].strip()
        q = question.lower()
        wants_clf = bool(re.search(r"\bclf\b|clf-c02|cloud practitioner", q))
        wants_saa = bool(re.search(r"\bsaa\b|saa-c02|solutions architect", q))
        book = "CLF-C02" if wants_clf and not wants_saa else "SAA-C02" if wants_saa and not wants_clf else None
        return {
            "question": question,
            "book_filter": book,
            "rewritten_question": question,
            "retrieval_attempts": 0,
            "context_sufficient": False,
        }

    # 2. Retrieve: FAISS similarity search with the current query
    def retrieve_documents(self, state: GraphState) -> GraphState:
        get_stream_writer()({"type": "status", "text": "Searching the study books..."})
        query = state.get("rewritten_question") or state["question"]
        docs = search(self.vectorstore, query, k=RETRIEVAL_K, book=state.get("book_filter"))
        return {
            "retrieved_documents": docs,
            "context": format_context(docs),
            "sources": unique_sources(docs),
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        }

    # 3. Evaluate: is the retrieved context good enough? (structured output, robust fallback)
    def evaluate_context(self, state: GraphState) -> GraphState:
        if not state.get("retrieved_documents"):
            return {"context_sufficient": False}
        messages = [
            SystemMessage(EVALUATE_SYSTEM_PROMPT),
            HumanMessage(EVALUATE_USER_TEMPLATE.format(question=state["question"], context=state["context"])),
        ]
        try:
            grade = self.llm.with_structured_output(ContextGrade).invoke(messages)
            if grade is not None:
                return {"context_sufficient": bool(grade.sufficient)}
        except Exception:
            pass  # model/provider can't do structured output reliably -> plain-text fallback
        raw = self.llm.invoke(messages + [HumanMessage(EVALUATE_JSON_REMINDER)])
        return {"context_sufficient": _parse_sufficient(_text(raw.content))}

    # 4. Rewrite: ask the LLM for a better search query
    def rewrite_query(self, state: GraphState) -> GraphState:
        get_stream_writer()({"type": "status", "text": "Refining the search..."})
        previous = state.get("rewritten_question") or state["question"]
        reply = self.llm.invoke([
            SystemMessage(REWRITE_SYSTEM_PROMPT),
            HumanMessage(REWRITE_USER_TEMPLATE.format(question=state["question"], previous_query=previous)),
        ])
        lines = _text(reply.content).strip().splitlines()
        new_query = lines[0].strip().strip('"\'') if lines else ""
        # Never retry with an empty or identical query
        return {"rewritten_question": new_query if new_query and new_query != previous else state["question"]}

    # 5. Generate: stream the final answer from Groq token by token
    def generate_answer(self, state: GraphState) -> GraphState:
        writer = get_stream_writer()
        context = state.get("context") or "(no relevant passages were retrieved)"
        note = "" if state.get("context_sufficient") else INSUFFICIENT_NOTE
        messages = [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage(ANSWER_USER_TEMPLATE.format(
                question=state["question"], context=context, sufficiency_note=note)),
        ]
        parts: list[str] = []
        for chunk in self.llm.stream(messages):  # real LLM streaming
            piece = _text(chunk.content)
            if piece:
                parts.append(piece)
                writer({"type": "token", "text": piece})
        return {"answer": "".join(parts)}
