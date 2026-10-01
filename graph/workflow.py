"""Builds the agentic LangGraph workflow."""

from __future__ import annotations

from typing import Literal

from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from config.settings import MAX_RETRIEVAL_ATTEMPTS
from graph.nodes import AgentNodes
from graph.state import GraphState


def route_after_evaluation(state: GraphState) -> Literal["generate_answer", "rewrite_query"]:
    """Answer if context is good OR the retry budget is spent; otherwise refine the query."""
    if state.get("context_sufficient") or state.get("retrieval_attempts", 0) >= MAX_RETRIEVAL_ATTEMPTS:
        return "generate_answer"
    return "rewrite_query"


def build_workflow(vectorstore: FAISS, llm: BaseChatModel):
    nodes = AgentNodes(vectorstore, llm)
    graph = StateGraph(GraphState)

    graph.add_node("analyze_question", nodes.analyze_question)
    graph.add_node("retrieve_documents", nodes.retrieve_documents)
    graph.add_node("evaluate_context", nodes.evaluate_context)
    graph.add_node("rewrite_query", nodes.rewrite_query)
    graph.add_node("generate_answer", nodes.generate_answer)

    graph.add_edge(START, "analyze_question")
    graph.add_edge("analyze_question", "retrieve_documents")
    graph.add_edge("retrieve_documents", "evaluate_context")
    graph.add_conditional_edges(
        "evaluate_context",
        route_after_evaluation,
        {"generate_answer": "generate_answer", "rewrite_query": "rewrite_query"},
    )
    graph.add_edge("rewrite_query", "retrieve_documents")
    graph.add_edge("generate_answer", END)

    return graph.compile()
