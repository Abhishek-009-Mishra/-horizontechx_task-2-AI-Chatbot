"""Groq chat model factory."""

from __future__ import annotations

from langchain_groq import ChatGroq

from config.settings import GROQ_TEMPERATURE, GROQ_TIMEOUT_SECONDS, get_groq_api_key, get_groq_model


def get_llm() -> ChatGroq:
    """Create the Groq chat model (key and model come from .env)."""
    return ChatGroq(
        model=get_groq_model(),
        api_key=get_groq_api_key(),
        temperature=GROQ_TEMPERATURE,
        timeout=GROQ_TIMEOUT_SECONDS,
        max_retries=2,
    )
