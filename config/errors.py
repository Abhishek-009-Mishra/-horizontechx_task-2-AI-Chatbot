"""User-facing errors and a helper that turns library exceptions into friendly messages."""

from __future__ import annotations

import groq
import httpx
import ollama


class PadhoError(Exception):
    """An error whose message is already written for the end user."""


def friendly_error(exc: BaseException) -> str:
    """Translate any exception into a beginner-friendly message."""
    if isinstance(exc, PadhoError):
        return str(exc)

    # ---- Groq ----
    if isinstance(exc, groq.AuthenticationError):
        return ("Groq rejected your API key. Check GROQ_API_KEY in your .env file "
                "(create a key at https://console.groq.com/keys).")
    if isinstance(exc, groq.APITimeoutError):
        return "The Groq request timed out. Check your internet connection and try again."
    if isinstance(exc, groq.APIConnectionError):
        return "Could not reach the Groq API. Check your internet connection and try again."
    if isinstance(exc, groq.RateLimitError):
        return "Groq rate limit reached. Wait a little and try again."
    if isinstance(exc, groq.NotFoundError) or (
        isinstance(exc, groq.BadRequestError) and "model" in str(exc).lower()
    ):
        return ("Groq does not accept the model in GROQ_MODEL (it may be misspelled or retired). "
                "See https://console.groq.com/docs/models for current model names.")
    if isinstance(exc, groq.APIStatusError):
        return f"Groq API error (HTTP {exc.status_code}): {exc}"

    # ---- Ollama ----
    if isinstance(exc, ollama.ResponseError):
        if getattr(exc, "status_code", None) == 404:
            return "The Ollama embedding model is not installed. Run: ollama pull nomic-embed-text"
        return f"Ollama error: {exc}"
    if isinstance(exc, (ConnectionError, httpx.ConnectError)):
        return "Could not connect to Ollama. Start it (open the Ollama app or run `ollama serve`) and try again."
    if isinstance(exc, httpx.TimeoutException):
        return "The request timed out. Check that Ollama / your network is responding and try again."

    return f"Unexpected error ({type(exc).__name__}): {exc}"
