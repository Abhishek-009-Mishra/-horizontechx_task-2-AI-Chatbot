"""AWS Padho - Agentic AWS Q&A CLI (LangGraph + FAISS + Ollama + Groq).

Usage:  python main.py
"""

from __future__ import annotations

import os
import sys

from config.errors import PadhoError, friendly_error
from graph.llm import get_llm
from graph.workflow import build_workflow
from retrieval.embeddings import get_embeddings
from retrieval.retriever import load_vectorstore

if os.name == "nt":
    os.system("")  # enables ANSI escape codes in Windows terminals

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
LINE = "=" * 50

BANNER = f"""{LINE}
AWS PADHO
Agentic AWS Q&A Assistant
{LINE}

Books:
  • AWS Certified Cloud Practitioner — CLF-C02
  • AWS Certified Solutions Architect — SAA-C02

Powered by:
  LangGraph + FAISS + Ollama + Groq

Commands:
  /help
  /source
  /clear
  /exit
{LINE}
"""

HELP = """
Ask any question about AWS. Tips: mention "CLF-C02" or "SAA-C02" in your
question to search only that book.

  /help    show this help
  /source  show the sources of the last answer
  /clear   clear the screen
  /exit    quit
"""


def clear_line() -> None:
    print("\r\033[K", end="", flush=True)


def show_status(text: str) -> None:
    clear_line()
    print(f"{DIM}  {text}{RESET}", end="", flush=True)


def print_sources(sources: list[str]) -> None:
    if sources:
        print("\nSources:")
        for i, label in enumerate(sources, 1):
            print(f"  {i}. {label}")


def ask(graph, question: str) -> list[str]:
    """Run the graph, printing tokens as they stream. Returns the source labels."""
    final: dict = {}
    streaming = False
    for mode, data in graph.stream({"question": question}, stream_mode=["custom", "values"]):
        if mode == "custom":
            if data["type"] == "status" and not streaming:
                show_status(data["text"])
            elif data["type"] == "token":
                if not streaming:
                    clear_line()
                    print(f"\n{BOLD}AWS Padho:{RESET}")
                    streaming = True
                print(data["text"], end="", flush=True)
        else:
            final = data
    if not streaming:  # nothing streamed (e.g. empty model output)
        clear_line()
        print(f"\n{BOLD}AWS Padho:{RESET}\n{final.get('answer') or 'I could not generate an answer.'}", end="")
    print()
    sources = final.get("sources", [])
    print_sources(sources)
    return sources


def start_up():
    """Validate configuration and build the agent. Raises PadhoError on problems."""
    llm = get_llm()                      # checks GROQ_API_KEY / GROQ_MODEL
    embeddings = get_embeddings()        # checks Ollama + nomic-embed-text
    vectorstore = load_vectorstore(embeddings)
    try:
        llm.invoke("Reply with the single word: OK")   # verifies key, model and network
    except Exception as exc:
        raise PadhoError(f"Groq check failed: {friendly_error(exc)}") from exc
    return build_workflow(vectorstore, llm)


def main() -> None:
    print("Starting AWS Padho...")
    try:
        graph = start_up()
    except Exception as exc:
        print(f"\nERROR: {friendly_error(exc)}")
        sys.exit(1)

    print(BANNER)
    last_sources: list[str] = []

    while True:
        try:
            user_input = input(f"{BOLD}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            print("Please type a question (or /help).")
            continue

        if user_input.startswith("/"):
            command = user_input.lower()
            if command in ("/exit", "/quit"):
                print("Goodbye!")
                break
            elif command == "/help":
                print(HELP)
            elif command == "/source":
                print_sources(last_sources) if last_sources else print("No sources yet - ask a question first.")
            elif command == "/clear":
                os.system("cls" if os.name == "nt" else "clear")
            else:
                print("Unknown command. Type /help.")
            continue

        try:
            last_sources = ask(graph, user_input)
        except KeyboardInterrupt:
            print("\n[interrupted]")
        except Exception as exc:
            clear_line()
            print(f"\nERROR: {friendly_error(exc)}")
        print()


if __name__ == "__main__":
    main()
