"""Model roles and service addresses.

A role is a job the app gives a model. Each role names exactly one model, and
every artifact the model produces records which one it was, so two results are
only ever compared when their configurations are known.

These are the M1 baseline defaults: the same models the hand-built project
measured. Later milestones change them only after the scorecard says so.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

OLLAMA_URL = os.environ.get("RAG_STUDIO_OLLAMA_URL", "http://127.0.0.1:11434")

API_PORT = int(os.environ.get("RAG_STUDIO_API_PORT", "8775"))
UI_PORT = int(os.environ.get("RAG_STUDIO_UI_PORT", "3100"))


@dataclass(frozen=True)
class Role:
    key: str
    label: str          # what the UI calls it
    purpose: str        # one sentence, plain language
    model: str | None   # Ollama tag; None until the milestone that adds it
    runs_on: str        # "ollama" or "python"
    arrives: str | None = None  # milestone that switches it on, while model is None


ROLES: tuple[Role, ...] = (
    Role("embed", "Embedding", "Turns passages and questions into vectors for search.",
         "nomic-embed-text:latest", "ollama"),
    Role("answer", "Answer", "Writes the answer from the retrieved passages.",
         "qwen3:1.7b", "ollama"),
    Role("rewrite", "Query rewriting", "Rewrites questions and decides when to search again.",
         "qwen3:1.7b", "ollama"),
    Role("context", "Context lines", "Writes a sentence placing each passage in its document.",
         "qwen3:1.7b", "ollama"),
    Role("judge", "Judge", "Checks answers against their passages during evaluation.",
         "qwen3:8b", "ollama"),
    Role("vision", "Vision", "Describes charts and figures as text.",
         None, "ollama", arrives="M7"),
    Role("rerank", "Reranker", "Re-orders search results by reading each passage with the question.",
         None, "python", arrives="M5"),
)


def ollama_roles() -> list[Role]:
    return [r for r in ROLES if r.runs_on == "ollama" and r.model]
