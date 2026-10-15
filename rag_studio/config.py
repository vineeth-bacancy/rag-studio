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
    # Since M4 each index profile names its own embedding model (Indexes);
    # this is the default index's.
    Role("embed", "Embedding", "Turns passages and questions into vectors. Each index has its own model; "
         "this is the default index's.", "nomic-embed-text:latest", "ollama"),
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


def role_model(key: str) -> str:
    """The model a role names. Raises if the role has no model yet."""
    role = next((r for r in ROLES if r.key == key), None)
    if role is None or role.model is None:
        raise KeyError(f"The {key} role has no model configured.")
    return role.model


# Measured behaviour worth showing beside a model in a picker (spec §10).
MODEL_NOTES: dict[str, str] = {
    "qwen3:4b": "This build ignores thinking off: 128-829 s per answer here. Not recommended.",
    "qwen3:8b": "About 141-246 s per answer on this CPU; a full scorecard takes about 34 minutes.",
    "qwen3:1.7b": "About 25 s per answer on this CPU. The baseline answer model.",
}
