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
    # Where it runs: "local" is this computer (Ollama or this app); "remote" is the
    # Colab runtime only, with no local fallback (it refuses loudly, with how to
    # connect, when Remote AI is down).
    where: str = "local"


ROLES: tuple[Role, ...] = (
    # Since M4 each index profile names its own embedding model (Indexes);
    # this is the default index's.
    Role("embed", "Embedding", "Turns passages and questions into vectors. Each index has its own model; "
         "this is the default index's.", "google/embeddinggemma-2", "python"),
    # gemma4:e4b since M7 (the user's choice, 2026-10-23, from ladder row 10: 24 · 1 · 0, the best
    # answer model measured here; ~47 s per answer and 8 GB). qwen3:1.7b stays the light option.
    Role("answer", "Answer", "Writes the answer from the retrieved passages.",
         "gemma4:e4b", "ollama"),
    Role("rewrite", "Query rewriting", "Rewrites questions, turns follow-ups into full questions and decides "
         "when to search again.", "qwen3:8b", "ollama", where="remote"),
    Role("context", "Context lines", "Writes a sentence placing each passage in its document.",
         "qwen3:8b", "ollama", where="remote"),
    # Not in this version (user, 2026-10-22: the deterministic scorecard decides every row; the
    # judge, RAGAS and tracing were cut from M6). No model, so nothing local is checked for it.
    Role("judge", "Judge", "Checks answers against their passages during evaluation.",
         None, "ollama", arrives="a later version"),
    # M7: chosen by the bake-off on Set C's measured figures (2026-10-23; docs/steps/11-figures.md).
    # Used only to write descriptions on Remote AI; a build reads them from the cache.
    Role("vision", "Vision", "Describes charts and figures as text, once, for the index with figures.",
         "gemma4:12b", "ollama", where="remote"),
    Role("rerank", "Reranker", "Re-orders search results by reading each passage with the question. "
         "Runs on this computer, or on Remote AI for the larger models.", "Qwen/Qwen3-Reranker-0.6B", "python"),
)


def ollama_roles() -> list[Role]:
    return [r for r in ROLES if r.runs_on == "ollama" and r.model and r.where == "local"]


def role_model(key: str) -> str:
    """The model a role names. Raises if the role has no model yet."""
    role = next((r for r in ROLES if r.key == key), None)
    if role is None or role.model is None:
        raise KeyError(f"The {key} role has no model configured.")
    return role.model


# Memory an answer model's runner takes once loaded, in GB: its peak resident size measured here
# (llama-server RSS at num_ctx 8192: row 10, the M5 freeze). Loading one with less than this free
# pushes the laptop into swap, which froze it twice; Ask and evaluation runs refuse instead.
ANSWER_RAM_GB: dict[str, float] = {"qwen3:1.7b": 5.4, "qwen3.5:4b": 6.7, "gemma4:e4b": 8.0}


# Measured behaviour worth showing beside a model in a picker (spec §10).
MODEL_NOTES: dict[str, str] = {
    "qwen3:4b": "This build ignores thinking off: 128-829 s per answer here. Not recommended.",
    "qwen3:8b": "About 141-246 s per answer on this CPU; a full scorecard takes about 34 minutes.",
    "qwen3:1.7b": "About 25 s per answer on this CPU. The baseline answer model, and the light option (5.4 GB).",
    # Ladder row 10 (2026-10-22), 6d + the 4B reranker, Sets A+D.
    "qwen3.5:4b": "Thinking off works. About 44 s per answer on this CPU (29-87 s), 6.7 GB in memory; "
                  "scored 22 · 1 · 2 like qwen3:1.7b, and its two Wrongs are refusals followed by an explanation.",
    "gemma4:e4b": "Thinking off works. About 47 s per answer on this CPU (29-102 s), 8 GB in memory: close other "
                  "apps first. Scored 24 · 1 · 0, the best answer model measured here (both traps refused). "
                  "The default since M7.",
}
