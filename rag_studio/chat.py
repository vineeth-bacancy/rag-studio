"""Conversations: a thread of questions and answers kept on this computer.

One JSON file per conversation in `data/chat/<id>.json`; the id is 32 hex
characters (so a request can never name a path), and every write is atomic.
Each turn keeps the question as typed, the question as searched (when a
follow-up was made whole by Remote AI), the answer, the citation map, the
sources (chunk, document, page), the setup, whether it was refused, and a short
trace summary: enough to show the thread again without asking anything.

**Memory acts only through the rewrite.** `history()` is what the follow-up
rewrite reads; the answer model is never given the conversation (answer.py).
This module logs nothing: not content, not even counts.
"""
from __future__ import annotations

import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_studio import paths
from rag_studio.files import read_json, write_json_atomic

ID = re.compile(r"^[0-9a-f]{32}$")
MAX_TURNS = 200                     # a thread this long is not a conversation any more; start another
TITLE_CHARS = 80


class BadChatId(ValueError):
    """Not a conversation id."""


def new_id() -> str:
    return secrets.token_hex(16)


def _path(chat_id: str) -> Path:
    if not isinstance(chat_id, str) or not ID.match(chat_id):
        raise BadChatId("That is not a conversation id.")
    return paths.CHAT / f"{chat_id}.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def load(chat_id: str) -> dict[str, Any] | None:
    return read_json(_path(chat_id))


def history(chat_id: str) -> list[dict[str, str]]:
    """The earlier turns as the follow-up rewrite reads them: what was asked, and what was answered."""
    convo = load(chat_id)
    return [{"question": t["question"], "answer": t.get("answer") or ""} for t in (convo or {}).get("turns", [])
            if t.get("answer")]


def add_turn(chat_id: str, done: dict[str, Any], setup: dict[str, Any]) -> dict[str, Any]:
    """Append a finished answer (the `done` payload of answer.ask) to a conversation, creating it if need be."""
    path = _path(chat_id)
    convo = read_json(path) or {"id": chat_id, "created_at": _now(), "title": None, "turns": []}
    if len(convo["turns"]) >= MAX_TURNS:
        raise ValueError(f"This conversation has {MAX_TURNS} questions. Start a new one.")
    sources = done.get("sources") or []
    turn = {
        "question": done["question"],
        "standalone_question": done.get("standalone_question"),
        "rewritten": done.get("rewritten"),
        "answer": done["answer"],
        "refused": done["refused"],
        "citations": done["citations"]["map"],
        "sources": [{"n": s["n"], "chunk_id": s["chunk_id"], "doc_id": s["doc_id"], "page": s["page"]} for s in sources],
        "setup": setup,
        "trace": [{"id": s["id"], "ms": s["ms"]} for s in done.get("trace") or []],
        "total_ms": done.get("total_ms"),
        "at": _now(),
    }
    convo["turns"].append(turn)
    convo["updated_at"] = turn["at"]
    convo["title"] = convo["title"] or " ".join(done["question"].split())[:TITLE_CHARS]
    write_json_atomic(path, convo)
    return convo


def list_chats() -> list[dict[str, Any]]:
    """Every conversation, newest first, without its turns."""
    if not paths.CHAT.is_dir():
        return []
    out = []
    for p in paths.CHAT.glob("*.json"):
        if not ID.match(p.stem):
            continue
        c = read_json(p)
        if c:
            out.append({"id": c["id"], "title": c.get("title"), "turns": len(c.get("turns", [])),
                        "created_at": c.get("created_at"), "updated_at": c.get("updated_at")})
    return sorted(out, key=lambda c: c.get("updated_at") or "", reverse=True)


def delete(chat_id: str) -> bool:
    path = _path(chat_id)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
