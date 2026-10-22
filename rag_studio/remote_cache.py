"""A memory for everything a remote language model wrote.

Query rewrites, follow-up questions made standalone and the agent's decisions
come from a model on the Colab runtime, and Ollama's output depends on what it
was asked just before (prefix reuse), so the same question can read slightly
differently on another day. Each output is therefore **kept**, keyed by what
was asked: the model's digest, the prompt's hash, `num_ctx` and the input. A
retrieval-only run fills it; the scorecard that follows replays the same
rewrites (and needs no runtime); and every result says whether it came from the
cache ("hit") or the model ("miss"), with the order it was generated in.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from rag_studio import paths
from rag_studio.files import read_json, write_json_atomic


def root() -> Path:
    return paths.CACHE / "remote"


def make_key(kind: str, model_digest: str, prompt_sha: str, num_ctx: int, payload: Any) -> str:
    blob = json.dumps([kind, model_digest, prompt_sha, num_ctx, payload], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _path(kind: str, key: str) -> Path:
    return root() / kind / key[:2] / f"{key}.json"


def next_seq() -> int:
    """A counter that only goes up, across processes: the order outputs were generated in."""
    root().mkdir(parents=True, exist_ok=True)
    fd = os.open(root() / ".seq", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        raw = os.read(fd, 32)
        n = int(raw or b"0") + 1
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, str(n).encode())
        return n
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def get(kind: str, key: str) -> dict[str, Any] | None:
    data = read_json(_path(kind, key))
    return data if isinstance(data, dict) and "value" in data else None


def put(kind: str, key: str, value: Any, **meta: Any) -> dict[str, Any]:
    entry = {"value": value, "seq": next_seq(), "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **meta}
    write_json_atomic(_path(kind, key), entry)
    return entry


def remember_digest(model: str, digest: str) -> None:
    """The last digest seen for a remote model, so a cache lookup can be made with no runtime connected."""
    path = root() / f".digest-{model.replace(':', '-').replace('/', '-')}"
    if not path.exists() or path.read_text().strip() != digest:
        root().mkdir(parents=True, exist_ok=True)
        path.write_text(digest + "\n")


def known_digest(model: str) -> str | None:
    path = root() / f".digest-{model.replace(':', '-').replace('/', '-')}"
    try:
        return path.read_text().strip() or None
    except OSError:
        return None
