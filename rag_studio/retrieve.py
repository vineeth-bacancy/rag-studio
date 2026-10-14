"""Search: find the passages most likely to answer a question.

M1 has dense search only ("by meaning"): embed the question the same way the
index embedded its passages, then take the nearest vectors. BM25, hybrid and
reranking arrive in M3 and M5 as switches beside it.

A `Retriever` opens one profile's **active** build and refuses to run if the
build and the profile disagree about how vectors were made (rule 3).
"""
from __future__ import annotations

import time
from typing import Any

from rag_studio import profiles, store
from rag_studio.embed import Embedder, ProfileMismatch
from rag_studio.llm import OllamaClient


class IndexMissing(RuntimeError):
    """No build of this profile exists yet."""


class Retriever:
    def __init__(self, profile_id: str = profiles.DEFAULT, client: OllamaClient | None = None):
        self.profile = profiles.get(profile_id)
        manifest = store.active_manifest(profile_id)
        if manifest is None:
            raise IndexMissing("No search index yet. Build one with ./rag ingest.")
        if manifest["config"] != self.profile.config():
            raise ProfileMismatch(
                f"The {self.profile.label} index was built with a different setup than the "
                "app now uses for it. Rebuild it with ./rag ingest.")
        self.manifest = manifest
        self.embedder = Embedder(self.profile.embedding, client=client,
                                 expect_digest=manifest["embedding"]["digest"])
        self.table = store.open_table(manifest["table"])

    @property
    def table_name(self) -> str:
        return self.manifest["table"]

    def index_info(self) -> dict[str, Any]:
        """What a trace or a run records about the index it searched."""
        m = self.manifest
        return {"profile": m["profile"], "label": m["label"], "table": m["table"],
                "built_at": m["built_at"], "rows": m["rows"], "config": m["config"],
                "embedding": m["embedding"]}

    def dense(self, question: str, k: int) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """The top k passages by cosine similarity, and how long each step took."""
        t = time.perf_counter()
        vector = self.embedder.query(question)
        embed_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        hits = store.search_dense(self.table, vector, k)
        search_ms = (time.perf_counter() - t) * 1000
        return hits, {"embed_ms": embed_ms, "search_ms": search_ms}


_cache: dict[str, Retriever] = {}


def retriever(profile_id: str = profiles.DEFAULT) -> Retriever:
    """A Retriever for the profile's active build, reused while that build
    stays active. A rebuild switches the pointer, and the next call opens the
    new table."""
    manifest = store.active_manifest(profile_id)
    if manifest is None:
        _cache.pop(profile_id, None)
        raise IndexMissing("No search index yet. Build one with ./rag ingest.")
    r = _cache.get(profile_id)
    if r is None or r.table_name != manifest["table"]:
        r = _cache[profile_id] = Retriever(profile_id)
    return r
