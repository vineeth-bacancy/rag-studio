"""Search: find the passages most likely to answer a question.

Three methods, each a switch the scorecard measures:

* **dense** ("by meaning"): embed the question the way the index embedded its
  passages, then take the nearest vectors (cosine, exact).
* **BM25** ("by words"): LanceDB's full-text index over the same text that was
  embedded.
* **hybrid**: both lists, 50 deep each, fused with **reciprocal rank fusion**
  (RRF): a passage scores Σ 1 / (60 + rank) over the lists it is in, 1-based
  ranks. Only ranks count, never the raw scores, which live on different
  scales (cosine 0.6-0.9 here, BM25 0-30). We fuse ourselves rather than use
  LanceDB's built-in hybrid query, which returns only the fused list: the
  Search lab and the trace show both lists and where each passage moved.

An optional **section cap** then moves passages behind the others when their
section already has `section_cap` passages above them, and contents pages and
reference lists after their first passage (spec §5.3: GEP's Chapter 3
bibliography outranked the chapter's own summary). Nothing is dropped: a run
searching 50 deep still finds a capped passage further down.

A `Retriever` opens one profile's **active** build and refuses to run if the
build and the profile disagree about how vectors were made (rule 3).
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

from rag_studio import profiles, store
from rag_studio.embed import Embedder, ProfileMismatch
from rag_studio.llm import OllamaClient

METHODS = ("dense", "bm25", "hybrid")
RRF_K = 60          # the constant from Cormack et al. 2009, and LanceDB's default
DEPTH = 50          # candidates taken from each list before fusing or capping
NOT_CAPPED = ("toc", "references")


class IndexMissing(RuntimeError):
    """No build of this profile exists yet."""


@dataclass(frozen=True)
class SearchSpec:
    """How a question is searched. Recorded with every answer and run."""
    method: str = "dense"
    section_cap: int | None = None   # at most N passages per section ahead of the rest

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"Unknown search method {self.method!r}. Known: {', '.join(METHODS)}.")
        if self.section_cap is not None and self.section_cap < 1:
            raise ValueError("The section cap is at least 1 passage per section.")

    def describe(self) -> dict[str, Any]:
        """What a run records. A plain dense search records exactly what M1-M2
        runs did, so they stay comparable."""
        d: dict[str, Any] = {"method": self.method}
        if self.method in ("dense", "hybrid"):
            d |= {"metric": "cosine", "exact": True}
        if self.method == "hybrid":
            d |= {"fusion": "rrf", "rrf_k": RRF_K, "depth": DEPTH}
        if self.section_cap is not None:
            d |= {"section_cap": self.section_cap, "depth": DEPTH}
        return d

    @property
    def label(self) -> str:
        name = {"dense": "Dense", "bm25": "BM25", "hybrid": "Hybrid"}[self.method]
        return name + (f", {self.section_cap} per section" if self.section_cap else "")


DENSE = SearchSpec()


def rrf(lists: dict[str, list[dict[str, Any]]], k: int = RRF_K) -> list[dict[str, Any]]:
    """Reciprocal rank fusion of ranked lists of passages (by chunk_id). Ties
    keep the better dense rank, then the better BM25 rank, so the order is
    deterministic."""
    score: dict[str, float] = {}
    rows: dict[str, dict[str, Any]] = {}
    ranks: dict[str, dict[str, int]] = {}
    for name, hits in lists.items():
        for h in hits:
            cid = h["chunk_id"]
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + h["rank"])
            rows.setdefault(cid, h)
            ranks.setdefault(cid, {})[name] = h["rank"]
    big = 10**9
    order = sorted(score, key=lambda c: (-score[c], ranks[c].get("dense", big), ranks[c].get("bm25", big)))
    return [{**rows[c], "score": score[c]} for c in order]


def cap_key(hit: dict[str, Any]) -> tuple | None:
    kind = hit.get("kind") or "body"
    if kind in NOT_CAPPED:
        return (hit["doc_id"], kind)
    if hit.get("section_path"):
        return (hit["doc_id"], hit["section_path"])
    return None                      # no section known (baseline): never capped


def cap_sections(hits: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    """Stable: passages within their section's cap keep their order; the rest
    follow, in their order. Contents pages and reference lists count as one
    section each per document, capped at 1."""
    seen: Counter[tuple] = Counter()
    kept, over = [], []
    for h in hits:
        key = cap_key(h)
        if key is None:
            kept.append(h)
            continue
        seen[key] += 1
        limit = 1 if key[1] in NOT_CAPPED else cap
        (kept if seen[key] <= limit else over).append({**h, "capped": seen[key] > limit})
    return kept + over


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

    def has_word_index(self) -> bool:
        return store.fts_index(self.table) is not None

    def index_info(self) -> dict[str, Any]:
        """What a trace or a run records about the index it searched."""
        m = self.manifest
        return {"profile": m["profile"], "label": m["label"], "table": m["table"],
                "built_at": m["built_at"], "rows": m["rows"], "config": m["config"],
                "embedding": m["embedding"]}

    def word_index_info(self) -> dict[str, Any] | None:
        idx = store.fts_index(self.table)
        return idx and {**idx, "settings": store.FTS_SETTINGS}

    def dense(self, question: str, k: int) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """The top k passages by cosine similarity, and how long each step took."""
        t = time.perf_counter()
        vector = self.embedder.query(question)
        embed_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        hits = store.search_dense(self.table, vector, k)
        search_ms = (time.perf_counter() - t) * 1000
        return hits, {"embed_ms": embed_ms, "search_ms": search_ms}

    def bm25(self, question: str, k: int) -> tuple[list[dict[str, Any]], dict[str, float]]:
        t = time.perf_counter()
        hits = store.search_bm25(self.table, question, k)
        return hits, {"bm25_ms": (time.perf_counter() - t) * 1000}

    def search(self, question: str, k: int, spec: SearchSpec = DENSE) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """The top k passages by the spec's method, each with `ranks` (its rank
        in each list it came from) and `rank` (its final rank), plus the input
        lists and timings for the trace. The first k of a deeper search are
        always the same passages as a search with k, so a run can search 50
        deep and score the top 5."""
        depth = k if spec.method == "dense" and spec.section_cap is None else max(k, DEPTH)
        lists: dict[str, list[dict[str, Any]]] = {}
        timings: dict[str, float] = {}
        if spec.method in ("dense", "hybrid"):
            lists["dense"], t = self.dense(question, depth)
            timings |= t
        if spec.method in ("bm25", "hybrid"):
            lists["bm25"], t = self.bm25(question, depth)
            timings |= t
        t0 = time.perf_counter()
        ranked = rrf(lists) if spec.method == "hybrid" else list(lists[spec.method])
        if spec.section_cap is not None:
            ranked = cap_sections(ranked, spec.section_cap)
        timings["fuse_ms"] = (time.perf_counter() - t0) * 1000
        by_list = {name: {h["chunk_id"]: h["rank"] for h in hits} for name, hits in lists.items()}
        hits = [{**h, "rank": i, "ranks": {name: r.get(h["chunk_id"]) for name, r in by_list.items()},
                 "capped": h.get("capped", False)}
                for i, h in enumerate(ranked[:k], start=1)]
        return hits, {"lists": lists, "timings": timings, "candidates": len(ranked)}


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
