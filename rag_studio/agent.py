"""Multi-step search (the "agentic loop"): look, decide, look again.

One search can miss when the answer is worded differently, or sits in a document
the first query did not favour. This loop lets a language model look at what was
found and choose: **answer** (enough evidence) or **search** once more, with a new
query and, if it wants, restricted to one document. At most a few rounds.

* **The controller is the `rewrite` role** (qwen3:8b on Remote AI; no new role).
  Its decision is a JSON object: `{action, query, doc, reason}`, checked by code.
* **Round 1 is always the normal search.** After that the controller sees the
  evidence (id, document, page, label, the first 300 characters: quoted data) and
  the searches already tried.
* **The document filter is a validated id.** It is passed to LanceDB as a
  prefilter built only from a document id the index knows (`store.doc_filter`);
  free text never reaches a `where` string.
* **It stops, and says why**: enough evidence, the round limit, the time budget,
  the token budget, a repeated query, or bad output. Nothing loops forever.
* **Answering.** All rounds' passages are pooled (reciprocal rank fusion); the
  pipeline then reranks the pool against the original question (when a reranker
  is on) and cuts to the top k, and the local answer model answers as always.

Every decision is cached (remote_cache.py), so a retrieval-only run and the
scorecard after it take the same steps.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterator

from rag_studio import rewrite
from rag_studio.trace import Step

EVIDENCE_PASSAGES = 6
EVIDENCE_CHARS = 300
DEFAULT_ROUNDS = 3
BUDGET_S = 90.0
BUDGET_TOKENS = 8000

STOPS = {
    "enough": "the controller said the passages answer it",
    "rounds": "the round limit",
    "time": "the time budget",
    "tokens": "the token budget",
    "repeat": "a query it had already tried",
    "bad_output": "output that was not a valid decision",
}


@dataclass(frozen=True)
class AgentSpec:
    rounds: int = DEFAULT_ROUNDS
    budget_s: float = BUDGET_S
    budget_tokens: int = BUDGET_TOKENS


def describe(rounds: int) -> dict[str, Any]:
    """What a run records about the loop (stable: no timings)."""
    spec = AgentSpec(rounds=rounds)
    return {**rewrite.describe("agent"), "rounds": spec.rounds, "budget_s": spec.budget_s,
            "budget_tokens": spec.budget_tokens, "evidence": f"{EVIDENCE_PASSAGES} passages x {EVIDENCE_CHARS} characters",
            "filter": "validated document id, as a LanceDB prefilter"}


def _evidence(hits: list[dict[str, Any]]) -> str:
    rows = []
    for h in hits[:EVIDENCE_PASSAGES]:
        label = " | ".join(str(h.get("label") or "").split("\n")[-1:]) or "-"
        text = " ".join(str(h.get("text") or "").split())[:EVIDENCE_CHARS]
        rows.append(f"[{h['chunk_id']}] {h['doc_id']} p{h['page']} | {label} | {text}")
    return "\n".join(rows) or "(nothing yet)"


def _schema(docs: list[str]) -> dict[str, Any]:
    base = rewrite.PROMPTS["agent-v1"]["schema"]
    props = {**base["properties"], "doc": {"type": ["string", "null"], "enum": [*docs, None]}}
    return {**base, "properties": props}


def decide(question: str, n: int, rounds: int, docs: list[str], tried: list[str], hits: list[dict[str, Any]], *,
           client: Any = None) -> rewrite.Rewrite:
    """The controller's decision for round `n`."""
    return rewrite.call(
        "agent", {"question": question, "docs": "\n".join(f"- {d}" for d in docs),
                  "tried": "\n".join(f"- {q}" for q in tried) or "(none)", "evidence": _evidence(hits),
                  "round": str(n), "rounds": str(rounds)},
        client=client, schema=_schema(docs), docs=docs)


def norm(q: str) -> str:
    return " ".join(q.lower().split())


def run_rounds(question: str, *, retriever: Any, base: Any, first: list[dict[str, Any]], rounds: int,
               depth: int, client: Any = None, fuse: Any, spec: AgentSpec | None = None
               ) -> Iterator[tuple[str, Any]]:
    """Rounds 2..`rounds`. Yields ("stage", Step) per round, then ("done", result) with the pooled hits, the
    per-round records, the stop reason and the tokens used. `first` is round 1's hits (the normal search);
    `fuse(lists)` fuses named ranked lists into one list (the pipeline's RRF)."""
    spec = spec or AgentSpec(rounds=rounds)
    docs = retriever.doc_ids()
    lists: dict[str, list[dict[str, Any]]] = {"r1": first}
    tried = [question]
    records: list[dict[str, Any]] = [{"round": 1, "action": "search", "query": question, "doc": None,
                                      "reason": "the normal search", "cache": None, "new": len(first)}]
    pooled = first
    tokens = 0
    started = time.perf_counter()
    stop = "rounds"
    briefs: list[dict[str, Any]] = []
    for n in range(2, spec.rounds + 1):
        if time.perf_counter() - started > spec.budget_s:
            stop = "time"
            break
        if tokens > spec.budget_tokens:
            stop = "tokens"
            break
        t0 = time.perf_counter()
        try:
            d = decide(question, n, spec.rounds, docs, tried, pooled, client=client)
        except rewrite.RewriteBadOutput as e:
            records.append({"round": n, "action": "stop", "reason": str(e), "cache": None})
            stop = "bad_output"
            yield "stage", Step("round", f"Round {n}: the controller's decision was not usable", (time.perf_counter() - t0) * 1000,
                                {"round": n, "stop": stop, "error": str(e)})
            break
        briefs.append(d.brief())
        tokens += (d.tokens.get("prompt") or 0) + (d.tokens.get("answer") or 0)
        v = d.value
        rec: dict[str, Any] = {"round": n, "action": v["action"], "query": v["query"], "doc": v["doc"],
                               "reason": v["reason"], "cache": d.cache, "seq": d.seq, "tokens": d.tokens}
        if v["action"] == "answer":
            records.append(rec)
            stop = "enough"
            yield "stage", Step("round", f"Round {n}: enough evidence ({v['reason']})", (time.perf_counter() - t0) * 1000,
                                {**rec, "stop": stop})
            break
        if norm(v["query"]) in {norm(q) for q in tried}:
            records.append(rec | {"action": "stop"})
            stop = "repeat"
            yield "stage", Step("round", f"Round {n}: it asked for a search it had already tried", (time.perf_counter() - t0) * 1000,
                                {**rec, "stop": stop})
            break
        from rag_studio import store
        where = store.doc_filter(v["doc"], docs) if v["doc"] else None
        hits, _ = retriever.search(v["query"], depth, base, where=where)
        tried.append(v["query"])
        lists[f"r{n}"] = hits
        before = {h["chunk_id"] for h in pooled}
        pooled = fuse(lists)
        rec["new"] = sum(1 for h in hits[:10] if h["chunk_id"] not in before)
        rec["found"] = len(hits)
        records.append(rec)
        where_txt = f" in {v['doc']}" if v["doc"] else ""
        yield "stage", Step("round", f"Round {n}: searched for \"{v['query'][:70]}\"{where_txt}",
                            (time.perf_counter() - t0) * 1000, {**rec, "network": d.network, "model": rewrite.MODEL,
                                                                "where": "remote"})
    yield "done", {"hits": pooled, "lists": lists, "rounds": records, "stop": stop, "stop_text": STOPS[stop],
                   "tokens": tokens, "decisions": briefs, "seconds": round(time.perf_counter() - started, 2)}
