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

After the first stage, `run()` (the one retrieval pipeline Ask, evaluation and
How it works share) can **rerank** the top 20 with a cross-encoder (rerank.py).
The order of operations is fixed and recorded: rewrite, search each query,
fuse, limit per section, rerank the top `rerank_depth`, cut to k. Passages
below the reranked top 20 keep their order, so `search(k)` is always the first
k of `search(50)`.

A `Retriever` opens one profile's **active** build and refuses to run if the
build and the profile disagree about how vectors were made (rule 3).
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterator

from rag_studio import profiles, rerank as reranking, store
from rag_studio.embed import Embedder, ProfileMismatch
from rag_studio.llm import OllamaClient
from rag_studio.trace import Step

METHODS = ("dense", "bm25", "hybrid")
RRF_K = 60          # the constant from Cormack et al. 2009, and LanceDB's default
DEPTH = 50          # candidates taken from each list before fusing or capping
NOT_CAPPED = ("toc", "references")
REWRITES = ("decompose", "multi", "hyde")
ORDER = "rewrite, search each query, fuse, limit per section, rerank, cut to k"


class IndexMissing(RuntimeError):
    """No build of this profile exists yet."""


@dataclass(frozen=True)
class SearchSpec:
    """How a question is searched. Recorded with every answer and run."""
    method: str = "dense"
    section_cap: int | None = None   # at most N passages per section ahead of the rest
    rerank: str | None = None        # an id in rerank.RERANKERS
    rerank_depth: int = reranking.DEFAULT_DEPTH
    rewrite: str | None = None       # decompose | multi | hyde (M5, remote)
    agent_rounds: int | None = None  # the agentic loop's round limit (M5, remote)

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"Unknown search method {self.method!r}. Known: {', '.join(METHODS)}.")
        if self.section_cap is not None and self.section_cap < 1:
            raise ValueError("The section cap is at least 1 passage per section.")
        if self.rerank is not None:
            reranking.get(self.rerank)
        if not 2 <= self.rerank_depth <= 50:
            raise ValueError("The reranker reads between 2 and 50 passages.")
        if self.rewrite is not None and self.rewrite not in REWRITES:
            raise ValueError(f"Unknown query rewriting {self.rewrite!r}. Known: {', '.join(REWRITES)}.")
        if self.agent_rounds is not None and not 2 <= self.agent_rounds <= 6:
            raise ValueError("Multi-step search takes 2 to 6 rounds.")

    def as_args(self) -> dict[str, Any]:
        """The constructor arguments, with each M5 switch only when it is on
        (what a run's `spec.search` stores)."""
        d: dict[str, Any] = {"method": self.method, "section_cap": self.section_cap}
        if self.rerank:
            d |= {"rerank": self.rerank, "rerank_depth": self.rerank_depth}
        if self.rewrite:
            d["rewrite"] = self.rewrite
        if self.agent_rounds:
            d["agent_rounds"] = self.agent_rounds
        return d

    @property
    def extended(self) -> bool:
        """Any M5 switch on: the search is more than a first-stage search."""
        return bool(self.rerank or self.rewrite or self.agent_rounds)

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
        # Each M5 key appears only when its switch is on, so every earlier
        # run's recorded setup stays byte-identical.
        if self.rerank:
            d |= {"rerank": self.rerank, "rerank_depth": self.rerank_depth}
        if self.rewrite:
            d["rewrite"] = self.rewrite
        if self.agent_rounds:
            d["agent_rounds"] = self.agent_rounds
        if self.extended:
            d["order"] = ORDER
        return d

    @property
    def label(self) -> str:
        name = {"dense": "Dense", "bm25": "BM25", "hybrid": "Hybrid"}[self.method]
        return (name + (f", {self.section_cap} per section" if self.section_cap else "")
                + (", reranked" if self.rerank else ""))


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
            raise IndexMissing(f"No search index yet. Build one with {profiles.build_command(profile_id)}.")
        if manifest["config"] != self.profile.config():
            raise ProfileMismatch(
                f"The {self.profile.label} index was built with a different setup than the "
                f"app now uses for it. Rebuild it with {profiles.build_command(profile_id)}.")
        self.manifest = manifest
        # Questions are embedded where the profile says (the 8B on the Colab
        # runtime), always with the build's model digest.
        self.embedder = Embedder(self.profile.embedding, client=client,
                                 expect_digest=manifest["embedding"]["digest"],
                                 on=self.profile.embedding.query_on)
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
                "embedding": m["embedding"], "query_on": self.embedder.on}

    def word_index_info(self) -> dict[str, Any] | None:
        idx = store.fts_index(self.table)
        return idx and {**idx, "settings": store.FTS_SETTINGS}

    def doc_ids(self) -> list[str]:
        return [d["doc_id"] for d in self.manifest["documents"]]

    def dense(self, question: str, k: int, where: str | None = None) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """The top k passages by cosine similarity, and how long each step took."""
        t = time.perf_counter()
        vector = self.embedder.query(question)
        embed_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        hits = store.search_dense(self.table, vector, k, where)
        search_ms = (time.perf_counter() - t) * 1000
        return hits, {"embed_ms": embed_ms, "search_ms": search_ms}

    def dense_vector(self, vector: list[float], k: int) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """The top k passages nearest an already-made vector (a drafted passage embedded as a document)."""
        t = time.perf_counter()
        hits = store.search_dense(self.table, vector, k)
        return hits, {"search_ms": (time.perf_counter() - t) * 1000}

    def bm25(self, question: str, k: int, where: str | None = None) -> tuple[list[dict[str, Any]], dict[str, float]]:
        t = time.perf_counter()
        hits = store.search_bm25(self.table, question, k, where)
        return hits, {"bm25_ms": (time.perf_counter() - t) * 1000}

    def search(self, question: str, k: int, spec: SearchSpec = DENSE,
               where: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """The top k passages by the spec's method, each with `ranks` (its rank
        in each list it came from) and `rank` (its final rank), plus the input
        lists and timings for the trace. The first k of a deeper search are
        always the same passages as a search with k, so a run can search 50
        deep and score the top 5."""
        depth = k if spec.method == "dense" and spec.section_cap is None else max(k, DEPTH)
        lists: dict[str, list[dict[str, Any]]] = {}
        timings: dict[str, float] = {}
        if spec.method in ("dense", "hybrid"):
            lists["dense"], t = self.dense(question, depth, where)
            timings |= t
        if spec.method in ("bm25", "hybrid"):
            lists["bm25"], t = self.bm25(question, depth, where)
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



# --------------------------------------------------------------------------- the pipeline

_BY = {"dense": "by meaning", "bm25": "by words (BM25)", "hybrid": "by meaning and by words (hybrid)"}


@dataclass
class Found:
    """What `run()` found for one question: everything Ask, an evaluation and
    How it works need, so none of them searches a second time."""
    question: str
    hits: list[dict[str, Any]]                  # the deep list (up to `deep`), final order
    lists: dict[str, list[dict[str, Any]]]      # each first-stage list
    timings: dict[str, float]
    candidates: int
    steps: list[Step] = field(default_factory=list)
    before_rerank: list[dict[str, Any]] | None = None   # the order the reranker was given, when it ran
    rerank: dict[str, Any] | None = None                # what the reranker did (spec, ms)
    rewrites: list[dict[str, Any]] | None = None
    rounds: list[dict[str, Any]] | None = None
    agent: dict[str, Any] | None = None             # how the multi-step search ended: stop reason, tokens, seconds
    network: dict[str, Any] | None = None   # what crossed the tunnel, per remote step
    standalone_question: str | None = None
    cache: dict[str, str] | None = None         # "hit" or "miss" per remote output


def apply_rerank(hits: list[dict[str, Any]], scores: list[float], depth: int) -> list[dict[str, Any]]:
    """Re-order the first `depth` hits by score (best first; ties keep their
    order). The rest follow unchanged. Each hit keeps the ranks it had and gains
    `ranks["reranked"]`, its place among the reranked ones, and `rerank_score`."""
    pool = hits[:depth]
    order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))
    out = []
    for place, i in enumerate(order, start=1):
        h = pool[i]
        out.append({**h, "ranks": {**(h.get("ranks") or {}), "reranked": place},
                    "rerank_score": scores[i], "rank_before": h["rank"]})
    out += [{**h, "ranks": {**(h.get("ranks") or {}), "reranked": None}} for h in hits[depth:]]
    return [{**h, "rank": n} for n, h in enumerate(out, start=1)]


def round_robin(lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """One passage from each list in turn (rank 1 of every list, then rank 2, ...), each passage once."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for rank in range(max((len(x) for x in lists), default=0)):
        for hits in lists:
            if rank < len(hits) and hits[rank]["chunk_id"] not in seen:
                seen.add(hits[rank]["chunk_id"])
                out.append(hits[rank])
    return out


def _renumber(hits: list[dict[str, Any]], lists: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Final ranks 1..n, each hit remembering where it ranked in every list that fed the fusion."""
    by_list = {name: {h["chunk_id"]: h["rank"] for h in hs} for name, hs in lists.items()}
    return [{**h, "rank": i, "ranks": {name: r.get(h["chunk_id"]) for name, r in by_list.items()},
             "capped": h.get("capped", False)} for i, h in enumerate(hits, start=1)]


def run(question: str, *, retriever: Retriever, k: int, spec: SearchSpec = DENSE, deep: int = 0,
        history: list[dict[str, str]] | None = None, remote_client: Any = None,
        scorer: Any = None, as_typed: bool = False) -> Iterator[tuple[str, Any]]:
    """The one retrieval pipeline. Yields ("stage", Step) for each step that
    finished, then ("found", Found). `deep` asks for a longer final list than
    k (an evaluation looks 50 deep to find where the answer page really ranked).
    `scorer` is anything with `.spec` and `.score(question, documents)`; the
    default is the reranker in this app.

    Order: make a follow-up a full question (`history`; skipped for `as_typed`),
    rewrite, search each query, fuse, limit per section, rerank the top
    `rerank_depth`, cut to k. With no switch on, it is one search, exactly as
    before M5 (the stage order is a contract the UI and the tests rely on)."""
    steps: list[Step] = []

    def step(id: str, label: str, ms: float, **detail: Any) -> Step:
        s = Step(id, label, ms, detail)
        steps.append(s)
        return s

    r = retriever
    depth = max(k, deep, spec.rerank_depth if spec.rerank else 0)
    t: dict[str, float] = {}
    rewrites: list[dict[str, Any]] = []
    cache_states: dict[str, str] = {}
    network: dict[str, Any] = {}
    search_q = question
    standalone_q = None

    if history and not as_typed:
        from rag_studio import rewrite
        t0 = time.perf_counter()
        rw = rewrite.standalone(question, history, client=remote_client)
        ms = (time.perf_counter() - t0) * 1000
        t["followup_ms"] = ms
        search_q = standalone_q = rw.value["standalone"]
        rewrites.append(rw.brief())
        cache_states["followup"] = rw.cache
        if rw.network:
            network["followup"] = rw.network
        yield "stage", step("followup", "Made the follow-up a full question", ms, asked=question,
                            searched_for=search_q, cache=rw.cache, model=rewrite.MODEL, where="remote",
                            prompt_version=rewrite.PROMPT_FOR["standalone"], network=rw.network,
                            changed=search_q != question)

    if not spec.rewrite:
        hits, found = r.search(search_q, depth, spec)
        t |= found["timings"]
        lists = found["lists"]
        candidates = found["candidates"]
        if "embed_ms" in t:
            yield "stage", step("embed", "Embedded the question", t["embed_ms"],
                                model=r.profile.embedding.model, prefix=r.profile.embedding.query_prefix,
                                backend=r.embedder.backend)
        search_ms = t.get("search_ms", 0) + t.get("bm25_ms", 0) + t.get("fuse_ms", 0)
        yield "stage", step("search", f"Searched {r.manifest['rows']:,} passages {_BY[spec.method]}", search_ms,
                            **spec.describe(), k=k,
                            lists={name: [{"rank": h["rank"], "chunk_id": h["chunk_id"]} for h in hits_[:max(k, 10)]]
                                   for name, hits_ in lists.items()},
                            capped=sum(1 for h in hits[:k] if h.get("capped")),
                            timings={key: round(v, 1) for key, v in t.items()})
    else:
        from concurrent.futures import ThreadPoolExecutor
        from rag_studio import rewrite
        base = SearchSpec(spec.method)                    # each query is searched plainly; fusion and the limit come after
        dq = max(depth, DEPTH)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=1) as pool:
            # The rewrite waits on the network and the GPU; the original question is searched here meanwhile.
            future = pool.submit(rewrite.plan, spec.rewrite, search_q, client=remote_client)
            orig, f0 = r.search(search_q, dq, base)
            local_ms = (time.perf_counter() - t0) * 1000
            rw = future.result()
        wait_ms = (time.perf_counter() - t0) * 1000
        t["rewrite_ms"] = wait_ms
        rewrites.append(rw.brief())
        cache_states["rewrite"] = rw.cache
        if rw.network:
            network["rewrite"] = rw.network
        queries = rewrite.queries_of(rw)
        yield "stage", step("rewrite", {"decompose": f"Split the question into {len(queries)} part(s)",
                                        "multi": "Wrote 3 other phrasings of the question",
                                        "hyde": "Drafted a passage that would answer the question"}[spec.rewrite],
                            wait_ms, technique=spec.rewrite, queries=queries,
                            passage=rw.value.get("passage"), cache=rw.cache, model=rewrite.MODEL, where="remote",
                            prompt_version=rewrite.PROMPT_FOR[spec.rewrite], tokens=rw.tokens, network=rw.network,
                            searched_original_meanwhile_ms=round(local_ms, 1))
        lists = {"original": orig}
        yield "stage", step("search", f"Searched {r.manifest['rows']:,} passages {_BY[spec.method]} for the question",
                            f0["timings"].get("search_ms", 0) + f0["timings"].get("bm25_ms", 0),
                            query=search_q, lists={"original": [{"rank": h["rank"], "chunk_id": h["chunk_id"]}
                                                                for h in orig[:max(k, 10)]]})
        embed_ms = f0["timings"].get("embed_ms", 0)
        for i, q in enumerate(queries, start=1):
            hs, fi = r.search(q, dq, base)
            lists[f"q{i}"] = hs
            embed_ms += fi["timings"].get("embed_ms", 0)
            yield "stage", step("search", f"Searched for: {q[:80]}", fi["timings"].get("search_ms", 0)
                                + fi["timings"].get("bm25_ms", 0), query=q,
                                lists={f"q{i}": [{"rank": h["rank"], "chunk_id": h["chunk_id"]} for h in hs[:max(k, 10)]]})
        if spec.rewrite == "hyde":
            t1 = time.perf_counter()
            vec = r.embedder.documents([rw.value["passage"]])[0]      # embedded as a document: the same model as the index
            hs, ti = r.dense_vector(vec, dq)
            lists["hyde"] = hs
            embed_ms += (time.perf_counter() - t1) * 1000 - ti["search_ms"]
            yield "stage", step("search", "Searched with the drafted passage (embedded as a document)",
                                ti["search_ms"], embed_as="document",
                                lists={"hyde": [{"rank": h["rank"], "chunk_id": h["chunk_id"]} for h in hs[:max(k, 10)]]})
        t["embed_ms"] = embed_ms
        t0 = time.perf_counter()
        if spec.rewrite == "decompose":
            pooled = round_robin(list(lists.values()))          # each sub-question's best, in turn; the original first
            fuse_how = "round robin over the lists"
        else:
            pooled = rrf(lists)
            fuse_how = f"reciprocal rank fusion (K={RRF_K})"
        hits = _renumber(pooled, lists)
        if spec.section_cap is not None:
            hits = _renumber(cap_sections(hits, spec.section_cap), lists)
        hits = hits[:max(depth, deep, k)]
        t["fuse_ms"] = (time.perf_counter() - t0) * 1000
        candidates = len(pooled)
        yield "stage", step("fuse", f"Combined {len(lists)} lists by {fuse_how}", t["fuse_ms"], how=fuse_how,
                            lists=list(lists), candidates=candidates, capped=sum(1 for h in hits[:k] if h.get("capped")),
                            top=[{"rank": h["rank"], "chunk_id": h["chunk_id"], "ranks": h["ranks"]}
                                 for h in hits[:max(k, 10)]])

    rounds_rec = agent_info = None
    if spec.agent_rounds:
        from rag_studio import agent
        result = None
        for kind, payload in agent.run_rounds(
                search_q, retriever=r, base=SearchSpec(spec.method), first=hits, rounds=spec.agent_rounds,
                depth=max(depth, DEPTH), client=remote_client,
                fuse=lambda ls: _renumber(rrf(ls), ls)):
            if kind == "stage":
                steps.append(payload)
                yield "stage", payload
            else:
                result = payload
        assert result is not None
        hits = result["hits"]
        if spec.section_cap is not None:
            hits = _renumber(cap_sections(hits, spec.section_cap), result["lists"])
        hits = hits[:max(depth, deep, k)]
        lists = result["lists"]
        candidates = len(hits)
        rounds_rec = result["rounds"]
        agent_info = {key: result[key] for key in ("stop", "stop_text", "tokens", "seconds")} | {
            "rounds_used": len(rounds_rec)}
        rewrites.extend(result["decisions"])
        cache_states["agent"] = "hit" if result["decisions"] and all(d["cache"] == "hit" for d in result["decisions"]) else (
            "miss" if result["decisions"] else "none")
        t["agent_ms"] = result["seconds"] * 1000

    before = rerank_info = None
    cache_status = net = None
    if spec.rerank:
        rspec = reranking.get(spec.rerank)
        scorer = scorer or reranking.scorer_for(rspec)
        pool = hits[:spec.rerank_depth]
        t0 = time.perf_counter()
        scores = scorer.score(search_q, [h["search_text"] for h in pool])      # read against the question as asked
        ms = (time.perf_counter() - t0) * 1000
        t["rerank_ms"] = ms
        cache_status = getattr(scorer, "last_cache", None)
        net = getattr(scorer, "last_net", None)
        compute_ms = (getattr(scorer, "last", None) or {}).get("compute_ms")
        before = hits
        hits = apply_rerank(hits, scores, spec.rerank_depth)
        rerank_info = rspec.describe(spec.rerank_depth)
        yield "stage", step("rerank", f"Reranked the top {len(pool)} with {rspec.label}", ms,
                            reranker=rspec.id, model=rspec.model, where=rspec.where, depth=len(pool),
                            cache=cache_status, compute_ms=compute_ms, network=net,
                            moves=[{"chunk_id": h["chunk_id"], "from": h["rank_before"], "to": h["rank"]}
                                   for h in hits[:max(k, 10)] if "rank_before" in h])
        if cache_status:
            cache_states["rerank"] = cache_status
        if net:
            network["rerank"] = net

    out = Found(question=question, hits=hits[:max(k, deep)], lists=lists, timings=t,
                candidates=candidates, steps=steps, before_rerank=before, rerank=rerank_info,
                rewrites=rewrites or None, network=network or None, standalone_question=standalone_q,
                cache=cache_states or None, rounds=rounds_rec, agent=agent_info)
    yield "found", out


_cache: dict[str, Retriever] = {}


def retriever(profile_id: str = profiles.DEFAULT) -> Retriever:
    """A Retriever for the profile's active build, reused while that build
    stays active. A rebuild switches the pointer, and the next call opens the
    new table."""
    manifest = store.active_manifest(profile_id)
    if manifest is None:
        _cache.pop(profile_id, None)
        raise IndexMissing(f"No search index yet. Build one with {profiles.build_command(profile_id)}.")
    r = _cache.get(profile_id)
    if r is None or r.table_name != manifest["table"]:
        r = _cache[profile_id] = Retriever(profile_id)
    return r
