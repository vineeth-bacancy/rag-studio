"""How it works: the engine behind the interactive sections.

Every widget on the How it works page calls one of these through the API, so
what it shows is what the pipeline does (rule 1): the chunk playground cuts
with the real chunkers, the search lab runs the real retriever, and the vector
store section reads the real LanceDB tables. Nothing here changes an index.

* `chunk_lab`: one page cut by any chunker (step 3).
* `search_lab`: dense, BM25 and hybrid lists for one question side by side,
  with the ranks each passage had in the other lists (step 7).
* `store_facts` and `ann_check`: the tables, versions and indexes of a build,
  and how much recall an approximate (ANN) index would lose against exact
  search on this corpus (step 6).
"""
from __future__ import annotations

import statistics
import time
from datetime import datetime, timezone
from typing import Any

from rag_studio import chunk, eval as evaluation, library, paths, profiles, store
from rag_studio.embed import Embedder, VectorCache
from rag_studio.files import read_json, write_json_atomic
from rag_studio.parse import pdf_title
from rag_studio.retrieve import SearchSpec, retriever

# --------------------------------------------------------------------------- step 3: chunk playground

SIZE_RANGE = (200, 2000)


def chunk_lab(doc_id: str, page: int, *, parser: str = "pymupdf", method: str = "structure",
              size: int = 1000, overlap: int = 150, percentile: float = 90.0) -> dict[str, Any]:
    """One cleaned page cut by `method`. Spans index the page's clean text.
    Semantic chunking embeds the page's sentences with the baseline embedding
    model (cached on disk) and returns the similarity curve it cut on."""
    if method not in chunk.PLAYGROUND_METHODS:
        raise ValueError(f"Unknown chunker {method!r}. Known: {', '.join(chunk.PLAYGROUND_METHODS)}.")
    if not SIZE_RANGE[0] <= size <= SIZE_RANGE[1] or not 0 <= overlap < size:
        raise ValueError(f"Size is {SIZE_RANGE[0]}-{SIZE_RANGE[1]} characters and overlap less than size.")
    doc = library.clean_doc(doc_id, parser)
    if doc is None:
        raise LookupError("Docling has not converted this document yet. "
                          "Run ./rag ingest --profile docling-clean.")
    if not 1 <= page <= len(doc.pages):
        raise IndexError(f"{doc_id} has {len(doc.pages)} pages, not {page}.")
    p = doc.pages[page - 1]
    cache = VectorCache(Embedder(profiles.BASELINE.embedding)) if method == "semantic" else None
    t = time.perf_counter()
    # Not `cache and cache.embed`: an empty cache has len 0, so it is falsy.
    out = chunk.playground(p, method, size, overlap, embed=cache.embed if cache is not None else None,
                           percentile_p=percentile)
    ms = (time.perf_counter() - t) * 1000
    model = None
    if cache is not None:
        cache.save()
        model = cache.embedder.spec.model
    spans = out["spans"]

    def mid_word(a: int) -> bool:
        return a > 0 and not p.text[a - 1].isspace() and not p.text[a].isspace()

    title = pdf_title(library.pdf_path(doc_id))
    return {
        "doc_id": doc_id, "page": page, "page_count": len(doc.pages), "parser": parser,
        "method": method, "size": size, "overlap": overlap, "percentile": percentile,
        "text": p.text, "near_empty": p.is_near_empty, "ms": round(ms, 1), "model": model,
        "passages": [{"start": a, "end": b, "chars": b - a, "starts_mid_word": mid_word(a)}
                     for a, b in spans],
        "baseline": ([{"start": a, "end": b, "starts_mid_word": mid_word(a)} for a, b in out["baseline_spans"]]
                     if "baseline_spans" in out else None),
        # Each structure-aware passage with the section label an index would
        # put in front of it (chunk.section_label), so the page shows exactly that.
        "sections": ([{**x, "label": chunk.section_label(title, x["path"], x["kind"])}
                      for x in out["sections"]] if out.get("sections") else None),
        "curve": out.get("curve"), "threshold": out.get("threshold"),
        "sentences": [{"start": a, "end": b} for a, b in out.get("sentences", [])] or None,
        "headings": p.headings, "page_sections": p.sections, "kinds": p.kinds,
    }


# --------------------------------------------------------------------------- step 7: search lab

LAB_K = 10


def _row(h: dict[str, Any], targets: list[dict[str, Any]]) -> dict[str, Any]:
    return {"rank": h["rank"], "chunk_id": h["chunk_id"], "doc_id": h["doc_id"], "page": h["page"],
            "page_label": h.get("page_label"), "section_path": h.get("section_path"),
            "kind": h.get("kind") or "body", "type": h.get("type"), "score": round(float(h["score"]), 4),
            "text": h["text"], "char_start": h["char_start"], "char_end": h["char_end"],
            "answer_page": any(h["doc_id"] == t["doc_id"] and h["page"] in t["pages"] for t in targets)}


def golden_targets(question: str) -> tuple[str | None, list[dict[str, Any]]]:
    """The test question with this exact wording, and its answer pages."""
    q = " ".join(question.split()).lower()
    for s in evaluation.load_sets().values():
        for item in s["questions"]:
            if " ".join(item["question"].split()).lower() == q:
                return item["id"], evaluation.targets(item)
    return None, []


def search_lab(question: str, profile_id: str, k: int = LAB_K, section_cap: int | None = None) -> dict[str, Any]:
    """Dense, BM25 and hybrid for one question, each `k` deep, plus where the
    first answer page ranked (searching 50 deep) when the question is a test
    question. One embedding and one BM25 query serve all three lists."""
    r = retriever(profile_id)
    qid, targets = golden_targets(question)
    lists: dict[str, Any] = {}
    timings: dict[str, float] = {}
    word_index = r.has_word_index()
    for method in ("dense", "bm25", "hybrid"):
        if method != "dense" and not word_index:
            lists[method] = None
            continue
        spec = SearchSpec(method, section_cap if method == "hybrid" else None)
        hits, info = r.search(question, evaluation.DEEP_K, spec)
        timings |= {key: round(v, 1) for key, v in info["timings"].items()}
        lists[method] = {
            "hits": [_row(h, targets) | {"ranks": h.get("ranks"), "capped": h.get("capped", False)}
                     for h in hits[:k]],
            "first_answer_rank": evaluation.first_answer_rank(hits, targets) if targets else None,
        }
    return {"question": question, "profile": profile_id, "label": r.profile.label,
            "rows": r.manifest["rows"], "k": k, "section_cap": section_cap,
            "test_question": qid, "targets": targets, "word_index": word_index,
            "lists": lists, "timings": timings, "rrf_k": 60}


# --------------------------------------------------------------------------- step 6: the vector store

def _size(path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


def store_facts(profile_id: str) -> dict[str, Any]:
    """A build's table as LanceDB keeps it: rows, columns, versions, indexes,
    size on disk, and every other build of the same profile."""
    manifest = store.active_manifest(profile_id)
    if manifest is None:
        raise LookupError("No search index yet. Build one with ./rag ingest.")
    table = store.open_table(manifest["table"])
    versions = [{"version": v["version"], "at": v["timestamp"].isoformat(timespec="seconds"),
                 "rows": int(v["metadata"].get("total_rows", 0))} for v in table.list_versions()]
    indexes = [{"name": i.name, "type": i.index_type, "columns": i.columns,
                "rows": i.num_indexed_rows, "bytes": i.size_bytes} for i in table.list_indices()]
    sample = table.to_arrow().slice(0, 1).to_pylist()[0]
    vector = sample.pop("vector")
    return {
        "profile": profile_id, "label": manifest["label"], "table": manifest["table"],
        "rows": table.count_rows(), "dimensions": len(vector),
        "bytes": _size(paths.LANCEDB / f"{manifest['table']}.lance"),
        "columns": [{"name": f.name, "type": str(f.type)} for f in table.schema],
        "sample": {"chunk_id": sample["chunk_id"], "text": sample["text"][:300],
                   "label": sample.get("label"), "vector_head": [round(x, 4) for x in vector[:8]]},
        "versions": versions, "indexes": indexes, "word_index": manifest.get("word_index"),
        "builds": [{"table": b["table"], "built_at": b["built_at"], "rows": b["rows"],
                    "active": b["table"] == manifest["table"]} for b in store.builds(profile_id)],
        "ann": read_json(_ann_path(manifest["table"])),
    }


def _ann_path(table: str):
    return paths.CACHE / "ann" / f"{table}.json"


# The approximate indexes compared with exact search. Each is built on a
# scratch copy of the vectors, never on a served table.
ANN_KINDS = ("IVF_FLAT", "IVF_PQ", "IVF_HNSW_SQ")
ANN_K = 10


def ann_check(profile_id: str) -> dict[str, Any]:
    """Recall@10 and latency of three approximate indexes against exact search,
    on this build's vectors, with the test questions as queries. Takes a few
    seconds; the result is cached per build."""
    import lancedb
    from lancedb.index import HnswSq, IvfFlat, IvfPq

    r = retriever(profile_id)
    sets = evaluation.load_sets()
    questions = [q["question"] for s in sets.values() for q in s["questions"]]
    vectors = [r.embedder.query(q) for q in questions]
    src = r.table.to_arrow().select(["chunk_id", "vector"])
    scratch = paths.CACHE / "ann" / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(scratch))

    def run(t: Any, exact: bool) -> tuple[list[list[str]], list[float]]:
        ids, ms = [], []
        for v in vectors:
            q = t.search(v).distance_type("cosine").limit(ANN_K).select(["chunk_id"])
            if exact:
                q = q.bypass_vector_index()
            t0 = time.perf_counter()
            rows = q.to_list()
            ms.append((time.perf_counter() - t0) * 1000)
            ids.append([x["chunk_id"] for x in rows])
        return ids, ms

    exact_ids, exact_ms = run(r.table, exact=True)
    results = [{"index": "exact", "label": "Exact (flat scan)", "recall": 1.0,
                "median_ms": round(statistics.median(exact_ms), 2), "build_s": 0.0}]
    configs = {"IVF_FLAT": IvfFlat(distance_type="cosine"),
               "IVF_PQ": IvfPq(distance_type="cosine"),
               "IVF_HNSW_SQ": HnswSq(distance_type="cosine")}
    for kind in ANN_KINDS:
        t = db.create_table(f"ann_{kind.lower()}", data=src, mode="overwrite")
        t0 = time.perf_counter()
        t.create_index("vector", config=configs[kind])
        build_s = time.perf_counter() - t0
        ids, ms = run(t, exact=False)
        recall = statistics.mean(len(set(a) & set(b)) / ANN_K for a, b in zip(ids, exact_ids))
        results.append({"index": kind, "label": kind, "recall": round(recall, 3),
                        "median_ms": round(statistics.median(ms), 2), "build_s": round(build_s, 2)})
        db.drop_table(f"ann_{kind.lower()}")
    out = {"table": r.table_name, "queries": len(questions), "k": ANN_K, "results": results,
           "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "lancedb_version": lancedb.__version__}
    write_json_atomic(_ann_path(r.table_name), out)
    return out
