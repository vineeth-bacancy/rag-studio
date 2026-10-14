"""Ingest: PDFs in, a searchable index out.

parse → clean → chunk → embed → store, for one index profile. The steps are
the engine's modules; this file only runs them in order and records what it
did in the build's manifest, so every later number can name the exact index
it came from (rule 2).

**Incremental by content hash.** A passage whose text is unchanged since the
active build keeps its vector, as long as the embedding model's digest is the
same. Only new or changed passages are embedded. If nothing changed at all, no
new build is made.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import lancedb

from rag_studio import chunk, paths, store
from rag_studio.embed import Embedder
from rag_studio.llm import OllamaClient
from rag_studio.parse import Document, parse_pdf
from rag_studio.profiles import IndexProfile

Progress = Callable[[str, dict[str, Any]], None]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pdf_files() -> list:
    return sorted(paths.PDFS.glob("*.pdf"))


def documents() -> list[Document]:
    return [parse_pdf(p) for p in pdf_files()]


@dataclass
class BuildResult:
    manifest: dict[str, Any]
    up_to_date: bool          # nothing changed; the active build was kept


def build(profile: IndexProfile, *, force: bool = False, progress: Progress | None = None,
          client: OllamaClient | None = None) -> BuildResult:
    say = progress or (lambda stage, info: None)
    client = client or OllamaClient()
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    t = time.perf_counter()
    docs = documents()
    if not docs:
        raise RuntimeError(f"No PDFs found in {paths.PDFS}. Add PDFs, then run ./rag ingest again.")
    timings["parse_s"] = time.perf_counter() - t
    say("parse", {"documents": len(docs), "pages": sum(d.page_count for d in docs),
                  "seconds": timings["parse_s"]})

    t = time.perf_counter()
    passages = [p for d in docs for p in chunk.chunk_pages(d.pages, profile.chunker)]
    timings["chunk_s"] = time.perf_counter() - t
    say("chunk", {"passages": len(passages), "seconds": timings["chunk_s"]})

    embedder = Embedder(profile.embedding, client=client)
    digest = embedder.check()
    config = profile.config()
    doc_rows = [{
        "doc_id": d.doc_id, "title": d.title, "file": d.path.name, "sha256": d.sha256,
        "pages": d.page_count,
        "near_empty_pages": [p.page for p in d.pages if p.is_near_empty],
        "passages": sum(1 for p in passages if p.doc_id == d.doc_id),
    } for d in docs]

    prior = store.active_manifest(profile.id)
    same_setup = bool(prior and prior["config"] == config and prior["embedding"]["digest"] == digest)
    if same_setup and not force and prior["documents"] == doc_rows:
        say("up_to_date", {"table": prior["table"]})
        return BuildResult(prior, up_to_date=True)

    # Reuse vectors from the active build when the setup is identical.
    reuse: dict[str, list[float]] = {}
    if same_setup:
        try:
            reuse = store.vectors_by_hash(store.open_table(prior["table"]))
        except (ValueError, FileNotFoundError, OSError):
            reuse = {}

    hashes = [content_hash(p.text) for p in passages]
    todo = [i for i, h in enumerate(hashes) if h not in reuse]
    say("embed_start", {"to_embed": len(todo), "reused": len(passages) - len(todo),
                        "model": profile.embedding.model})
    t = time.perf_counter()
    fresh = embedder.documents([passages[i].text for i in todo],
                               progress=lambda done, total: say("embed", {"done": done, "total": total}))
    timings["embed_s"] = time.perf_counter() - t
    vectors = dict(zip(todo, fresh))

    rows = [{
        "chunk_id": p.chunk_id, "doc_id": p.doc_id, "page": p.page, "section_path": None,
        "type": "text", "text": p.text, "context": None, "content_hash": h,
        "vector": vectors[i] if i in vectors else reuse[h],
        "parser": profile.parser, "chunker": profile.chunker.id,
        "char_start": p.char_start, "char_end": p.char_end,
    } for i, (p, h) in enumerate(zip(passages, hashes))]

    t = time.perf_counter()
    name = store.new_table_name(profile.id)
    table = store.write_table(name, rows, profile)
    # Smoke test before switching: the right row count, and a stored vector
    # must find its own passage first.
    if table.count_rows() != len(rows):
        raise RuntimeError(f"The new index has {table.count_rows()} rows, expected {len(rows)}.")
    probe = store.search_dense(table, rows[0]["vector"], 1)
    if not probe or probe[0]["chunk_id"] != rows[0]["chunk_id"]:
        raise RuntimeError("The new index failed its self-search check; the old one is still active.")
    timings["write_s"] = time.perf_counter() - t
    timings["total_s"] = time.perf_counter() - t0

    info = client.model_info(profile.embedding.model)
    manifest = {
        "profile": profile.id,
        "label": profile.label,
        "table": name,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config,
        "embedding": {
            "model": profile.embedding.model, "digest": digest,
            "quantization": info.quantization if info else None,
            "dimensions": profile.embedding.dimensions,
            "doc_prefix": profile.embedding.doc_prefix,
            "query_prefix": profile.embedding.query_prefix,
            "backend": profile.embedding.backend, "ollama_version": client.version(),
        },
        "documents": doc_rows,
        "rows": len(rows),
        "embedded": len(todo),
        "reused": len(rows) - len(todo),
        "timings": {k: round(v, 2) for k, v in timings.items()},
        "lancedb_version": lancedb.__version__,
        "table_version": table.version,
    }
    store.activate(manifest)
    dropped = store.prune(profile.id)
    say("done", {"table": name, "rows": len(rows), "dropped": dropped,
                 "seconds": timings["total_s"]})
    return BuildResult(manifest, up_to_date=False)
