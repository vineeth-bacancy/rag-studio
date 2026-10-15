"""Ingest: PDFs in, a searchable index out.

parse → clean (and guard) → chunk → embed → store, for one index profile. The steps are
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
from pathlib import Path
from typing import Any, Callable

import lancedb

from rag_studio import chunk, clean, guard, paths, store
from rag_studio.embed import Embedder
from rag_studio.files import read_json, write_json_atomic
from rag_studio.llm import OllamaClient
from rag_studio.parse import Document, file_sha256, parse_pdf
from rag_studio.profiles import IndexProfile

Progress = Callable[[str, dict[str, Any]], None]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pdf_files() -> list:
    return sorted(paths.PDFS.glob("*.pdf"))


def documents() -> list[Document]:
    return [parse_pdf(p) for p in pdf_files()]


# --------------------------------------------------------------------------- cleaned documents

_CODE = ("clean.py", "guard.py", "structure.py", "parse.py", "parse_docling.py")


def _code_hash() -> str:
    """The cleaning code itself, so a changed rule never serves a stale cache."""
    h = hashlib.sha256()
    for name in _CODE:
        h.update((Path(__file__).parent / name).read_bytes())
    return h.hexdigest()[:8]


def clean_cache_path(path: Path, parser: str, sha: str | None = None) -> Path:
    sha = sha or file_sha256(path)
    return paths.CACHE / "clean" / f"{sha[:16]}-{parser}-{clean.FULL_CLEANER_ID}-{_code_hash()}.json"


def cleaned_document(path: Path, parser: str, *, progress: Progress | None = None,
                     cached_only: bool = False) -> clean.CleanDoc | None:
    """A PDF through the full cleaning order with one parser, cached by the
    file's content, the parser, the cleaner and the code. `cached_only` never
    runs a parser (the API uses it for Docling, which takes minutes)."""
    path = Path(path)
    out = clean_cache_path(path, parser)
    data = read_json(out)
    if data is not None:
        return clean.CleanDoc.from_dict(data)
    if parser == "pymupdf":
        doc = clean.clean_pdf(path)
    elif parser == "docling":
        from rag_studio import parse_docling
        if cached_only and not parse_docling.is_converted(path):
            return None
        ref = cleaned_document(path, "pymupdf")
        doc = parse_docling.clean_docling(path, ref, progress=progress)
    else:
        raise ValueError(f"Unknown parser {parser!r}.")
    write_json_atomic(out, doc.as_dict())
    return doc


def _passage_rows_meta(page: clean.CleanPage, p: chunk.Passage) -> dict[str, Any]:
    """What a cleaned page knows about one passage: its section, kind, type
    and any instruction-like phrase in it."""
    mid = (p.char_start + p.char_end) // 2
    path = page.at(page.sections, mid, "path") or []
    in_table = any(i["label"] in ("table", "document_index") and i["start"] is not None
                   and i["start"] <= mid < i["end"] for i in page.items)
    return {"section_path": " > ".join(path) or None, "kind": page.at(page.kinds, mid, "kind") or "body",
            "type": "table" if in_table else "text", "page_label": page.label,
            "flags": [f["kind"] for f in guard.instruction_flags(p.text)]}


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
    meta: dict[str, dict[str, Any]] = {}
    cleaned: dict[str, clean.CleanDoc] = {}
    if profile.cleaner != clean.CLEANER_ID:
        for d in docs:
            say("clean", {"doc_id": d.doc_id, "parser": profile.parser})
            cd = cleaned_document(d.path, profile.parser, progress=say)
            assert cd is not None
            cleaned[d.doc_id] = cd
    timings["parse_s"] = time.perf_counter() - t
    say("parse", {"documents": len(docs), "pages": sum(d.page_count for d in docs),
                  "seconds": timings["parse_s"]})

    t = time.perf_counter()
    passages: list[chunk.Passage] = []
    for d in docs:
        pages: Any = cleaned[d.doc_id].pages if d.doc_id in cleaned else d.pages
        cut = chunk.chunk_pages(pages, profile.chunker)
        if d.doc_id in cleaned:
            by_page = {pg.page: pg for pg in cleaned[d.doc_id].pages}
            for p in cut:
                meta[p.chunk_id] = _passage_rows_meta(by_page[p.page], p)
        passages += cut
    timings["chunk_s"] = time.perf_counter() - t
    say("chunk", {"passages": len(passages), "seconds": timings["chunk_s"]})

    embedder = Embedder(profile.embedding, client=client)
    digest = embedder.check()
    config = profile.config()
    doc_rows = []
    for d in docs:
        pages = cleaned[d.doc_id].pages if d.doc_id in cleaned else d.pages
        row = {
            "doc_id": d.doc_id, "title": d.title, "file": d.path.name, "sha256": d.sha256,
            "pages": d.page_count,
            "near_empty_pages": [p.page for p in pages if p.is_near_empty],
            "passages": sum(1 for p in passages if p.doc_id == d.doc_id),
        }
        if d.doc_id in cleaned:
            row["cleaning"] = cleaned[d.doc_id].counts
            row["flagged_passages"] = sum(1 for p in passages if p.doc_id == d.doc_id
                                          and meta[p.chunk_id]["flags"])
        doc_rows.append(row)

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

    none = {"section_path": None, "kind": "body", "type": "text", "page_label": None, "flags": []}
    rows = [{
        "chunk_id": p.chunk_id, "doc_id": p.doc_id, "page": p.page,
        "text": p.text, "context": None, "content_hash": h,
        "vector": vectors[i] if i in vectors else reuse[h],
        "parser": profile.parser, "chunker": profile.chunker.id,
        "char_start": p.char_start, "char_end": p.char_end,
        **meta.get(p.chunk_id, none),
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
