"""Ingest: PDFs in, a searchable index out.

parse → clean (and guard) → chunk → embed → store, for one index profile. The steps are
the engine's modules; this file only runs them in order and records what it
did in the build's manifest, so every later number can name the exact index
it came from (rule 2). Every build also gets a word index for BM25 (M3).

**Incremental by content hash.** A passage whose text is unchanged since the
active build keeps its vector, as long as the embedding model's digest is the
same. Only new or changed passages are embedded. If nothing changed at all, no
new build is made. "Nothing" means every stored row: the manifest keeps a
fingerprint of each passage's id, content hash and metadata, because a fixed
section (M3's box ends) changes labels without changing a single count.

**Resumable (M4).** New vectors go through the on-disk vector cache, saved
every 256 passages: a build cut off (Colab reclaimed the runtime, Ctrl-C, a
cancel in the app) starts again where it stopped. **One build at a time**, and
a build on this computer never starts beside an evaluation run: both need the
CPU and Ollama, and each would make the other's timings meaningless. A build's
progress is written to `data/indexes/jobs/<profile>.json`, so the Indexes page
shows a build started in the terminal too.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import lancedb

from rag_studio import chunk, clean, guard, paths, store
from rag_studio.embed import Embedder, VectorCache, model_name
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
    and any instruction-like phrase in it. A structure-aware passage lies in
    one section by construction and says so itself."""
    if p.meta:
        return {**p.meta, "page_label": page.label,
                "flags": [f["kind"] for f in guard.instruction_flags(p.text)]}
    mid = (p.char_start + p.char_end) // 2
    path = page.at(page.sections, mid, "path") or []
    in_table = any(i["label"] in ("table", "document_index") and i["start"] is not None
                   and i["start"] <= mid < i["end"] for i in page.items)
    return {"section_path": " > ".join(path) or None, "kind": page.at(page.kinds, mid, "kind") or "body",
            "type": "table" if in_table else "text", "page_label": page.label,
            "flags": [f["kind"] for f in guard.instruction_flags(p.text)]}


def _sentence_vectors(embedder: Embedder, pages: list[clean.CleanPage], say: Progress):
    """Embed every sentence semantic chunking will compare, once, through the
    on-disk cache, and return a lookup that never calls the model again."""
    cache = VectorCache(embedder)
    texts = [pg.text[a:b] for pg in pages if not pg.is_near_empty
             for block in chunk.blocks(chunk.units(pg)) if not all(u.role == "heading" for u in block)
             and block[0].role != "table"
             for a, b in chunk.sentences(pg.text, block[0].start, block[-1].end)]
    unique = list(dict.fromkeys(texts))
    say("sentences", {"sentences": len(unique), "cached": sum(1 for x in unique if x in cache)})
    cache.embed(unique, progress=lambda done, total: say("embed", {"done": done, "total": total}))
    cache.save()
    return cache.embed


def rows_fingerprint(passages: list[chunk.Passage], hashes: list[str],
                     meta: dict[str, dict[str, Any]], none: dict[str, Any]) -> str:
    """sha256 over everything a build stores per passage except its vector."""
    h = hashlib.sha256()
    for p, ch in zip(passages, hashes):
        m = meta.get(p.chunk_id, none)
        h.update(json.dumps([p.chunk_id, ch, p.char_start, p.char_end, p.label, m["section_path"],
                             m["kind"], m["type"], m["page_label"], m["flags"]],
                            ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()


def _add_word_index(manifest: dict[str, Any], say: Progress) -> dict[str, Any]:
    """Builds made before M3 get their word index in place: no row or vector
    changes, so the build stays the one earlier runs used."""
    if manifest.get("word_index"):
        return manifest
    try:
        table = store.open_table(manifest["table"])
    except (ValueError, FileNotFoundError, OSError):
        return manifest
    t = time.perf_counter()
    fts = store.ensure_fts(table)
    say("word_index", {"column": fts["column"], "seconds": time.perf_counter() - t})
    manifest = {**manifest, "word_index": {k: v for k, v in fts.items() if k != "created"}}
    store.activate(manifest)
    return manifest


@dataclass
class BuildResult:
    manifest: dict[str, Any]
    up_to_date: bool          # nothing changed; the active build was kept


class BuildRefused(RuntimeError):
    """The build cannot start: wrong machine for this profile, or busy."""


class BuildCancelled(RuntimeError):
    """Stopped on request. Vectors made so far are kept in the cache."""


# --------------------------------------------------------------------------- one build at a time

def job_path(profile_id: str) -> Path:
    return paths.INDEXES / "jobs" / f"{profile_id}.json"


class _BuildLock:
    """An flock on data/indexes/.build.lock, held for the whole build, by the
    CLI or the API alike. `holder.json` beside it says which build it is."""

    def __init__(self) -> None:
        paths.INDEXES.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(paths.INDEXES / ".build.lock", os.O_RDWR | os.O_CREAT, 0o644)

    def acquire(self) -> None:
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self._fd)
            held = active_build() or {}
            raise BuildRefused(f"Another index is being built ({held.get('profile', 'unknown')}). "
                               "Wait for it to finish, or cancel it.") from None

    def release(self) -> None:
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)


def active_build() -> dict[str, Any] | None:
    """The build in progress (its job record), or None. Asks the lock, so a
    build whose process died is never reported as running."""
    paths.INDEXES.mkdir(parents=True, exist_ok=True)
    fd = os.open(paths.INDEXES / ".build.lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        holder = read_json(paths.INDEXES / "jobs" / "holder.json") or {}
        return read_json(job_path(holder["profile"])) if holder.get("profile") else {"status": "running"}
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    finally:
        os.close(fd)


def last_job(profile_id: str) -> dict[str, Any] | None:
    """The latest build job of a profile. One still marked running while no
    process holds the lock was interrupted, and says so."""
    job = read_json(job_path(profile_id))
    if job and job.get("status") == "running":
        active = active_build()
        if not active or active.get("profile") != profile_id:
            job = {**job, "status": "interrupted",
                   "error": "The build stopped before it finished (the app or terminal closed). "
                            "Build again: it resumes from the vectors already made."}
    return job


class _Job:
    """The job record a build keeps up to date: stage, counts, timings.
    Throttled to about two writes a second; stage changes always write."""

    def __init__(self, profile: IndexProfile, on: str, force: bool):
        self.data: dict[str, Any] = {
            "profile": profile.id, "label": profile.label, "on": on, "force": force,
            "model": profile.embedding.model, "status": "running", "stage": "start",
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pid": os.getpid(), "done": 0, "total": None, "message": None, "error": None}
        self._last = 0.0
        write_json_atomic(paths.INDEXES / "jobs" / "holder.json", {"profile": profile.id})
        self.write()

    def write(self) -> None:
        write_json_atomic(job_path(self.data["profile"]), self.data)
        self._last = time.monotonic()

    def update(self, stage: str, info: dict[str, Any]) -> None:
        changed = stage != self.data["stage"]
        self.data["stage"] = stage
        if stage == "embed":
            self.data["done"], self.data["total"] = info["done"], info["total"]
        elif stage == "embed_start":
            self.data["done"], self.data["total"] = 0, info["to_embed"]
            self.data["reused"] = info["reused"]
            self.data["cached"] = info.get("cached", 0)
        elif stage == "chunk":
            self.data["passages"] = info["passages"]
        if changed or time.monotonic() - self._last > 0.5:
            self.write()

    def finish(self, status: str, **extra: Any) -> None:
        self.data.update(status=status, finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         **extra)
        self.write()


def available_ram_gb() -> float | None:
    """MemAvailable from /proc/meminfo, in GB, or None where there is none."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024**2
    except (OSError, ValueError, IndexError):
        return None
    return None


def check_memory(profile: IndexProfile, on: str) -> dict[str, Any] | None:
    """Refuse a build on this computer that would push it into swap. On
    2026-10-15 a build loading a 2.3 GB model with 2.7 GB free froze the
    desktop and ended the session. Returns what was checked, for the UI."""
    if on != "local":
        return None
    from rag_studio.profiles import BUILD_RAM_GB
    need = BUILD_RAM_GB.get(profile.embedding.model)
    free = available_ram_gb()
    if need is None or free is None:
        return None
    info = {"need_gb": need, "free_gb": round(free, 1)}
    if free < need:
        raise BuildRefused(
            f"Only {free:.1f} GB of memory is free, and building this index here needs about {need:.1f} GB "
            f"({model_name(profile.embedding)} plus the build itself). Close some apps and build again"
            + (", or build it on the Colab runtime." if profile.embedding.builds_on("colab") else "."))
    return info


def check_where(profile: IndexProfile, on: str) -> None:
    """Refuse a build on a machine this profile does not build on."""
    spec = profile.embedding
    name = model_name(spec)
    if on not in ("local", "colab"):
        raise BuildRefused("Build on this computer (local) or on the Colab runtime (colab).")
    if spec.builds_on(on):
        return
    if on == "local":
        raise BuildRefused(f"{profile.label} is built on the Colab GPU: {name} is too large to embed "
                           "1,500 passages on this CPU. Connect a runtime, then build it there.")
    raise BuildRefused(f"{name} runs in this app on this computer, not on the Colab runtime. "
                       "Build it here.")


# --------------------------------------------------------------------------- the build

def build(profile: IndexProfile, *, force: bool = False, progress: Progress | None = None,
          client: Any = None, on: str = "local", cancel: Callable[[], bool] | None = None) -> BuildResult:
    """Build the profile's index on `on` ("local" or "colab"). `client` is an
    Ollama client for that machine (tests pass a fake); `cancel` is polled
    while embedding."""
    check_where(profile, on)
    check_memory(profile, on)
    lock = _BuildLock()
    lock.acquire()
    try:
        if on == "local":
            from rag_studio import eval as evaluation
            if evaluation.run_is_active():
                raise BuildRefused("An evaluation run is using the AI on this computer. Wait for it to "
                                   "finish, or build on the Colab runtime.")
        job = _Job(profile, on, force)

        def say(stage: str, info: dict[str, Any]) -> None:
            job.update(stage, info)
            if progress:
                progress(stage, info)
            if stage == "embed" and cancel and cancel():
                raise BuildCancelled("The build was cancelled. Vectors made so far are kept; "
                                     "building again resumes from them.")

        try:
            result = _build(profile, force=force, say=say, client=client, on=on)
        except BuildCancelled as e:
            job.finish("cancelled", error=str(e))
            raise
        except BaseException as e:  # noqa: BLE001 - recorded, then re-raised
            job.finish("failed", error=str(e) or type(e).__name__)
            raise
        m = result.manifest
        job.finish("done", up_to_date=result.up_to_date, table=m["table"], rows=m["rows"],
                   embedded=m.get("embedded"), seconds=(m.get("timings") or {}).get("total_s"))
        return result
    finally:
        lock.release()


@dataclass
class Cut:
    """A profile's documents cut into passages: what both a build and the
    context-line job need, so they always cut the same passages."""
    docs: list[Document]
    passages: list[chunk.Passage]
    meta: dict[str, dict[str, Any]]
    cleaned: dict[str, clean.CleanDoc]
    timings: dict[str, float]
    figures: dict[str, Any] | None = None     # a figures profile's summary (vision.passages)


def passages_for(profile: IndexProfile, *, say: Progress = lambda stage, info: None,
                 make_embed_fn: Callable[[dict[str, clean.CleanDoc]], Any] | None = None) -> Cut:
    """parse -> clean (and guard) -> chunk for one profile. `make_embed_fn`
    gives the semantic chunker its sentence vectors (the only chunker that
    needs a model); every other profile cuts without one."""
    timings: dict[str, float] = {}
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
    embed_fn = make_embed_fn(cleaned) if make_embed_fn and profile.chunker.method == "semantic" else None
    passages: list[chunk.Passage] = []
    for d in docs:
        pages: Any = cleaned[d.doc_id].pages if d.doc_id in cleaned else d.pages
        cut = chunk.chunk_document(pages, profile.chunker, title=d.title, embed=embed_fn)
        if d.doc_id in cleaned:
            by_page = {pg.page: pg for pg in cleaned[d.doc_id].pages}
            for p in cut:
                meta[p.chunk_id] = _passage_rows_meta(by_page[p.page], p)
        passages += cut
    figures = None
    if profile.figures:
        # M7: a passage per chart, from the cached descriptions (never a model call here).
        from rag_studio import vision
        extra, extra_meta, figures = vision.passages(profile, {d.doc_id: d.title for d in docs})
        passages += extra
        meta |= extra_meta
        say("figures", figures)
    timings["chunk_s"] = time.perf_counter() - t
    say("chunk", {"passages": len(passages), "seconds": timings["chunk_s"]})
    return Cut(docs, passages, meta, cleaned, timings, figures)


def _build(profile: IndexProfile, *, force: bool, say: Progress, client: Any, on: str) -> BuildResult:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    if client is None and profile.embedding.backend == "ollama" and on == "local":
        client = OllamaClient()
    embedder = Embedder(profile.embedding, client=client, on=on)
    digest = embedder.check()
    cut = passages_for(profile, say=say, make_embed_fn=lambda cleaned: _sentence_vectors(
        embedder, [pg for cd in cleaned.values() for pg in cd.pages], say))
    docs, passages, meta, cleaned = cut.docs, cut.passages, cut.meta, cut.cleaned
    timings |= cut.timings
    if profile.contextual:
        from rag_studio import context as context_lines
        found = context_lines.attach(profile, passages, meta)
        say("context", {"lines": len(passages), "digest": found["digest"]})

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

    none = {"section_path": None, "kind": "body", "type": "text", "page_label": None, "flags": []}
    # The hash covers the section label too: a changed label means a new vector.
    hashes = [content_hash(p.embed_text) for p in passages]
    fingerprint = rows_fingerprint(passages, hashes, meta, none)

    prior = store.active_manifest(profile.id)
    # Vectors are only ever reused from a build made the same way on the same
    # machine: a Colab build never feeds a local one, or the other way round.
    same_setup = bool(prior and prior["config"] == config and prior["embedding"]["digest"] == digest
                      and prior["embedding"].get("built_on", "local") == on)
    if (same_setup and not force and prior["documents"] == doc_rows
            and prior.get("rows_sha256") == fingerprint):
        prior = _add_word_index(prior, say)
        say("up_to_date", {"table": prior["table"]})
        return BuildResult(prior, up_to_date=True)

    # Reuse vectors from the active build when the setup is identical.
    reuse: dict[str, list[float]] = {}
    if same_setup:
        try:
            reuse = store.vectors_by_hash(store.open_table(prior["table"]))
        except (ValueError, FileNotFoundError, OSError):
            reuse = {}

    todo = [i for i, h in enumerate(hashes) if h not in reuse]
    cache = VectorCache(embedder)
    texts = [passages[i].embed_text for i in todo]
    cached = sum(1 for x in dict.fromkeys(texts) if x in cache)
    say("embed_start", {"to_embed": len(set(texts)) - cached, "reused": len(passages) - len(todo),
                        "cached": cached, "model": profile.embedding.model, "on": on})
    t = time.perf_counter()
    fresh = cache.embed(texts, progress=lambda done, total: say("embed", {"done": done, "total": total}))
    timings["embed_s"] = time.perf_counter() - t
    vectors = dict(zip(todo, fresh))

    rows = [{
        "chunk_id": p.chunk_id, "doc_id": p.doc_id, "page": p.page,
        "text": p.text, "context": p.context, "content_hash": h,
        "vector": vectors[i] if i in vectors else reuse[h],
        "parser": profile.parser, "chunker": profile.chunker.id,
        "char_start": p.char_start, "char_end": p.char_end,
        **meta.get(p.chunk_id, none),
        "label": p.label, "search_text": p.embed_text,
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
    t = time.perf_counter()
    fts = store.ensure_fts(table)
    timings["word_index_s"] = time.perf_counter() - t
    say("word_index", {"column": fts["column"], "seconds": timings["word_index_s"]})
    timings["total_s"] = time.perf_counter() - t0

    runtime = embedder.runtime()
    manifest = {
        "profile": profile.id,
        "label": profile.label,
        "table": name,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config,
        "embedding": {
            "model": profile.embedding.model, "digest": digest,
            "quantization": runtime.pop("quantization", profile.embedding.dtype),
            "dimensions": profile.embedding.dimensions,
            "doc_prefix": profile.embedding.doc_prefix,
            "query_prefix": profile.embedding.query_prefix,
            **runtime,
        },
        "documents": doc_rows,
        "rows_sha256": fingerprint,
        "rows": len(rows),
        "embedded": len(set(texts)) - cached,
        "from_cache": cached,
        "reused": len(rows) - len(todo),
        "timings": {k: round(v, 2) for k, v in timings.items()},
        "lancedb_version": lancedb.__version__,
        "table_version": table.version,
        "word_index": {k: v for k, v in fts.items() if k != "created"},
    }
    if profile.chunker.method == "semantic":
        manifest["semantic"] = {"sentence_vectors": "cached by sha256 of the sentence, per model digest"}
    if cut.figures is not None:
        manifest["figures"] = cut.figures
    store.activate(manifest)
    dropped = store.prune(profile.id)
    say("done", {"table": name, "rows": len(rows), "dropped": dropped,
                 "seconds": timings["total_s"]})
    return BuildResult(manifest, up_to_date=False)
