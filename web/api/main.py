"""HTTP routes only: validate, call the engine, return. No RAG logic lives here.

Run with `./rag web`, which binds 127.0.0.1 and exports the ports the request
guard trusts. Interactive schema at http://127.0.0.1:8775/api/_docs.
"""
from __future__ import annotations

import re
import threading
from typing import NoReturn

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sse
from guard import RequestGuard, allowed_origins
from rag_studio import __version__, answer, eval as evaluation, health, learn, library, paths, profiles, store
from rag_studio.config import MODEL_NOTES, role_model
from rag_studio.embed import ProfileMismatch
from rag_studio.llm import OllamaClient, OllamaUnavailable
from rag_studio.retrieve import IndexMissing, SearchSpec
from rag_studio.parse import aligned_boxes, file_sha256, passage_boxes, render_page_png
from rag_studio.prompt import DEFAULT_VERSION, SYSTEM_PROMPTS

app = FastAPI(
    title="RAG Studio API",
    version=__version__,
    docs_url="/api/_docs",
    redoc_url=None,
    openapi_url="/api/openapi.json",
)

# The browser never makes a cross-origin call in practice: next.config.ts
# rewrites /api/* to this process, so everything is same-origin. CORS is the
# fallback for opening the UI directly against this port.
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins(),
                   allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
# Added last, so it runs first: a refused request never reaches CORS or a route.
app.add_middleware(RequestGuard)


def _fail(status: int, message: str) -> NoReturn:
    raise HTTPException(status_code=status, detail=message)


# --------------------------------------------------------------------------- status

@app.get("/api/status")
def get_status() -> dict:
    """What the sidebar and top bar poll: is the AI up, which model answers."""
    return health.status()


@app.get("/api/health")
def get_health() -> dict:
    """The full System Health report. Cheap enough to compute on every request."""
    return health.report()


@app.get("/api/models")
def get_models() -> dict:
    """Installed Ollama models that can write answers, for the model pickers."""
    try:
        models = OllamaClient().installed()
    except OllamaUnavailable as e:
        _fail(503, f"AI unavailable. {e.reason}")
    return {"default": role_model("answer"), "answer_models": [
        {"name": m.name, "parameter_size": m.parameter_size, "quantization": m.quantization,
         "note": MODEL_NOTES.get(m.name)}
        for m in models if "completion" in m.capabilities
    ]}


# --------------------------------------------------------------------------- documents

@app.get("/api/documents")
def get_documents() -> dict:
    """The PDF library, with page and passage counts once indexed."""
    return {"documents": library.documents()}


def _doc_path(doc_id: str):
    try:
        return library.pdf_path(doc_id)
    except KeyError as e:
        _fail(404, str(e.args[0]))


@app.get("/api/documents/{doc_id}/pages/{page}/image")
def get_page_image(doc_id: str, page: int, dpi: int = Query(110, ge=36, le=220)) -> Response:
    """The page rendered as a PNG, cached by the PDF's content hash."""
    path = _doc_path(doc_id)
    cache = paths.CACHE / "pages" / file_sha256(path)[:16] / f"{page}@{dpi}.png"
    if cache.is_file():
        png = cache.read_bytes()
    else:
        try:
            png = render_page_png(path, page, dpi)
        except IndexError as e:
            _fail(404, str(e))
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        tmp.write_bytes(png)
        tmp.replace(cache)
    return Response(png, media_type="image/png",
                    headers={"Cache-Control": "private, max-age=86400"})


@app.get("/api/documents/{doc_id}/pages/{page}/boxes")
def get_passage_boxes(doc_id: str, page: int, start: int = Query(..., ge=0),
                      end: int = Query(..., ge=1), profile: str = Query(profiles.DEFAULT, max_length=40),
                      chunk: str | None = Query(None, max_length=200)) -> dict:
    """Where a passage sits on the page. The baseline index stores exact
    character spans of PyMuPDF's text; a cleaned index's text differs, so its
    passage is found by aligning words (PyMuPDF) or by Docling's item boxes."""
    if end <= start:
        _fail(400, "The passage end must come after its start.")
    path = _doc_path(doc_id)
    prof = _profile(profile)
    try:
        if prof.cleaner == profiles.BASELINE.cleaner and prof.parser == profiles.BASELINE.parser:
            return passage_boxes(path, page, start, end)
        manifest = store.active_manifest(prof.id)
        passage = (store.get_passage(store.open_table(manifest["table"]), chunk)
                   if manifest and chunk else None)
        if passage is None or passage["doc_id"] != doc_id or passage["page"] != page:
            _fail(404, "That passage is not in the index any more. Ask the question again.")
        if prof.parser == "docling":
            return _docling_boxes(doc_id, page, start, end)
        return aligned_boxes(path, page, passage["text"])
    except IndexError as e:
        _fail(404, str(e))


def _docling_boxes(doc_id: str, page: int, start: int, end: int) -> dict:
    doc = library.clean_doc(doc_id, "docling")
    if doc is None:
        _fail(404, "Docling has not converted this document yet. Run: ./rag ingest --profile docling-clean")
    pg = doc.pages[page - 1]
    boxes = [{"x0": i["box"][0] / pg.width, "y0": i["box"][1] / pg.height,
              "x1": i["box"][2] / pg.width, "y1": i["box"][3] / pg.height}
             for i in pg.items if i["start"] is not None and i["start"] < end and i["end"] > start]
    return {"width": pg.width, "height": pg.height, "boxes": boxes, "aligned": 1.0 if boxes else 0.0}


@app.get("/api/documents/{doc_id}/pages/{page}/inspect")
def get_page_inspection(doc_id: str, page: int) -> dict:
    """One page through both parsers and every cleaning rule, for the page
    inspector. The first PyMuPDF view of a document takes a few seconds (the
    guard renders each page twice); later views come from the cache."""
    _doc_path(doc_id)
    try:
        return library.inspect_page(doc_id, page)
    except IndexError as e:
        _fail(404, str(e))


@app.get("/api/documents/{doc_id}/cleaning")
def get_cleaning(doc_id: str) -> dict:
    """What cleaning and the guard did to one document, per parser."""
    _doc_path(doc_id)
    return library.cleaning_summary(doc_id)


@app.get("/api/indexes")
def get_indexes() -> dict:
    """Each index profile and its active build's manifest (None until built).
    `word_index` says whether BM25 and hybrid search can run on it; `labels`
    whether its passages carry section labels (prompt v4 expects them)."""
    out = []
    for p in profiles.PROFILES.values():
        active = store.active_manifest(p.id)
        out.append({"id": p.id, "label": p.label, "description": p.description,
                    "parser": p.parser, "chunker": p.chunker.method, "labels": p.chunker.labels,
                    "word_index": bool(active and active.get("word_index")),
                    "active": active, "builds": len(store.builds(p.id))})
    return {"profiles": out, "search_methods": [
        {"id": "dense", "label": "Dense", "explain": "By meaning: the nearest vectors."},
        {"id": "bm25", "label": "BM25", "explain": "By words: the full-text index."},
        {"id": "hybrid", "label": "Hybrid", "explain": "Both lists, fused by rank (RRF)."},
    ]}


# --------------------------------------------------------------------------- ask

SEARCH_PATTERN = "^(dense|bm25|hybrid)$"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(answer.DEFAULT_K, ge=1, le=20)
    profile: str = Field(profiles.DEFAULT, max_length=40)
    prompt: str = Field(DEFAULT_VERSION, max_length=40)
    search: str = Field("dense", pattern=SEARCH_PATTERN)
    section_cap: int | None = Field(None, ge=1, le=10)


def _profile(profile_id: str) -> profiles.IndexProfile:
    try:
        return profiles.get(profile_id)
    except KeyError as e:
        _fail(400, str(e.args[0]))


def _prompt(version: str) -> str:
    if version not in SYSTEM_PROMPTS:
        _fail(400, f"No prompt version {version!r}. Known: {', '.join(SYSTEM_PROMPTS)}.")
    return version


@app.post("/api/ask")
def post_ask(req: AskRequest):
    """One question, answered as server-sent events (see rag_studio/answer.py
    for the event contract): stage, ready, delta, done or error."""
    return sse.stream(answer.ask(req.question, k=req.k, profile_id=_profile(req.profile).id,
                                 prompt_version=_prompt(req.prompt),
                                 search=SearchSpec(req.search, req.section_cap)))


@app.get("/api/prompts")
def get_prompts() -> dict:
    """The system prompt versions, with their full text: the prompt is part of
    what a run measures, so it is never hidden."""
    from rag_studio.prompt import prompt_sha256
    return {"default": DEFAULT_VERSION, "versions": [
        {"version": v, "sha256": prompt_sha256(v), "text": t} for v, t in SYSTEM_PROMPTS.items()]}


# --------------------------------------------------------------------------- evaluation

@app.get("/api/golden")
def get_golden() -> dict:
    """The test-question sets, as stored in data/golden/."""
    try:
        sets = evaluation.load_sets()
    except evaluation.GoldenError as e:
        _fail(500, f"A test-question file is invalid: {e}")
    return {"sets": list(sets.values())}


@app.get("/api/eval/runs")
def get_runs() -> dict:
    return {"runs": evaluation.list_runs(), "active": evaluation.run_is_active()}


_RUN_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")


@app.get("/api/eval/ladder")
def get_ladder() -> dict:
    """The upgrade ladder (spec §8): one row per change, filled from tagged runs."""
    return {"rows": evaluation.ladder()}


@app.get("/api/eval/compare")
def get_compare(a: str = Query(..., max_length=40), b: str = Query(..., max_length=40)) -> dict:
    """Two runs side by side: what differs in their setup, and which
    questions changed outcome or retrieval. A is "before", B is "after"."""
    if not (_RUN_ID.match(a) and _RUN_ID.match(b)):
        _fail(404, "No such run. It may have been deleted.")
    try:
        return evaluation.compare(a, b)
    except ValueError as e:
        _fail(404, str(e))


@app.get("/api/eval/runs/{run_id}")
def get_run(run_id: str) -> dict:
    run = evaluation.get_run(run_id) if _RUN_ID.match(run_id) else None
    if run is None:
        _fail(404, "No such run. It may have been deleted.")
    return run


class RunRequest(BaseModel):
    sets: list[str] = Field(min_length=1)
    mode: str = Field("answer", pattern="^(answer|retrieval)$")
    model: str | None = Field(None, max_length=100)
    k: int = Field(answer.DEFAULT_K, ge=1, le=20)
    label: str | None = Field(None, max_length=80)
    profile: str = Field(profiles.DEFAULT, max_length=40)
    prompt: str = Field(DEFAULT_VERSION, max_length=40)
    search: str = Field("dense", pattern=SEARCH_PATTERN)
    section_cap: int | None = Field(None, ge=1, le=10)


# Cancel flags for runs started by this process. A run started from the
# terminal is cancelled there (Ctrl-C).
_cancel: dict[str, threading.Event] = {}
_start_lock = threading.Lock()


@app.post("/api/eval/runs", status_code=202)
def post_run(req: RunRequest) -> dict:
    """Start a run in the background and return its id at once. The run saves
    itself after every question; poll GET /api/eval/runs/{id} for progress."""
    spec = evaluation.RunSpec(sets=req.sets, mode=req.mode, k=req.k, model=req.model,
                              label=req.label, profile=_profile(req.profile).id,
                              prompt_version=_prompt(req.prompt),
                              search={"method": req.search, "section_cap": req.section_cap})
    with _start_lock:
        if evaluation.run_is_active():
            _fail(409, "An evaluation run is already in progress. Wait for it to finish, or cancel it.")
        run_id = evaluation.new_run_id()
        started = threading.Event()
        failure: list[str] = []
        cancel = _cancel[run_id] = threading.Event()

        def work() -> None:
            try:
                evaluation.execute(spec, run_id=run_id, cancel=cancel.is_set,
                                   on_start=started.set)
            except Exception as e:  # noqa: BLE001 - reported to the request, or saved below
                failure.append(str(e) or type(e).__name__)
                if started.is_set():
                    evaluation.mark_failed(run_id, failure[0])
            finally:
                started.set()
                _cancel.pop(run_id, None)

        threading.Thread(target=work, name=f"run-{run_id}", daemon=True).start()
        started.wait(timeout=30)
    if failure and not evaluation.run_path(run_id).exists():
        _fail(409 if "already in progress" in failure[0] else 400, failure[0])
    return {"id": run_id}


@app.post("/api/eval/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict:
    flag = _cancel.get(run_id)
    if flag is None:
        _fail(404, "This run is not running here. A run started in the terminal stops with Ctrl-C.")
    flag.set()
    return {"id": run_id, "cancelling": True}



# --------------------------------------------------------------------------- how it works (M3)

def _learn_call(fn, *args, **kwargs):
    """Engine errors as plain HTTP answers: bad input 400, missing things 404,
    a mismatched index 409, AI down 503."""
    try:
        return fn(*args, **kwargs)
    except (IndexMissing, LookupError, IndexError) as e:
        _fail(404, str(e.args[0]) if e.args else str(e))
    except ProfileMismatch as e:
        _fail(409, str(e))
    except store.WordIndexMissing as e:
        _fail(409, str(e))
    except OllamaUnavailable as e:
        _fail(503, f"AI unavailable. {e.reason}")
    except ValueError as e:
        _fail(400, str(e))


@app.get("/api/learn/chunk")
def get_chunk_lab(doc_id: str = Query(..., max_length=100), page: int = Query(..., ge=1),
                  parser: str = Query("pymupdf", pattern="^(pymupdf|docling)$"),
                  method: str = Query("structure", max_length=20),
                  size: int = Query(1000, ge=learn.SIZE_RANGE[0], le=learn.SIZE_RANGE[1]),
                  overlap: int = Query(150, ge=0, le=1000),
                  percentile: float = Query(90.0, ge=50.0, le=99.0)) -> dict:
    """One cleaned page cut by any chunker, for the chunk playground."""
    _doc_path(doc_id)
    return _learn_call(learn.chunk_lab, doc_id, page, parser=parser, method=method, size=size,
                       overlap=overlap, percentile=percentile)


@app.get("/api/learn/search")
def get_search_lab(q: str = Query(..., min_length=1, max_length=500),
                   profile: str = Query(profiles.DEFAULT, max_length=40),
                   k: int = Query(learn.LAB_K, ge=1, le=20),
                   cap: int | None = Query(None, ge=1, le=10)) -> dict:
    """Dense, BM25 and hybrid lists for one question, for the search lab."""
    return _learn_call(learn.search_lab, q, _profile(profile).id, k, cap)


@app.get("/api/learn/store")
def get_store_facts(profile: str = Query(profiles.DEFAULT, max_length=40)) -> dict:
    """A build's LanceDB table: columns, versions, indexes, size, other builds."""
    return _learn_call(learn.store_facts, _profile(profile).id)


_ann_lock = threading.Lock()


@app.post("/api/learn/store/ann-check")
def post_ann_check(profile: str = Query(profiles.DEFAULT, max_length=40)) -> dict:
    """Measure approximate (ANN) indexes against exact search on a scratch copy
    of this build's vectors. Takes a few seconds; one at a time."""
    if not _ann_lock.acquire(blocking=False):
        _fail(409, "A measurement is already running. Wait for it to finish.")
    try:
        return _learn_call(learn.ann_check, _profile(profile).id)
    finally:
        _ann_lock.release()
