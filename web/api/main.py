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
from rag_studio import __version__, answer, eval as evaluation, health, library, paths, profiles, store
from rag_studio.config import MODEL_NOTES, role_model
from rag_studio.llm import OllamaClient, OllamaUnavailable
from rag_studio.parse import file_sha256, passage_boxes, render_page_png

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
                      end: int = Query(..., ge=1)) -> dict:
    """Where a passage (a span of the page's clean text) sits on the page."""
    if end <= start:
        _fail(400, "The passage end must come after its start.")
    try:
        return passage_boxes(_doc_path(doc_id), page, start, end)
    except IndexError as e:
        _fail(404, str(e))


@app.get("/api/indexes")
def get_indexes() -> dict:
    """Each index profile and its active build's manifest (None until built)."""
    return {"profiles": [
        {"id": p.id, "label": p.label, "description": p.description,
         "active": store.active_manifest(p.id), "builds": len(store.builds(p.id))}
        for p in profiles.PROFILES.values()
    ]}


# --------------------------------------------------------------------------- ask

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(answer.DEFAULT_K, ge=1, le=20)


@app.post("/api/ask")
def post_ask(req: AskRequest):
    """One question, answered as server-sent events (see rag_studio/answer.py
    for the event contract): stage, ready, delta, done or error."""
    return sse.stream(answer.ask(req.question, k=req.k))


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


# Cancel flags for runs started by this process. A run started from the
# terminal is cancelled there (Ctrl-C).
_cancel: dict[str, threading.Event] = {}
_start_lock = threading.Lock()


@app.post("/api/eval/runs", status_code=202)
def post_run(req: RunRequest) -> dict:
    """Start a run in the background and return its id at once. The run saves
    itself after every question; poll GET /api/eval/runs/{id} for progress."""
    spec = evaluation.RunSpec(sets=req.sets, mode=req.mode, k=req.k, model=req.model,
                              label=req.label)
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

