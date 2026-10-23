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
from rag_studio import (__version__, answer, colab, embed, eval as evaluation, health, indexes, ingest, learn,
                        library, paths, profiles, remote, store)
from rag_studio import chat, vision
from rag_studio import context as context_mod
from rag_studio import rerank as rerank_mod
from rag_studio import rewrite as rewrite_mod
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
                      end: int = Query(..., ge=0), profile: str = Query(profiles.DEFAULT, max_length=40),
                      chunk: str | None = Query(None, max_length=200)) -> dict:
    """Where a passage sits on the page. The baseline index stores exact
    character spans of PyMuPDF's text; a cleaned index's text differs, so its
    passage is found by aligning words (PyMuPDF) or by Docling's item boxes.
    A figure passage (M7) is model-written text with no span on the page: its
    box is the picture's own."""
    path = _doc_path(doc_id)
    prof = _profile(profile)
    if prof.figures and chunk and _FIGURE_ID.search(chunk):
        return _figure_boxes(doc_id, page, chunk)
    if end <= start:
        _fail(400, "The passage end must come after its start.")
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


_FIGURE_ID = re.compile(r":p\d+:f\d+$")


def _figure_boxes(doc_id: str, page: int, chunk: str) -> dict:
    try:
        r = vision.region(chunk)
    except (KeyError, vision.FiguresMissing):
        _fail(404, "That figure is not in the document any more. Ask the question again.")
    if r.doc_id != doc_id or r.page != page:
        _fail(404, "That figure is not on this page.")
    doc = library.clean_doc(doc_id, "docling")
    if doc is None:
        _fail(404, "Docling has not converted this document yet. Run: ./rag ingest --profile docling-clean")
    pg = doc.pages[page - 1]
    box = {"x0": r.box[0] / pg.width, "y0": r.box[1] / pg.height, "x1": r.box[2] / pg.width, "y1": r.box[3] / pg.height}
    return {"width": pg.width, "height": pg.height, "boxes": [box], "aligned": 1.0, "figure": True}


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
    """Each index profile: its embedding setup, active build's manifest (None
    until built), build history, last build job and vector match check, and
    whether it can be built here, built on Colab and queried now (with the
    reason when not). `word_index` says whether BM25 and hybrid search can run
    on it; `labels` whether its passages carry section labels."""
    return indexes.overview()


class BuildRequest(BaseModel):
    on: str = Field("local", pattern="^(local|colab)$")
    force: bool = False


# Cancel flags for builds started by this process. A build started in the
# terminal stops there (Ctrl-C).
_build_cancel: dict[str, threading.Event] = {}


@app.post("/api/indexes/{profile_id}/build", status_code=202)
def post_build(profile_id: str, req: BuildRequest) -> dict:
    """Start building one profile's index in the background. Poll
    GET /api/indexes/{id}/job for progress. Refused (409) while another build
    runs, while an evaluation uses this computer's AI, or when memory is short."""
    prof = _profile(profile_id)
    try:
        ingest.check_where(prof, req.on)
        ingest.check_memory(prof, req.on)
    except ingest.BuildRefused as e:
        _fail(409, str(e))
    if prof.contextual:
        have = _context_progress(prof)
        if have["written"] < have["passages"]:
            _fail(409, f"{have['passages'] - have['written']:,} passages have no context line yet. "
                       "Write them on Remote AI first.")
    if ingest.active_build():
        _fail(409, "Another index is being built. Wait for it to finish, or cancel it.")
    started = threading.Event()
    failure: list[str] = []
    cancel = _build_cancel[prof.id] = threading.Event()

    def work() -> None:
        def progress(stage: str, info: dict) -> None:
            started.set()
        try:
            ingest.build(prof, force=req.force, on=req.on, progress=progress, cancel=cancel.is_set)
        except Exception as e:  # noqa: BLE001 - recorded in the job file by ingest.build
            failure.append(str(e) or type(e).__name__)
        finally:
            started.set()
            _build_cancel.pop(prof.id, None)

    threading.Thread(target=work, name=f"build-{prof.id}", daemon=True).start()
    started.wait(timeout=30)
    job = ingest.last_job(prof.id)
    if failure and not (job and job.get("started_at")):
        _fail(409, failure[0])
    return {"profile": prof.id, "job": job}


@app.get("/api/indexes/{profile_id}/job")
def get_build_job(profile_id: str) -> dict:
    """The latest build job of a profile: stage, counts, and how it ended."""
    return {"profile": _profile(profile_id).id, "job": ingest.last_job(profile_id)}


@app.post("/api/indexes/{profile_id}/cancel")
def cancel_build(profile_id: str) -> dict:
    flag = _build_cancel.get(_profile(profile_id).id)
    if flag is None:
        _fail(404, "This build is not running here. A build started in the terminal stops with Ctrl-C.")
    flag.set()
    return {"profile": profile_id, "cancelling": True}


# --------------------------------------------------------------------------- context lines (M5)

_context_cancel: dict[str, threading.Event] = {}
_context_seen: dict[str, tuple[float, dict]] = {}


def _context_progress(prof: profiles.IndexProfile) -> dict:
    """How many passages have a line. Reads the cleaned documents and the cache (a few seconds), so
    a recent answer is reused for 20 s."""
    import time
    hit = _context_seen.get(prof.id)
    if hit and time.monotonic() - hit[0] < 20:
        return hit[1]
    out = context_mod.progress_for(prof)
    _context_seen[prof.id] = (time.monotonic(), out)
    return out


def _contextual(profile_id: str) -> profiles.IndexProfile:
    prof = _profile(profile_id)
    if not prof.contextual:
        _fail(400, f"{prof.label} has no context lines.")
    return prof


@app.get("/api/indexes/{profile_id}/context")
def get_context(profile_id: str) -> dict:
    """Context lines for a profile: how many exist, the last job, and whether they can be written now."""
    prof = _contextual(profile_id)
    info = remote.worker_info() if remote.connected() else None
    can = ({"ok": True, "reason": None} if info and info.get("context") else
           {"ok": False, "reason": "Needs Remote AI running the context job. Connect it in Settings."})
    return {"profile": prof.id, **_context_progress(prof), "job": context_mod.last_job(prof.id),
            "can": can, "model": context_mod.CONTEXT_MODEL}


@app.post("/api/indexes/{profile_id}/context", status_code=202)
def post_context(profile_id: str) -> dict:
    """Start writing the missing context lines on the connected runtime, in the background."""
    prof = _contextual(profile_id)
    try:
        client = remote.client()
    except remote.RemoteNotConnected as e:
        _fail(409, f"Context lines are written by Remote AI, which is not connected. {e}")
    info = remote.worker_info(client)
    if not info or not info.get("context"):
        _fail(409, "The connected runtime is not running the context job. Run the setup cell with the context job "
                   "(./rag llm colab --worker context), then connect.")
    started = threading.Event()
    failure: list[str] = []
    cancel = _context_cancel[prof.id] = threading.Event()

    def work() -> None:
        try:
            started.set()
            context_mod.write_lines(prof, client=client, cancel=cancel.is_set)
            context_mod.export(prof)
        except Exception as e:  # noqa: BLE001 - recorded in the job file by write_lines
            failure.append(str(e) or type(e).__name__)
        finally:
            started.set()
            _context_cancel.pop(prof.id, None)
            _context_seen.pop(prof.id, None)

    threading.Thread(target=work, name=f"context-{prof.id}", daemon=True).start()
    started.wait(timeout=10)
    import time
    time.sleep(0.5)
    job = context_mod.last_job(prof.id)
    if failure and not (job and job.get("status") == "running"):
        _fail(409, failure[0])
    return {"profile": prof.id, "job": job}


@app.post("/api/indexes/{profile_id}/context/cancel")
def cancel_context(profile_id: str) -> dict:
    flag = _context_cancel.get(_contextual(profile_id).id)
    if flag is None:
        _fail(404, "Context lines are not being written here. A job started in the terminal stops with Ctrl-C.")
    flag.set()
    return {"profile": profile_id, "cancelling": True}


_check_lock = threading.Lock()


@app.post("/api/indexes/{profile_id}/check")
def post_match_check(profile_id: str) -> dict:
    """The vector match check: 50 passages and 5 questions embedded one per
    request on this computer and on the Colab runtime, each machine's noise
    floor (itself in another order), and the stored vectors. Up to four
    minutes on this CPU (Qwen3 0.6B embeds about 0.6 passages a second)."""
    prof = _profile(profile_id)
    building = ingest.active_build()
    if building and building.get("on", "local") == "local":
        _fail(409, "An index is being built on this computer. Check again when it finishes.")
    if not _check_lock.acquire(blocking=False):
        _fail(409, "A check is already running. Wait for it to finish.")
    try:
        return _learn_call(embed.match_check, prof.id)
    finally:
        _check_lock.release()


# --------------------------------------------------------------------------- remote AI (M4)
#
# The browser half of `./rag llm`: every route calls rag_studio.remote, the
# same functions the CLI runs. A credential travels browser -> API, never API
# -> browser: the engine never returns the token, and _no_secrets strips any
# key that could hold one anyway.

_SECRET_KEYS = {"token", "authorization", "bearer", "secret", "password"}


def _no_secrets(node):
    if isinstance(node, dict):
        return {k: _no_secrets(v) for k, v in node.items() if str(k).lower() not in _SECRET_KEYS}
    if isinstance(node, list):
        return [_no_secrets(v) for v in node]
    return node


def _llm_snapshot(probe: bool = False) -> dict:
    return _no_secrets({"connection": remote.connection(), "remote": remote.probe() if probe else None})


@app.get("/api/llm/status")
def get_llm_status(probe: bool = Query(False, description="also ask the runtime for its GPU and models")) -> dict:
    """The connection, and with probe=true the runtime's GPU, models and where
    each is loaded (one request through the tunnel)."""
    return _llm_snapshot(probe and remote.connected())


@app.get("/api/llm/colab")
def get_colab_cell(models: str | None = Query(None, max_length=300),
                   worker: str | None = Query(None, max_length=200)) -> dict:
    """The setup cell to paste into Colab. Tokenless on purpose: the runtime
    makes its own token and hands it back inside the connect code, which is
    pasted into Connect. `./rag llm colab` is the variant with a remembered
    token; it prints to a terminal, not into an HTTP response."""
    jobs = worker.split(",") if worker else []
    wanted = models.split() if models else ([] if jobs else list(colab.DEFAULT_MODELS))
    try:
        version = OllamaClient().version()
    except OllamaUnavailable:
        version = None
    try:
        cell = colab.build_cell(models=wanted, token=None, ollama_version=version, worker=jobs)
    except ValueError as e:
        _fail(400, str(e))
    except colab.MissingPayload as e:
        _fail(500, str(e))
    return {"cell": cell, "models": wanted, "worker": jobs, "ollama_version": version,
            "files": list(colab.PAYLOAD_FILES), "tokenless": True,
            "cli_equivalent": "./rag llm colab --copy", **colab.cell_stats(cell)}


class ConnectRequest(BaseModel):
    code: str = Field("", max_length=remote.MAX_CODE_CHARS)
    url: str = Field("", max_length=remote.MAX_URL_CHARS)
    token: str = Field("", max_length=remote.TOKEN_CHARS[1])


@app.post("/api/llm/connect")
def post_llm_connect(req: ConnectRequest) -> dict:
    """Connect to a Colab runtime with its connect code (or a URL), then test
    it with a real request. The same checks, in the same order, as
    `./rag llm connect`. A failed test still saves the connection."""
    try:
        result = remote.connect(code=req.code, url=req.url, token=req.token)
    except ValueError as e:
        _fail(400, str(e))
    return _no_secrets({**result, "connection": remote.connection()})


@app.post("/api/llm/test")
def post_llm_test() -> dict:
    """A real request: embed one sentence with each model the runtime serves."""
    result = remote.smoke_test()
    return _no_secrets({**result, **_llm_snapshot(probe=bool(result["ok"]))})


class DisconnectRequest(BaseModel):
    forget_token: bool = False


@app.post("/api/llm/disconnect")
def post_llm_disconnect(req: DisconnectRequest | None = None) -> dict:
    """Back to this computer only."""
    remote.disconnect(forget=bool(req and req.forget_token))
    return {"ok": True, **_llm_snapshot()}


# --------------------------------------------------------------------------- ask

SEARCH_PATTERN = "^(dense|bm25|hybrid)$"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(answer.DEFAULT_K, ge=1, le=20)
    profile: str = Field(profiles.DEFAULT, max_length=40)
    prompt: str = Field(DEFAULT_VERSION, max_length=40)
    search: str = Field("dense", pattern=SEARCH_PATTERN)
    section_cap: int | None = Field(None, ge=1, le=10)
    rerank: str | None = Field(None, max_length=40)
    rerank_depth: int = Field(rerank_mod.DEFAULT_DEPTH, ge=2, le=50)
    rewrite: str | None = Field(None, max_length=20)
    agent_rounds: int | None = Field(None, ge=2, le=6)
    # A conversation: omit for a new one (the answer says its id). `as_typed` skips the follow-up rewrite.
    chat_id: str | None = Field(None, max_length=32)
    as_typed: bool = False


def _search_spec(method: str, section_cap: int | None, rerank: str | None, rerank_depth: int,
                 rewrite: str | None = None, agent_rounds: int | None = None) -> SearchSpec:
    """The request's switches as a search spec; a switch that is not valid is a 400."""
    try:
        return SearchSpec(method, section_cap, rerank=rerank, rerank_depth=rerank_depth, rewrite=rewrite,
                          agent_rounds=agent_rounds)
    except ValueError as e:
        _fail(400, str(e))


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
    prof = _profile(req.profile)
    prompt = _prompt(req.prompt)
    spec = _search_spec(req.search, req.section_cap, req.rerank, req.rerank_depth, req.rewrite, req.agent_rounds)
    chat_id = req.chat_id
    past: list[dict[str, str]] | None = None
    if chat_id:
        try:
            if chat.load(chat_id) is None:
                _fail(404, "No such conversation. It may have been deleted.")
            past = chat.history(chat_id) or None
        except chat.BadChatId as e:
            _fail(400, str(e))
    else:
        chat_id = chat.new_id()
    setup = {"profile": prof.id, "prompt": prompt, "search": spec.describe(), "k": req.k}

    def events():
        for name, data in answer.ask(req.question, k=req.k, profile_id=prof.id, prompt_version=prompt, search=spec,
                                     history=past, as_typed=req.as_typed):
            if name in ("ready", "done", "error"):
                data = {**data, "chat_id": chat_id}
            if name == "done":
                try:
                    chat.add_turn(chat_id, data, setup)
                except (ValueError, KeyError, TypeError, OSError) as e:
                    # An answer is never lost because its thread could not be saved: say so beside it.
                    data = {**data, "chat_warning": str(e) if isinstance(e, ValueError)
                            else "This answer could not be saved to the conversation."}
            yield name, data

    return sse.stream(events())


@app.get("/api/rerankers")
def get_rerankers() -> dict:
    """The rerankers, where each runs, and whether it can run now."""
    from rag_studio import rewrite
    return {"depth": rerank_mod.DEFAULT_DEPTH, "rerankers": rerank_mod.availability(),
            "rewrite": rewrite.availability()}


@app.get("/api/chats")
def get_chats() -> dict:
    return {"chats": chat.list_chats()}


@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: str) -> dict:
    try:
        convo = chat.load(chat_id)
    except chat.BadChatId as e:
        _fail(404, str(e))
    if convo is None:
        _fail(404, "No such conversation. It may have been deleted.")
    return convo


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str) -> dict:
    try:
        gone = chat.delete(chat_id)
    except chat.BadChatId as e:
        _fail(404, str(e))
    if not gone:
        _fail(404, "No such conversation. It may have been deleted.")
    return {"id": chat_id, "deleted": True}


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
    rerank: str | None = Field(None, max_length=40)
    rerank_depth: int = Field(rerank_mod.DEFAULT_DEPTH, ge=2, le=50)
    rewrite: str | None = Field(None, max_length=20)
    agent_rounds: int | None = Field(None, ge=2, le=6)
    as_typed: bool = False


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
                              search=_search_spec(req.search, req.section_cap, req.rerank, req.rerank_depth,
                                                  req.rewrite, req.agent_rounds).as_args(), as_typed=req.as_typed)
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
        busy = "already in progress" in failure[0] or "being built" in failure[0]
        _fail(409 if busy else 400, failure[0])
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
    except rerank_mod.RerankError as e:
        _fail(503, str(e))
    except rewrite_mod.RewriteUnavailable as e:
        _fail(503, str(e))
    except rewrite_mod.RewriteBadOutput as e:
        _fail(502, f"Remote AI's answer was not usable: {e} Try again.")
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
                   cap: int | None = Query(None, ge=1, le=10),
                   rerank: str | None = Query(None, max_length=40)) -> dict:
    """Dense, BM25 and hybrid lists for one question, for the search lab; with
    `rerank`, also the dense top 20 re-ordered by that reranker."""
    return _learn_call(learn.search_lab, q, _profile(profile).id, k, cap, rerank)


@app.get("/api/learn/embed")
def get_embed_lab(profile: str = Query(profiles.DEFAULT, max_length=40),
                  q: str | None = Query(None, max_length=500),
                  mix: str | None = Query(None, max_length=40)) -> dict:
    """An index's passages as points (PCA of their vectors); with a question,
    where it lands, its nearest passages, Matryoshka cuts, and (with `mix`)
    the same question embedded by another index's model, searched here."""
    return _learn_call(learn.embed_lab, _profile(profile).id, (q or "").strip() or None,
                       _profile(mix).id if mix else None)


@app.get("/api/learn/context")
def get_context_lab(profile: str = Query(learn.CONTEXT_DEFAULT, max_length=60),
                    question: str | None = Query(None, pattern=r"^[A-Z][0-9]{1,2}$"),
                    chunk: str | None = Query(None, max_length=120, pattern=r"^[a-z0-9_]+:p[0-9]{1,4}:c[0-9]{1,4}$")
                    ) -> dict:
    """Step 4: a passage, its context line and the text embedded with and
    without it; with a test question id, where its answer ranks on the plain
    and the -ctx index. Reads the context-line cache only: no model call."""
    return _learn_call(learn.context_lab, _profile(profile).id, question, chunk)


_FIGURE_PATTERN = r"^[a-z0-9_]+:p[0-9]{1,4}:f[0-9]{1,3}$"


@app.get("/api/learn/figures")
def get_figures_lab(figure: str | None = Query(None, max_length=120, pattern=_FIGURE_PATTERN)) -> dict:
    """Step 11: one figure before (the text a search already had) and after (the vision model's
    description and the passage it becomes; for a Set C figure, every value it read beside the value
    measured from the PDF's drawing). Reads the description cache only: no model call."""
    return _learn_call(learn.figures_lab, figure)


@app.get("/api/figures/{figure_id}/image")
def get_figure_image(figure_id: str) -> Response:
    """A figure region as the vision model saw it (PNG, drawn from the PDF)."""
    if not re.match(_FIGURE_PATTERN, figure_id):
        _fail(400, "A figure is named document:p<page>:f<n>.")
    try:
        png, _ = vision.image_for(vision.region(figure_id))
    except (KeyError, vision.FiguresMissing):
        _fail(404, "That figure is not in the documents.")
    return Response(png, media_type="image/png", headers={"Cache-Control": "private, max-age=86400"})


@app.get("/api/learn/rewrite")
def get_rewrite_lab(question: str = Query(..., pattern=r"^[A-Z][0-9]{1,2}$"),
                    profile: str = Query(profiles.DEFAULT, max_length=60)) -> dict:
    """Step 8: one test question as each rewriting technique and the
    multi-step search searched it, replayed from recorded runs (no model call)."""
    return _learn_call(learn.rewrite_lab, question, _profile(profile).id)


@app.get("/api/learn/rewrite/live")
def get_rewrite_live(q: str = Query(..., min_length=1, max_length=500),
                     technique: str = Query(..., pattern="^(decompose|multi|hyde|agent)$"),
                     profile: str = Query(profiles.DEFAULT, max_length=60)) -> dict:
    """Step 8, live: one question searched now with one technique. Needs
    Remote AI; a rewrite made before for the same words is replayed."""
    return _learn_call(learn.rewrite_live, q, _profile(profile).id, technique)


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


# --------------------------------------------------------------------------- how it works (M6)

QUESTION_ID = r"^[A-Z][0-9]{1,2}$"


def _run_id(run_id: str) -> str:
    if not _RUN_ID.match(run_id):
        _fail(404, "No such run. It may have been deleted.")
    return run_id


@app.get("/api/learn/answers")
def get_recorded_answers(question: str = Query(..., pattern=QUESTION_ID)) -> dict:
    """Steps 9 and 10: the runs that answered one test question, the default setup first."""
    return _learn_call(learn.recorded_answers, question)


@app.get("/api/learn/prompt")
def get_prompt_lab(run: str = Query(..., max_length=40), question: str = Query(..., pattern=QUESTION_ID)) -> dict:
    """Step 9: the exact messages of a recorded answer, its context budget, and
    prompts v2, v3 and v4 for the same passages. Read from the run: no model call."""
    return _learn_call(learn.prompt_lab, _run_id(run), question)


@app.get("/api/learn/cite")
def get_cite_lab(run: str = Query(..., max_length=40), question: str = Query(..., pattern=QUESTION_ID)) -> dict:
    """Step 10: a recorded answer's citations mapped to pages, the refusal rule,
    and the same passages answered by the other answer models."""
    return _learn_call(learn.cite_lab, _run_id(run), question)


class AnswerCheckRequest(BaseModel):
    run: str = Field(max_length=40)
    question: str = Field(pattern=QUESTION_ID)
    text: str = Field(max_length=4000)


@app.post("/api/learn/answer-check")
def post_answer_check(req: AnswerCheckRequest) -> dict:
    """Steps 10 and 12: an edited answer read by the same citation, refusal and
    scoring code as a recorded one. Changes nothing."""
    return _learn_call(learn.answer_check, _run_id(req.run), req.question, req.text)


@app.get("/api/learn/evaluate")
def get_evaluate_lab(run: str = Query(..., max_length=40), against: str | None = Query(None, max_length=40)) -> dict:
    """Step 12: one run's scorecard taken apart; with `against`, the sign test
    on the questions that changed between the two runs."""
    return _learn_call(learn.evaluate_lab, _run_id(run), _run_id(against) if against else None)


def _safety_source(source: str) -> str:
    if source != learn.CRAFTED:
        _doc_path(source)
    return source


@app.get("/api/learn/safety")
def get_safety_lab(source: str = Query(learn.CRAFTED, max_length=100), page: int = Query(1, ge=1)) -> dict:
    """Step 13: one page through the guard (the crafted test page, or a real
    page where hidden text was dropped). The first view of a page draws it
    twice; later views come from memory."""
    return _learn_call(learn.safety_lab, _safety_source(source), page)


@app.get("/api/learn/safety/pages")
def get_safety_pages() -> dict:
    """Step 13: the crafted page and every real page where hidden text was dropped."""
    return _learn_call(learn.safety_pages)


@app.get("/api/learn/safety/image")
def get_safety_image(source: str = Query(learn.CRAFTED, max_length=100), page: int = Query(1, ge=1),
                     view: str = Query(..., pattern="^(page|no_text|change)$")) -> Response:
    """One of the guard's three images of a page, as PNG."""
    png = _learn_call(learn.safety_image, _safety_source(source), page, view)
    return Response(png, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


@app.get("/api/learn/safety/flags")
def get_safety_flags(text: str = Query(..., min_length=1, max_length=2000)) -> dict:
    """Any text through the guard's instruction check."""
    return learn.flag_check(text)


_live_lock = threading.Lock()


@app.post("/api/learn/safety/live")
def post_safety_live(version: str = Query(..., pattern="^(v2|v3)$")) -> dict:
    """Step 13, live: the crafted page as the only passage, answered now by the
    Answer role's model with prompt v2 or v3. One at a time."""
    if not _live_lock.acquire(blocking=False):
        _fail(409, "An answer is already being written. Wait for it to finish.")
    try:
        return _learn_call(learn.safety_live, version)
    except learn.LiveFailed as e:
        _fail(503, str(e))
    finally:
        _live_lock.release()
