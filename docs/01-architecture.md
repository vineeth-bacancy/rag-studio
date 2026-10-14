# Architecture

## Processes

```
 Browser ──► Next.js UI 127.0.0.1:3100 ──/api/*──► FastAPI 127.0.0.1:8775 ──► rag_studio (engine)
                (renders only)          rewrite          (routes only)              │
                                                                                    ├──► Ollama 127.0.0.1:11434 (CPU)
                                                                                    ├──► LanceDB folder data/lancedb/
                                                                                    └──► Colab gateway over HTTPS (M4, optional)
```

Everything binds 127.0.0.1. The browser only ever talks to the UI's origin; Next
rewrites `/api/*` to the API, so there is no cross-origin request and no CORS
surface in normal use. The Colab gateway is reached only by the Python process,
never by the browser.

## The one rule

**The engine computes; the API routes; the UI renders.** Every parse, chunk,
rank, score and citation is produced by `rag_studio/`. `web/api/main.py` only
validates input, calls the engine and returns JSON. `web/ui/src/lib/api.ts` is
the only place the UI fetches data, and it holds no RAG logic. A number cannot
be right in the terminal (`./rag …`) and wrong in the browser, because there is
only one implementation.

## Engine modules

Present now (M1). Each RAG step has a doc in `docs/steps/`.

| Module | Job |
|---|---|
| `paths.py` | Every on-disk location. `RAG_STUDIO_DATA` points the engine at another folder (tests) |
| `config.py` | Model roles (one model per role), service addresses, measured notes per model |
| `files.py` | Atomic writes: temp file in the same folder, fsync, `os.replace` |
| `llm.py` | The one Ollama client: version, installed and loaded models, `embed` (`truncate: false`), streamed `chat` |
| `parse.py` | PyMuPDF extraction per 1-based page; page render to PNG; a passage's word boxes on its page |
| `clean.py` | NFKC (M2 adds the full cleaning order) |
| `chunk.py` | The baseline fixed-size chunker, defect kept; each passage stores its character span |
| `profiles.py` | Index profiles: parser + cleaner + chunker + embedding model, prefixes, dimensions, metric |
| `embed.py` | The embedder for a profile; refuses a missing model or a different digest |
| `store.py` | LanceDB: one table per build, atomic pointer switch, exact cosine search, build history |
| `ingest.py` | parse → clean → chunk → embed → store, incremental by content hash, a manifest per build |
| `retrieve.py` | Opens a profile's active build; refuses a build made with another setup; dense search |
| `prompt.py` | Versioned system prompts (v2, v2-handbuilt) with sha256; the context block; truncation check |
| `answer.py` | The one answering path: events stage → ready → delta → done/error; citations built by code |
| `trace.py` | The in-app trace: each finished step with its duration |
| `eval.py` | Test-question sets, the scorer, runs (saved after every question, one at a time), the ladder |
| `library.py` | The PDFs, their titles, and what the active index holds of each |
| `health.py` | System Health: every check returns a status from one vocabulary and never raises |
| `cli.py` | Terminal views behind `./rag`: health, ingest, eval |

Planned, each arriving with its milestone: `structure`, `guard`, `context`,
`rewrite`, `agent`, `rerank`, `vision`, `judge`.

## API surface (M1)

| Endpoint | Returns |
|---|---|
| `GET /api/status` | What the sidebar polls: AI up, answer model, another app's model loaded, documents, the active index |
| `GET /api/health` | The full System Health report |
| `GET /api/models` | Installed answer models with measured notes |
| `GET /api/documents` | The PDF library: titles, pages, passages, near-empty pages |
| `GET /api/documents/{doc}/pages/{n}/image?dpi=` | The page as PNG, cached by the PDF's content hash |
| `GET /api/documents/{doc}/pages/{n}/boxes?start=&end=` | A passage's lines on its page, in 0..1 coordinates |
| `GET /api/indexes` | Each profile and its active build's manifest |
| `POST /api/ask` | One question as server-sent events (`web/api/sse.py`) |
| `GET /api/golden` | The test-question sets |
| `GET /api/eval/runs`, `GET /api/eval/runs/{id}` | Run history; one run with every question |
| `POST /api/eval/runs`, `POST /api/eval/runs/{id}/cancel` | Start a run in the background (409 if one is running); cancel it after the current question |
| `GET /api/eval/ladder` | The 11 ladder rows, filled from tagged runs |

Interactive schema: http://127.0.0.1:8775/api/_docs while the app is running.

A document is named only by `doc_id` and resolved against the files on disk
(`library.pdf_path`); a run only by an id matching `YYYYMMDD-HHMMSS-xxxx`. No
path a browser sends is ever opened.

## Security model

The model is *reachability*, as in the trading app:

1. **Loopback only.** Both servers bind 127.0.0.1 (`--host`/`--hostname` in
   every launch path; `tests/test_ui_contract.py` fails if one stops). Next
   binds 0.0.0.0 by default, which would make it an open proxy to the API.
2. **The request guard** (`web/api/guard.py`, pure ASGI, outermost):
   - every request must name a loopback host in both `Host` and
     `X-Forwarded-Host` (DNS rebinding);
   - every non-GET request must not be cross-site (`Sec-Fetch-Site`, `Origin`).
   `tests/test_api_guard.py` sends each attack.
3. **Credentials travel browser → API, never API → browser** (M4). The Colab
   token is stored at `~/.local/state/rag-studio/` with mode 0600, outside the
   repository.
4. **PDF text is data, never instructions** (M2): hidden text is dropped,
   instruction-like passages are flagged, and passages are wrapped as data in the
   prompt.
5. **Nothing leaves the machine** except calls to your own Colab runtime.
   Tracing (M6) is self-hosted on loopback. Logs hold sizes and counts, never
   document or answer content.

## Data on disk

| Path | Committed | Contents |
|---|---|---|
| `data/pdfs/` | yes | The source PDFs |
| `data/golden/` | yes | Test questions with verified answer pages |
| `data/lancedb/` | no | One LanceDB table per index build |
| `data/indexes/` | no | Each profile's active-build pointer (= its manifest), written atomically, and the build history |
| `data/cache/` | no | Parsed pages, context lines, figure descriptions, keyed by content hash |
| `data/runs/` | no | Evaluation runs with their full configuration |
| `data/chat/` | no | Conversations, one JSON file each |
| `~/.local/state/rag-studio/` | no (outside repo) | Colab token and connection, mode 0600 |

## Streaming

Answers stream as server-sent events: `stage` per completed step, `ready` with
everything code-built (sources, the prompt budget), `delta` per piece of model
text, then `done` or `error`. `web/api/sse.py` runs the engine generator in a
thread and writes `Cache-Control: no-cache, no-transform`, `X-Accel-Buffering:
no` and a `: ping` comment after 15 s of silence; the Next rewrite proxy
(330 s idle timeout) neither gzips nor drops it. FastAPI's own
`EventSourceResponse` is not used: it sets `Cache-Control: no-cache` itself and
a route cannot replace it, only add a second header. Measured in the browser:
the sources arrive within 0.5 s, before the model starts.

Evaluation runs do not stream. A run saves itself after every question and the
page polls it every 2 s while it runs, so a reload or a second tab shows the
same progress.
