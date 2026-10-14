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

Present now (M0):

| Module | Job |
|---|---|
| `paths.py` | Every on-disk location. `RAG_STUDIO_DATA` points the engine at another folder (tests) |
| `config.py` | Model roles (one model per role) and service addresses |
| `llm.py` | The one Ollama client: version, installed models, loaded models. Chat and embed arrive in M1 |
| `health.py` | System Health: every check returns a status from one vocabulary and never raises |
| `cli.py` | Terminal views behind `./rag` |

Planned, each arriving with its milestone: `parse`, `clean`, `structure`, `guard`,
`chunk`, `context`, `embed`, `store`, `retrieve`, `rewrite`, `agent`, `rerank`,
`prompt`, `answer`, `vision`, `judge`, `eval`, `trace`.

## API surface (M0)

| Endpoint | Returns |
|---|---|
| `GET /api/status` | What the sidebar polls: is the AI up, which answer model, is another app's model loaded, document count, index built |
| `GET /api/health` | The full System Health report: sections of checks, installed and loaded models, roles |
| `GET /api/documents` | The PDF library (file and size; pages and parser views arrive in M1-M2) |

Interactive schema: http://127.0.0.1:8775/api/_docs while the app is running.

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
| `data/lancedb/` | no | One table per index profile build, plus an atomically written pointer to the active one |
| `data/cache/` | no | Parsed pages, context lines, figure descriptions, keyed by content hash |
| `data/runs/` | no | Evaluation runs with their full configuration |
| `data/chat/` | no | Conversations, one JSON file each |
| `~/.local/state/rag-studio/` | no (outside repo) | Colab token and connection, mode 0600 |

## Streaming (M1)

Answers stream as server-sent events: `stage` per completed step, `ready` with
everything code-built (sources, citation map), `delta` per piece of model text,
then `done` or `error`. Responses carry `Cache-Control: no-cache, no-transform`
and `X-Accel-Buffering: no`, and a heartbeat comment every 15 s keeps the Next
rewrite proxy (330 s idle timeout) from dropping a slow CPU answer. FastAPI
0.143 has SSE built in (`fastapi.sse.EventSourceResponse`).
