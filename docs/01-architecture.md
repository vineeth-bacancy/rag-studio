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

Present now (M4). Each RAG step has a doc in `docs/steps/`.

| Module | Job |
|---|---|
| `paths.py` | Every on-disk location. `RAG_STUDIO_DATA` points the engine at another folder (tests) |
| `config.py` | Model roles (one model per role), service addresses, measured notes per model |
| `files.py` | Atomic writes: temp file in the same folder, fsync, `os.replace` |
| `llm.py` | The one Ollama client, local or remote (the Colab gateway, with a bearer token and its own timeouts and error wording): version, installed and loaded models, the GPU probe, `embed` (`truncate: false`, options), streamed `chat` |
| `remote.py` | Remote AI: the connection and the remembered token (0600, outside the repo), the `rsc1.` connect code, connect checks, probe, smoke test, disconnect. The CLI and the API both call it |
| `colab.py` | The Colab setup cell, built from `tools/colab/server.py` and `start.sh` (gzip + base64, sha256-checked), with or without a token |
| `parse.py` | PyMuPDF: plain text per 1-based page (baseline); the layout reader (lines, characters, invented spaces) whose lines rebuild `get_text()` exactly; ligature glyph names read from the fonts; bookmarks; page render; a passage's boxes on its page (exact span, or aligned by words for cleaned text) |
| `parse_docling.py` | Docling: converts a PDF once (cached by content and options), then its body text through the cleaning rules, reusing the PyMuPDF path's hidden text, page labels and hyphen decisions |
| `guard.py` | The injection guard: hidden text found by drawing the page with and without its text; instruction-like phrases flagged |
| `clean.py` | NFKC (baseline) and the full order: hidden text, NFKC, "Th" glyphs, footnote markers, letter spacing, headers and footers, dot leaders, hyphens, whitespace. Every change is a mark with the original text |
| `structure.py` | Section paths (bookmarks, then headings by font size), printed page labels, contents pages and reference lists tagged |
| `chunk.py` | The baseline fixed-size chunker (defect kept) and its word-boundary fix; structure-aware passages (paragraphs and tables packed within one section) and semantic passages (cut where neighbouring sentences stop being similar), both with a section label; recursive and sentence chunkers for the playground. Every passage is an exact slice of its page |
| `profiles.py` | Index profiles: parser + cleaner + chunker (+ section labels) + embedding model, prefixes, dimensions, options, metric; where each builds and is queried (policy, not config); measured RAM needs and embedding rates |
| `embed.py` | The embedder for a profile, on Ollama (this computer or the Colab runtime) or sentence-transformers in this process (EmbeddingGemma 2, pinned revision, loaded only with 2.2 GB free); refuses a missing model, a different digest, or a Colab index with no runtime. The vector cache (per model digest, prefix, options and machine) that semantic chunking and resumable builds share. The vector match check |
| `store.py` | LanceDB: one table per build, atomic pointer switch, exact cosine search, the word index (native full-text, BM25) over exactly what was embedded, build history |
| `ingest.py` | parse → clean (and guard) → chunk → embed → store → word index, incremental by content hash (of the section label and the text), a manifest per build; cleaned documents cached by file, parser, cleaner and the cleaning code's hash; builds made before M3 get their word index in place. Builds on this computer or on Colab (`on`), resumable, one at a time (a lock), never beside an evaluation, refused when memory is short; a job file per profile for the Indexes page |
| `indexes.py` | The Indexes page's facts: per profile, can it build here, on Colab, can a question be embedded now (each with the reason), its job, its match check, measured time estimates |
| `retrieve.py` | Opens a profile's active build; refuses a build made with another setup; dense, BM25 and hybrid search (our own RRF, K = 60, 50 deep per list); the optional section cap. `run()` is **the one retrieval pipeline** Ask, evaluation and How it works share (M5): follow-up rewrite → query rewriting → search each query → fuse → limit per section → multi-step rounds → rerank the top 20 → cut to k. Each switch adds its stage only when on, so a plain search is byte-identical to M4 |
| `rerank.py` | The rerankers (Qwen3-Reranker 0.6B here; 0.6B, 4B and 8B 8-bit on Remote AI), pinned by Hugging Face revision, one recorded instruction; the load guard, the NaN guard, the inference lock; scores remembered in `data/cache/rerank/` so a scorecard replays a retrieval run's scores |
| `rewrite.py` | Query rewriting on Remote AI only (qwen3:8b): split into parts, several phrasings, draft answer first, and the follow-up rewrite; versioned prompts with JSON-schema output; junk output is an error |
| `agent.py` | Multi-step search: after round 1, the rewrite role answers or searches again (a new query, optionally inside one validated document) until enough evidence, the round, time or token budget, a repeated query or bad output |
| `remote_cache.py` | Every Remote AI output (rewrites, follow-ups, round decisions), keyed by model digest, prompt hash, num_ctx and input, with the order it was generated in: a retrieval run and its scorecard see the same rewrites |
| `context.py` | Contextual retrieval: windows (same-section blocks of at most 6,000 tokens), prompt context-v1, the line cache, the resumable Remote AI job (2 batches in flight, a timed-out batch asked again), the export to `data/context/` |
| `vision.py` | Figures as text (M7): regions from Docling's picture items (widened to a title printed above, panel-title sections dropped), PNGs drawn from the PDF and identified by hash, prompt vision-v1, the description cache (bad outputs kept, so a rebuild replays exactly), the resumable Remote AI job (one per model), figure passages for the `-fig` index (photos and logos left out by the model's Kind line), export to `data/figures/`, and `truth_check` against Set C's values measured from the drawing |
| `chat.py` | Conversations in `data/chat/<32 hex>.json`, written atomically; memory acts only through the follow-up rewrite |
| `prompt.py` | Versioned system prompts (v2, v2-handbuilt, v3 with passages fenced as data, v4 with rule 4 relaxed for labelled passages) with sha256; the context block (with the section label when a passage has one); truncation check |
| `answer.py` | The one answering path: events stage → ready → delta → done/error; the guard step; citations built by code; a figure source says which model described it; the memory check before a local answer model loads (M7) |
| `trace.py` | The in-app trace: each finished step with its duration |
| `eval.py` | Test-question sets, the scorer, runs (saved after every question, one at a time; two passes when a reranker or rewrite is on: search every question, release the search models, then answer from the same passages), MRR and recall@20, Set E's follow-up chain, the ladder with variants, two-run comparison |
| `library.py` | The PDFs, their titles, what the active index holds of each, and the page inspector's data |
| `learn.py` | How it works: the chunk playground, the search lab (three lists side by side), the vector store's tables, versions and indexes, the exact-vs-ANN recall check on a scratch copy, the embedding lab (PCA map, Matryoshka cuts, mixed models), the search lab's reranked column, context lines (step 4) and rewriting replayed from runs or run live (step 8); M6: a recorded answer replayed from its run's own index build (steps 9 and 10: exact messages, budget, citations, an editable answer read by the same code), a run's scorecard taken apart with a Wilson interval and a sign test (step 12), and the guard on the crafted page and the real hidden-text pages, with a live v2/v3 demo through `answer.ask` (step 13); M7: one figure before and after, with every value read beside the value measured from the drawing and Set C's recorded answers (step 11) |
| `health.py` | System Health: every check returns a status from one vocabulary and never raises |
| `cli.py` | Terminal views behind `./rag`: health, ingest, index, llm, context, figures, rerank, eval |

Cut from this version (2026-10-22, `docs/plans/m6-m8.md`): the `judge`,
RAGAS/DeepEval, tracing export, figure image embeddings (M7 is vision
descriptions only) and answer models on Colab. Deferred to a later version:
upload from the browser (add a PDF by copying it into `data/pdfs/` and
rebuilding the index) and editing test questions in the browser. Outside the
package: `tools/colab/` (the gateway, the GPU worker and the start script, which
run on the Colab runtime; docs/02-remote-gpu.md).

## API surface

| Endpoint | Returns |
|---|---|
| `GET /api/status` | What the sidebar polls: AI up, answer model, another app's model loaded, documents, the active index |
| `GET /api/health` | The full System Health report |
| `GET /api/models` | Installed answer models with measured notes |
| `GET /api/prompts` | The system prompt versions, with their full text and sha256 |
| `GET /api/documents` | The PDF library: titles, pages, passages, near-empty pages |
| `GET /api/documents/{doc}/pages/{n}/image?dpi=` | The page as PNG, cached by the PDF's content hash |
| `GET /api/documents/{doc}/pages/{n}/boxes?start=&end=&profile=&chunk=` | A passage's lines on its page, in 0..1 coordinates: an exact span (baseline), words aligned (PyMuPDF + cleaning) or Docling's item boxes |
| `GET /api/documents/{doc}/pages/{n}/inspect` | The page inspector: raw and clean text with every rule's marks, headers and footers, hidden text, sections, kinds, for PyMuPDF and (once converted) Docling |
| `GET /api/documents/{doc}/cleaning` | Per-document counts per rule and parser, and the pages where a rule acted |
| `GET /api/indexes` | Each profile, its active build's manifest and history, whether it has a word index and section labels, its embedding setup, whether it can build here / on Colab / be queried now (with reasons), its last job and match check; Remote AI, free memory, the active build; the search methods |
| `POST /api/indexes/{id}/build`, `GET …/job`, `POST …/cancel` | Start a build in the background (`on`: local or colab; 409 when refused), follow it, cancel it (vectors kept) |
| `POST /api/indexes/{id}/check` | The vector match check (up to two minutes) |
| `GET /api/llm/status?probe=` | The Colab connection (never the token); with `probe`, the runtime's GPU, models and where each is loaded |
| `GET /api/llm/colab` | The tokenless setup cell |
| `POST /api/llm/connect`, `POST /api/llm/test`, `POST /api/llm/disconnect` | Connect with a connect code or URL and test with a real request; test again; back to this computer |
| `POST /api/ask` | One question as server-sent events (`web/api/sse.py`); `profile`, `prompt`, `search` (dense, bm25, hybrid), `section_cap`, `rerank`, `rewrite`, `agent_rounds` choose the setup; `chat_id` continues a conversation and `as_typed` skips the follow-up rewrite |
| `GET /api/rerankers` | Each reranker, where it runs and whether it can run now; whether Remote AI can rewrite now |
| `GET /api/chats`, `GET /api/chats/{id}`, `DELETE /api/chats/{id}` | Saved conversations (ids are 32 hex characters; anything else is a 404) |
| `GET /api/golden` | The test-question sets |
| `GET /api/eval/runs`, `GET /api/eval/runs/{id}` | Run history; one run with every question |
| `POST /api/eval/runs`, `POST /api/eval/runs/{id}/cancel` | Start a run in the background (409 if one is running), with the same setup fields as Ask; cancel it after the current question |
| `GET /api/eval/ladder` | The 11 ladder rows, each with its variants, filled from tagged runs |
| `GET /api/eval/compare?a=&b=` | Two runs side by side: setup differences and the questions whose outcome or retrieval changed |
| `GET /api/learn/chunk?doc_id=&page=&parser=&method=&size=&overlap=&percentile=` | The chunk playground: one cleaned page cut by any chunker; semantic adds the similarity curve |
| `GET /api/learn/search?q=&profile=&k=&cap=&rerank=` | The search lab: dense, BM25 and hybrid lists with each passage's rank in the others, and the first answer page's rank for a test question; with `rerank`, the dense top 20 reranked |
| `GET /api/learn/context?profile=&question=&chunk=` | Step 4: a passage, its context line, the exact text embedded with and without it, coverage, and a test question's rank on the plain and the `-ctx` index (reads the cache only) |
| `GET /api/learn/rewrite?question=&profile=` | Step 8: one test question as each rewriting technique searched it, replayed from recorded runs |
| `GET /api/learn/rewrite/live?q=&technique=&profile=` | Step 8, live: one question searched now with one technique (needs Remote AI) |
| `GET /api/indexes/{id}/context`, `POST …/context`, `POST …/context/cancel` | Context lines for a `-ctx` profile: progress, write the missing ones on Remote AI, cancel |
| `GET /api/learn/store?profile=` | A build's LanceDB table: columns, a sample row, versions, indexes, size, other builds, the last ANN check |
| `POST /api/learn/store/ann-check?profile=` | Measure IVF_FLAT, IVF_PQ and IVF_HNSW_SQ against exact search on a scratch copy (recall@10, latency) |
| `GET /api/learn/embed?profile=&q=&mix=` | The embedding lab: every passage as a 2-D point (PCA), a question's landing point and true nearest passages, Matryoshka cuts, and the question embedded by another index's model |
| `GET /api/learn/answers?question=` | Steps 9 and 10 (M6): the runs that answered a test question, the app's defaults first; runs whose build was pruned are left out |
| `GET /api/learn/prompt?run=&question=` | Step 9: the exact messages rebuilt from the run's own build, the budget, and v2/v3/v4 for the same passages with measured outcomes |
| `GET /api/learn/cite?run=&question=` | Step 10: citations mapped to pages, the refusal rule, timings, the same passages answered by other answer models |
| `POST /api/learn/answer-check` | Steps 10 and 12: `{run, question, text}` read by the citation, refusal and scoring code (changes nothing) |
| `GET /api/learn/evaluate?run=&against=` | Step 12: a run's grid, facts with spans, reciprocal ranks, Set E chain, Wilson interval, sign test against another run |
| `GET /api/learn/safety?source=&page=`, `…/pages`, `…/image?view=`, `…/flags?text=` | Step 13: a page through the guard (the crafted page or a real hidden-text page), its three images, the instruction check on any text |
| `POST /api/learn/safety/live?version=v2\|v3` | Step 13, live: the crafted page answered now by the Answer role's model through `answer.ask` (one at a time) |
| `GET /api/learn/figures?figure=` | Step 11 (M7): one figure before and after (description cache only, no model call) |
| `GET /api/figures/{figure_id}/image` | A figure region as the vision model saw it (PNG); a figure passage's page boxes are its picture's box |

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
4. **PDF text is data, never instructions** (M2, `rag_studio/guard.py`,
   `docs/steps/13-safety.md`): hidden text is dropped before any step reads the
   page (found by drawing the page with and without its text), instruction-like
   phrases in passages and questions are flagged and shown, and prompt v3 fences
   passages off as quoted data. `tests/test_guard.py` builds a PDF with each trick.
5. **Nothing leaves the machine** except calls to your own Colab runtime.
   There is no tracing service (Phoenix/Langfuse were cut, 2026-10-22): the
   in-app trace (`trace.py`) is the only one. Logs hold sizes and counts, never
   document or answer content.

## Data on disk

| Path | Committed | Contents |
|---|---|---|
| `data/pdfs/` | yes | The source PDFs |
| `data/golden/` | yes | Test questions with verified answer pages |
| `data/lancedb/` | no | One LanceDB table per index build |
| `data/indexes/` | no | Each profile's active-build pointer (= its manifest), written atomically, and the build history |
| `data/cache/` | no | Rendered pages; `docling/` converted documents (about 15 min to rebuild for all three); `clean/` cleaned documents per parser (seconds). Keyed by content hash (and options, and the cleaning code's hash) |
| `data/runs/` | no | Evaluation runs with their full configuration |
| `data/chat/` | no | Conversations, one JSON file each |
| `data/figures/` | ask | Figure descriptions exported per figures profile (`./rag figures export`): GPU time, not regenerable word for word |
| `data/context/` | ask | Context lines exported per profile (`./rag context export`): hours of T4 time that cannot be regenerated word for word, so worth keeping |
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
