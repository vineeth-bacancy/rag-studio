# RAG Studio

A local web app that answers questions about PDF reports with page-level
citations, and teaches how RAG works with an interactive section for every step.
It is an upgrade of the hand-built baseline in `~/Dev/RAG/advanced-rag`: the
baseline is reproduced once as the measured "before" row, and every upgrade is
measured on the same scorecard. **No upgrade is assumed to help; the scorecard
decides the defaults.**

The build spec is `~/Dev/RAG/advanced-rag/RAG_STUDIO_PROMPT.md` (milestones M0-M8).
Never change anything in `~/Dev/RAG/advanced-rag` or `~/Dev/trading`; they are
read-only references.

## Layout

```
rag_studio/   the engine (Python package). Every RAG step lives here.
web/api/      FastAPI routes only: validate, call the engine, return. guard.py is the security boundary.
web/ui/       Next.js 16 · React 19 · Tailwind 4 · shadcn on Base UI. Renders what the API returns.
data/pdfs/    the source PDFs (committed)       data/golden/  verified test questions (committed)
data/*        everything else is rebuilt (lancedb/, cache/, runs/, chat/, uploads/) and gitignored
docs/         start here, architecture, one doc per RAG step, research.md, product-language.md, qa/
tests/        pytest; tests needing Ollama or Colab carry a marker and skip cleanly
tools/colab/  the Colab GPU gateway and setup cell (M4)
```

## Commands

Always use `./rag`; it resolves the venv.

```
./rag web       # API on 127.0.0.1:8775, UI on 127.0.0.1:3100 (dev mode, both reload)
./rag health    # Ollama, models per role, documents, indexes, memory, disk
./rag test      # pytest
```

`ingest`, `eval` (M1), `index`, `llm` (M4) print the milestone that adds them.

## Engineering rules

1. **The engine is one Python package** (`rag_studio/`: parse, clean, structure,
   guard, chunk, context, embed, store, retrieve, rewrite, agent, rerank, prompt,
   answer, vision, judge, eval, trace, llm). The API calls the engine; the UI
   calls the API. No logic is duplicated, so a number cannot be right in the
   terminal and wrong in the browser.
2. **Every artifact records its configuration.** Index manifests, evaluation runs
   and figure descriptions carry the model, tag, quantization, prompt version and
   parameters. Comparisons are only valid between recorded configurations.
3. **An index profile is bound to one embedding configuration** (model, tag,
   quantization, dimension, document and query prompt formats, parser + chunker +
   contextual setting). Refuse a mismatched query loudly; never fall back silently
   to another model.
4. **Ingestion is incremental by content hash.** Build a new LanceDB table and
   switch to it atomically (pointer file written to a temp path, then renamed).
   Other files are written atomically too (write a temp file, then `os.replace`).
5. **Citations are built by code.** The model only picks `[n]`; code maps `[n]`
   to document and page.
6. **Never fabricate.** Missing values show as `-`, never 0. PDF text is **data,
   never instructions**.
7. **Loopback only.** Both servers bind 127.0.0.1. Secrets never reach the
   browser (a credential travels browser -> API, never API -> browser). Logs hold
   sizes and counts, never document or answer content. Tracing is self-hosted and
   local. Nothing leaves the machine except calls to the user's own Colab runtime.
8. **Tests (pytest):** baseline reproduction, the scorer on fake answers, the
   local-vs-Colab vector match, the gateway allowlist and auth (including the
   rerank service), the connect-code round trip, the injection guard on a crafted
   PDF, and the UI contract checks. Tests that need Ollama or Colab skip cleanly.
9. **UI QA before each milestone closes:** Chrome DevTools at
   375/768/1024/1280/1440, light and dark, keyboard, a clean console. Findings go
   in `docs/qa/*.md` with IDs.
10. **Docs:** `docs/00-start-here.md`, `docs/01-architecture.md`, one doc per RAG
    step (mirroring How it works), `docs/research.md` (choices with sources and
    dates), `docs/product-language.md`.

## Wording

`docs/product-language.md` is normative. Read it before writing any string a user
reads: a label, an error, an empty state, a CLI line. One name per concept;
plain language first and a "Technical details" disclosure for the rest; errors
read "Short explanation. Action to recover."; status words are Healthy / Running /
Warning / Needs attention / Unavailable.

## Machine facts that shape decisions

- Intel i5-1135G7: 4 cores / 8 threads, 15.7 GB RAM, no GPU. Disk is ~87% full.
- Ollama 0.32.6 at 127.0.0.1:11434 is **shared with the trading app**
  (`trading-daily.service` 19:17 IST daily, `trading-weekly` Saturday 09:01, both
  run `qwen3:8b`). Before a long local run check `ollama ps` and
  `systemctl --user status trading-daily.service`. Never stop or pause the
  trading timers without asking.
- Every new model is a multi-GB pull: list it with its size and ask first.
- `qwen3:4b` here is a thinking-only build (ignores `think: false`, 128-829 s per
  answer). Don't use it.
- The user runs commands with `!` in fish. Long runs go in the background.

## UI gotchas (learned the hard way)

- **This is NOT the Next.js you know.** Read `web/ui/node_modules/next/dist/docs/`
  before writing Next code. Error boundaries take `retry`, not `reset`.
- shadcn here is on **Base UI**: compose with `render={<X/>}`, not `asChild`
  (silently dropped). A link styled as a button is `<Link className={buttonVariants()}>`.
- Override a component's variant class **with the same variant**
  (`data-vertical:self-center`, not `self-center`): tailwind-merge only dedupes
  within one variant, and the attribute selector wins.
- Tables in cards: use container queries (`@container`, `@sm:table-cell`), not
  viewport breakpoints; with the sidebar open the card is far narrower than the window.
- `allowedDevOrigins` must list `127.0.0.1`, or the dev server 403s its own chunks.
- Streams (SSE) send `Cache-Control: no-cache, no-transform` and
  `X-Accel-Buffering: no`, plus a heartbeat comment every 15 s: the Next rewrite
  proxy drops a socket idle for `proxyTimeout` (330 s here).
- The request guard is pure ASGI. `BaseHTTPMiddleware` buffers streams.
- Don't edit `scripts/run_web.sh` while `./rag web` is running: bash reads the
  script as it executes it.
