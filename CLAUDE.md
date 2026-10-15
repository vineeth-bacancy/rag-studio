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
./rag ingest    # build the baseline index (≈7 min; a no-op when nothing changed; --force)
./rag eval      # the scorecard (--set a b d, --mode retrieval, --model, --prompt, --ladder ROW, --tag RUN_ID)
./rag health    # Ollama, models per role, documents, indexes, memory, disk
./rag test      # pytest
```

`index` and `llm` (M4) print the milestone that adds them.

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
- SSE goes through `web/api/sse.py`, not FastAPI's `EventSourceResponse`: that
  one hardcodes `Cache-Control: no-cache` and a route can only add a second
  header, while `no-transform` is what stops Next gzipping the stream.
- Table rows with a toggle are tinted by `has-aria-expanded:bg-muted/50` in
  `ui/table.tsx`; to change an open row's tint, override that same variant.
- A child component imported only by a client page needs no `"use client"`;
  adding it makes the TS plugin warn about every function prop.

## Engine gotchas (learned in M1)

- `prompt_eval_count` counts the whole prompt even when Ollama reuses a cached
  prefix, so characters/tokens > 8 reliably means the start was dropped.
- Whitespace in a system prompt changes small-model answers: `v2` vs
  `v2-handbuilt` (one blank line) move two Set A rows. Record the prompt hash.
- A rebuild of the baseline index reuses every vector by content hash; only a
  changed passage or a new embedding digest costs embedding time.
- Answer pages in a test set must list **every** page that states the answer,
  and a key can be wrong: D3's was widened when p44 turned out to call the
  adjusted figure plain "net income". Check the text before blaming the model.

## Engine gotchas (learned in M2)

- **Hidden text is decided by pixels, not rules.** "White text" and "a later
  fill covers it" flagged visible text (iMac p6, GEP table headers). The guard
  renders the page with and without its text (`guard.py`); keep it that way.
- Text placed outside the page (GPO p15) is not in `get_text()` (mediabox
  clip) and survives a page-rect redaction; the guard checks the page rect first.
- PyMuPDF's `rawdict` marks invented spaces (`synthetic`); the layout reader's
  lines rebuild `get_text()` exactly (append "\n" unless the line already ends
  with one). Don't switch to `sort=True`: it interleaves columns.
- A ligature glyph repair must check that the rest is really missing: GPO maps
  `f_f_i` to "f" and carries "fi" after it ("Traffiicking" otherwise).
- Line-end fragments ("sugges", "tions") must not count as dictionary words;
  "debt-to-GDP" attests "debt-to".
- Docling: 866 s and 6.4 GB for the 294 pages. Only `./rag ingest` converts;
  everything else reads `data/cache/docling/`. Iterate with
  `traverse_pictures=True` or chart text disappears. Its bboxes are bottom-left.
- The cleaned-document cache key includes a hash of the cleaning code, so any
  edit to `clean/guard/structure/parse/parse_docling.py` recomputes (seconds;
  Docling's conversion itself stays cached). A built index keeps the text it was
  built with: rebuild it (`./rag ingest --profile …`, vectors reused by hash)
  before measuring a cleaning change.
- `dcg` blocks `shutil.rmtree` and some heredocs with backslashes: use a fresh
  scratch folder and the Edit tool instead.

## Engine gotchas (learned in M3)

- **A profile's recorded config must not change for builds that exist.**
  `ChunkerSpec.describe()` adds a key only when it differs from the default;
  a new field on `IndexProfile` itself would make every existing build refuse
  to open ("built with a different setup").
- **"Up to date" compares every stored row**, not just counts: manifests carry
  `rows_sha256` (ids, content hashes, labels, sections, kinds, flags). A
  manifest without it rebuilds once, every vector reused.
- **A section label is a fact the model reads.** A wrong section (GEP boxes ran
  past their end until M3) becomes a wrong attribution in every passage after
  it. Check labels in the chunk playground before measuring.
- **Labels blur passages that share them**: every Chapter 3 passage carries
  the chapter title, so its bibliography outranks its summary (A5). The limit
  per section fixes that and costs single-section answers (A4). Measure both.
- `VectorCache` has `__len__`, so an empty one is falsy: never write
  `cache and cache.embed`.
- A background `./rag ingest` starves `./rag eval` of Ollama (each question's
  embedding waits behind a batch). Never run a scorecard beside a build;
  retrieval-only runs are deterministic, timings are not.
- `dcg` also blocks heredocs whose text contains `>` or `||` (TypeScript): edit
  UI files with the Edit tool.

## UI gotchas (learned in M3)

- SWR `keepPreviousData` keeps the last result in `data` while the new key
  loads or fails: show `error` whenever it is set (not `error && !data`), and
  draw a result only if it matches the current key's parameters.
- With the sidebar open, a 1280 px window leaves a content container of about
  960 px: pick container-query breakpoints (`@4xl` = 56rem) for that, not 64rem.
