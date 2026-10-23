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
./rag ingest    # build the BASELINE index (≈7 min; a no-op when nothing changed; --force); --profile for others.
                # The app's default index is docling-structure-embeddinggemma2-fig (row 11a = 5d + figures):
                # ./rag figures import <id> (on a fresh data/), then ./rag index build <id>
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

## Engine gotchas (learned in M4)

- **A local build can freeze this laptop.** Ollama's default 4,096 context
  gives qwen3-embedding:0.6b a 1.2 GB compute buffer (2.26 GB total); with
  2.7 GB free and deep swap the desktop froze and the session ended
  (2026-10-15). Check `free -m` (not just `ollama ps`) before a long local
  run; `ingest.check_memory` refuses below `profiles.BUILD_RAM_GB`.
- **Through Ollama a text's vector depends on what was embedded just before
  it** (the server reuses a shared prefix's cached keys/values, computed in a
  differently shaped pass): the same text moves to cosine 0.9996 (0.6B) or
  0.9985 (the 4-bit 8B on a T4). Batches are worse than one text per request.
  Compare vectors only one per request, same order, on a fresh cache, and
  measure a noise floor first (`embed.match_check`). An early "num_ctx moves
  vectors (0.99963)" was this noise: compared fairly, num_ctx changes nothing.
- `EmbeddingSpec.options` (`num_ctx: 1024` for Qwen3) is sent on every embed
  call and recorded; it is there for memory, not meaning.
- Identical test texts benchmark wrong: Ollama's prompt cache made 32 copies
  of one passage embed at 16.5/s; real passages run at 0.6/s (Qwen3 0.6B).
- EmbeddingGemma 2 runs in-process (sentence-transformers, pinned revision,
  `local_files_only`); pass `prompt=""` to `encode` so no config prompt is
  added on top of the profile's prefix.

## Engine gotchas (learned in M5)

- **Every Remote AI output is remembered** (`data/cache/rerank/`,
  `remote_cache.py`, `data/cache/context/`): fill the caches with a
  retrieval-only run while Colab is up, disconnect, then run the scorecards
  here; they replay the same scores and rewrites. Query rewrites and follow-up
  rewrites depend only on the question, so cache them once for every index;
  multi-step decisions and rerank scores depend on the index.
- **qwen3:8b (about 5.6 GB at num_ctx 8192) and the 4B reranker (9.1 GB peak)
  do not fit a T4 together**; the 0.6B (2 GB) does. Do all qwen3:8b work
  first, unload it (`POST /api/chat` with `messages: []`, `keep_alive: 0`),
  then rerank with the 4B. The worker serves only models downloaded at start
  (a download inside a request would pass the tunnel's 100 s).
- A `/context` batch can time out on the tunnel (once in about 120 batches);
  the runtime keeps finished lines, so the client asks again. Never make a
  Colab job fail on one timeout, or on one bad output: a line over the junk
  limit (it was 400 characters, below what `num_predict` allows) stopped the
  job at 1,211/1,473. A rejected line is now counted and its passage left
  without one, so the build refuses loudly instead.
- **Back-to-back scorecards starve each other**: the answer model stays
  resident for Ollama's 5-minute keep-alive (qwen3:1.7b's runner held 5.4 GB
  RSS), so the next run could not load EmbeddingGemma 2 and refused three times
  in a row (2026-10-22). A two-pass run now unloads our own answer model first
  (`OllamaClient.unload`); never unload another app's model.
- **A replay must reproduce every step, failures included.** Bad multi-step
  decisions were not cached, so a later run on a runtime without qwen3:8b could
  not replay them (2026-10-22). An `agent` bad output is now cached (raw +
  error) and replays as the same stop; a query rewrite's junk stays an
  uncached error. A rerank-only runtime (no Ollama) replays cached rewrites.
- Asked again in a later session, one of four bad controller decisions came
  back valid: never assume a remote LLM output can be regenerated; cache first.
- Every rebuilt data folder must be in `tests/conftest.py`'s `tmp_data` list:
  `data/context/` was not, and a test export landed in the real folder.
- Runs with a reranker or a rewrite evaluate in two passes (search everything,
  release EmbeddingGemma 2 and the reranker, then answer from the same
  passages). `./rag eval` with no `--set` leaves out Set E (remote-only).
- Context lines are written in windows (same-section blocks); Ollama's prefix
  reuse showed **no** speed-up in practice (first line of a block 2.3 s
  median, later 2.9 s), and 43% of lines still begin "The passage…". The
  generation, not the tunnel, is the cost: about 0.3-0.4 lines/s on a T4.

## Engine gotchas (learned in M6)

- **`ollama ps` understates a model's memory.** On 0.32.6 the model runs as
  `llama-server` (not `ollama runner`); its RSS counts the mapped weights and
  the KV cache: qwen3.5:4b 6.7 GB RSS while `ps` said 3.4 GB, gemma4:e4b 7.5 GB
  (`ps`: 1.9 GB). Plan memory from RSS; `scripts/eval_watchdog.sh` measures it.
- **gemma4:e4b has no Go template** (`{{ .Prompt }}`, `RENDERER gemma4`, a
  DRAFT model with `draft_num_predict 3`), so its system-prompt handling can't
  be read: it was verified by behaviour (system "reply PINEAPPLE" → PINEAPPLE).
  Both new models honour `think: false` (0 thinking characters).
- **The strict refusal rule hides a difference between models.** qwen3.5:4b
  refuses and then explains ("I could not find this in the documents. The
  provided text only contains…"): scored Wrong, like qwen3:1.7b inventing a
  2031 forecast on the same trap. Read a Wrong's answer before calling a model
  less safe.
- **A run records passages by chunk id only.** Replaying a recorded answer
  (How it works 9/10) reads the rows back from the run's own LanceDB table;
  `KEEP_BUILDS = 2` prunes old tables, so old runs cannot be replayed (they are
  left out of the pickers). Since M6 each answer records `messages_sha256`.
- **Editing `guard.py` re-cleans every PDF** on the next read (it is in the
  cleaning cache key, `ingest._code_hash`); `./rag test` triggers that read
  (here at 21:31 on 2026-10-22). Don't do it beside a scorecard whose timings
  matter.

## Engine gotchas (learned in M7)

- **Plan a Colab cell's GPU memory before the user runs it, step by step**, not just that it builds:
  every model `start.sh` loads, the worker's first reranker, and what stays resident between them.
  On 2026-10-23 two startup-checked vision models stayed resident (6.2 + 8.0 GB) and the worker
  OOMed on a T4 after ~40 min of downloads. `start.sh` now releases each model after its check and
  everything before the worker, and fails before the downloads below 3000 MiB free. Ollama does not
  evict for another process; it spills its own models to the CPU instead.
- **Ask for the smallest session that answers the question**: every new runtime downloads everything
  again (three vision models + two rerankers was ~27 GB). Run what fits here (qwen3.5:4b) locally.
- **A chart value is measured, never eyeballed**: `get_drawings()` rect/path heights calibrated by
  the axis's tick labels, then a 300 dpi crop. A 90 dpi visual read of a GEP bar was 0.4 points off;
  the last bar of a series is often a line path (`l` items), not a `re`.
- GEP draws ~57 chart panels as raster images with no text layer, and Docling leaves chart-panel
  titles ("D. Poverty") outside the picture box and makes them headings (wrong section labels).
- **Read every vision description yourself; the checker is a hint.** Models write tables loosely (no
  outer pipes, "Data:" on the header, the first column's value inside the row label, series named by
  colour); `truth_check` missed or misplaced values three times before it learnt each form.
- A `try` process imports `vision.py` when it starts: a checker fixed meanwhile is not the one it
  prints. Re-run `./rag figures check` (no model call) after changing the checker.
- Small vision models read plain bars to ~1% of the axis and fail stacked bars (qwen3.5:4b/9b); a
  prompt with stacked-bar instructions (vision-v2) did not fix that and hurt gemma4:12b.
- Evaluation runs are two-pass when the answer model is large (`ANSWER_RAM_GB` ≥ 6): EmbeddingGemma 2
  (2.2 GB) beside gemma4:e4b (8 GB) is too much for this laptop. Ask/eval refuse to load an answer
  model without its `ANSWER_RAM_GB` free (`answer.check_memory`).
- Figure passages can take top-5 slots from text passages without moving any answer page's rank
  (7 of 25 A/D questions): compare the top 5, not just recall, and check the trap questions.
- **gemma4:e4b is not word-for-word repeatable at temperature 0**: 2 of 18 byte-identical prompts
  (same `messages_sha256`) gave differently worded answers (row 11; likely its draft model). Never
  complete a run with another run's answers; measure the whole set.
- Another app can hold the shared Ollama: Hermes Agent loaded qwen3:8b (6.8 GB) on 2026-10-23 and is
  now disabled at the user's request. Wait for a foreign model's `expires_at`; never unload it.
- `scripts/run_progress.sh` shows a running evaluation (bar, ETA, outcomes, free memory) without the
  web app; give it to the user when the UI is stopped to save memory.

## UI gotchas (learned in M3)

- SWR `keepPreviousData` keeps the last result in `data` while the new key
  loads or fails: show `error` whenever it is set (not `error && !data`), and
  draw a result only if it matches the current key's parameters.
- With the sidebar open, a 1280 px window leaves a content container of about
  960 px: pick container-query breakpoints (`@4xl` = 56rem) for that, not 64rem.

## UI gotchas (learned in M8)

- **The QA matrix runs from `scripts/qa/`** (`build.cjs` bundles axe-core into
  `harness-src.js`): inject it into an open app page with DevTools
  `evaluate_script` (sourcePath, format "script"); it loads each page in an
  iframe of the width under test. Keep a call to 2-3 pages: past DevTools' 180 s
  limit the tab can stop responding (open a fresh one). Bring the harness tab to
  the front: background tabs throttle its timers.
- Scrollbars: once `scrollbar-width` or `scrollbar-color` is set, Chrome ignores
  `::-webkit-scrollbar`, and `thin` is 10 px on Linux. Chromium gets the webkit
  rules (6 px); the standard ones are gated to Firefox by
  `@supports not selector(::-webkit-scrollbar)` (`globals.css`).
- Light `--muted-foreground` is oklch 0.54, not shadcn's 0.556: 0.556 on
  `bg-muted` is 4.34:1. Text on a translucent tint must be measured on the real
  background (chart-1 on `bg-chart-1/15` was 3.7-4.4:1); white text needs a
  600-700 step, black text in dark a 400-500 step (`chunk-text.tsx`).
- `ui/select.tsx`'s list is `w-max` within `--anchor-width`..`--available-width`:
  shadcn's `w-(--anchor-width)` clipped long items under the check mark.
- A saved conversation keeps each passage's number, document and page, not its
  text: `AnswerText` takes them as `saved`, or every `[n]` reads as invented.
- axe's `scrollable-region-focusable` is moot in Chrome 130+ and Firefox (they
  make scrollers keyboard-focusable); it still applies to Safari. In dev mode the
  first Tab can land on the `nextjs-portal` dev-tools button.
