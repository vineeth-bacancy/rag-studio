# Session handoff: RAG Studio

Updated 2026-10-15 ~16:20 IST, at the end of M3. A new session should read, in
order: this file, `CLAUDE.md`, then the build spec
`~/Dev/RAG/advanced-rag/RAG_STUDIO_PROMPT.md` (§16-17 for how to work and the
milestones). **Claude builds everything here**; the user reviews, runs and asks
questions.

## Rules the user set (don't re-ask, don't break)

- **Never run `git commit` or push.** At a milestone's end: `git add`, update
  this file, report neatly with every number, and end with a suggested commit
  message. The user commits and pushes. (Repeated by the user during M2.)
- Repo `~/Dev/RAG/rag-studio`; UI 127.0.0.1:**3100**, API 127.0.0.1:**8775**
  (trading uses 3000/8765); Phoenix later on 6006.
- Local model pulls approved, each when its milestone starts: qwen3-embedding:0.6b,
  EmbeddingGemma 2 (via sentence-transformers, see research §4), Qwen3-Reranker-0.6B
  (HF), gemma4:e4b (M6). Docling's models are now downloaded (M2). qwen3.5:4b is already pulled.
- Git: PDFs and test questions are committed; indexes, runs and caches are rebuilt.
- Check `ollama ps` and `systemctl --user status trading-daily.timer` (19:17 IST)
  before any pull or long run. Never stop the trading timers.
- The user's own `./rag web` is usually running (dev mode, both servers
  reload). Test against it; don't start a second copy. Check free memory too:
  with VS Code open, only 2-4 GB is free; don't run browser QA beside a scorecard run.

## M3 status: complete (staged, not committed)

**Engine:** `chunk.py` (word-boundary fix `FIXED_SNAP`; `STRUCTURE`: paragraphs
and Docling tables packed ≤ 1,000 chars within one section/kind, heading-only
pieces dropped, fragments < 100 chars joined; `SEMANTIC`: same sections, cut
above the document's 90th-percentile neighbour distance; the section label
"Document: …\nSection: …"; playground chunkers recursive/sentence);
`structure.py` **GEP box sections end** where pages stop printing "BOX n.n";
`store.py` `label` + `search_text` columns and the **word index** (LanceDB FTS,
stop words kept, stemming, positions); `retrieve.py` `SearchSpec` (dense | bm25
| hybrid, our RRF K=60 depth 50, `section_cap`); `ingest.py` hashes label+text,
`rows_sha256` fingerprint (fixed an "up to date" bug), word index added in
place to old builds, `VectorCache` for sentence vectors (`embed.py`);
`prompt.py` v4 (rule 4 relaxed) and labels in the passage block; runs record
`retrieval`, ladder variants split by it; `learn.py` (chunk lab, search lab,
store facts, ANN check). Profiles: `pymupdf-snap`, `pymupdf-structure`,
`docling-structure`, `pymupdf-semantic` (all built).
**API:** `search`/`section_cap` on ask and runs; `/api/indexes` says
`word_index`/`labels`; `/api/learn/{chunk,search,store,store/ann-check}`.
**UI:** How it works steps 1-3, 6, 7 at `/learn/[slug]` (parse, clean, chunk
playground with semantic curve, store with exact-vs-ANN, search lab with rank
lines); Ask and Evaluation search pickers, per-list ranks on sources.

| Check | Result |
|---|---|
| Tests | `./rag test` **172 passed** (133 at M2) |
| UI | typecheck and lint clean; QA 40/40 page×width, console clean, 8 findings (7 fixed): `docs/qa/2026-10-15-m3-how-it-works-search.md` |
| M3 target (A6 a Hit) | rank 45 (row 1) → 5 (row 2) → **2 / 1** (structure, PyMuPDF / Docling), answered with the right attribution |

**Scorecard, Sets A+D (Pass · Refused · Wrong), qwen3:1.7b** (all in `docs/ladder.md`):
row 1 18 · 2 · 5; 2c 17 · 1 · 7; 2e (Docling v3) 16 · 3 · 6; 3a PyMuPDF+structure v3
19 · 0 · 6; **3b Docling+structure v3 20 · 1 · 4 (beats row 1 on every count)**;
3c PyMuPDF+structure v4 19 · 1 · 5; 3d Docling+structure v4 19 · 1 · 5; 4a 3c+hybrid
18 · 2 · 5 (Set A 11 · 0 · 0); 4b 3b+hybrid 19 · 2 · 4; 4c 3c+limit 18 · 1 · 6; 4d 3b+limit
19 · 1 · 5 (best recall: A 9/9, B 6/6, D 10/11).

Runs: 2e `20261015-081851-1b4e`; 3a `…083915-c607`; 3b `…085032-9b8e`; 3c
`…090249-35cc`; 3d `…091402-86d6`; 4a `…092513-fcb5`; 4b `…093623-6539`; 4c
`…094757-1c5f`; 4d `…100205-ae34`. Retrieval-only runs are labelled
"M3 retrieval: …" (before the box fix) and "M3 retrieval (boxes fixed): …".

**Decision waiting for the user:** make 3b (Docling + structure-aware, dense,
prompt v3) the app's default? Recommended, with caveats (2 rows of 25, one run
each; Docling's 15-min conversion is needed to build it). Defaults are still
baseline + v2 + dense.

If `data/` is wiped: the M2 commands, then `./rag ingest --profile
pymupdf-structure` (~7 min), `--profile docling-structure` (~8 min),
`--profile pymupdf-snap` (~8 min), `--profile pymupdf-semantic` (~15 min).

## M2 status: complete (committed by the user as d0921ee)

**Engine:** `parse.py` (layout reader, glyph names, bookmarks, word-aligned
passage boxes), `parse_docling.py` (Docling, cached), `guard.py` (hidden text by
pixels, instruction flags), `clean.py` (the full order, marks), `structure.py`
(sections, labels, contents and references), `prompt.py` v3, profiles
`pymupdf-clean` and `docling-clean`, the guard step in `answer.py`, ladder
variants and `compare()` in `eval.py`. **API:** `/inspect`, `/cleaning`,
`/eval/compare`, `/prompts`, profile + prompt on ask and runs, boxes for cleaned
indexes. **UI:** the Documents page inspector (page + overlays, raw / cleaned /
Docling text with every rule's marks, cleaning report, page facts); Ask index and
prompt pickers, source badges (printed page, contents/reference list, guard
flags), the question's guard note; Evaluation index/prompt pickers, ladder
variants, two-run comparison.

**Numbers (all measured 2026-10-15; details in `docs/steps/01-parse.md`,
`02-clean.md`, `13-safety.md`, `docs/ladder.md`):**

| Target | Result |
|---|---|
| Ligatures → 0 | GEP 1,507 → 0 (both parsers) |
| Broken "Th" → 0 | 325 → 0, repaired from the font's glyph name `T_h` (no word list); Docling reads it itself |
| Compounds intact | an independent oracle finds 140 (FINDINGS said 127); all 140 kept. Docling had merged them; 162 restored in GEP |
| Hidden text dropped and flagged | 9 pages, by pixel test: GPO p1, 5, 15, 28, 29, 30; GEP p125, 195, 200. Paint-order rules alone were wrong (iMac p6) |
| Docling | 294 pages in 866 s, 6.4 GB peak, ~570 MB of models |
| Tests | `./rag test` **133 passed** (89 at M1), incl. the crafted-PDF guard test |
| UI | typecheck and lint clean; QA 25/25 page×width, console clean, 5 findings (4 fixed, 1 not reproduced): `docs/qa/2026-10-15-m2-inspector-guard.md` |

**Ladder row 2** (qwen3:1.7b; Sets A+D Pass · Refused · Wrong; baseline 18·2·5):

| Variant | Recall A / B / D | A+D scorecard |
|---|---|---|
| 2a PyMuPDF + cleaning, v2 | 9/9 · 6/6 · 10/11 | 15 · 1 · 9 |
| 2b Docling + cleaning, v2 | 9/9 · 6/6 · 10/11 | 14 · 3 · 8 |
| 2c PyMuPDF + cleaning, v3 | 9/9 · 6/6 · 10/11 | 17 · 1 · 7 |
| 2d Baseline, v3 | 8/9 · 5/6 · 9/11 | 18 · 1 · 6 |

Retrieval up everywhere (A6 rank 45 → 5, B4 rank 11 → 1); answers down
(A6 misattribution, A8 391 vs 346 kg, D10, D6 table rows). Prompt v3 fixes T2/T3
on the cleaned index with no regression, but is mixed on the baseline.
**Defaults unchanged** (baseline + v2): no variant beats row 1 on answers.
**M3 base (user, 2026-10-15): both parsers side by side.** Rows 3-4 are built and measured on `pymupdf-clean` and `docling-clean` alike (I had recommended PyMuPDF only).

Runs: 2a `20261015-064121-03ef`, 2b `20261015-065220-7583`, 2c
`20261015-070309-a697`, 2d `20261015-071550-c31c`. An interrupted 2a attempt
(`20261014-203245-1caa`) is in history; it does not count.

If `data/` is wiped: `./rag ingest` (baseline), `./rag ingest --profile
pymupdf-clean` (~6 min), `./rag ingest --profile docling-clean` (~15 min of
Docling + ~7 min embedding), then the row-1 commands below and
`./rag eval --set a b d --profile <p> [--prompt v3] --ladder 2 --label "<variant>"`
(~11 min each). Row 1: `./rag eval --set a b --ladder 1 --label Baseline` and
`./rag eval --set d --ladder 1 --label "Baseline: edge cases"`.

## Decisions settled (M1-M3)

- **The spec wins over M0 research** where they differed (research.md §0).
- **No Ollama upgrade**; Ollama stays 0.32.6 (EmbeddingGemma 2 goes through Python).
- **Hidden text is decided by pixels** (render with and without text), not rules.
- **Docling runs only in `./rag ingest`**; everything else reads the cache.
- **Ladder rows can have variants**, grouped by setup.
- **Prompt v3 is opt-in**; v2 stays the default (scorecard, row 2d).
- Set D's D3 key widened in M1 (p44 says plain "net income" for the adjusted figure).
- **M3 base: both parsers side by side** (user, 2026-10-15).
- **Hybrid search and the limit per section are switches, off by default**
  (row 4: neither improves the scorecard on labelled passages).
- **Prompt v4 is opt-in** (row 3: a wash against v3).

## Not done, on purpose (where it goes)

- Docling's letter-spaced fragments and spaced footnote numbers: shown, not repaired.
- Editable test questions in the UI: M8. Set C (figure question): M7.
- How it works widgets: steps 4, 5, 8 (M4-M5), 9, 10, 12 (M6), 11, 13 (M7).
- PyMuPDF tables are still flattened (no `find_tables`): Docling's whole-table
  passages are what helped D6/A8. An exact-match column for numbers in BM25
  was not built (research.md §8).
- Semantic chunking's document-wide threshold (90th percentile) misses the p19
  cut the page-level 80th finds; not tuned (it is an option, not a default).
- Scrollbars tested in Chrome only.

## Next: M4 (Colab and index profiles)

Spec §17 M4: the Colab gateway and setup cell (adapt `~/Dev/trading`'s design,
spec §13), the connect code, settings card and CLI; **Qwen3-Embedding 0.6B
locally** (`qwen3-embedding:0.6b`, approved pull: check `ollama ps` first) and
**8B on Colab**; **EmbeddingGemma 2 text** via sentence-transformers (research
§4); the Indexes page; the local-vs-Colab vector-match check. Done when: remote
connect and test from the UI; the match check passes; ladder row 5 per profile.
Row 5 should build on the default the user picks after M3 (3b recommended).
D9 (Spanish) is the question the multilingual model is expected to fix.

## Gotchas hit so far

- `pkill -f <pattern>` killed the Bash tool's own shell. Stop servers by port:
  `ss -ltnpH "sport = :3100"`.
- Don't edit `scripts/run_web.sh` while `./rag web` runs (bash reads it as it goes).
- `uv sync` can stall when other uv processes hold the cache lock.
- The dcg shell hook blocks `shutil.rmtree` and some heredocs (backslashes,
  `>`-like text): use a fresh scratch folder and the Edit/Write tools.
- Editing an engine file while `./rag ingest`/`eval` runs is fine (the process
  already imported it) but the run uses the old code; rerun if it matters.
- Any edit to `clean/guard/structure/parse/parse_docling.py` changes the
  cleaned-document cache key; a built index keeps its old text until
  `./rag ingest --profile … --force` (vectors are reused by hash).
- A background job dies with the Claude session: an 11-min run was cut at 25/31
  when a session ended. Check `data/runs/` for `interrupted` and rerun.
- `./rag ingest` once said "up to date" when only labels changed (fixed with
  `rows_sha256`). If an index ever looks stale, `--force` (vectors reused).
- Never run a scorecard beside a build: Ollama serves one at a time.
- Chrome DevTools MCP: `resize_page` fails on a maximised window; use
  `emulate` with a viewport instead. The app keeps its own theme (next-themes):
  use its toggle, not `colorScheme` emulation.
