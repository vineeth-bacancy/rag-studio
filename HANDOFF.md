# Session handoff: RAG Studio

Updated 2026-10-15 ~13:50 IST, at the end of M2. A new session should read, in
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

## M2 status: complete (staged, not committed)

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
**Recommended base for M3: PyMuPDF + cleaning** (ask the user to confirm).

Runs: 2a `20261015-064121-03ef`, 2b `20261015-065220-7583`, 2c
`20261015-070309-a697`, 2d `20261015-071550-c31c`. An interrupted 2a attempt
(`20261014-203245-1caa`) is in history; it does not count.

If `data/` is wiped: `./rag ingest` (baseline), `./rag ingest --profile
pymupdf-clean` (~6 min), `./rag ingest --profile docling-clean` (~15 min of
Docling + ~7 min embedding), then the row-1 commands below and
`./rag eval --set a b d --profile <p> [--prompt v3] --ladder 2 --label "<variant>"`
(~11 min each). Row 1: `./rag eval --set a b --ladder 1 --label Baseline` and
`./rag eval --set d --ladder 1 --label "Baseline: edge cases"`.

## Decisions settled (M1 and M2)

- **The spec wins over M0 research** where they differed (research.md §0).
- **No Ollama upgrade**; Ollama stays 0.32.6 (EmbeddingGemma 2 goes through Python).
- **Hidden text is decided by pixels** (render with and without text), not rules.
- **Docling runs only in `./rag ingest`**; everything else reads the cache.
- **Ladder rows can have variants**, grouped by setup.
- **Prompt v3 is opt-in**; v2 stays the default (scorecard, row 2d).
- Set D's D3 key widened in M1 (p44 says plain "net income" for the adjusted figure).

## Not done, on purpose (where it goes)

- GEP box sections run past the box's end (no end in the bookmarks): M3, with
  structure-aware passages.
- Docling's letter-spaced fragments and spaced footnote numbers: shown, not repaired.
- Editable test questions in the UI: M8. Set C (figure question): M7.
- How it works widgets: M3 onward (steps 1-3, 6, 7 in M3).
- Scrollbars tested in Chrome only.

## Next: M3 (chunking and hybrid search)

Spec §17: the word-boundary chunker fix, structure-aware passages with section
labels (use `CleanPage.sections`/`kinds`/`items`), semantic chunking as an
option, LanceDB full-text + vector hybrid (our own RRF, research C §3), a
per-section cap (reference lists and contents pages are tagged, `kind`), How it
works steps 1-3, 6, 7. Targets: Q6 (A6) becomes a Hit — **it already is (rank 5)
on the cleaned index**; the M3 bar is that it is answered, not misattributed.
Ladder rows 3-4, built on the base the user confirms (recommended
`pymupdf-clean`). Consider row 3's effect on A8/D6 (table rows) and A6
(section labels let rule 4 of the prompt be relaxed, spec §6).

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
- Chrome DevTools MCP: `resize_page` fails on a maximised window; use
  `emulate` with a viewport instead. The app keeps its own theme (next-themes):
  use its toggle, not `colorScheme` emulation.
