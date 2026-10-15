# QA: M3 How it works (steps 1-3, 6, 7), search settings on Ask and Evaluation (2026-10-15)

**Method** (as in M1-M2): Chrome DevTools MCP on `./rag web` (dev mode), the
real indexes (baseline, both row-2 and both structure-aware builds) and the
real `nomic-embed-text`. Run while the snap and semantic index builds held
Ollama (so the first semantic playground request took about 9 s), never beside
a scorecard run.

- The overview and the five step pages (`/learn/parse`, `/learn/clean`,
  `/learn/chunk`, `/learn/store`, `/learn/search`), Ask and Evaluation × 375 /
  768 / 1024 / 1280 / 1440 px, measured in a sized iframe: page overflow and
  elements wider than the viewport.
- Screenshots in dark and light, desktop (1280) and phone (375 × 812 @2x).
- Each widget on its teaching case: GPO p29 (hidden footer), GEP p19 ("Th",
  every chunker, the semantic curve at the 80th percentile), the baseline's
  table and versions with the exact-vs-ANN table, and B4 "global growth" in
  the search lab.
- Keyboard order through the chunk playground, and the console.

## Result

| Check | Result |
|---|---|
| Horizontal overflow, 8 views × 5 widths | none (40/40) |
| Console errors / warnings | 0 / 0 (over the last three navigations) |
| Parse, GPO p29 | the slate page with the footer outlined in red; three page numbers (Python index 28, PDF page 29, printed "-"); "Painted over by a later fill" × 2; PyMuPDF's raw letter-spaced text beside Docling's |
| Clean, GEP | the nine rules in order with counts (hidden 25, ligature 1,516, "Th" 325 on 75 pages, …); p19 before/after: "Te global economy" → "The global economy", only that rule highlighted |
| Chunk, GEP p19 | fixed: "Starts mid-word: baseline 4 of 5 · fixed 0 of 5"; structure-aware: 6 passages, each under its section label (the engine's own string); semantic at the 80th percentile: 9 passages, the curve with 6 cuts, passage 4 starts at "Risks to the outlook remain skewed to the downside" |
| Chunk, keyboard | document, previous / page number / next, parser tabs, chunker tabs, Size, Overlap, the baseline checkbox, each with a focus ring |
| Store, baseline | 1,190 rows × 768 dimensions; v1 created, v2 word index added, both 1,190 rows; the word index's settings in words; two builds, the active one marked; exact vs IVF_FLAT / IVF_PQ / IVF_HNSW_SQ: recall@10 1.000 / 1.000 / 0.800 / 0.987; "Measure again" (Ollama idle): 1.000 / 1.000 / 0.794 / 0.987, exact 3.95 ms |
| Search lab, B4 on the baseline | Dense rank 11 · Hybrid rank 2 · BM25 rank 9 (searching 50 deep), matching the runs; the answer page's rank line drawn from dense 11 to hybrid 2; columns side by side at 1280 with the sidebar open, stacked without lines at 375 |
| Ask | index, search (Dense / BM25 / Hybrid) and "Limit per section" beside the prompt picker; the intro names the index and how it is searched ("1,473 passages in the Docling + structure-aware passages index, searched by meaning and by words (hybrid)") |
| Ask, A6 end to end | Docling + structure-aware, Hybrid, prompt v3: answered in 29 s ("renewed escalation of hostilities, prolonged disruptions to commodity flows, and financial stress" [1]); source 1 is p19 (printed xvii), "rank 1 (dense rank 1 · BM25 rank 2) · Section Executive Summary"; console clean |

## Findings

Severity: P0 broken · P1 wrong or misleading · P2 confusing or ugly · P3 polish.

| ID | Sev | Where | Finding | Fix |
|---|---|---|---|---|
| M3-01 | P1 | Chunk playground, Semantic (engine) | HTTP 500 the first time: `cache and cache.embed` is the cache itself when the sentence cache is empty, because `VectorCache` has a length and an empty one is falsy | `cache.embed if cache is not None`; `tests/test_learn_api.py` now asks for a semantic page through the API with an empty cache |
| M3-02 | P1 | All five step widgets | An error after a first success was never shown: with SWR's `keepPreviousData` the old result is still `data`, and the widgets showed errors only `if error && !data`. The semantic 500 looked like a slow success | Every widget shows the error (with Try again) whenever the current request failed |
| M3-03 | P2 | Chunk playground | While a new chunker loaded, the previous chunker's passages were drawn under the new one's name: "Semantic" showed the structure-aware passages and "Too few sentences on this page to compare." | A result is drawn only if its page, parser and chunker are the current ones; otherwise the loading state |
| M3-04 | P2 | Search lab | The three columns sat side by side only from a 64rem container, so at 1280 px with the sidebar open they were stacked and the rank-movement lines (the point of the widget) were hidden | Side by side from 56rem; the "Reranked" placeholder is a full-width row until there is room for a fourth column |
| M3-05 | P3 | Parse, GPO p29 | Hidden letter-spaced text was boxed letter by letter: the guard marks hidden text glyph by glyph | Marks of one rule that only whitespace separates are drawn as one box |
| M3-06 | P3 | Store | "Searches the section label and the passage" on the baseline, which has no section labels | Says "the passage text … (this index has no section labels)" when the sample row has no label |
| M3-07 | P3 | Strings the UI agent invented | "Moved down: section limit", "Answers come from this one", "Pages worth a look" | "Moved down (limit per section)", "Active", "Suggested pages" (docs/product-language.md) |
| M3-08 | P3 | Ask, index picker | For a moment after load the trigger shows the raw id ("baseline") until the index list arrives | Not fixed: it settles to "Baseline" within a second. Observation |

## Not fixed, on purpose

- **The rank lines are hidden below a 56rem container** (phones, narrow
  windows, the sidebar open on a 1024 px screen): stacked columns have no
  meaningful line geometry. Each passage still shows its rank in every list.
- **Next's dev-mode badge** sits over content at 375 px; it does not exist in a
  production build.
