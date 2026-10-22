# QA: M5 reranker, rewriting, conversations, How it works 4 and 8 (2026-10-22)

**Method** (as in M1-M4): Chrome DevTools MCP on `./rag web` (dev mode), the
real indexes and runs, a real Colab T4 runtime. Never beside a scorecard run or
an index build on this computer (the app was stopped while the `-ctx` index
built). Overflow is measured with a script (`scrollWidth` against the
viewport, and every element whose right edge passes it); screenshots at 1280
and at 375 × 812 @2x.

## Result

| Check | Result |
|---|---|
| Settings (1280) | Roles say where each runs ("Local AI", "Local AI, in this app", "Remote AI only"); python roles show Downloaded; remote roles "On Remote AI"; the Defaults card lists the engine's defaults (5d, v3, dense, 5, every switch Off) |
| Evaluation (1280) | No overflow; the ladder shows MRR and recall@20 for every variant (old runs too); row 6's setup lines name the reranker ("reranked by Remote AI · 0.6B") |
| How it works › Query rewriting (1280) | No overflow; B4 replays "As asked" from a recorded ladder run (Hit@1, run id, reranker); techniques without a run read "Not measured yet" |
| Ask (1280, 375) | No overflow at either width; the conversation list is a side column at 1280 and a "Conversations" Sheet at 375 (opens from the left, 281 px, focus on "New conversation", Escape closes it) |
| Ask, a real follow-up (Remote AI connected) | "What is the growth outlook for South Asia in this report?" answered with 3 citations and saved (the list shows "1 question · now" at once); "And for 2027?" shows "Searched for: What is the growth outlook for South Asia in 2027?" with **Ask as typed**, and is answered with a citation |
| Indexes | The new "Context lines" section: Docling `-ctx` "Ready to ask", 1,473 of 1,473 lines ("Every passage"), last job speed; PyMuPDF "1,044 missing" with "Write 1,044 on Remote AI" |
| Search lab, Rerank (this computer, 0.6B, Colab off) | B4 on Docling: the answer passage (p19) moves from dense rank 8 to **reranked rank 1** ("dense 8 · up 7"); "Lines: Fusion / Reranking" switches; the column sits beside Dense at a 1,200 px container, full width below the three at 950 px, stacked at 700 and 360 px; no overflow at any width; light and dark |
| How it works 4 and 8 | Both load with no overflow and no console errors or warnings; step 4 shows D10's answer at rank 13 on the plain index and 24 on the `-ctx` one (row 7's loss, seen live); step 8 replays D10 on PyMuPDF: as asked refused (rank 21), split into parts Pass (rank 4), several phrasings refused, draft answer first Pass, multi-step Pass, all with the same 4B reranker |

**Not covered by this pass** (the Chrome DevTools connection was lost when the laptop had to be
restarted at 18:20, and the window could not be resized while maximised): page-level widths 768,
1024 and 1440 for the new pages (container widths were measured instead), a keyboard-only pass over
the new controls, and dark theme on How it works 4 and 8.

## Findings

Severity: P0 broken · P1 wrong or misleading · P2 confusing or ugly · P3 polish.

| ID | Sev | Where | Finding | Fix |
|---|---|---|---|---|
| M5-01 | P2 | Settings › Model backend | A runtime set up for rewriting and reranking showed **Warning** and "A model the Colab indexes need is missing" (qwen3-embedding:8b). That is expected for such a runtime, not a fault | A neutral note: "This runtime does not serve … so the indexes built on Colab cannot be asked while it is connected. To use them, run the setup cell for Embeddings."; the status no longer counts it |
| M5-02 | P1 | Settings › Where each job runs | Said "Answer, others: Local AI", but query rewriting and context lines run on Remote AI only, and the reranker can run on either | Rows for Rewriting (Remote AI only), Reranker (the 0.6B here or 0.6B/4B/8B on the GPU worker) and Answer, judge (Local AI) |
| M5-03 | P2 | Settings › Model backend | The GPU worker was not shown, and Test said "Embedded a test sentence with each model" on a runtime that embeds nothing (it chats and reranks) | A "GPU worker" row (what it serves, what is loaded, peak memory); Test lists each embedding, chat and worker check in its own words |
| M5-05 | P1 | Indexes | The two `-ctx` indexes were **never drawn**: each row-5 card took the first profile with its model and parser (the plain one), and "Earlier indexes" left out every EmbeddingGemma 2 profile. Their context lines could not be followed or written from the browser although the API existed | `contextual` on each profile row; a "Context lines" section (ladder row 7) with each `-ctx` index card and a context-lines panel: N of M passages, the last job and its speed, "Write N on Remote AI" (disabled with Remote AI's reason), Cancel while writing, polling every 3 s while a job runs. Checked: Docling 1,473 of 1,473 "Every passage", PyMuPDF "1,044 missing" with the button |
| M5-06 | P3 | Indexes › Context lines | A `-ctx` index with missing lines still offers "Build here"; the build refuses with "N passages have no context line yet. Write them on Remote AI first." | Open (the refusal is clear; the panel beside it says the same) |
| M5-07 | P2 | Search lab › Rerank | One rerank on this computer took 87 s instead of 20-35 s: a second request for the same question (from a browser click during a script check) waited behind the inference lock, and both loaded the model cold | Not changed: the lock is what keeps two loads from exhausting memory; the scores are remembered, so the next rerank of the same question is instant |
| M5-04 | P3 | Ask, 375 px | The composer's switches wrap to four rows and take nearly half the screen | Below 640 px they fold behind a "Search setup" button (`aria-expanded`); from 640 px they stay inline as before |
