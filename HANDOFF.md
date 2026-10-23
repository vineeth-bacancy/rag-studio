# Session handoff: RAG Studio

Updated 2026-10-23 ~23:45 IST, at the end of M8: **the reduced plan (M0-M8) is complete.**
M8 was committed by Claude at the user's request (M7 = 4734d19, M6 = 9a9780d). A new session should read, in
order: this file, `CLAUDE.md`, then the build spec
`~/Dev/RAG/advanced-rag/RAG_STUDIO_PROMPT.md` (§16-17 for how to work and the
milestones). **Claude builds everything here**; the user reviews, runs and asks
questions.

## Rules the user set (don't re-ask, don't break)

- **Never run `git commit` or push.** At a milestone's end: `git add`, update
  this file, report neatly with every number, and end with a suggested commit
  message. The user commits and pushes. (Repeated by the user during M2.)
  One-off exception, 2026-10-23: the user asked Claude to commit M8 itself (the
  same one-line "Mn complete: ..." format) and then shut the laptop down.
- Repo `~/Dev/RAG/rag-studio`; UI 127.0.0.1:**3100**, API 127.0.0.1:**8775**
  (trading uses 3000/8765). Tracing (Phoenix) was cut.
- Local model pulls approved, each when its milestone starts: qwen3-embedding:0.6b,
  EmbeddingGemma 2 (via sentence-transformers, see research §4), Qwen3-Reranker-0.6B
  (HF), gemma4:e4b (pulled in M6, 6.6 GB). Docling's models are now downloaded (M2). qwen3.5:4b is already pulled.
- Git: PDFs and test questions are committed; indexes, runs and caches are rebuilt.
- Check `ollama ps` and `systemctl --user status trading-daily.timer` (19:17 IST)
  before any pull or long run. Never stop the trading timers.
- The user's own `./rag web` is usually running (dev mode, both servers
  reload). Test against it; don't start a second copy. Check free memory too:
  with VS Code open, only 2-4 GB is free; don't run browser QA beside a scorecard run.

## M8 status: complete (committed at the user's request, 2026-10-23)

**Built:** the full QA matrix, run in the user's DevTools Chrome (chrome-devtools-mcp 1.11.0, Chrome 155,
watched live): 20 pages (7 main + 13 How it works steps) × 375/768/1024/1280/1440 × light/dark = 200
renders per pass, each checked for page overflow, one `<h1>`, axe-core 4.14.0 (WCAG 2.1 AA + best
practice), console errors, failed requests and the applied theme; a focus audit on every page; real key
presses for the skip link, Ctrl+K, a Base UI select, the phone sidebar and the scroll boxes. The harness
is committed (`scripts/qa/harness-src.js`, `scripts/qa/build.cjs`; how to run it in CLAUDE.md, M8 UI
gotchas). Report: `docs/qa/2026-10-23-m8-full-matrix.md` (M8-01..13).

| Check | Result |
|---|---|
| Matrix after the fixes | **200 / 200 clean**: overflow 0, console 0, failed requests 0, one `<h1>` each, axe 0 apart from `scrollable-region-focusable` (M8-11, Safari only) |
| Keyboard | every focusable control on every page shows a visible focus and has a name; skip link, Ctrl+K, select, phone sidebar all work by keyboard |
| Tests | `./rag test` **573 passed, 1 skipped**; UI typecheck and lint clean |

**Fixed:** M8-01 scrollbar 10 → 6 px, transparent track (the user asked); **M8-02 (P1)** a reopened
conversation drew every citation as invented ("given 0 passages"): saved citations now name document
and page; M8-03 light `--muted-foreground` 0.556 → 0.54 (4.34 → 4.62:1 on muted); M8-04 badge and
chunk-chip colours (all ≥ 4.66:1); M8-05 chart-blue on chart-blue tints (embed prefix mark, Ask
citation chips); M8-06 select lists clipped long names (now as wide as their longest item); M8-07..10
(ARIA role, memory-notice wording, an empty table header, the sidebar landmark). **Open (P3):** M8-11
scroll boxes for Safari, M8-12 focus after Ctrl+K navigation; earlier M5-06, M5-07 (P2, kept), M6-03,
M6-04, M7-04. No P0-P1 open anywhere.

**Docs:** 00-start-here (all 13 steps interactive, M8 done, a "Left out of this version" table),
01-architecture (no tracing service; cut and deferred), 02-remote-gpu and research (no future tense
for cut features; research §11 QA tooling and scrollbars), ladder (row 11b cut; the final defaults),
steps/10 (the default answer model), steps/12 (Set C), plans/m6-m8 (M8 done), CLAUDE.md (M8 UI gotchas).

**Not covered:** a live Ask answer with a figure source (gemma4:e4b needs 8 GB; 6.4 GB was free with
VS Code open); Safari and Firefox.

## M6 status: complete (committed by the user as 9a9780d)

**Built:** ladder row 10 (two local answer models, thinking off verified live), How it works 9, 10,
12 and 13 interactive. Engine: `learn.py` (`recorded_answers`, `prompt_lab`, `cite_lab`,
`answer_check`, `evaluate_lab`, `safety_lab`/`safety_pages`/`safety_image`/`flag_check`,
`safety_live`, `write_crafted_pdf` shared with `tests/test_guard.py`), `eval.py` (`fact_spans`,
`pass_interval` Wilson 95%, `sign_test` exact McNemar, `run_head`, each answer records
`prompt: {chars, messages_sha256}`), `prompt.py` (`passage_entry`, `messages_sha256`), `answer.py`
(`citation_marks`), `guard.py` (`visibility_views`, `_renders` shared with the decision),
`config.MODEL_NOTES` for both models. API: `/api/learn/{answers, prompt, cite, answer-check,
evaluate, safety, safety/pages, safety/image, safety/flags, safety/live}`. UI: `prompt-lab`,
`cite-lab`, `evaluate-lab`, `safety-lab`, `recorded-picker`, `marked-answer`, `answer-try`; step
copy in `lib/stages.ts`. Tools: `scripts/eval_watchdog.sh` (memory watchdog, measures
`llama-server` RSS), `scripts/think_check.py`. Judge/RAGAS/tracing cut and said so everywhere.

| Check | Result |
|---|---|
| Tests | `./rag test` **541 passed, 1 skipped** (531 at M5); `tests/test_colab_worker.py`'s log test was racy under load (the worker logs after replying) and now waits for the line |
| UI | typecheck and lint clean; QA (`docs/qa/2026-10-22-m6-answer-models-how-it-works.md`): 4 pages at 390/768/1024/1440/1806 px no overflow, light + dark, console clean; findings M6-01..04 (01 P2 and 02 P1 fixed, 03-04 P3 open) |
| Thinking off | qwen3.5:4b and gemma4:e4b: 0 thinking characters; gemma4's system prompt verified by behaviour |

**Row 10** (Sets A+D Pass · Refused · Wrong, 6d + the 4B's cached scores, prompt v3; `docs/ladder.md`):

| Row | Answer model | Run | A+D | Set E | Answer median | Peak RSS |
|---|---|---|---|---|---|---|
| 6d | qwen3:1.7b | `20261016-160533-549e` | 22·1·2 | 7·0·1 | 17.2 s | - |
| 10a | qwen3.5:4b | `20261022-153229-8e7b` | 22·1·2 | 7·0·1 (`…155217-25e2`) | 43.7 s | ~6.7 GB |
| **10b** | **gemma4:e4b** | `20261022-160852-1695` | **24·1·0** | **8·0·0** (`…163004-803a`) | 47.3 s | 8.0 GB |

10b vs 6d: T4, T5 Wrong → Pass, nothing worse (sign test p = 0.5; with Set E 3/0, p = 0.25).
10a = 6d question for question on A+D, but its Wrongs are refusals followed by an explanation where
qwen3:1.7b invented facts. Injection inside a passage (crafted page, one call each): qwen3:1.7b gives
the injected 99 with v2 and v3; qwen3.5:4b and gemma4:e4b keep to 42 with both.

**Defaults were unchanged** (5d, prompt v3, dense, qwen3:1.7b, every switch off). **Decided by the
user 2026-10-23: gemma4:e4b.** (Was: which answer model becomes the default (candidate: gemma4:e4b, 24·1·0 but ~47 s and 8 GB;
qwen3:1.7b stays the light option).

**If `data/` is wiped:** the M2-M5 commands, then the row 10 scorecards (no Colab, the 4B scores must
be cached first, see M5): `scripts/eval_watchdog.sh r10b --set a b d --prompt v3 --rerank qwen3-4b
--profile docling-structure-embeddinggemma2 --model gemma4:e4b --ladder 10 --label "6d + gemma4:e4b
answers (10b)"` (10a: `--model qwen3.5:4b`, label "(10a)"), and Set E with `--set e` and labels
"Set E: follow-ups rewritten, Docling + 4B, <model>". Close VS Code/Vivaldi first for gemma4.

## M7 status: complete (committed by the user as 4734d19)

**Built:** figures as text. `rag_studio/vision.py` (regions from Docling's picture items, crop widened to a
title printed above, PNG drawn from the PDF and keyed by hash, prompt `vision-v1`, a description cache that
keeps bad outputs, a resumable Remote AI job per model, figure passages, export/import, `truth_check`
against values measured from the drawing); profile `docling-structure-embeddinggemma2-fig` (row 11;
`IndexProfile.figures`, recorded only when set); `./rag figures list|status|try|describe|check|export|import`;
Set C (`data/golden/set_c.json`, 4 questions, every key measured from the PDF's vector paths);
How it works 11 (`learn.figures_lab`, `/api/learn/figures`, `/api/figures/{id}/image`,
`components/learn/figures-lab.tsx`); figure sources "Figure (described by gemma4:12b)" and their page
box. **Default answer model = gemma4:e4b** (user) with `config.ANSWER_RAM_GB` and `answer.check_memory`
(Ask and eval refuse rather than swap), two-pass runs for large models, `/api/status` `memory.parts`.
Colab: `start.sh` releases each model after its check and everything before the worker, and fails
before the downloads below 3000 MiB free. `scripts/run_progress.sh` (a terminal view of a run).

| Check | Result |
|---|---|
| Vision model | gemma4:12b + vision-v1, chosen by a bake-off read by hand against the drawing (`docs/steps/11-figures.md`) |
| Figures | 288 found, 286 described in 50 min on the T4, 237 indexed, 49 photos/logos, 2 bad (cut at 900 tokens) |
| Tests | `./rag test` (see the close-out run below); UI typecheck and lint clean |
| UI | QA `docs/qa/2026-10-23-m7-figures.md`: M7-01 (P1), 02, 03 fixed; 04 (P3) open |

**Row 11** (gemma4:e4b answers, prompt v3, no reranker; `docs/ladder.md`):

| Variant | Run | Set C | Sets A+D | Answer median |
|---|---|---|---|---|
| 11 (before): 5d | `20261023-150923-8680` | 0 · 3 · 1 | 21 · 1 · 3 | 42.2 s |
| **11a: 5d + figures** | `20261023-153310-8b12` | **4 · 0 · 0** | 21 · 1 · 3 | 45.3 s |

No A or D question changed; the traps T1, T4, T5 stay refused with a figure passage in their top 5; C1's
pass is weak (it lists the parts in order). **The `-fig` index is now the default** (user, 2026-10-23:
`profiles.DEFAULT`; Ask, `./rag eval` and the labs use it). Open: the C1 key could demand "largest"
wording; not changed.

**If `data/` is wiped:** the M2-M5 commands, then `./rag figures import docling-structure-embeddinggemma2-fig`
(from the committed export) and `./rag index build docling-structure-embeddinggemma2-fig` (~5 min with the
vector cache, ~20 without), then the two row-11 scorecards (commands in `docs/ladder.md`, row 11).

## Next: nothing scheduled (the reduced plan is complete)

Candidates if the user wants more: the deferred browser upload and test-question editing; M8-11
(`tabIndex={0}` and a label on each scroll box, for Safari), M8-12 and the earlier P3s; a live Ask pass
with a figure source when ≥ 8 GB is free (close VS Code/Vivaldi); the 4B reranker as a default (needs
Colab). The figures picker's keyboard pass, carried over from M7, was done in M8.

## M7 log (how it went; the status is above)

### M7 progress: 2026-10-23 14:10 IST checkpoint

**Read this first when resuming M7.** M6 is committed by the user (9a9780d). Nothing staged for M7.

**14:10 — everything but the answer scorecards is done.** Colab session 2 finished and the user
deleted the runtime: 286/288 figures described by gemma4:12b (237 indexed, 49 photos/logos, 2 bad:
GEP p85 C, p114 A, cut at 900 tokens), exported to `data/figures/docling-structure-embeddinggemma2-fig.jsonl`
(303 KB, 288 rows; commit it like the context export). `-fig` index built here (1,710 rows, 237
embedded, 282 s). Retrieval-only (`20261023-084058-7910` 5d, `…084112-afa1` -fig, Sets A+C+D): every
A/D answer page at the same rank, Set C figure passage rank 1 for all 4, MRR 0.871 → 0.892; 7 A/D
questions get a figure in their top 5 (A3, A8, D4, D9, traps T1, T4, T5), all answerable facts still
in the top 5. Browser QA of How it works 11 done (`docs/qa/2026-10-23-m7-figures.md`, M7-01..03
fixed, 04 open P3). Docs: ladder row 11 (retrieval), steps/11 (run + bake-off), CLAUDE.md M7.
**Waiting for the user** (busy in VS Code/Vivaldi; chose "Later, with me"): when they close both,
run, one after the other (≥ 8 GB free, check `free -m`, trading not active):
`scripts/eval_watchdog.sh r11-before --set a c d --prompt v3 --profile docling-structure-embeddinggemma2 --model gemma4:e4b --ladder 11 --label "Default (5d), gemma4:e4b answers"`
then `scripts/eval_watchdog.sh r11a --set a c d --prompt v3 --profile docling-structure-embeddinggemma2-fig --model gemma4:e4b --ladder 11 --label "5d + figures, gemma4:e4b answers (11a)"`
(~25 min each; two-pass now applies to gemma4). Then: read every Set C answer and the traps T1/T4/T5,
ladder row 11 table + summary row, steps/11 Measured, Ask with a figure source (optional, 8 GB),
stage (incl. `data/figures/`, `data/golden/set_c.json`), HANDOFF, report, commit message.

**User decisions (2026-10-22/23, don't re-ask):** the vision pass runs on the **Colab T4, one
session** (bake-off qwen3.5:9b / gemma4:12b / qwen3.5:4b on the Set C figures, keep the one whose
values match the drawing, describe everything, same session fills the 4B reranker cache for the
figures index). **gemma4:e4b becomes the default answer model** (not switched in code yet).
The user stressed: **image analysis must be accurate first**, before judging the RAG: every Set C
key comes from the PDF's vector geometry + a 300 dpi crop + a full-text search, never a visual guess
(my 90-dpi reading of a poverty bar was off by 0.4 points; the drawing is the truth).

**Done:**
- `data/golden/set_c.json` (new, unstaged): C1 iMac p2 largest part of the 445 kg baseline =
  materials and process (171 of 401 pt ≈ 190 kg; iMac's own bar is led by product use 156 kg);
  C2 GEP fig 2.7.D South Asia poverty 2028 = 18.93% (bars 20.78/20.07/19.33/18.93; key 18.6-19.2);
  C3 GEP fig 3.1.C non-investment-grade spreads 2020 pandemic ≈ 525 bp (310 in 2022; IG 191/36;
  key 495-555; text has distractors 190/114 bp); C4 GEP fig 3.1.D largest 2026 risk category =
  moderate risk 40.0% (high 34.3, distress 13.4, low 12.1). Each has a `truth` block (region id,
  every value, axis span, tolerance). Keys tested on 15 sample answers, 0 mismatches. None of the
  values occurs in the text.
- `rag_studio/vision.py` (new): regions from Docling pictures (≥0.5% of the page; crop widened to a
  short title printed ≤24 pt above, e.g. "D. Poverty"; figure heading "FIGURE 2.7 SAR: Outlook"
  added by code; panel-title sections stripped), PNG render 150-400 dpi (long side ~1024 px),
  render-key index so images are drawn once, prompt `vision-v1` (Title/Kind/Axes/Series/Data table
  with '~' for estimates/Largest/Takeaway), cache `data/cache/figures/<model>/<digest>/<prompt>/
  <image sha>.json` incl. bad outputs, resumable job, `passages()` for the build (photos/logos by
  the model's Kind line are not indexed), export/import `data/figures/<profile>.jsonl`,
  `truth_check` (reading per value, error as a share of the axis span).
- `profiles.py`: `IndexProfile.figures` (config key only when set; 16 profiles, 0 recorded configs
  changed) and `docling-structure-embeddinggemma2-fig` (row 11). `ingest.passages_for` adds figure
  passages; the manifest records `figures`. `paths.FIGURES`; `tests/conftest.py` tmp_data has it.
- CLI `./rag figures list|status|try|describe|check|export|import` (and in `./rag`'s router).
- Measured locally on iMac p2 (old prompt): qwen3.5:4b 66 s, 3.7 GB, named the largest baseline part
  right but misordered the iMac bar; gemma4:e4b 47 s, 7.4 GB, gave no segment sizes.
- 288 regions (iMac 17, GPO 49, GEP 222). **GEP draws ~57 chart panels as raster images with no text
  layer**, so neither text nor vector paths can separate charts from photos: the model's Kind decides.
- Research (agent, 2026-10-22): no MCP/skill beats the PyMuPDF vector truth; adopted the data-table
  prompt (WB-ChartExtract, arXiv 2605.27298), axis-span error (PaperUnPlot NMS), per-panel crops
  (Gemma 4 caps an image at 1,120 tokens). Optional second reader: Docling's Granite Vision 4.1 4B
  chart extraction (in our docling 2.139.0; ~8 GB on Colab) — not asked yet.

**12:30 — Colab session 1 FAILED (my fault, ~40 min of the user's T4 lost):** the cell
`--models qwen3.5:9b gemma4:12b qwen3.5:4b --worker qwen3-0.6b-gpu qwen3-4b` pulled ~27 GB, then the
worker OOMed: start.sh left every startup-checked model resident (keep-alive 30m; 6.2 + 8.0 GB) and
the worker had 60 MB of the T4. Ollama does not evict for the worker (it pushed qwen3.5:4b to the CPU
instead). The user deleted that runtime. **Fixed in `tools/colab/start.sh`:** `release_model` after
each model's check, `release_all_models` + a free-GPU check (die below 3000 MiB) BEFORE the worker
downloads anything; the CLI's `figures try/describe` unload the vision model when done. Tested:
syntax, 147 Colab/remote tests, release against the local Ollama 0.32.6, the cell embeds the fix
(start.sh sha ccb0d9287d09). **User decision (12:27): vision only, 2 models** —
`./rag llm colab --models qwen3.5:9b gemma4:12b --copy` (~15 GB, no worker); qwen3.5:4b's bake-off
runs locally; **row 11 is measured without the reranker** (5d + figures vs 5d, gemma4:e4b answers);
a reranker row only later, via a small second cell on the same runtime, if the user asks.

**12:50 IST — Colab session 2 (vision only) RUNNING:** runtime `times-famous-driver-fly`
(`--models qwen3.5:9b gemma4:12b`, no worker). Bake-off done (`docs/steps/11-figures.md` has the
whole table, read by hand AND `./rag figures check`): **gemma4:12b + vision-v1 chosen** (C1 4/8 but the
answer right, C2 7/7, C3 4/4, C4 8/8 incl. stacked; ~19 s/figure); qwen3.5:9b cut off on C1 and
botched C4; qwen3.5:4b (local) fails both stacked charts; vision-v2 (stacked-bar hints) helped no one
and hurt gemma4 on C1 (kept as a record). `vision.VISION_MODEL = "gemma4:12b"`, config vision role =
gemma4:12b (remote). **The full run started 07:18 UTC:** `./rag figures describe --model gemma4:12b
--prompt vision-v1 --on colab` (288 figures, resumable: if cut, rerun the same command). After it:
`./rag figures status --model gemma4:12b`, `./rag figures export docling-structure-embeddinggemma2-fig`,
disconnect (`./rag llm disconnect`; tell the user the runtime can be deleted), then build the `-fig`
index here (`./rag index build docling-structure-embeddinggemma2-fig`, EmbeddingGemma 2, ~3.2 GB free
needed; the other 1,473 passage vectors come from the vector cache), then row 11 (no reranker):
`--set a c d --profile docling-structure-embeddinggemma2-fig --prompt v3 --model gemma4:e4b --ladder 11`
and the same on `docling-structure-embeddinggemma2` (5d + gemma4, the before row), both under
`scripts/eval_watchdog.sh`, VS Code/Vivaldi closed.

**13:10 IST checkpoint:** describe at 135/288 (0 bad, 0 cut, ~12 s/figure; Kind lines cross-checked
against colour counts: no chart wrongly called a photo/logo so far). Done since 12:50: How it works 11
(`learn.figures_lab`, `/api/learn/figures`, `/api/figures/{id}/image`, `components/learn/figures-lab.tsx`
built by the builder agent and reviewed: fixed `after.bad` (an object, would have crashed), container
queries, wording), step 11 `explain` in `lib/stages.ts`, docs: `docs/steps/11-figures.md` (bake-off),
research.md §5.1, 02-remote-gpu.md (the failed session + fix), 01-architecture.md, product-language.md
(figure passage, figure). `tests/test_figures_api.py` (7). **Full suite 573 passed, 1 skipped**
(`test_local_and_colab_vectors_match` now skips when the runtime lacks the model). UI tsc + lint clean.
Not done yet: browser QA of /learn/figures and a figure source in Ask (needs `./rag web` + the built
`-fig` index), ladder.md row 11, 00-start-here.md, CLAUDE.md M7 UI gotchas if any.

**Also done this session:** default answer model = gemma4:e4b (`config.ROLES`), `config.ANSWER_RAM_GB`
(qwen3:1.7b 5.4, qwen3.5:4b 6.7, gemma4:e4b 8.0 GB), `answer.check_memory` before generating (Ask and
eval refuse with "Not enough free memory to load …"; an eval run FAILS on it, keeping finished
questions), `/api/status` `memory.parts` + Ask's notice names both parts; figure sources
`described_by` + the badge "Figure (described by …)"; the boxes endpoint returns a figure's own box
(`:pN:fN` chunk ids); table-aware `truth_check`; per-model figures job lock. Tests: affected suites 118
passed; tests/test_health.py updated for the new default.

`tests/test_vision.py`: 22 passed (11:00). **Superseded next step:** ask the user to start Colab with
`./rag llm colab --models qwen3.5:9b gemma4:12b qwen3.5:4b --worker qwen3-0.6b-gpu qwen3-4b --copy`
(the cell builds, 43 KB; the worker loads its FIRST reranker at start, so the 0.6B (2 GB) goes first
and the 4B (~9 GB peak) swaps in only after the vision model is unloaded: they do not fit together); `./rag figures
try --models qwen3.5:9b gemma4:12b qwen3.5:4b`; read every description against the truth; pick
`vision.VISION_MODEL`; `./rag figures describe`; export; build `-fig` here; retrieval-only runs with
`--rerank qwen3-4b` on Colab for Sets A, C, D on `-fig` and Set C on 5d; disconnect; scorecards
(gemma4:e4b, watchdog) row 11; switch the default answer model; API/UI (figure sources, boxes,
How it works 11); docs; QA.

## M5 status: complete (committed by the user as 178c0d4)

**Built:** the reranker (`rerank.py`: Qwen3-Reranker 0.6B here, 0.6B/4B/8B-8bit on the Colab GPU
worker, pinned revisions, scores cached), contextual retrieval (`context.py`: windows, prompt
context-v1, resumable Remote AI job, export), query rewriting (`rewrite.py`: split into parts,
several phrasings, draft answer first, follow-up rewrite; JSON schema; cached), multi-step search
(`agent.py`: 3 rounds, validated document filter, stop reasons), conversations (`chat.py`), the one
pipeline `retrieve.run`, two-pass evaluation, MRR and recall@20, Set E (8 follow-ups). Colab: the
gateway v2 + `worker.py` (rerank, context), gzip both ways, keep-alive. **UI:** Ask as a
conversation ("Searched for", Ask as typed, saved list), reranker/rewriting/multi-step switches;
Search lab Reranked column; How it works 4 and 8; Evaluation switches, Set E table, "How it was
searched"; Indexes "Context lines" section; Settings roles by machine, Defaults, setup-cell jobs.

| Check | Result |
|---|---|
| Tests | `./rag test` **531 passed, 1 skipped** (288 at M4) |
| UI | typecheck and lint clean; QA findings M5-01..07 (01-05 fixed, 06-07 open, P3/P2) in `docs/qa/2026-10-22-m5-rerank-rewrite-chat.md` |
| M5 target: B4 "global growth" a Hit on 5d | **rank 8 → 1** with every reranker (`20261016-154536-14cd` and later) |

**Ladder (Sets A+D Pass · Refused · Wrong, qwen3:1.7b; all in `docs/ladder.md`):**

| Row | Best variant | A+D | Note |
|---|---|---|---|
| 5d (start) | Docling + EmbeddingGemma 2 | 20·3·2 | B4 missed |
| **6d** | + Qwen3-Reranker 4B (Remote AI) | **22·1·2** | best row; 0.6B 20·2·3, 8B 21·1·3 |
| 7b | 6d + context lines | 21·2·2 | MRR up, D10 lost; 7a not measured (user) |
| 8a/8e | 6c + split into parts / draft answer first | 21·1·3 | PyMuPDF +1 (D10); Docling 8b/8d/8f all = 6d |
| 9a / 9b | + multi-step search | 21·1·3 / 22·1·2 | no answer changed; 10% bad controller decisions |
| Set E | follow-ups rewritten vs as typed | 7·0·1 vs 5·1·2 | memory worth +2 Pass, +3 hits |

**Defaults are unchanged** (5d, prompt v3, dense, every switch off). **Waiting for the user:** which
techniques become defaults (candidates: the 4B reranker on Remote AI, or the 0.6B here), and
whether to commit `data/context/docling-structure-embeddinggemma2-ctx.jsonl` (1.0 MB, 1.38 h of T4,
not regenerable word for word; left unstaged).

**If `data/` is wiped:** the M2-M4 commands, then (Colab with `--models qwen3:8b --worker context
qwen3-0.6b-gpu qwen3-4b`) `./rag context import docling-structure-embeddinggemma2-ctx` (from the
export, if committed) or `./rag context write …`, `./rag index build docling-structure-embeddinggemma2-ctx`,
then each row's retrieval-only run on Colab (fills the caches) and its scorecard here:
`./rag eval --set a b d --prompt v3 --rerank qwen3-4b [--rewrite …] [--agent 3] --profile <p>
--ladder N --label "<…>"` (run ids and labels in `docs/ladder.md`).

## M6 log (how it went; the status is above)

### M6 progress (update after every phase): 2026-10-22 21:11 IST checkpoint

**Read this first when resuming M6.** M5 is committed (178c0d4). Nothing staged yet for M6.

- **Thinking off, verified live:** qwen3.5:4b with `think: false`: 0 thinking chars, no `<think>`
  tag, 67 answer tokens on the FDsys question (prompt 1,339 tok in 54 s ≈ 25 tok/s, writing 5.8 tok/s,
  ~65-77 s per answer, 3.4 GB resident at num_ctx 8192). **gemma4:e4b pulled** (6.6 GB, 7.5B, Q4_K_M,
  capabilities include thinking **and vision**: a candidate for M7 locally); its thinking check is
  NOT done yet (`.venv/bin/python scripts/think_check.py gemma4:e4b`: streams `/api/chat` with `think: false`, count
  `message.thinking` chars and `<think>` in content, on a real v3 prompt). Disk: 41 GB free.
- **21:22 Row 10a DONE:** `20261022-153229-8e7b` qwen3.5:4b **22·1·2**, recall 9/6/11 as 6d; **no
  outcome changed vs 6d on any of the 25 scored questions**; answer median 43.7 s (28.7-87.4) vs
  6d's 17.2 s (9.3-41.2), answer_seconds 1,162 vs 6d's; median 59 answer tokens. Its Wrongs T4, T5 are
  refusals followed by an explanation ("I could not find this in the documents. The provided text
  only contains…"): no invented fact, but not the exact refusal (strict rule, spec §5.7). D11
  refused. Memory: llama-server RSS ~6.7 GB observed at 21:19 (the running watchdog matched
  `ollama runner`, so its JSON says 0; fixed in `scripts/eval_watchdog.sh`), MemAvailable min 1,476 MB,
  eval process peak 2,043 MB. Set E for qwen3.5:4b started 21:22.
- **21:40:** Set E qwen3.5:4b `20261022-155217-25e2` **7·0·1** (= qwen3:1.7b's 7·0·1; E6 W→P, E7 P→W:
  refusal + explanation again; all 3 of qwen3.5:4b's Wrongs are that habit), median 51 s/answer.
  UI **typecheck + lint clean**; `./rag test` **541 passed, 1 skipped** (5 min 22 s).
  **gemma4:e4b thinking off verified:** 0 thinking chars, no tag; FDsys answer right with [1, 3];
  prompt 1,370 tok in 43.2 s (32 tok/s), writing 58 tok in 7.7 s (7.5 tok/s), load 28.9 s;
  llama-server peak RSS **7,490 MB**, MemAvailable min 2,035 MB (from 9.8 GB, VS Code open, Vivaldi
  closed). Template is `{{ .Prompt }}` with `RENDERER gemma4` + a DRAFT model (speculative,
  draft_num_predict 3); **system prompt verified** (system "reply PINEAPPLE" → "PINEAPPLE").
  Next: row 10b + Set E (asked the user to close VS Code first), browser QA, docs, ladder.
- **22:08: ROW 10 MEASURED.** 10b `20261022-160852-1695` gemma4:e4b **24·1·0** (A 11·0·0, D 13·1·0;
  recall 9/6/11, MRR 0.894 = 6d), only changes vs 6d and 10a: T4, T5 Wrong → Pass (exact refusals);
  D11 still Refused. Median answer 47.3 s (29.4-101.6), answer_seconds 1,256, run 1,269 s; llama-server
  peak **8,024 MB**, MemAvailable min 2,074 MB. Set E `20261022-163004-803a` **8·0·0** (rewrite_ok 6/8,
  hits 7/7 with pages), median 60.1 s; E6 fixed vs qwen3:1.7b, E7 fixed vs qwen3.5:4b; runner peak
  8,065 MB. Sign test 10b vs 6d on A+D: 2 better, 0 worse, p = 0.5 (a hint). **Next:** MODEL_NOTES
  for gemma4, ladder.md row 10, docs/10 Measured, browser QA, stage, report, ask the user which answer
  model becomes the default (defaults unchanged until then).
- (was) **21:42: row 10b running** (started 21:38 with 11.3 GB available; Vivaldi/VS Code still open but
  mostly swapped): run `20261022-160852-1695` "6d + gemma4:e4b answers (10b)", then Set E "Set E:
  follow-ups rewritten, Docling + 4B, gemma4:e4b", both under `scripts/eval_watchdog.sh` (JSON in
  `data/cache/eval-watch/r10b*.mem.json`). Rerun commands = the 10a ones with `--model gemma4:e4b`.
  **Key 10a finding for the report:** 6d's T4/T5 Wrongs (qwen3:1.7b) invent facts (MacBook Air
  question answered with the iMac's 346 kg; "2.8 percent for 2031"); 10a's T4/T5 Wrongs are refusals
  followed by an explanation. Same scorecard, different safety. Docs done: 09, 10 (How it works part),
  12, 13, 01-architecture (API M6 + learn.py), CLAUDE.md M6 gotchas, trace.py/settings "judge" wording.
- **21:27: all M6 UI written (typechecked/linted 21:35; NOT browser-tested):** `lib/api.ts` M6 types +
  `api.answerCheck`/`api.safetyLive`; `components/learn/recorded-picker.tsx` (hook `useRecordedAnswer`),
  `marked-answer.tsx`, `answer-try.tsx`, `prompt-lab.tsx`, `cite-lab.tsx`, `evaluate-lab.tsx`,
  `safety-lab.tsx`; WIDGETS in `app/learn/[slug]/page.tsx` (prompt, generate, evaluate, safety);
  `explain` triplets in `lib/stages.ts`. API: `/api/learn/{answers,prompt,cite,answer-check,evaluate,
  safety,safety/pages,safety/image,safety/flags,safety/live}`. Tests: `tests/test_learn_answer.py`
  (new, not run yet) and `tests/test_guard.py` now builds the PDF with `learn.write_crafted_pdf`.
  Set E for qwen3.5:4b: run `20261022-155217-25e2` (4/8 at 21:26, 4 Pass, rewrite_ok 6/8, hits 7).
  **Next:** after Set E: `cd web/ui && npx tsc --noEmit && npm run lint`, `./rag test`, gemma4 thinking
  check, ask the user to close VS Code/Vivaldi, row 10b + Set E with `scripts/eval_watchdog.sh`,
  browser QA of /learn/{prompt,generate,evaluate,safety}, docs 09/10/12/13, ladder row 10, MODEL_NOTES.
- (was) **Row 10a running** (background, dies with the session): run `20261022-153229-8e7b`
  "6d + qwen3.5:4b answers (10a)" (17/31 at 21:11), then Set E "Set E: follow-ups rewritten, Docling +
  4B, qwen3.5:4b". If either is missing or `interrupted`, rerun exactly (no Colab needed; all 4B scores
  and rewrites are cached):
  `./rag eval --set a b d --prompt v3 --rerank qwen3-4b --profile docling-structure-embeddinggemma2 --model qwen3.5:4b --ladder 10 --label "6d + qwen3.5:4b answers (10a)"`
  and `./rag eval --set e --prompt v3 --rerank qwen3-4b --profile docling-structure-embeddinggemma2 --model qwen3.5:4b --label "Set E: follow-ups rewritten, Docling + 4B, qwen3.5:4b"`.
  Then the same two with `--model gemma4:e4b`, label "(10b)". Compare against 6d `20261016-160533-549e`
  (22·1·2, qwen3:1.7b) and Set E `20261022-092055-d263` (7·0·1). Report Pass·Refused·Wrong, seconds
  per answer (`summary.answer_seconds` / scored), peak memory (watchdog JSON).
  Memory watchdog: `scripts/eval_watchdog.sh <name> <eval args>` (output in `data/cache/eval-watch/`) runs `./rag eval` in the background, samples MemAvailable and
  the RSS of `ollama runner` + the eval every 3 s, SIGINTs the eval below 800 MB, writes min/peak
  to a JSON. gemma4:e4b likely needs ~7+ GB: check `free -m` and ask the user to close VS Code /
  Vivaldi first (8.2 GB available with both open at 21:00).
- **Engine edits done (unstaged, untested):** `prompt.messages_sha256()` and `prompt.passage_entry()`
  (context_block uses it); `eval.py`: docstring (judge cut), `fact_spans()`, `pass_interval()` (Wilson
  95%), `sign_test()` (exact two-sided McNemar), `run_head()` (was `_run_head`), and `_answer_one`
  now records `res["prompt"] = {chars, messages_sha256}` per answer; `answer.citation_marks(text, k)`;
  `guard.py`: `_renders()` split out of `_visibility()`, `visibility_views()` (page / text removed /
  changed pixels as PNGs at the guard's 96 dpi and threshold 16), `_png()`.
- **21:30: the `learn.py` block below is WRITTEN and smoke-tested on real runs** (prompt_lab on 6d A1:
  1,502 tokens measured, 4.08 chars/token; 6d vs 5d: 2 better, 0 worse, sign test p = 0.5; T3 v2 Wrong
  → v3 Pass on 3 setups; crafted page: all 5 tricks found, 3 spans kept). Not yet run: `_safety_pages()`
  (the guard.py edit changed the cleaning cache key, so it re-cleans all PDFs once, ~30 s+; do it
  after the scorecards), `safety_live`. **Next: API routes, then the UI labs, tests, docs.**
- Design of that block (as built):
  `_done_answer_runs()` (full done answer runs, memoised by run-file mtimes), `recorded_answers(qid)`
  (picker: runs that answered it, default setup first, ladder next; skip runs whose LanceDB table is
  gone), `_recorded(run, qid)` (rebuild sources from the run's own table via `store.get_passage` by the
  recorded `hits` chunk ids, refuse if the prompt sha changed or the table is gone; `exact` =
  identical / rebuilt / differs via `messages_sha256`), `_setup_key(cfg, prompt=, model=)` to find
  the same passages answered with another prompt version or answer model;
  `prompt_lab` (messages, parts + budget bar split by chars of the real `prompt_tokens`, v2/v3/v4
  side by side with changed lines and measured outcomes), `cite_lab` + `answer_check(run, qid, text)`
  (POST; citations, marks, strict refusal, score + fact spans: the user edits the answer),
  `evaluate_lab(run, against)` (Hit/Miss/no-pages × outcome grid, facts with spans, MRR/recall@20 per
  question, Set E chain, Wilson interval, paired sign test via `evaluation.compare`; the two
  2026-10-14 baseline Set D runs gave word-for-word identical D3 answers: temperature 0 repeats, the
  key changed), `safety_lab(source, page)` (crafted page from a shared `write_crafted_pdf()` that
  `tests/test_guard.py`'s fixture will also use, cached in `data/cache/learn/`; real hidden-text pages
  from `library.clean_doc(..., "pymupdf").pages[].hidden`; images via `guard.visibility_views`;
  flags; v2 vs v3 fence on the crafted passage; T3 measured v2 vs v3), `flag_check(text)`, and
  `safety_live(version)` through the real `answer.ask` with a `retrieve.Found` holding the crafted
  passage and a stand-in object with `index_info()`. Then API routes `/api/learn/{answers,prompt,cite,
  answer-check,evaluate,safety,safety/image,safety/flags,safety/live}`, UI labs
  (`prompt-lab.tsx`, `cite-lab.tsx`, `evaluate-lab.tsx`, `safety-lab.tsx`, WIDGETS in
  `learn/[slug]/page.tsx`, `explain` triplets in `lib/stages.ts`), tests, docs 09/10/12/13,
  `config.MODEL_NOTES` for the new models, QA doc, ladder row 10.

### The M6 scope as decided (2026-10-22)

**Read `docs/plans/m6-m8.md` first: it is the scope now and wins over the spec.** The user's
answers (2026-10-22, don't re-ask): defaults stay as they are (all M5 switches off); the
context-line export is committed; M6 = qwen3.5:4b **and** gemma4:e4b locally (the 6.6 GB pull is
approved) + How it works 9, 10, 12, 13; the judge, RAGAS and tracing are **cut**; M7 = figures as
vision descriptions only (Set C, row 11) + How it works 11; browser upload is **deferred**; M8 =
the full QA matrix + docs. Estimate: about 21 h of work, 3 sessions, 1-2 short Colab sessions.

Before any long local run: `free -m` (≥ 7 GB available; ask the user to close VS Code/Vivaldi),
`ollama ps`, the trading timer (19:15 IST; a skip needs the user's word, and a reboot wipes it), and
run scorecards with the memory watchdog (`phase_s9.sh` pattern in the M5 log).

## M4 status: complete (committed by the user as 8f3d2e4)

**User decisions (2026-10-15):** row 5 on **both parsers** (6 profiles:
{pymupdf, docling}-structure × {Qwen3-Embedding 0.6B, EmbeddingGemma 2,
Qwen3-Embedding 8B}); the **8B is built and queried on Colab only** (no local
pull). **Defaults stay baseline + prompt v2**: the user did not pick "3b
default"; ask again with row 5 (candidates 3b and 5d).

**Engine:** `remote.py` (connection + remembered token in
`~/.local/state/rag-studio/`, 0600 atomic; `rsc1.` connect code; connect →
smoke test → probe; disconnect), `colab.py` (the setup cell from
`tools/colab/`), `llm.py` remote client (own timeouts and wording, 401 → auth,
tunnel error pages, `/api/gpu`, embed `options`), `embed.py` (backends: Ollama
local/Colab and sentence-transformers in-process with a 2.2 GB free-memory
guard; `RemoteUnavailable`; the vector cache per digest/prefix/options/machine
now also makes builds resumable; the vector match check), `profiles.py`
(row-5 profiles; `EmbeddingSpec.describe()` keeps every earlier config
byte-identical; `build_on`/`query_on` are policy, not config; `BUILD_RAM_GB`,
`EMBED_RATE`), `ingest.py` (`on=local|colab`, one build at a time, never
beside an eval, memory guard, job files, cancel), `indexes.py` (the Indexes
page's facts), `learn.embed_lab` (step 5), CLI `./rag llm …` and
`./rag index list|build|check|download`, `ingest --on`. **Colab:**
`tools/colab/server.py` (allowlist with 404 before auth, bearer token, 8 MB
cap, gzip for big replies (not yet measured on a runtime), content-free log)
and `start.sh` (7 checked steps, Ollama pinned to the laptop's version, many
models, exact-tag GPU residency). **API:** `/api/llm/{status,colab,connect,
test,disconnect}`, `/api/indexes/{id}/{build,job,cancel,check}`,
`/api/learn/embed`. **UI:** Settings › Model backend (connect dialog, setup
cell), the Indexes page (model × parser, live builds, match check), How it
works step 5 (PCA map, Matryoshka, mixed models), pickers honour "can be asked
now". New dependency: sentence-transformers 6.1.0 + transformers 5.19.0.
Models: qwen3-embedding:0.6b pulled; EmbeddingGemma 2 in the HF cache
(revision 914f7f89…).

| Check | Result |
|---|---|
| Tests | `./rag test` **288 passed** (172 at M3), incl. the real local-vs-Colab test while connected |
| UI | typecheck and lint clean; QA 40/40 page × width × theme after fixes, console clean, 11 findings all fixed (M4-01 the user's connect dialog bug): `docs/qa/2026-10-15-m4-remote-indexes-embed.md` |
| Remote connect and test from the UI | done on the real T4 (URL + remembered token); the user connected first with the CLI |
| Vector match check (spec) | Qwen3 0.6B, this computer vs T4: **0.99954 / 0.99953** (passages), 0.99963 (questions): passed |

**Row 5** (Sets A+D, Pass · Refused · Wrong; vs 3a 19·0·6, 3b 20·1·4):
5a PyMuPDF+Qwen3 0.6B 17·2·6 (`20261015-162847-f165`); 5b Docling+0.6B
15·1·9 (`…164313-5e5b`); 5c PyMuPDF+EmbeddingGemma 2 18·2·5
(`…152925-06aa`); **5d Docling+EmbeddingGemma 2 20·3·2** (`…153719-e38d`);
5e PyMuPDF+8B 17·3·5 (`…160114-55ff`); 5f Docling+8B 19·2·4
(`…161859-51c0`). D9 (Spanish) fixed by every new model; only EmbeddingGemma 2
finds A5 and has Set A recall 9/9; Qwen3 does not beat nomic. Details and
reasons: `docs/ladder.md` (Row 5).

**Findings worth keeping** (CLAUDE.md, M4 gotchas): a local build at Ollama's
default context froze the laptop (memory guard now); through Ollama a
vector depends on the previous request (prompt-cache prefix reuse; 0.9996 for
0.6B, 0.9985 for the 8B), so the match check embeds one text per request and
measures a noise floor; an early "num_ctx moves vectors" claim was that noise
(corrected everywhere); EmbeddingGemma 2 in-process is exactly reproducible;
the tunnel, not the T4, limits Colab builds (1.7 MB JSON per 8B batch).

If `data/` is wiped: the M2/M3 commands, then `./rag index build
<parser>-structure-embeddinggemma2` (~15-17 min each, here), connect Colab and
`./rag index build <parser>-structure-qwen3-0.6b --on colab` (~4 min) and
`…-qwen3-8b --on colab` (~16-18 min), then the row-5 evals:
`./rag eval --set a b d --profile <p> --prompt v3 --ladder 5 --label
"<Parser> + structure, <Model>"`.

## M3 status: complete (committed by the user as eb917ab)

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
(~11 min each). Row 1 (the defaults changed in M5, so pin them): `./rag eval --set a b --profile
baseline --prompt v2 --ladder 1 --label Baseline` and `./rag eval --set d
--profile baseline --prompt v2 --ladder 1 --label "Baseline: edge cases"`.

## Decisions settled (M1-M4)

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
- **Row 5 on both parsers; the 8B queried on Colab only** (user, M4).
- **Where an index embeds is policy, not config**; the match check embeds one
  text per request and judges only across machines.

## Not done, on purpose (where it goes)

**From M5 (2026-10-22):**
- Context lines for PyMuPDF: 170 of 1,214 written; 7a not measured (user: save T4 time).
- `agent-v2` (require `query` when the controller says "search"): the 10% bad decisions; a new
  prompt version means a new measurement.
- The 0.6B reranker on this CPU (fp32) vs on the T4 (fp16): agreement not measured; rows 6a/6b used
  the T4. Dynamic int8 on the CPU not measured. Gzip on replies over 16 KB still unmeasured on a runtime.
- QA gaps (see the QA doc): page-level 768/1024/1440 for the new pages, a keyboard-only pass over the
  new controls, dark theme on How it works 4 and 8.
- Context-line quality: 42% still begin "The passage…" (prompt context-v1 not revised).

- Docling's letter-spaced fragments and spaced footnote numbers: shown, not repaired.
- Editable test questions in the UI: M8. Set C (figure question): M7.
- How it works widgets: steps 4, 8 (M5), 9, 10, 12 (M6), 11, 13 (M7). Step 5 done in M4.
- M4: the gateway's gzip is unmeasured on a real runtime; Qwen3 Ollama-vs-ST
  parity unverified (needs a 1.2 GB HF download); a document-specific Qwen3
  query instruction is an untried experiment (row 5 used the card's default);
  no 4B profile (optional in research); no local 8B (user's choice); the
  rerank service on Colab is M5; answer/vision/judge on the GPU and their
  optional loud fallback are M6; EmbeddingGemma 2 images are M7. A connect
  code was never pasted into the browser on a real runtime (the URL +
  remembered token path was).
- PyMuPDF tables are still flattened (no `find_tables`): Docling's whole-table
  passages are what helped D6/A8. An exact-match column for numbers in BM25
  was not built (research.md §8).
- Semantic chunking's document-wide threshold (90th percentile) misses the p19
  cut the page-level 80th finds; not tuned (it is an option, not a default).
- Scrollbars tested in Chrome only.

## M5 log (how it went; the status is above)

### M5 progress (update after every phase)

- **2026-10-22 session (resumed after the weekly limit cut the 2026-10-16 one):**
  - State found: 507 tests pass; P0-P2 done; **row 6 measured** (6a-6f, all on the T4, scores
    cached; written into `docs/ladder.md`; **B4 is a Hit on 5d**: rank 8 → 1, run
    `20261016-154536-14cd`; best row 6d Docling + 4B 22 · 1 · 2). Engine for P3-P5 (context,
    rewrite, agent, chat, Set E) was done; Ask's conversation UI was half done.
  - User decisions (2026-10-22): **context lines for Docling only** (7a not measured, to save T4
    time); **row 9 in a short second Colab session** after the row-8 scorecards.
  - Done today: Ask conversation UI finished (list refresh, save warning); Search lab **Reranked
    column** + "Lines: Fusion | Reranking"; How it works **step 4** (`context-lab.tsx`,
    `learn.context_lab`) and **step 8** (`rewrite-lab.tsx`, `learn.rewrite_lab` replays runs,
    `rewrite_live`); Evaluation: rewrite/multi-step/as-typed switches, Set E table, MRR and
    recall@20, per-question "How it was searched"; Settings: roles say where they run (Local AI /
    Remote AI only), python roles show downloaded/loaded, a read-only **Defaults** card, the setup
    cell's "What the runtime is for" picker; Health shows the GPU worker. Fixed: the context export
    test leaked into `data/context/` (`paths.CONTEXT` now redirected by `tmp_data`); a timed-out
    `/context` batch is asked again instead of failing the job (seen live once in ~120 batches).
  - Colab session 1 (runtime `journals-formed-…`, `--models qwen3:8b --worker context
    qwen3-0.6b-gpu qwen3-4b`): Docling context lines resumed from 769/1,473. Next on it:
    `precache_rewrites.py` (every rewrite + Set E follow-up, index-independent), local build of
    `docling-structure-embeddinggemma2-ctx`, then `phase_rerank.sh` (unload qwen3:8b; 4B scores for
    7b, row 8 on 6d / 7b / 6c bases, Set E). Both scripts are in the session scratchpad; their
    commands are plain `./rag eval --mode retrieval …` runs and are easy to retype.
  - **13:21 checkpoint:** all 1,473 Docling context lines written (1.38 h of T4 model time in all, median
    2.9 s/line; exported to `data/context/docling-structure-embeddinggemma2-ctx.jsonl`, 1.0 MB; the
    first runtime died at 1,335 and the user restarted it, new URL `nations-snow-con-avatar`);
    `docling-structure-embeddinggemma2-ctx` built here in 1,284 s; all 101 rewrites cached (305 s);
    QA findings M5-01..06 in `docs/qa/2026-10-22-m5-rerank-rewrite-chat.md` (05 fixed: the Indexes page
    never drew the `-ctx` indexes). 7b retrieval (`20261022-075111-2af7`): MRR 0.894 → 0.925 but D10
    (two documents) rank 4 → 24.
  - **14:40 CHECKPOINT (session limit near). Resume from here.**
    - Colab is **disconnected** (session 1 done: 14 retrieval runs, 0 errors, every 4B score,
      rewrite and Set E follow-up rewrite **cached**: scorecards replay with Colab off).
    - **Scorecards done** (all `--prompt v3 --rerank qwen3-4b`, Sets A+D = Pass · Refused · Wrong):
      | Row | Run | A+D | Recall A/B/D | MRR |
      |---|---|---|---|---|
      | 7b Docling ctx + 4B | `20261022-081914-e582` | 21·2·2 | 9/6/10 | 0.925 |
      | 8a PyMuPDF + 4B + decompose | `20261022-082934-1484` | 21·1·3 | 9/6/11 | 0.952 |
      | 8c PyMuPDF + 4B + multi | `20261022-083900-5409` | 20·2·3 | 9/6/10 | 0.925 |
      | 8e PyMuPDF + 4B + hyde | `20261022-084756-90f9` | 21·1·3 | 9/6/11 | 0.933 |
      | 8b Docling (6d) + 4B + decompose | `20261022-085553-67f6` | 22·1·2 | 9/6/11 | 0.894 |
      | 8d Docling (6d) + 4B + multi | `20261022-090353-c357` | 22·1·2 | 9/6/11 | 0.914 |
      | 8f Docling (6d) + 4B + hyde | `20261022-091217-0f71` | 22·1·2 | 9/6/11 | 0.878 |
      **Row-9 bases fixed:** PyMuPDF 8a (`--rewrite decompose`), Docling **6d** (no rewrite: 8b/8d/8f all tie it).
      **Set E done** (Docling + 4B): rewritten `20261022-092055-d263` 7·0·1, rewrite OK 6/8, hits 7/8; as typed `20261022-092354-264e` 5·1·2, hits 4/8. All local scorecards finished, none failed.
      Bases: 6c 20·2·3 (PyMuPDF), 6d 22·1·2 (Docling). Rows 6 and 7 are written in `docs/ladder.md`;
      **row 8 is not written yet** (use the table above + 8d/8f).
    - **Running in the background when this was written** (dies with the session; check
      `data/runs/` for status `done` vs `running`/`interrupted`): 8d Docling + multi, then 8f
      Docling + hyde, then Set E rewritten and Set E as typed, all on `docling-structure-embeddinggemma2`.
      If any is missing, rerun it exactly (Colab not needed; unload the answer model first):
      `ollama stop qwen3:1.7b; ./rag eval --set a b d --prompt v3 --rerank qwen3-4b --profile
      docling-structure-embeddinggemma2 --rewrite multi --ladder 8 --label "Docling + 4B + several
      phrasings (8d)"` (8f: `--rewrite hyde`, label "… draft answer first (8f)"); Set E:
      `./rag eval --set e --prompt v3 --rerank qwen3-4b --profile docling-structure-embeddinggemma2
      --label "Set E: follow-ups rewritten, Docling + 4B"` and the same with `--as-typed` and label
      "Set E: follow-ups as typed, Docling + 4B". Three refused attempts (`20261022-082839/41/43`,
      not enough memory) are failed runs that do not count.
    - **Decided from the numbers:** Docling's row-8 base is **6d** (7b lost D10); PyMuPDF's row-9
      base is **8a** (ties 8e on A+D, higher MRR, cheaper). Docling's row-9 base = the best of
      6d/8b/8d/8f (if none beats 6d's 22·1·2, use 6d: no rewrite).
    - **Row 9 so far (17:55):** phase A decisions cached in `20261022-093248-77a7` (PyMuPDF 8a: 27 enough,
      4 bad output) and `20261022-093611-1a8c` (Docling 6d: 28 enough, 3 bad output); every bad output is "search
      without a usable query" (the schema does not require `query` for `search`: the obvious next fix, a new
      prompt version). A usage-limit pause lost that runtime; a rerank-only runtime could not replay the 7 bad
      decisions because bad outputs were never cached (phase B runs `…121932-835e`, `…122003-202b` failed: they do
      not count). Fixed: `rewrite.call` replays cached outputs on a runtime without the rewrite model, and an
      `agent` bad output is cached (raw + error) and replays as the same stop (tests in test_rewrite/test_agent).
      Now running `session2c.sh` on runtime `advert-helpful-…` (qwen3:8b + 0.6B/4B): re-asks only the missing
      decisions, unloads qwen3:8b, reranks both bases with the 4B. Then disconnect and score 9a/9b locally:
      `ollama stop qwen3:1.7b; ./rag eval --set a b d --prompt v3 --rerank qwen3-4b --agent 3 --profile
      pymupdf-structure-embeddinggemma2 --rewrite decompose --ladder 9 --label "PyMuPDF 8a + multi-step (9a)"`
      and `… --profile docling-structure-embeddinggemma2 --ladder 9 --label "Docling 6d + multi-step (9b)"`.
    - **18:20 the laptop froze** during the 9a scorecard (swap thrash: VS Code ~6.9 GB + Vivaldi + the answer
      model); the user restarted. 9a's run `20261022-123841-ae4f` (22/31) is interrupted and does not count.
      Colab session 2 finished before that: final decisions + 4B scores in `20261022-123453-98c3` (PyMuPDF) and
      `20261022-123714-e629` (Docling); Colab is disconnected for good. The trading skip for today was
      re-created after the reboot (`/run/user/1000/systemd/user/trading-daily.service.d/skip-2026-10-22.conf`).
      9a/9b rerun at 18:4x with a memory watchdog (`phase_s9.sh`) after the user closed VS Code's flux window
      and Vivaldi.
    - Browser QA done today: Ask follow-up live ("And for 2027?" → "Searched for: What is the growth outlook for
      South Asia in 2027?", Ask as typed shown, conversation saved and listed at once).
    - **Next: Colab session 2 for row 9.** Ask the user to run
      `./rag llm colab --models qwen3:8b --worker qwen3-0.6b-gpu qwen3-4b --copy` in a new T4 cell and
      connect. Phase A (qwen3:8b, decisions cached): `./rag eval --mode retrieval --set a b d --agent 3
      --profile pymupdf-structure-embeddinggemma2 --rewrite decompose --label "M5 retrieval: PyMuPDF 8a +
      multi-step (decisions)"` and the same for Docling's base. While qwen3:8b is loaded, QA Ask's
      follow-up (conversation + "Searched for" + Ask as typed) in the browser. Phase B: unload qwen3:8b
      (`remote.client().post_json("/api/chat", {"model": "qwen3:8b", "messages": [], "keep_alive": 0})`),
      then the same two runs with `--rerank qwen3-4b`. Disconnect. Then scorecards 9a/9b
      (`--agent 3 --rerank qwen3-4b --prompt v3 --ladder 9`).
    - Then: QA still to do (Search lab Reranked column live, How it works 4 with a question, dark
      theme, 768/1024/1440 widths, keyboard; findings so far M5-01..06 in the QA doc, 01-05 fixed);
      write rows 8-9 + Set E into `docs/ladder.md` and `docs/steps/08-rewrite-agent.md` "Measured";
      `./rag test` (last full run 507 passed; since then +~10 tests in test_context, test_rewrite_lab,
      test_health, test_runs, all passing individually); UI typecheck + lint (clean at 14:00);
      `git add`; HANDOFF M5 status; report; ask which techniques become defaults and whether to commit
      `data/context/docling-structure-embeddinggemma2-ctx.jsonl`; suggest a commit message.
    - **After M5, the user asked** to cut scope for M6-M8 and to be asked "a bunch of questions" with
      time estimates per choice. Prepared options (hours of work, this laptop): answer model (keep
      qwen3:1.7b 0 h / qwen3.5:4b already pulled ~1.5 h / + gemma4:e4b ~3 h / full Colab plan ~7 h);
      judge + RAGAS + tracing (skip 0 h / judge on Colab ~3 h / + RAGAS ~6 h / + Phoenix ~8 h);
      figures M7 (skip ~0.5 h / vision descriptions only ~5-6 h / both paths ~10 h); upload M8 (CLI
      only ~1 h / browser upload to the default index ~4 h / full ~8 h); How it works 9-13 (text
      only ~1.5 h / widgets for 9, 10, 13 ~4 h / all ~7 h); final QA (light ~1.5 h / full matrix ~3.5 h).
    - Engine fixes today worth knowing: `OllamaClient.unload` + two-pass runs free our answer model
      first; context job retries a timed-out batch and skips a rejected line; `MAX_LINE_CHARS` 720;
      `paths.CONTEXT`; `indexes.overview` rows carry `contextual`; Settings/Health/Indexes UI fixes.
  - Four empty runs `20261016-160027…160033` (0 of 31 questions, dead pids) are duplicate starts
    from 2026-10-16; the app lists them as interrupted. They do not count.

- **P1 + P2 code (2026-10-16 evening, staged; 359 tests pass):** `rerank.py` (specs with
  pinned HF revisions, `LocalReranker`, `RemoteReranker`, `CachedScorer`),
  `retrieve.run` (the one pipeline; Ask, evaluation and the lab share it),
  two-pass evaluation when a reranker is on, MRR/recall@20 on old runs too,
  `config_key` on canonical retrieval JSON (equivalence tested on all real runs),
  `--rerank/--rerank-depth/--fresh-rerank`, `/api/rerankers`, Colab `worker.py`
  (`/rerank`, `/worker`, loopback only, pinned models), gateway v2 (worker routes,
  gzip requests with bomb cap, `Server-Timing`), `start.sh` step 4b and a
  rerank-only mode (no Ollama), kept-alive remote client, 524 mapped.
  **User chose (2026-10-16): finish P1 on Colab** (rows 6a-6f in one session).
  **Rerank scores are cached** (`data/cache/rerank/`): a retrieval-only run on
  Colab fills it and the scorecards replay the same scores with no Colab; each
  result records hit/miss. `--fresh-rerank` forces a live call.
  **Local 0.6B fp32 measured** (6 questions, 20 candidates, beside EmbeddingGemma 2):
  depth 20: 39 s median (batch 4), 42 s (batch 8); depth 10: 15 s; peak RSS 5.95 GB
  with the embedder, so `RERANK_LOAD_GB` = 4.0. bf16 was abandoned (slower, and
  my test runs contaminated it); int8 not measured.

- **P0 done (2026-10-16, staged):** `profiles.DEFAULT` = `docling-structure-embeddinggemma2`
  (5d), `prompt.DEFAULT_VERSION` = v3, the `embed` role = EmbeddingGemma 2 (python).
  `./rag ingest --profile` stays `baseline`; "not built" messages use
  `profiles.build_command()`. The five tests that relied on the old defaults pin
  baseline + v2. The UI reads `default_profile` (indexes) and `default` (prompts),
  no hard-coded `baseline`/`v2`; Ask has a "no index" and a "not enough free
  memory" empty state (`/api/status` gains `default_profile`, `build_command`,
  `memory`). `./rag test`: 291 passed, 1 skipped. UI typecheck and lint clean.

**The full build prompt is `docs/plans/m5.md`. Read it end to end, then start at
P0.** Spec §17 M5: reranker, contextual retrieval, query rewriting, the agentic
loop, chat memory; the Search lab's reranked column; How it works 4 and 8. Done
when "global growth" (B4) is a Hit on 5d.

Decisions made 2026-10-16 (don't re-ask):
1. **5d is the app default** (Docling + structure + EmbeddingGemma 2, prompt v3,
   dense, 20·3·2) and the base for rows 6-9.
2. **Rows 6-9 run on both parsers:** 5c (PyMuPDF) and 5d (Docling).
3. **Rewriting, follow-up rewriting, context lines and the agentic loop run on
   Colab only** (Remote AI, `qwen3:8b`). The answer model stays local `qwen3:1.7b`.
4. **The network is compressed and never waits on the tunnel:** gzip both ways,
   keep-alive, batching on the runtime, pipelining.
5. The prompt lives in `docs/plans/m5.md`.

Approved pull: Qwen3-Reranker-0.6B (HF, 1.19 GB). Measure the gzip gateway on the
next Colab session.

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
