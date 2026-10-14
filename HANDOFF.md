# Session handoff: RAG Studio

Updated 2026-10-15 ~01:50 IST, at the end of M1. A new session should read, in
order: this file, `CLAUDE.md`, then the build spec
`~/Dev/RAG/advanced-rag/RAG_STUDIO_PROMPT.md` (§16-17 for how to work and the
milestones). **Claude builds everything here**; the user reviews, runs and asks
questions.

## Rules the user set (don't re-ask, don't break)

- **Never run `git commit` or push.** At a milestone's end: `git add`, update
  this file, report neatly with every number, and end with a suggested commit
  message. The user commits and pushes.
- Repo `~/Dev/RAG/rag-studio`; UI 127.0.0.1:**3100**, API 127.0.0.1:**8775**
  (trading uses 3000/8765); Phoenix later on 6006.
- Local model pulls approved, each when its milestone starts: qwen3-embedding:0.6b,
  EmbeddingGemma 2 (via sentence-transformers, see research §4), Qwen3-Reranker-0.6B
  (HF), Docling models, gemma4:e4b (M6). qwen3.5:4b is already pulled.
- Git: PDFs and test questions are committed; indexes, runs and caches are rebuilt.
- Check `ollama ps` and `systemctl --user status trading-daily.timer` (19:17 IST)
  before any pull or long run. Never stop the trading timers.
- The user's own `./rag web` is usually running (dev mode, both servers
  reload). Test against it; don't start a second copy.

## Decisions settled in M1 (from the audit, research.md §0)

- **The spec wins over M0 research** where they differed: qwen3-embedding:**8b**
  on Colab is in the plan (4b optional); reranker **top 20 → top 5** first,
  measured (top 10 only if 20 is too slow), 8B kept in 8-bit on the T4;
  **gemma4:e4b** stays a local candidate, benchmarked alone at M6.
- **No Ollama upgrade.** EmbeddingGemma 2's Ollama build is nvfp4 safetensors
  needing Ollama ≥ 0.36 and does not run on Linux at any version; spec §2.2
  already sends image embeddings through Python. Ollama stays 0.32.6.
- **Set D (edge cases)** was added at the user's request ("extend them to push
  the model to more edge"): 11 questions + traps T3-T5, every answer page
  verified in the text (`scripts/find_pages.py`). D3's key was widened once
  (p44 calls the adjusted figure plain "net income"); it is documented in the
  question's note.
- The user's 9 questions are Set A, pinned word for word by `tests/test_golden.py`.

## M1 status: complete (staged, not committed)

**Engine** (`rag_studio/`): parse, clean, chunk, profiles, embed, store,
ingest, retrieve, prompt, answer, trace, eval, library, files. **API**: ask
(SSE via `web/api/sse.py`), page image + passage boxes, golden, runs (start,
cancel, list, get), ladder, models, indexes. **UI**: Ask (streaming, `[n]`
chips, page preview with the passage outlined, sources, "What happened" trace,
calm refusal state), Evaluation (start a run, live progress, history, per-question
detail panel, the 11-row ladder), Documents (library stats). Thin scrollbars,
set once in `globals.css`. **CLI**: `./rag ingest`, `./rag eval`.

**Numbers (all measured 2026-10-15, details in `docs/ladder.md`):**

| Check | Result |
|---|---|
| Passages | 1,190 (34/233/923), identical to `advanced-rag/outputs/chunks.jsonl` |
| Index build | 406 s for 1,190 passages (2.9/s); a rebuild with no changes is a no-op |
| Set A recall@5 | 8/9 (A6 at rank 45, as by hand) |
| Set B recall@5 | 5/6 (B4 at rank 11, as by hand) |
| Set D recall@5 | 9/11 (D6 rank 6, D9 rank 9) |
| Set A scorecard, qwen3:1.7b, prompt v2 | PASS 10 · REFUSED 1 · WRONG 0 |
| Same, prompt `v2-handbuilt` (byte-identical to `answer.py`) | PASS 9 · REFUSED 0 · WRONG 2, row for row with the reference, same answers |
| Set D scorecard | PASS 8 · REFUSED 1 · WRONG 5 (all three traps answered: "Tim Cook", a made-up MacBook Air footprint, a 2031 "forecast") |
| Determinism | Set D run twice: 14/14 answers identical |
| Tests | `./rag test` 89 passed (2 need Ollama + the index) |
| UI | typecheck and lint clean; QA 25/25 page×width, console clean, 6 findings fixed (`docs/qa/2026-10-15-m1-ask-evaluation.md`) |

Ladder row 1 = runs `20261014-190323-98cc` (Sets A, B) and
`20261014-192721-3958` (Set D). They live in `data/runs/` (gitignored); the
numbers are in `docs/ladder.md`. If `data/` is wiped: `./rag ingest`, then
`./rag eval --set a b --ladder 1 --label Baseline` and
`./rag eval --set d --ladder 1 --label "Baseline: edge cases"` (~12 min).

## Not done, on purpose (where it goes)

- Two-run comparison view (spec §9.5): M2, when there is a row 2 to compare.
- Editable test questions in the UI (§9.5): later (M8 polish); the JSON files are the source now.
- Set C (figure question): M7, after confirming the answer from `get_drawings()`.
- How it works interactive widgets: M3 onward, per the spec's milestone table.
- Scrollbars in Firefox and Safari: written for both, **tested only in Chrome**.
- Chat memory / follow-up rewriting: M5. Each Ask question is standalone now.

## Next: M2 (parsing, cleaning, the injection guard)

Docling 2.139 (CPU torch index, ~424 MB) as a second parser beside PyMuPDF; the
§4 cleaning order with counts (ligatures → 0, "Th" → 0, 127 compounds intact,
headers/footers moved to metadata with the printed page label); section paths
and reference tagging; the injection guard (hidden text dropped, instruction-like
passages flagged, passages wrapped as data; a crafted PDF test). T3 ("Ignore
all previous rules…") is a live target. The Documents page inspector. Ladder
row 2 measured; add the two-run comparison then. Ask before pulling Docling's
models (~530 MB).

## Gotchas hit so far

- `pkill -f <pattern>` killed the Bash tool's own shell. Stop servers by port:
  `ss -ltnpH "sport = :3100"`.
- Don't edit `scripts/run_web.sh` while `./rag web` runs (bash reads it as it goes).
- `uv sync` can stall when other uv processes hold the cache lock.
- The dcg shell hook blocks some heredocs containing `>`-like text; use the
  editor (Edit/Write tools) for those changes.
- Editing an engine file while `./rag ingest`/`eval` runs is fine (the process
  already imported it) but the run uses the old code; rerun if it matters.
- Chrome DevTools MCP: `resize_page` fails on a maximised window; use
  `emulate` with a viewport instead.
