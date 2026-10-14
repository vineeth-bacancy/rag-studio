# Session handoff: RAG Studio

Written 2026-10-14 ~21:30 IST, mid-M0, because the session may hit its limit.
A new session should read, in order: this file, `CLAUDE.md`, then the build spec
`~/Dev/RAG/advanced-rag/RAG_STUDIO_PROMPT.md` (§16-17 for how to work and the
milestones). Unlike `advanced-rag`, **Claude builds everything here**; the user
reviews, runs and asks questions.

## Decisions the user already made (don't re-ask)

- Repo `~/Dev/RAG/rag-studio`; UI 127.0.0.1:**3100**, API 127.0.0.1:**8775** (trading uses 3000/8765); Phoenix later on 6006.
- Local model pulls approved, each when its milestone starts: qwen3-embedding:0.6b, embeddinggemma-2, Qwen3-Reranker-0.6B (HF), Docling models, qwen3.5:4b (**already pulled** 2026-10-14 by research), gemma4:e4b (M6; research says RAM is tight, see research.md §0).
- Git: commit PDFs + golden sets; LanceDB/caches/runs are gitignored and rebuilt.
- Still check `ollama ps` and the trading timer before any pull or long run. Ask before committing.

## M0 status

**Done (all verified):**
- Repo, uv (Python 3.12), pnpm, `./rag` (web, health, test; ingest/eval/index/llm print their milestone).
- Engine: `rag_studio/{paths,config,llm,health,cli}.py`. API: `web/api/{main,guard}.py` with `/api/status`, `/api/health`, `/api/documents`.
- UI: trading's design system (copied primitives, globals.css), shell (sidebar with status rows, top bar, ⌘K), 7 pages. System Health is complete; the other pages are honest empty states (How it works lists the 13 steps with real examples).
- Tests: `./rag test` → 23 passed (guard attacks, health with Ollama down/busy, UI contract: loopback binding, one route list).
- `pnpm typecheck` and `pnpm lint` clean.
- Browser QA: `docs/qa/2026-10-14-m0-shell.md` (9 findings, all fixed; 35/35 page×width checks clean).
- Docs: `CLAUDE.md`, `docs/00-start-here.md`, `docs/01-architecture.md`, `docs/product-language.md`, `docs/research.md` (§0-8 written), `docs/research/` appendices A-C + probe scripts.

**M0 is complete** (2026-10-14 ~21:50 IST): `docs/research.md` §0-10 written,
appendices A-D and probe scripts in `docs/research/`. Committed (checkpoint
commits on `main`, local only). Next session: start M1 after the user says go.

## Research findings that change the plan (details in docs/research.md)

- **EmbeddingGemma 2 cannot run in Ollama on Linux** (MLX-only tags, Ollama ≥ 0.36). Plan: a sentence-transformers profile (fp32, text-only).
- **Ollama has no rerank endpoint.** Qwen3-Reranker-0.6B via sentence-transformers `CrossEncoder`, top 10 by default; 4B fp16 on Colab; skip 8B (not better than 4B on English).
- **qwen3.5:4b with `think:false` works** (measured): warm 37 s, cold 72 s on the 1,318-token RAG prompt; exact refusal on an unanswerable question.
- **num_ctx overflow silently drops the START of the prompt** (HTTP 200). Detect via `prompt_eval_count` vs our own count.
- CPU is an i5-1135G7: **4 cores / 8 threads** (not 8 cores).
- LanceDB: default metric is squared L2 (always set cosine); built-in hybrid hides the two lists, so fuse with our own RRF (K=60); `rename_table` unsupported, so swap builds via an atomic `active.json`.
- Docling reads the PDF text layer: "Th" (#7), footnote markers (#9) and hidden text (#13) likely survive it.
- Evaluation: **DeepEval 4.2** (RAGAS is idle and fails to import fresh). Set `DEEPEVAL_TELEMETRY_OPT_OUT=1`, `DEEPEVAL_DISABLE_DOTENV=1`, `DEEPEVAL_HOME`, `eval_mode="llm"`, `penalize_ambiguous_claims=True`, and check verdict count == claim count ourselves (it silently dropped one in a test).
- Tracing: own in-app trace + optional **Phoenix** sidecar (own venv, `PHOENIX_HOST=127.0.0.1 PHOENIX_ALLOW_EXTERNAL_RESOURCES=false PHOENIX_TELEMETRY_ENABLED=false`, OTLP over HTTP only; gRPC 4317 binds all interfaces regardless). Langfuse rejected.
- Judges: temperature 0 + seed is not reproducible on CPU Ollama; repeat k=3-5 and report spread.
- **User decisions pending (not blocking M1):** upgrade Ollama 0.32.6 → 0.40.4 (shared with trading)? gemma4:e4b vs e2b locally at M6?

## Next: M1 (baseline parity)

PyMuPDF + NFKC + fixed 1,000/150 chunks (1,190 chunks: iMac 34, GPO 233, GEP 923) + nomic-embed-text + exact cosine search in LanceDB + system prompt v2 + the 11-question scorecard; Ask page with SSE streaming and code-built citations; Evaluation page with runs and history. Done when Set A 8/9 and Set B 5/6 recall@5 reproduce and the qwen3:1.7b scorecard is within ±1 row of PASS 9 / REFUSED 0 / WRONG 2. Golden sets are in spec §7 → `data/golden/`.

## Gotchas hit this session

- `pkill -f <pattern>` killed the Bash tool's own shell (the pattern was in its command line). Stop servers by port: `ss -ltnpH "sport = :3100"`.
- Editing `scripts/run_web.sh` while `./rag web` runs breaks the running script (bash reads incrementally).
- `uv sync` can stall for minutes when other uv processes hold the cache lock.
