# Research: choices, evidence and what is still unverified

Researched **2026-10-14** for M0, as the build spec's §14 asks. Several models
here are newer than any training data, so nothing was taken from memory: every
fact was checked against official docs, model cards, source code at a pinned tag,
or a live test on this machine.

**Evidence labels** (used throughout and in the appendices):
**VERIFIED-LIVE** run here, with the measured output · **VERIFIED-DOC** official
docs, model card or source · **CLAIM** vendor, blog or community · **ESTIMATE**
calculation with its reasoning · **UNVERIFIED**.

The full reports, with every table row sourced, are in `docs/research/`:
[A: Ollama and answer LLMs](research/A-ollama-llms.md) ·
[B: embeddings and reranker](research/B-embeddings-rerank.md) ·
[C: Docling and LanceDB](research/C-docling-lancedb.md) ·
[D: evaluation, tracing, techniques](research/D-eval-tracing-techniques.md).
The probe scripts are in `docs/research/scripts/`.

**This document picks defaults to test, not winners.** Every choice below
becomes a default only after it moves our own scorecard (§8 of the spec).

---

## 0. Decisions at a glance

| Stage | Plan | Why (short) | Milestone |
|---|---|---|---|
| Web stack | Next 16.3.0, React 19.2.8, Tailwind 4.3.3, shadcn base-nova on Base UI 1.9, FastAPI 0.143.1 | Same stack as the trading app; versions checked live | M0 ✓ |
| Streaming | FastAPI's built-in SSE (`fastapi.sse`) + 15 s heartbeat | Native since FastAPI 0.13x; the Next proxy drops idle sockets | M1 |
| Parsing | PyMuPDF baseline + **Docling 2.139** as a second parser profile, CPU torch | Layout, tables, headers moved out of the body; not a fix for "Th" or hidden text | M2 |
| Vector store | **LanceDB 0.40**, exact (flat) search, our own RRF fusion | 7-10 ms exact at 1k rows; built-in hybrid hides the two input lists | M1/M3 |
| Embeddings | nomic (baseline) → **qwen3-embedding:0.6b** (Ollama) → **EmbeddingGemma 2** (sentence-transformers, not Ollama) → qwen3-embedding:4b on Colab | EmbeddingGemma 2 on Ollama is MLX-only (no Linux) | M4 |
| Reranker | **Qwen3-Reranker-0.6B** via sentence-transformers `CrossEncoder` on CPU (top 10); **4B** fp16 on Colab. Skip 8B | Ollama has no rerank API; 8B is no better than 4B on English | M5 |
| Techniques | Defaults: fixed-size and structure-aware chunks, hybrid + rerank. Opt-in: contextual retrieval (Colab), HyDE, multi-query, decomposition, agentic loop (Colab) | The evidence for each comes from large models; our scorecard decides | M3-M5 |
| Answer LLM | **qwen3.5:4b** with `think:false` (measured: 37 s warm); qwen3:1.7b fast fallback | Thinking-off verified live; gemma4:e4b too big for free RAM | M6 |
| Evaluation | **DeepEval 4.2** behind our own judge wrapper, telemetry off, `penalize_ambiguous_claims=True`, our own claim/verdict count check; k = 3-5 repeats with spread | RAGAS idle since Feb 2026 and fails to import fresh; DeepEval's default faithfulness passes unsupported claims | M6 |
| Tracing | Our own trace record in the app; **Phoenix 20.21** as an optional local sidecar, locked down, OTLP over HTTP only | Langfuse needs 5 services and has an update check that cannot be turned off | M6 |

**Decisions that are yours** (none blocks M1):
1. **Upgrade Ollama 0.32.6 → 0.40.4?** It would enable image embeddings through
   `/api/embed` and newer model support, but Ollama is shared with the trading
   app's daily run. Not needed before M7.
2. **gemma4:e4b locally** (approved for M6): measured free RAM is often 2-4 GB
   and e4b needs ~6.6 GB of weights + KV cache. Research A suggests `gemma4:e2b`
   (4.6 GB) locally, or Gemma only on Colab. Decide at M6.

## 1. Web stack (Next.js 16, shadcn on Base UI, Tailwind v4, FastAPI)

| Fact | Status | Source |
|---|---|---|
| Installed: next 16.3.0, react 19.2.8, @base-ui/react 1.9.0, tailwindcss 4.3.3, recharts 3.10.1, swr 2.5.1, zustand 5.0.15, lucide-react 1.57.0 | VERIFIED-LIVE | `node -e require('<pkg>/package.json').version` in `web/ui` |
| Error boundaries take `retry()` (it was `reset` up to Next 15) | VERIFIED-DOC | `node_modules/next/dist/docs/01-app/03-api-reference/03-file-conventions/error.md` |
| The rewrite proxy's timeout is `proxyTimeout \|\| 30000` ms; `null` disables it. Set to 330 s here | VERIFIED-DOC (source) | `next/dist/server/lib/router-utils/proxy-request.js` L37 |
| `experimental.proxyClientMaxBodySize` (default 10 MB) applies to **proxy** (the renamed middleware), not to rewrites. Whether a > 10 MB PDF upload passes the `/api` rewrite intact | VERIFIED-DOC / **UNVERIFIED** for rewrites (test in M8) | `docs/.../proxyClientMaxBodySize.md` |
| `allowedDevOrigins` must list `127.0.0.1` or the dev server 403s its own chunks | VERIFIED-DOC + trading measurement | `docs/.../allowedDevOrigins.md`; trading `docs/14-web-dashboard.md` |
| `devIndicators: { position }` moves the dev badge; `false` hides it (errors still show) | VERIFIED-DOC | `docs/.../devIndicators.md` |
| `next dev` writes `AGENTS.md`/`CLAUDE.md` into the app folder (`agentRules: false` disables) | VERIFIED-LIVE | dev server log |
| Tailwind 4 container queries (`@container`, `@sm:`, `@xl:`, `@2xl:`) work without a plugin | VERIFIED-LIVE | used in `app/health/page.tsx`, measured in QA M0-04 |
| shadcn base-nova composes with `render={<X/>}`; `asChild` is dropped silently | VERIFIED-DOC (trading) | trading `docs/14-web-dashboard.md` |
| tailwind-merge only dedupes classes **within one variant**: a component's `data-vertical:self-stretch` beats a call site's `self-center` | VERIFIED-LIVE | QA M0-02 (computed `align-self: stretch`) |
| FastAPI 0.143.1 has SSE built in: `fastapi.sse.EventSourceResponse` and `ServerSentEvent(data, event, id, retry, comment)` | VERIFIED-DOC | ctx7 `/websites/fastapi_tiangolo`, fastapi.tiangolo.com/reference/sse |
| Whether `EventSourceResponse` sets `Cache-Control: no-cache, no-transform` and `X-Accel-Buffering: no` by itself | UNVERIFIED (set explicitly in M1) | |
| Starlette 1.8.0's `TestClient` prefers `httpx2` (2.13.1, pydantic org) and warns on httpx | VERIFIED-LIVE | test run warning; PyPI |

**Recommendation:** keep the engine's Ollama client on httpx 0.28 for now
(stable, already used); `httpx2` is a dev dependency for the test client only.
Revisit when httpx2's API is documented as a drop-in.

## 2. The Ollama API (installed 0.32.6; latest 0.40.4)

| Fact | Status |
|---|---|
| `think` accepts a bool or `"low"/"medium"/"high"/"max"`; 0.32.6 treats any level as `true`. Reasoning arrives in `message.thinking` | VERIFIED-DOC + LIVE |
| Chat `images`: a per-message list of base64 strings. Works with qwen3.5:4b (918 image tokens for an 850×1100 page) | VERIFIED-LIVE |
| `/api/embed`: `input` string or list; `truncate` (default true: cuts the **end**); `dimensions`; `keep_alive` | VERIFIED-DOC |
| `dimensions: 256` returns `full[:256]` **re-normalised**; a value above native (1024 on nomic) silently returns all 768 | VERIFIED-LIVE |
| `truncate: false` + over-long input → HTTP 400 "the input length exceeds the context length" (nomic's context stays 2048 even with `num_ctx: 8192`) | VERIFIED-LIVE |
| Image or audio input to `/api/embed` exists only from 0.40.4; on 0.32.6 → 400 "invalid input type" | VERIFIED-LIVE + DOC |
| **`num_ctx` overflow is silent:** HTTP 200, the server keeps the first 4 tokens and the tail, and **drops the start**. Only `prompt_eval_count` (258 for a 1,792-token prompt at `num_ctx` 512) and the server log reveal it. The model then answered from the wrong text | VERIFIED-LIVE |
| Multi-message overflow drops whole older turns and keeps the system prompt | VERIFIED-LIVE |
| Changing `num_ctx` between requests restarts llama-server and reloads the model (2-4 s for qwen3:1.7b) | VERIFIED-LIVE |
| `/api/ps` returns `size`, `size_vram`, `expires_at`, `context_length`; `/api/tags` returns `capabilities` and `details.context_length` | VERIFIED-LIVE |
| `keep_alive: 0` unloads after the request; default 5 min | VERIFIED-DOC + LIVE |
| **No rerank endpoint:** absent from docs and source; `/api/rerank` and `/v1/rerank` → 404; issue #3368 open since 2024 | VERIFIED-DOC + LIVE |

**Recommendations:**
- Always send `think` explicitly (Qwen 3.5 thinks unless told not to; Gemma 4 in
  Ollama is the opposite).
- One fixed `num_ctx` per model (8192 for answer models) so requests never
  trigger a reload.
- **Detect truncation in code:** compare `prompt_eval_count` with our own token
  count and show "the start of the prompt was dropped" in the trace. This is
  also a How it works lesson (step 9, Prompt).
- Embed with `truncate: false` so an over-long passage is an error we see, not
  a silent cut.
- Use `/api/ps` for the loaded-models panel and the "AI busy" check (done in M0).

## 3. Qwen3-Embedding (0.6B / 4B / 8B)

| Fact | Status |
|---|---|
| Query: `Instruct: {task}\nQuery:{query}` (**no space after `Query:`**, per the reference code and sentence-transformers config; the card's curl example has a space). Documents: **no prefix** | VERIFIED-DOC |
| Native dims 1024 / 2560 / 4096; Matryoshka from 32 up | VERIFIED-DOC |
| Last-token pooling on an appended `<|endoftext|>`; the Ollama GGUFs have `pooling_type=LAST` and `add_eos_token=True` | VERIFIED-DOC + LIVE (GGUF header) |
| Ollama tags: `0.6b` = Q8_0 (639 MB), `4b` = Q4_K_M (2.5 GB), `8b` = Q4_K_M (4.7 GB) | VERIFIED-LIVE (registry) |
| MTEB English v2 retrieval: 61.83 / 68.46 / 69.44; MMTEB retrieval 64.64 / 69.60 / 70.88 | VERIFIED-DOC (self-reported) |
| 8B in fp16 produces NaN on some inputs (maintainer advises fp32/bf16) | VERIFIED-DOC (issues) |
| Ollama vectors match sentence-transformers vectors | **UNVERIFIED** (M4 parity check) |

**Recommendation:** use the Ollama library tags. Store the literal query
template and one fixed RAG task string per index (e.g. "Given a question about
a PDF report, retrieve passages that answer the question"); test the default vs
custom task as a scorecard row, never as a silent change.

## 4. EmbeddingGemma 2

| Fact | Status |
|---|---|
| Official card `google/embeddinggemma-2`, released 2026-10-06, Apache 2.0 (plus the Gemma prohibited-use policy) | VERIFIED-DOC |
| 740M total = 270M text + 170M vision + 300M audio; mean pooling; 768 dims, Matryoshka 512/256/128 | VERIFIED-DOC |
| Query: `task: search result \| query: ` (or `task: question answering \| query: `). Document: `title: {title or none} \| text: ` | VERIFIED-DOC |
| Text and images share one 768-d space, compared by cosine. Images: the card says no prefix; the transformers docs example applies the document prompt. **Conflict**, to measure in M7 | VERIFIED-DOC |
| fp16 gives NaN or degraded vectors; use fp32 (CPU and T4) or bf16 | VERIFIED-DOC |
| **All Ollama tags are MLX safetensors requiring Ollama ≥ 0.36; on Linux the pull fails** ("requires MLX support"). Open issue #18825 | VERIFIED-LIVE (registry) + DOC |
| Runs in sentence-transformers ≥ 6.1 / transformers ≥ 5.19; text-only load is 271M parameters | VERIFIED-DOC |
| Roughly 4× cheaper per token than qwen3-embedding:0.6b on CPU | ESTIMATE |

**Recommendation:** an in-process **sentence-transformers** profile (fp32,
text-only, explicit prefixes, renormalise after Matryoshka truncation). This
brings in the torch stack the reranker needs anyway. Add images in M7 by loading
with `audio_config=None`.

## 5. Answer and vision LLMs: Gemma 4 vs Qwen 3.5

| Fact | Status |
|---|---|
| qwen3.5:4b: 4.2B, Q4_K_M, 262K context, vision, Apache 2.0 | VERIFIED-LIVE (`ollama show`) |
| **`think:false` turns thinking off cleanly**; `think:true` → reasoning in `message.thinking`; level strings behave like `true` | VERIFIED-LIVE |
| RAG prompt of 1,318 tokens (system prompt v2 + 8 passages), temperature 0, `num_ctx` 8192: **cold 71.6 s** (25 s load), **warm 37.4 s**; prompt ~35 tok/s, generation ~7 tok/s; resident 3.36 GB | VERIFIED-LIVE |
| Warm unanswerable question → exactly "I could not find this in the documents." | VERIFIED-LIVE |
| qwen3:1.7b for comparison: ~70-90 tok/s prompt, ~21 tok/s generation | VERIFIED-LIVE |
| iMac p2 chart via qwen3.5:4b: 67.5 s; read 445 / -99 / 346 kg correctly, with small wording flaws | VERIFIED-LIVE |
| Gemma 4 in Ollama 0.32.6 adds `<|think|>` only when `think` is true, so thinking is **off by default** | VERIFIED-DOC (source); not run |
| `gemma4:latest` = e4b, Q4_K_M, 5.5 GB weights + 1.0 GB projector | VERIFIED-DOC (registry) |
| Neither card publishes DocVQA or ChartQA. Qwen3.5-4B: OCRBench 85.0, CharXiv(RQ) 70.8, MMMU-Pro 66.3. Gemma 4 E4B: MMMU-Pro 52.6; its OmniDocBench figure is an edit distance, not comparable | VERIFIED-DOC (vendor) |
| No T4 or llama.cpp CPU speed published for either | VERIFIED-DOC (absence) |

**Recommendation:** qwen3.5:4b with `think:false` as the first new answer
candidate, 4-6 passages of context, streamed. Prompt reading dominates latency
on this CPU (~35 tok/s), so the passage count is the main speed lever. Treat
figure reading as an ingest-time job (about a minute per page), never per query.
Gemma 4 locally is a RAM question for M6 (see §0).

## 6. Qwen3-Reranker

| Fact | Status |
|---|---|
| Score = P("yes") from the last-position logits, with a fixed system prompt and an `<Instruct>/<Query>/<Document>` template | VERIFIED-DOC |
| The official repos ship sentence-transformers `CrossEncoder` support; its chat template produces token-identical input to the card's manual format (yes = 9693, no = 2152) | VERIFIED-DOC + LIVE (tokenizer check) |
| The reranker tokenizer does **not** append EOS; the embedding tokenizer does | VERIFIED-LIVE |
| MTEB-R 65.80 / 69.76 / 69.02 (0.6B / 4B / 8B): **8B is not better than 4B on English** | VERIFIED-DOC (self-reported) |
| Weights 1.19 / 8.04 / 16.38 GB; 4B fp16 fits a T4, 8B needs 8-bit | VERIFIED-LIVE (HF API) / ESTIMATE |
| Community "Qwen3-Reranker" Ollama models are not real rerankers (their own readme says so) | VERIFIED-DOC |
| CPU: this laptop is an i5-1135G7, **4 cores / 8 threads**. Estimated 20-35 s for 20 pairs of ~390 tokens (proxy benchmark, contended: ~70 s) | ESTIMATE |
| `llama-server --rerank` with the ggml-org Q8_0 GGUF is a possibly faster CPU path | VERIFIED-LIVE (metadata) / UNVERIFIED (runtime) |

**Recommendation:** `CrossEncoder("Qwen/Qwen3-Reranker-0.6B")` in-process, fp32,
batch 4-8, **top 10 by default** (top 20 as a "slow" option, with the measured
time in the UI). On Colab, 4B fp16 behind an allowlisted `POST /rerank` with a
NaN guard. Do not build 8B.

## 7. Docling

| Fact | Status |
|---|---|
| docling 2.139.0 (released 2026-10-14), MIT; now a meta-package over `docling-slim[standard]` | VERIFIED-LIVE (PyPI) |
| Install: ~3.2 GB of wheels by default (CUDA), **~424 MB with the PyTorch CPU index** (uv: explicit index + `[tool.uv.sources]`) | VERIFIED-LIVE (resolved, not installed) |
| Body text comes from the **PDF text layer**; OCR runs only where a region has no text | VERIFIED-DOC (source) |
| page_header / page_footer go to the FURNITURE layer, skipped by default; `page_no` is 1-based; bboxes have a bottom-left origin; printed page labels are not carried | VERIFIED-DOC |
| Line-end hyphens are joined only within one block, and real compounds get merged ("post-"/"pandemic") | VERIFIED-DOC (source) |
| No occlusion test, so text hidden under a later fill survives | VERIFIED-DOC (source) |
| Models: layout 172 MB, TableFormer accurate 213 MB (+ fast 146 MB) | VERIFIED-LIVE (HF API) |
| Speed: 0.6-1.3 pages/s on CPU (Docling v1 report), so ~4-8 min for 294 pages | CLAIM |

**Defect hypotheses, to measure in M2** (full table in appendix C): likely fixed
1, 2, 12 and 4 (when ligatures are real Unicode); partly 3, 5, 6, 8, 10, 15;
**probably not fixed 7 ("Th"), 9 (footnote markers) and 13 (hidden text)**. So
our own cleaning and guard steps stay, whichever parser wins.

**Recommendation:** Docling as a second parser profile beside PyMuPDF, not a
replacement: `do_ocr=False`, ACCURATE tables, page images for the inspector
overlays, heading hierarchy on, the DoclingDocument JSON cached per PDF.

## 8. LanceDB

| Fact | Status |
|---|---|
| lancedb 0.40.0, Apache 2.0; no pandas or pylance needed for search, FTS or hybrid | VERIFIED-LIVE |
| **Default metric is squared L2**; always set `distance_type("cosine")` | VERIFIED-LIVE |
| Exact (flat) search, 1,000 × 768, top 10: **median 7-10 ms**; FTS 4.6 ms; hybrid 13.7 ms | VERIFIED-LIVE |
| FTS is native only (Tantivy removed); BM25 in `_score`; phrase queries need `with_position=True` | VERIFIED-LIVE |
| Tokenising: "2.5%" → 2 + 5; "FY2018" one token; phrase "FY 2018" works; stop words removed by default | VERIFIED-LIVE |
| Built-in hybrid (RRF, K=60, 1-based ranks) returns only the fused list, cut to `limit` | VERIFIED-LIVE |
| ANN recall@10 vs exact on random vectors: IVF_PQ 0.32-0.52 (0.84-0.97 with refine), IVF_HNSW_SQ up to 0.985, IVF_FLAT 1.0, and none faster than exact at this size | VERIFIED-LIVE |
| `rename_table` is not supported in LanceDB OSS; `mode="overwrite"` drops the indexes | VERIFIED-LIVE |
| Rows appended after an FTS index are still found (flat scan of unindexed rows); `optimize()` folds them in | VERIFIED-LIVE |
| Versions, `checkout`, `restore`, tags all work; tags block cleanup until deleted | VERIFIED-LIVE |

**Recommendations:**
- Exact search, always cosine. ANN only as the teaching toggle in the Search lab,
  with its measured recall beside exact.
- **Run dense and BM25 separately and fuse with our own RRF** (K = 60, 1-based
  ranks). It reproduces LanceDB's order (verified) and lets the UI show both
  lists. LanceDB's hybrid stays as a test oracle.
- FTS settings to start from: simple tokenizer, English stemming, **stop words
  kept**, positions on. Test an extra exact-match column for numbers in M3.
- Rebuild = a new table per build, wait for its indexes, smoke-test, tag, then
  swap an atomically written `active.json` pointer; keep the previous build for
  rollback.

## 9. Evaluation (RAGAS vs DeepEval) and tracing (Phoenix vs Langfuse)

All live tests ran in a loopback-only network namespace with a socket guard
that logged every outside connection attempt, so nothing left the machine.
Judge: qwen3:1.7b on CPU with `think:false`, temperature 0, seed 42 and a
JSON-schema `format`, which is the request shape the Colab gateway will expose.

### 9.1 RAGAS vs DeepEval

| Fact | Status |
|---|---|
| ragas 0.4.3 (2026-01-13), no commit since 2026-02-24. **A fresh install fails on import** (langchain-community 0.4.2 removed a module); works only with `langchain-community<0.4`. 92 packages, 586 MB | VERIFIED-LIVE |
| deepeval 4.2.8 (2026-10-02), roughly weekly releases, no langchain. 66 packages, 69 MB | VERIFIED-LIVE |
| Both phone home by default. RAGAS → `t.explodinggradients.com` (opt out: `RAGAS_DO_NOT_TRACK=true`). DeepEval → PostHog `us.i.posthog.com`, including your **public IP**: 12 blocked lookups in one metric run (opt out: `DEEPEVAL_TELEMETRY_OPT_OUT=1`) | VERIFIED-LIVE + DOC |
| DeepEval also auto-loads `.env` files from the working directory (`DEEPEVAL_DISABLE_DOTENV=1`) and keeps state in `~/.deepeval` (`DEEPEVAL_HOME`). `eval_mode` "hybrid"/"system_one" call a **hosted judge**; the default "llm" does not | VERIFIED-LIVE (source) |
| A custom judge works in both: DeepEval `DeepEvalBaseLLM` (passes the pydantic schema, so `/api/chat` `format` works), RAGAS `InstructorBaseRagasLLM`. 0 malformed-JSON failures in 15 calls | VERIFIED-LIVE |
| **The faithfulness metrics disagree on meaning.** DeepEval marks a claim unfaithful only if the context *contradicts* it; an unsupported claim is "borderline" and counts as faithful unless `penalize_ambiguous_claims=True`. RAGAS marks anything not inferable from the context as unfaithful | VERIFIED-LIVE (source) |
| One test case, truth 0.5 (2 supported + 2 unsupported claims): RAGAS 0.50 with correct verdicts; DeepEval default **0.67**, because it returned 3 verdicts for 4 claims and **did not notice** | VERIFIED-LIVE (n = 1) |
| Per-claim output for the UI: DeepEval exposes `claims`, `verdicts`, `reason` as attributes (aligned by list index only). RAGAS returns just a number; the statements are reachable only through private methods | VERIFIED-LIVE |
| Answer relevancy: RAGAS needs an embedding model; DeepEval does not. Context precision and recall need a reference answer in both (RAGAS also has a reference-free precision) | VERIFIED-DOC |
| CPU cost with qwen3:1.7b: RAGAS, all 4 metrics, 11 calls, ~130 s; DeepEval faithfulness, 4 calls, ~45 s | VERIFIED-LIVE |

**Recommendation: DeepEval 4.2**, wrapped by our own `judge` module that calls
`/api/chat` (local or the Colab gateway), with:
- env `DEEPEVAL_TELEMETRY_OPT_OUT=1`, `DEEPEVAL_DISABLE_DOTENV=1`,
  `DEEPEVAL_HOME=<data dir>`; metric args `eval_mode="llm"`,
  `penalize_ambiguous_claims=True`; our prompts via `evaluation_template=`;
- **our own structural check:** if the verdict count differs from the claim
  count, the sample is a `judge_error` (retried, and reported as an error rate),
  never a score;
- a permanent test that runs the judge path under the network guard, so a
  library upgrade cannot add a phone-home call unnoticed.

The RAGAS metric definitions stay as teaching material: How it works step 12
compares the two faithfulness definitions on the same answer.

### 9.2 Phoenix vs Langfuse

| Fact | Status |
|---|---|
| arize-phoenix 20.21.3: pip package, `phoenix serve`, **SQLite** storage, ~415-450 MB RAM, 12-20 s start. Licence **Elastic-2.0** (source-available, not OSI); `arize-phoenix-otel` is Apache-2.0. 152 packages, 602 MB, so it gets its own venv | VERIFIED-LIVE |
| Default Phoenix binds `0.0.0.0`, makes 8 DNS lookups at start, and its UI loads Google Fonts and a Scarf pixel | VERIFIED-LIVE |
| Locked down (`PHOENIX_HOST=127.0.0.1 PHOENIX_ALLOW_EXTERNAL_RESOURCES=false PHOENIX_TELEMETRY_ENABLED=false`): 0 DNS lookups, no fonts, no pixel. **But OTLP/gRPC port 4317 still binds all interfaces** (hardcoded) | VERIFIED-LIVE |
| Manual OpenInference spans (chain, retriever with documents and scores, reranker, LLM with token counts) were stored and read back | VERIFIED-LIVE |
| No OpenInference auto-instrumentation exists for raw httpx calls; spans are created by hand | VERIFIED-LIVE (PyPI) |
| Langfuse v4 (current): web + worker + Postgres + ClickHouse + Redis + S3, recommended 4 cores / 16 GiB; web UI and MinIO bind all interfaces by default; an **update check to langfuse.com that cannot be turned off**; telemetry off with `TELEMETRY_ENABLED=false` except on EE licences | VERIFIED-DOC |

**Recommendation: Phoenix, optional.** The app keeps its own per-question
trace (the "What happened" panel on Ask), which needs nothing extra. Phoenix is
a richer viewer you can start when you want it, from its own venv, locked down
as above, with `PHOENIX_WORKING_DIR` under `data/`. The app exports OTLP over
**HTTP to 127.0.0.1:6006 only**. Because 4317 cannot be moved to loopback, M6
must either block it with a firewall rule or confirm a supported way to switch
gRPC off (UNVERIFIED). Langfuse is rejected: it cannot meet "nothing leaves the
machine", and its footprint does not fit next to Ollama on 15.7 GB.

### 9.3 Judges are inconsistent: the reporting protocol

| Fact | Status |
|---|---|
| Even the best judges fall short of human agreement and lean lenient; small judges still give a usable *ranking* signal (Thakur et al., GEM2 2025) | VERIFIED-DOC |
| "Deterministic" settings vary up to 15% in accuracy across runs (Atil et al. 2025); server-side variation comes from batch-size-dependent kernels (Thinking Machines, 2025-09) | VERIFIED-DOC |
| Here: same request, temperature 0, seed 42, qwen3:1.7b on CPU. The cold call gave 3 verdicts, the warm calls gave 1 (2 claims dropped). RAGAS gave 0.50, then 0.33, on one case | VERIFIED-LIVE (small n) |

**Protocol for every judge score:** record the judge tag **and digest**, the
Ollama version, the options, the sha256 of the prompt template and the library
version; never compare across different pins. Repeat k = 3 (interactive) or
k = 5 (scorecard), and report mean ± SD plus the verdict agreement rate; flag
samples whose range exceeds 0.25. Calibrate each judge once against 30-50
hand-labelled claims from our corpus and show "judge = <model>, κ = x" in the
UI. Present small-judge scores as A/B signals, not absolute quality. This
matches spec §5.8: keyword scoring is not faithfulness, and neither is one
judge call.

## 10. Techniques: what the evidence shows, and our defaults

| Technique | Evidence | Status | Default here |
|---|---|---|---|
| **Fixed-size vs semantic chunking** | Qu et al. 2024 (arXiv 2410.13070): fixed-size won evidence retrieval on 3 of 5 datasets; semantic helped only on artificially stitched documents; answer quality equal; "embedding quality matters more" | VERIFIED-DOC | Fixed-size (with the word-boundary fix) and **structure-aware** chunks; semantic chunking is an option to measure, plus the GEP p19 teaching case |
| **Hybrid search** | BEIR (2021): BM25 is a robust zero-shot baseline. Anthropic: adding contextual BM25 to contextual embeddings took failures from -35% to -49% | VERIFIED-DOC | On (M3), fused with our own RRF |
| **Reranking** | BEIR: rerankers give the best zero-shot results at high compute cost. Anthropic: reranking took the reduction from -49% to -67% (Cohere reranker, top 150 → 20) | VERIFIED-DOC | Top 10 on CPU (latency, §6), top 30-50 on Colab. The gain from a 0.6B local model must be measured |
| **Contextual retrieval** | Anthropic, 2024-09-19: 1 − recall@20 went 5.7% → 3.7% (contextual embeddings) → 2.9% (+ contextual BM25) → 1.9% (+ rerank), with Gemini embeddings on codebases, fiction and papers; one LLM call per chunk with the whole document in context | VERIFIED-DOC (vendor's own test) | Opt-in, built in bulk on Colab, cached by content hash. Targets the Q6 "executive summary" miss |
| **HyDE** | Gao et al. 2022: DL19 nDCG@10 44.5 → 61.3 with text-davinci-003 + unsupervised Contriever; the paper warns weaker generators can hurt | VERIFIED-DOC | Off; a scorecard row |
| **Multi-query / RAG-Fusion** | Rackauckas 2024: manual evaluation, no numbers; off-topic drift reported | VERIFIED-DOC (weak) | Off; a scorecard row |
| **Decomposition** | Least-to-most (Zhou et al., ICLR 2023): SCAN length split ≥ 99% vs 16% for chain-of-thought. Self-ask (Press et al. 2023) narrows the compositionality gap (numbers not retrieved) | VERIFIED-DOC | Off; targets two-part questions like "What replaced FDsys, and when…" |
| **Agentic RAG** | A-RAG (arXiv 2602.03442): big gains with GPT-4o-mini and GPT-5-mini (MuSiQue 52.8 → 74.1), no small-model data. "Do We Still Need GraphRAG?" (2604.09666): dense RAG is fine for single-hop, GraphRAG leads multi-hop by ~27 points, agentic search with 7B models is mixed (Search-o1 hurt). Pi-Serini (2605.10848): BM25 + a frontier model is strong | VERIFIED-DOC | Optional mode with hard round and time limits, run on Colab; whether qwen3 4B/8B can drive a 3-tool loop is the first thing to measure. No GraphRAG |
| **ColPali** | Faysse et al., ICLR 2025: page-image multi-vector retrieval, strong on ViDoRe (numbers not retrieved) | VERIFIED-DOC (claim) | Not pursued: needs a GPU VLM and multi-vector storage; our corpus is text-first |

**What the scorecard must measure** to decide each one: recall@k (and
1 − recall@20) and MRR, split into single-hop and multi-hop questions; judged
faithfulness with spread (§9.3); latency p50/p95 and model calls per question;
tokens sent to the answer model; ingest time and index size; for the agentic
loop, tool calls and the loop or failure rate. Every technique is an A/B against
the same configuration, judge and prompt hash.

## Sources

Every source is listed, with its access date (2026-10-14), at the end of each
appendix. The main ones:

- Ollama: github.com/ollama/ollama `docs/api.md` (tag v0.32.6 and `main`),
  docs.ollama.com, release notes v0.20-v0.40.4, issues #3368 #18825 #12921,
  registry.ollama.ai manifests.
- Models: huggingface.co `Qwen/Qwen3-Embedding-*`, `Qwen/Qwen3-Reranker-*`,
  `Qwen/Qwen3.5-4B`, `google/embeddinggemma-2`, `google/gemma-4-E4B-it`;
  ollama.com/library `qwen3-embedding`, `embeddinggemma-2`, `gemma4`, `qwen3.5`.
- Docling: github.com/docling-project/docling @v2.139.0, docling-core @v2.101.1,
  docling-parse @v7.22.3, arXiv 2408.09869v5.
- LanceDB: docs.lancedb.com (search, indexing, reranking, versioning), PyPI.
- Next.js 16.3.0: the docs bundled in `web/ui/node_modules/next/dist/docs/`.
- FastAPI: fastapi.tiangolo.com/reference/sse (via ctx7).
- Evaluation and tracing: PyPI (ragas, deepeval, arize-phoenix, langfuse),
  github.com/vibrantlabsai/ragas, deepeval.com/docs/data-privacy,
  langfuse.com/self-hosting (and /security/telemetry), the installed Phoenix
  20.21.3 source.
- Papers: Anthropic "Contextual Retrieval" (2024-09-19); arXiv 2212.10496 (HyDE),
  2402.03367 (RAG-Fusion), 2205.10625 (least-to-most), 2210.03350 (self-ask),
  2602.03442 (A-RAG), 2604.09666 (GraphRAG), 2605.10848 (Pi-Serini),
  2410.13070 (semantic chunking), 2104.08663 (BEIR), 2407.01449 (ColPali),
  2406.12624 (judging the judges), 2408.04667 (non-determinism), 2412.12509;
  thinkingmachines.ai "Defeating Nondeterminism in LLM Inference".
