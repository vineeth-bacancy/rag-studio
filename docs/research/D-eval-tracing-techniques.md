# D. Evaluation, tracing and retrieval techniques: verified research for RAG Studio

Date: 2026-10-14. Status keys: VERIFIED-DOC (official docs or paper, URL given), VERIFIED-LIVE (run here), CLAIM (vendor, blog or secondary write-up), UNVERIFIED.

**How the live tests ran.** Each test used its own scratch venv (`scratchpad/venvs/{ragas,deepeval,phoenix}-env`, Python 3.12, uv) and ran inside `unshare -rn`, a network namespace with loopback only. Ollama was reachable only through a unix-socket bridge. A Python socket guard (`scratchpad/scripts/netguard.py`) blocked and logged every non-loopback DNS lookup or connect, and Phoenix was additionally run under `strace -e connect`. Nothing left the machine during the tests.

**Judge setup.** Local Ollama 0.32.6 running `qwen3:1.7b` (Q4_K_M, digest 8f68893c…) on CPU. We called it through our own wrapper around `/api/chat` with `think:false`, `temperature:0`, `seed:42` and a JSON-schema `format`, which is the same request shape the Colab gateway exposes. Ollama was shared with other workloads during the tests, so timings are indicative only. All models were unloaded afterwards (`keep_alive:0`; `/api/ps` came back empty).

## 1. RAGAS vs DeepEval

| Fact | Status | Source |
|---|---|---|
| ragas latest is 0.4.3 (2026-01-13). The repo, now vibrantlabsai/ragas, has had no commit since 2026-02-24, so it has been idle about 8 months | VERIFIED-LIVE | PyPI JSON; GitHub API |
| deepeval latest is 4.2.8 (2026-10-02) and ships releases roughly weekly | VERIFIED-LIVE | PyPI JSON |
| Both are licensed Apache-2.0 | VERIFIED-DOC | PyPI metadata; GitHub licence |
| ragas core deps: langchain, langchain-core, langchain-community, langchain_openai, openai, instructor, tiktoken, datasets, scikit-network, diskcache. A fresh venv holds 92 packages and 586 MB | VERIFIED-LIVE | `uv pip install ragas==0.4.3` |
| deepeval core deps: openai, posthog, grpcio, opentelemetry-sdk, pytest plus plugins, aiohttp. No langchain. A fresh venv holds 66 packages and 69 MB | VERIFIED-LIVE | `uv pip install deepeval==4.2.8` |
| **A fresh ragas 0.4.3 install does not import.** It fails with `ModuleNotFoundError: langchain_community.chat_models.vertexai` because langchain-community 0.4.2 removed that module. Pinning `langchain-community<0.4` (0.3.31) fixes it | VERIFIED-LIVE | run log |
| ragas telemetry is on by default and POSTs to `https://t.explodinggradients.com`. Opt out with `RAGAS_DO_NOT_TRACK=true`. It also writes an anonymous id to `appdirs.user_data_dir("ragas")/uuid.json` | VERIFIED-DOC (README "Open Analytics") and VERIFIED-LIVE (`ragas/_analytics.py`) | github.com/vibrantlabsai/ragas |
| With the default env, the ragas `llm_factory`/`InstructorLLM` path tried to reach t.explodinggradients.com (blocked by the guard). With `RAGAS_DO_NOT_TRACK=true` it made 0 attempts. A custom `InstructorBaseRagasLLM` subclass running the v2 `metrics.collections` made 0 attempts even without the opt-out, because `track()` lives in InstructorLLM, the legacy metric base and the testset generator | VERIFIED-LIVE | netguard log |
| ragas imports tiktoken, and `get_tokenizer()` defaults to `o200k_base`. The four v2 metrics never triggered a tiktoken download in our runs. Whether a download happens on first use is tiktoken behaviour we did not check here | VERIFIED-LIVE (no attempt seen) / UNVERIFIED (download) | `ragas/tokenizers.py` |
| deepeval telemetry is on by default and goes to PostHog `us.i.posthog.com`. It collects event names, metric names, an anonymous UUID and the **public IP**. Opt out with `DEEPEVAL_TELEMETRY_OPT_OUT=1`. The id lives in `~/.deepeval`, which `DEEPEVAL_HOME` can relocate | VERIFIED-DOC and VERIFIED-LIVE (12 blocked lookups of us.i.posthog.com in one metric run) | deepeval.com/docs/data-privacy |
| deepeval also: auto-loads `.env`, `.env.{APP_ENV}` and `.env.local` from CWD (disable with `DEEPEVAL_DISABLE_DOTENV=1`); runs a PyPI update check only when `DEEPEVAL_UPDATE_WARNING_OPT_IN=1`; uses `eval_mode="llm"` by default. `hybrid` or `system_one` mode calls the hosted "Jev" judge, so pin `eval_mode="llm"` | VERIFIED-LIVE (source: `__init__.py`, `config/settings.py`, `config/eval_mode.py`) | deepeval 4.2.8 |
| Custom judge in deepeval: subclass `DeepEvalBaseLLM` and implement `load_model`, `generate(prompt, schema=None)`, `a_generate` and `get_model_name`. deepeval passes in the pydantic schema, so `/api/chat format=schema` works. The built-in `OllamaModel(model, base_url, temperature=0 by default, generation_kwargs, **kwargs)` forwards `**kwargs` to `ollama.Client`, so `headers={"Authorization": "Bearer …"}` should work for the gateway (needs `pip install ollama`) | VERIFIED-LIVE (custom class ran); headers path is source-read only | deepeval.com/guides/guides-using-custom-llms; /integrations/models/ollama |
| Custom judge in ragas: either subclass `InstructorBaseRagasLLM` (`generate(prompt, response_model)` / `agenerate`), or call `llm_factory(model, provider="openai", client=OpenAI(base_url=…/v1, default_headers=…))`. Factory defaults are temperature 0.01, top_p 0.1 and **max_tokens 1024**, which is too small for thinking models. `mode=instructor.Mode.MD_JSON` covers backends without response_format | VERIFIED-LIVE (both paths ran) | `ragas/llms/base.py`; docs.ragas.io customize_models |
| Inputs per metric. **Faithfulness**: question, answer and contexts, in both libraries. **Answer relevancy**: ragas needs an **embedding model** (it generates N=3 questions and averages their cosine similarity to the real one); deepeval needs no embeddings (an LLM classifies each statement). **Context precision**: ragas has `WithReference` and `WithoutReference` variants (the latter compares contexts to the response); deepeval requires `expected_output`. **Context recall**: both need a reference answer | VERIFIED-DOC and source | docs.ragas.io metric pages; deepeval.com metric pages |
| **The two faithfulness metrics mean different things.** deepeval marks a claim `no` only when the context *directly contradicts* it. Unsupported claims are `borderline` and count as faithful unless `penalize_ambiguous_claims=True`. ragas marks any statement that cannot be inferred from the context as unfaithful | VERIFIED-LIVE (`generate_verdicts.txt`, `_calculate_score`) | deepeval source; docs.ragas.io faithfulness |
| Live run on one case (2 supported plus 2 unsupported claims, so the true score is 0.5). ragas with the custom judge scored **0.50** with correct per-statement verdicts. deepeval at defaults scored **0.67**: it extracted 4 claims but returned **3 verdicts**, did not detect the mismatch, and passed an unsupported claim as `borderline` | VERIFIED-LIVE (n=1, qwen3:1.7b) | run log |
| Per-claim output for the UI. deepeval exposes `metric.claims`, `truths`, `verdicts`, `reason` and `verbose_logs`, but verdicts line up with claims only by list index. ragas v2 `ascore()` returns only `MetricResult(value)` with `reason=None`; the statement/verdict/reason triples are reachable only through the private `_create_statements` / `_create_verdicts` | VERIFIED-LIVE | run log |
| Fixing prompts. deepeval accepts `evaluation_template=` (a subclass of e.g. `FaithfulnessTemplate`) and ships its prompts as .txt files. ragas metrics keep prompt objects (`statement_generator_prompt`, `nli_statement_prompt`) that can be swapped | VERIFIED-LIVE (source) | both packages |
| CPU time, qwen3:1.7b, thinking off: ragas ran all 4 metrics in 11 calls and about 130 s; deepeval faithfulness took 4 calls and 45 s. ragas through `llm_factory` and `/v1`, with thinking left on, took 210-248 s for faithfulness alone | VERIFIED-LIVE | run log |
| JSON reliability: 0 schema-parse failures in 15 constrained `/api/chat` calls. The failures we saw were semantic (dropped verdicts), not malformed JSON | VERIFIED-LIVE (small n) | run log |

### Still to verify (and in which milestone)
- **M-eval (scorecard):** run each metric at least 5 times on at least 20 golden cases, with qwen3 1.7b/4b/8b locally and a larger judge on Colab. Measure spread and the claim/verdict mismatch rate.
- **M-eval:** check `OllamaModel(headers=…)` or our wrapper against the real bearer gateway.
- **M-CI:** keep the netns plus netguard test permanently, so that a library upgrade cannot add new phone-home calls unnoticed.

### Recommendation for RAG Studio
**Use DeepEval 4.2.x behind our own `Judge` wrapper (`/api/chat` plus bearer token), and override its prompts.** It is actively maintained, while ragas is idle and does not install cleanly. It has a small footprint with no langchain. Its claims and verdicts are public attributes, and answer relevancy needs no embedding model. Required settings:
- env: `DEEPEVAL_TELEMETRY_OPT_OUT=1`, `DEEPEVAL_DISABLE_DOTENV=1`, `DEEPEVAL_HOME=<app data dir>`
- metric arguments: `eval_mode="llm"`, `penalize_ambiguous_claims=True`
- our own guard: if `len(verdicts) != len(claims)`, mark the sample `judge_error` and retry
- `async_mode=False` per sample when the judge is CPU-bound

Keep the ragas metric definitions as teaching material only. If ragas is ever used, set `RAGAS_DO_NOT_TRACK=true`, write a custom `InstructorBaseRagasLLM`, and pin `langchain-community<0.4`.

## 2. Arize Phoenix vs Langfuse (self-hosted)

| Fact | Status | Source |
|---|---|---|
| arize-phoenix is at 20.21.3 (2026-10-14) and requires Python >=3.11,<3.15. Its licence metadata says **Elastic-2.0** (ELv2, source-available, not OSI). arize-phoenix-otel is Apache-2.0 | VERIFIED-LIVE | PyPI JSON |
| Phoenix installs as a pip package: 152 packages, a 602 MB venv. Deps include anthropic, pydantic-ai-slim, fastmcp-slim, pandas, pyarrow, grpcio, strawberry-graphql. Start it with `phoenix serve` | VERIFIED-LIVE | uv install |
| Phoenix stores data in **SQLite** at `$PHOENIX_WORKING_DIR/phoenix.db` (default `~/.phoenix`). Postgres is optional via `PHOENIX_SQL_DATABASE_URL` and the `[pg]` extra | VERIFIED-LIVE (db file created) | `phoenix/config.py` |
| Ports: 6006 serves the UI, REST and OTLP/HTTP `/v1/traces`; 4317 is OTLP/gRPC. The default host is `0.0.0.0`. `PHOENIX_HOST=127.0.0.1` moves 6006 to loopback, but **gRPC 4317 still binds `[::]`** because it is hardcoded in `grpc_server.py:109`. Block 4317 with the firewall or a container port mapping | VERIFIED-LIVE (`ss -ltn`) | source |
| Phoenix RSS is about 415-450 MB idle and after ingesting a test trace. Startup takes 12-20 s on this CPU | VERIFIED-LIVE | `ps` |
| Phoenix at default settings: the server made **8 DNS lookups** at startup (we did not capture the hostnames; the source points to the WASM sandbox download and the agent assistant). Its UI HTML loads **Google Fonts** and a **static.scarf.sh pixel** in the browser | VERIFIED-LIVE (strace, `curl /`) | — |
| Phoenix in lockdown (`PHOENIX_ALLOW_EXTERNAL_RESOURCES=false`, `PHOENIX_TELEMETRY_ENABLED=false`, `PHOENIX_HOST=127.0.0.1`): **0 DNS lookups**, and the fonts and scarf pixel are gone. Only an `og:image` meta tag pointing at raw.githubusercontent.com remains; browsers normally do not fetch that, which we did not verify | VERIFIED-LIVE | config.py docstrings |
| Manual OpenInference spans worked. We created CHAIN, RETRIEVER (with `retrieval.documents.N.document.{id,score,content}`), RERANKER and LLM (with model name and token counts) spans through `phoenix.otel.register(...)` plus `openinference.semconv`, then read all 4 back from `GET /v1/projects/{name}/spans` | VERIFIED-LIVE | `scripts/phoenix_client.py` |
| Auto-instrumentation for raw httpx calls to Ollama does not exist: there is no `openinference-instrumentation-httpx` on PyPI (404). `opentelemetry-instrumentation-httpx` 0.66b1 exists but produces generic HTTP spans with no LLM attributes. `openinference-instrumentation-ollama` 0.1.12 exists for the `ollama` client library (what it patches is unverified) | VERIFIED-LIVE (PyPI) | PyPI JSON |
| Langfuse **server v4** is current (v4.59.0, 2026-10-14). v3 gets security patches only, through January 2027. v4 still needs web and worker containers plus **Postgres (>=15), ClickHouse (>=25.12), Redis (>=7.0) and S3/blob storage** | VERIFIED-DOC | langfuse.com/self-hosting; upgrade-v3-to-v4 |
| The Langfuse compose file runs langfuse-web, langfuse-worker, clickhouse 25.12, minio, redis 7 and postgres 17. **The web UI (3000) and minio (9090) bind all interfaces by default**; the other services bind 127.0.0.1. The docs recommend a VM with at least **4 cores and 16 GiB** | VERIFIED-DOC | github.com/langfuse/langfuse docker-compose.yml; docker-compose docs |
| Langfuse telemetry: `TELEMETRY_ENABLED` defaults to `true` and sends aggregate counts plus email *domains* to eu.posthog.com every 12 h or less; set `TELEMETRY_ENABLED=false` to stop it. **EE-licensed instances cannot disable it.** The **update check** to `langfuse.com/api/latest-releases` is "not configurable". Browsers load img.shields.io and static.langfuse.com. Image pulls from docker.langfuse.com are observed by Reo.dev | VERIFIED-DOC | langfuse.com/self-hosting/security/telemetry |
| Langfuse is MIT except the `ee` features, which need a license key | VERIFIED-DOC | langfuse.com/self-hosting/license-key |
| Langfuse Python SDK 5.0.1 (released today, MIT) is OTel-based, with deps httpx, otel-sdk and otlp-http exporter, and no analytics library | VERIFIED-LIVE (PyPI deps) | PyPI JSON |
| RAM use of the Langfuse stack on this box was not measured | UNVERIFIED | — |
| This machine has 15 GiB RAM, about 6 GiB free while Ollama runs, and 8 CPUs | VERIFIED-LIVE | `free`, `nproc` |

### Still to verify (and in which milestone)
- **M-tracing:** capture the hostnames behind Phoenix's 8 default DNS lookups (strace `-e sendto -s 200`), and confirm a browser with lockdown settings makes 0 external requests (DevTools network log).
- **M-tracing:** check whether the `openinference-instrumentation-ollama` spans are useful if we ever switch to the `ollama` client.
- Langfuse needs no further work unless requirements change.

### Recommendation for RAG Studio
**Phoenix, as an optional dev sidecar rather than an app dependency.** The app emits OTel spans with OpenInference attributes (retrieval, rerank and LLM spans created manually around our httpx calls) and keeps its own per-query trace record for the in-app UI. When a learner wants a richer trace viewer, Phoenix runs from its own venv (Phoenix needs 3.11+ and pulls about 150 packages) with this environment:

`PHOENIX_HOST=127.0.0.1 PHOENIX_ALLOW_EXTERNAL_RESOURCES=false PHOENIX_TELEMETRY_ENABLED=false PHOENIX_WORKING_DIR=<app data>/phoenix`

Port 4317 must also be firewalled, or OTLP over HTTP used only. Langfuse is rejected: five services, a 16 GiB VM recommendation, and an update check that cannot be turned off, all on a 15 GiB laptop that already runs Ollama, fail the requirement that nothing leaves the machine. Note the ELv2 licence if RAG Studio is ever offered as a hosted service.

## 3. Retrieval techniques: what the evidence actually shows

| Technique / fact | Status | Source |
|---|---|---|
| **Contextual Retrieval** (Anthropic, 2024-09-19). Metric is 1 − recall@20. Failure rate went from 5.7% at baseline to 3.7% with contextual embeddings (−35%), to 2.9% with contextual BM25 added (−49%), and to 1.9% with reranking added (−67%). Setup: Gemini Text 004 embeddings, Cohere reranker over the top 150 → 20, datasets of codebases, fiction, arXiv and science papers. Cost was $1.02 per million document tokens with prompt caching on Claude 3 Haiku | VERIFIED-DOC (vendor's own experiment) | anthropic.com/news/contextual-retrieval |
| The context prompt wraps the full document in `<document>` and the chunk in `<chunk>`, then asks: "Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else." | VERIFIED-DOC (quoted fragments; copy the full template from the post or cookbook) | same |
| **HyDE** (Gao, Ma, Lin, Callan; arXiv 2212.10496, Dec 2022). text-davinci-003 writes a hypothetical document, which unsupervised Contriever then embeds. DL19 nDCG@10: Contriever 44.5 → HyDE 61.3 (BM25 50.6, fine-tuned ContrieverFT 62.1). DL20: 42.1 → 57.9 (ContrieverFT 63.2). The paper itself says that with a fine-tuned encoder, "less powerful instruction LMs can negatively impact" results | VERIFIED-DOC | arxiv.org/abs/2212.10496 (Tables 1 and 4, §5.2) |
| **RAG-Fusion** (Rackauckas, arXiv 2402.03367, Jan 2024): multiple generated queries plus reciprocal rank fusion. Evaluation was manual on an Infineon chatbot, and the abstract gives no numbers. It reports answers going off-topic when the generated queries drift | VERIFIED-DOC (weak evidence) | arxiv.org/abs/2402.03367 |
| **Least-to-most** (Zhou et al., ICLR 2023): SCAN length split ≥99% versus 16% for chain-of-thought (code-davinci-002) | VERIFIED-DOC | arxiv.org/abs/2205.10625 |
| **Self-ask** (Press et al., Findings of EMNLP 2023): describes the "compositionality gap" in GPT-3 and shows self-ask with search narrowing it. Specific numbers were not retrieved | VERIFIED-DOC (claim) / UNVERIFIED (numbers) | arxiv.org/abs/2210.03350 |
| **A-RAG** (Du et al., arXiv 2602.03442, 2026-02-03): an agent with keyword search, semantic search and chunk-read tools. With **GPT-4o-mini and GPT-5-mini** it beats Naive RAG everywhere: e.g. GPT-5-mini MuSiQue 52.8 → 74.1 and HotpotQA 81.2 → 94.5 LLM-Acc. With GPT-4o-mini it loses to HippoRAG2 on HotpotQA (77.1 vs 80.7) and 2Wiki (60.2 vs 64.7). Retrieved tokens are similar to or lower than Naive RAG. Removing one tool costs only 0.5-4.7 points. The judge is GPT-5-mini, and there are no small or local models | VERIFIED-DOC | arxiv.org/html/2602.03442v1 |
| **"Do We Still Need GraphRAG?"** (Fan et al., arXiv 2604.09666, April 2026), using Qwen2.5 3B/7B/32B agents. Single-shot dense RAG versus the best GraphRAG on 7B: NQ 46.6 vs 48.3, HotpotQA 19.0 vs 46.7, MuSiQue 21.0 vs 48.0. The multi-hop gap averages +27.2. Agentic search on dense RAG is agent-dependent: Search-o1 *hurt*, GraphSearch helped, but the gap barely moved (+26.6). Larger backbones narrow it slightly. MS GraphRAG costs $13.19 and 1.72 h per 1M tokens to build. Authors' advice: dense RAG for general QA, GraphRAG for complex multi-hop | VERIFIED-DOC | arxiv.org/html/2604.09666v1 |
| **Pi-Serini** (Hsu, Yang, Lin; arXiv 2605.10848, May 2026): BM25 plus a frontier LLM (gpt-5.5) with retrieve, browse and read tools reaches 83.1% on BrowseComp-Plus. Tuning BM25 adds +18.0% and greater retrieval depth adds +25.3% evidence recall. It depends on a frontier model | VERIFIED-DOC (abstract) | arxiv.org/abs/2605.10848 |
| A secondary source says Search-R1 with BM25 scores 3.86% on BrowseComp-Plus while GPT-5 with BM25 scores 55.9%, so agent capability dominates | CLAIM (not opened) | aiweekly.co; BrowseComp-Plus |
| **Semantic chunking** (Qu, Tu, Bao; arXiv 2410.13070, Oct 2024): fixed-size chunking won evidence retrieval on 3 of 5 datasets. Semantic chunkers helped only on artificially "stitched" multi-topic documents. Answer BERTScore was essentially equal (e.g. 0.65/0.65/0.65). Conclusion: "fixed-size chunking remains a more efficient and reliable choice", and embedding quality matters more | VERIFIED-DOC | arxiv.org/pdf/2410.13070 |
| **Reranking:** BEIR (NeurIPS 2021) found "re-ranking and late-interaction models on average achieve the best zero-shot performances, however, at high computational costs", with BM25 "a robust baseline". In Anthropic's setup, adding a reranker took the failure reduction from 49% to 67% | VERIFIED-DOC | arxiv.org/abs/2104.08663; Anthropic post |
| **ColPali** (Faysse et al., ICLR 2025): a VLM produces multi-vector page-image embeddings with late interaction, introduced together with the ViDoRe benchmark, and "largely outperforms" text pipelines. Specific numbers, GPU cost and storage cost were not retrieved | VERIFIED-DOC (claim) / UNVERIFIED (numbers) | arxiv.org/abs/2407.01449 |

### Defaults this suggests for a small local system
- **Chunking:** fixed-size with overlap as the default. Semantic chunking is an experiment, not a default.
- **Retrieval:** hybrid BM25 plus dense (BM25 is consistently strong), followed by a small local cross-encoder rerank of the top 30-50. The Anthropic and BEIR evidence used cloud or large rerankers, so the gain from ours must be measured.
- **Contextual Retrieval:** opt-in ingestion experiment, run on the Colab T4. It costs one LLM call per chunk with the whole document in context, which is impractical on CPU.
- **HyDE, multi-query/RAG-Fusion, decomposition:** off by default and available as toggles. The evidence comes from large models, and HyDE's own paper warns that weak generators can hurt results.
- **Agentic RAG:** an optional mode with a tool loop (keyword search, semantic search, read chunk), run on Colab. The gains in 2026 papers come from GPT-4o-mini or stronger, and 7B agents were mixed. No GraphRAG.
- **ColPali:** not pursued, because it needs a GPU VLM and multi-vector storage, and our corpus is text-first.

### What the scorecard must measure to decide
- recall@k and 1−recall@20, nDCG@10 and MRR on a labelled golden set, split into **single-hop and multi-hop**
- answer faithfulness and correctness from the judge, reported with spread (see §4)
- added latency per query (p50 and p95) and LLM calls per query
- ingestion time per 1k chunks, and index size
- tokens sent to the generator
- for agentic mode, the number of tool calls and the failure or loop rate

Each technique is an A/B against the same baseline config, under the same judge and prompt hash.

### Still to verify
- **M-retrieval:** a local cross-encoder's actual CPU latency and gain on our corpus.
- **M-agentic:** whether qwen3 4B/8B can drive a 3-tool loop at all (A-RAG gives no small-model data).
- ColPali and self-ask numbers, only if those features are reconsidered.

## 4. LLM-as-judge reliability and how to report it

| Fact | Status | Source |
|---|---|---|
| Thakur et al., "Judging the Judges" (GEM2 2025): 13 judges were compared. Only the largest and best reach reasonable human alignment, and even they fall well below inter-human agreement. Scores differ from human scores by up to 5 points. Judges tend toward **leniency** and are sensitive to prompt length and complexity. Small judges can still give a usable *ranking* signal. Percent agreement hides disagreement, so the authors recommend proper alignment metrics | VERIFIED-DOC | arxiv.org/abs/2406.12624 |
| Atil et al., "Non-Determinism of 'Deterministic' LLM Settings" (v5, Apr 2025): 5 LLMs, 8 tasks, 10 runs. Accuracy varied **up to 15%** across runs and the best-to-worst gap reached 70%. They propose the TARr@N / TARa@N agreement metrics | VERIFIED-DOC | arxiv.org/abs/2408.04667 |
| Thinking Machines, "Defeating Nondeterminism in LLM Inference" (2025-09-10): server-side variation comes from **batch-size-dependent kernels** (no batch invariance). Qwen3-235B at T=0 produced 80 unique completions out of 1000; batch-invariant kernels made all 1000 identical at a 1.6-2.1× slowdown. Single-request CPU inference was not tested | VERIFIED-DOC | thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference |
| Schroeder & Wood-Doughty (arXiv 2412.12509): measure judge reliability with McDonald's omega over multiple samples. Fixed seeds do not guarantee reliability, and single-shot judgments mislead | VERIFIED-DOC (abstract) | arxiv.org/abs/2412.12509 |
| **Local evidence:** Ollama 0.32.6 on CPU, qwen3:1.7b, T=0, seed 42, schema-constrained, the same request 3 times. The first call (cold) returned 3 verdicts `[1,0,0]`; the 2nd and 3rd returned an identical single verdict `[1]`, i.e. **2 claims dropped**. seed=7 produced the same output as the warm seed=42 calls | VERIFIED-LIVE (n=4) | `scripts/determinism.py` |
| **Local evidence:** ragas faithfulness through `/v1` (T=0, thinking on) scored 0.50 on one run and 0.33 on the next for the same case | VERIFIED-LIVE (n=2) | run log |
| A JSON schema guarantees valid JSON but not the semantics: array lengths were not enforced, and both deepeval and the raw test dropped items | VERIFIED-LIVE | run logs |

### Recommendation (reporting protocol)
1. **Pin and record every judge run**: judge model name **and digest** (from `/api/tags`), Ollama version, options (temperature 0, seed, num_ctx, think=false), the sha256 of the prompt template, the library version, and the gateway target (local or Colab). Store all of this with each score, and never compare scores across different pins.
2. **Repeat runs:** k=3 per sample during interactive use and k=5 for scorecard and release runs. Report mean ± SD (or min-max) per metric, plus the **agreement rate** (TARa@k on verdicts). Flag samples whose range exceeds 0.25.
3. **Validate structure:** require claim and verdict counts to match, or put a `minItems`/`maxItems` constraint in the schema. Count failures as `judge_error` rather than scores, and report the error rate.
4. **Calibrate once per judge:** build about 30-50 human-labelled claims from our corpus, report Cohen's kappa against the judge, and show "judge = qwen3:1.7b, κ = x" in the UI. Use the larger Colab judge for release scorecards.
5. **Rank, don't promise:** present small-judge scores as relative A/B signals rather than absolute quality, in line with Thakur et al.

### Still to verify
- **M-eval:** whether warm-call outputs are stable over at least 20 repeats, and whether the cold/warm difference comes from prompt caching (test with `/api/chat` `keep_alive` and cache-reuse settings).
- **M-eval:** how big the spread is on the Colab T4 GPU compared with CPU.

## Sources (all accessed 2026-10-14)
- https://pypi.org/pypi/ragas/json · https://pypi.org/pypi/deepeval/json · https://pypi.org/pypi/arize-phoenix/json · https://pypi.org/pypi/langfuse/json · https://pypi.org/pypi/arize-phoenix-otel/json · https://pypi.org/pypi/openinference-instrumentation-ollama/json · https://pypi.org/pypi/opentelemetry-instrumentation-httpx/json
- https://github.com/vibrantlabsai/ragas (README "Open Analytics"; GitHub API commits/releases)
- https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/ · …/answer_relevance/ · …/context_precision/ · …/context_recall/ · https://docs.ragas.io/en/stable/howtos/customizations/customize_models/
- https://deepeval.com/docs/data-privacy · https://deepeval.com/integrations/models/ollama · https://deepeval.com/guides/guides-using-custom-llms · https://deepeval.com/docs/metrics-faithfulness · https://deepeval.com/docs/metrics-answer-relevancy · https://deepeval.com/docs/metrics-contextual-precision · https://deepeval.com/docs/metrics-contextual-recall
- https://langfuse.com/self-hosting · https://langfuse.com/self-hosting/deployment/docker-compose · https://langfuse.com/self-hosting/security/telemetry · https://langfuse.com/self-hosting/license-key · https://langfuse.com/self-hosting/upgrade/upgrade-guides/upgrade-v3-to-v4 · https://langfuse.com/changelog/2026-08-17-langfuse-v4 · https://raw.githubusercontent.com/langfuse/langfuse/main/docker-compose.yml
- Phoenix: arize.com/docs/phoenix returned HTTP 403 to our fetcher, so Phoenix facts come from the installed 20.21.3 source and live runs
- https://www.anthropic.com/news/contextual-retrieval
- https://arxiv.org/abs/2212.10496 · https://arxiv.org/abs/2402.03367 · https://arxiv.org/abs/2205.10625 · https://arxiv.org/abs/2210.03350
- https://arxiv.org/abs/2602.03442 (html v1) · https://arxiv.org/abs/2604.09666 (html v1; alphaxiv mirror https://www.alphaxiv.org/abs/2604.09666) · https://arxiv.org/abs/2605.10848 · https://aiweekly.co/alerts/bm25-beats-dense-retrieval-and-agents-by-20-points-at-scale (CLAIM)
- https://arxiv.org/abs/2410.13070 · https://arxiv.org/abs/2104.08663 · https://arxiv.org/abs/2407.01449
- https://arxiv.org/abs/2406.12624 · https://arxiv.org/abs/2408.04667 · https://arxiv.org/abs/2412.12509 · https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
