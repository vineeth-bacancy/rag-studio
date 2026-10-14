# A. Ollama API + local LLMs (Qwen 3.5 / Gemma 4) for RAG Studio

Researched 2026-10-14, 20:11 to 20:45 IST. Machine: 8 CPUs, 15.7 GB RAM, no GPU, Ollama **0.32.6**, run as the `ollama` systemd *user* service.
Status tags: **VERIFIED-DOC** (official doc or model card), **VERIFIED-LIVE** (run here), **CLAIM** (vendor or blog), **UNVERIFIED**.
Scripts and raw outputs are in `scratchpad/research/work/` (`embed_test.py`, `ol.py`, `*.out`). Server-side evidence comes from `journalctl --user -u ollama`.
Safety checks: before every load, `ollama ps` was empty or held only my own model, and `trading-daily.service` was `inactive` (it last ran at 19:19 and runs next on 15 Oct at 19:17). Every model was unloaded with keep_alive 0 afterwards.
Other clients (probably sibling agents) also used Ollama during the window. At 20:13 something else loaded nomic-embed-text, and around 20:30 something else loaded qwen3:1.7b at ctx 8192. I left that one loaded because it isn't mine. Timings may carry some noise from this.

## 1. Ollama API (0.32.6)

| Fact | Status | Source / command |
|---|---|---|
| Installed 0.32.6 was released 2026-08-04. The latest is **v0.40.4** (2026-10-12). The docs on `main` and docs.ollama.com are **ahead** of 0.32.6. | VERIFIED-LIVE + DOC | `curl /api/version` → `{"version":"0.32.6"}`; `gh api repos/ollama/ollama/releases` |
| `think` = bool or level string `"low"\|"medium"\|"high"\|"max"` (on /api/chat and /api/generate). docs.ollama.com adds `null` = model default, says level strings are model-defined, and says numbers are not supported. | VERIFIED-DOC | api.md at tag v0.32.6 (l.49, l.498); docs.ollama.com/api/chat; docs.ollama.com/capabilities/thinking |
| In 0.32.6 source, `ThinkValue.Bool()` returns true for any of "low/medium/high/max". A renderer that only knows on/off therefore treats a level as `true`. | VERIFIED-DOC (source) | github ollama v0.32.6 `api/types.go` l.1166-1180 |
| The reasoning text is returned in `message.thinking` (chat) or `thinking` (generate). With `think:false` the field is absent. | VERIFIED-DOC + LIVE | capabilities/thinking doc; qwen3:1.7b and qwen3.5:4b runs (`thinking field present: False`) |
| `/api/show` exposes `thinking.values/default` only from **v0.34.3**. On 0.32.6 the response has no `thinking` key. | VERIFIED-LIVE + DOC | `curl /api/show -d '{"model":"qwen3.5:4b"}'` → keys without `thinking`; v0.34.3 release notes |
| Chat `images` = a per-message list of base64 strings. | VERIFIED-DOC + LIVE | api.md; qwen3.5:4b image test (section 2) |
| `/api/embed`: `input` = string or list of strings. Options are `truncate` (default true; "truncates the **end** of each input"; false → error), `dimensions`, `keep_alive`, `options`. | VERIFIED-DOC | api.md v0.32.6 l.1700-1707; docs.ollama.com/api/embed |
| **Image/audio embedding input** (`{"text","image","audio"}` objects) is documented only on `main`. It shipped in **v0.40.4** ("Added support for multimodal embeddings through llama-server"). On 0.32.6 it is rejected. | VERIFIED-DOC + LIVE | diff of api.md v0.32.6 vs main; v0.40.4 notes; live: `{"input":{"text":..,"image":..}}` → **400 `invalid input type`** |
| nomic-embed-text default: 768 dims, L2 norm 1.000000, 23 prompt tokens, 0.06 s warm. | VERIFIED-LIVE | `python3 -I work/embed_test.py` |
| `dimensions:256` → 256 dims with L2 norm 1.0. This equals `full[:256]` **renormalised** (max abs diff 1.7e-8; the raw prefix had norm 0.618). | VERIFIED-LIVE | same |
| `dimensions:1024` (above the native 768) → HTTP 200 with **768 dims, silently**. all-minilm `dimensions:128` → 128 dims, norm 1.0, accepted even though MiniLM is not Matryoshka-trained. | VERIFIED-LIVE | same |
| `truncate:false` with input over 2048 tokens → **HTTP 400 `{"error":"the input length exceeds the context length"}`**. Adding `options.num_ctx:8192` gives the same 400. The default `truncate:true` → 200 with `prompt_eval_count` = 2048 (4.2 s). | VERIFIED-LIVE | same; `/api/ps` showed `context_length: 2048` for nomic-embed-text |
| Why num_ctx 8192 has no effect on nomic (presumably capped at the model's trained context of 2048) | UNVERIFIED | inference |
| `keep_alive`: default 5m. Accepts a duration string or a number in seconds. `0` unloads after the request; a negative value keeps the model loaded indefinitely. `OLLAMA_KEEP_ALIVE` sets the server default, and the per-request value overrides it. | VERIFIED-DOC | docs.ollama.com/faq; docs.ollama.com/api/chat |
| `keep_alive:0` unloads (`/api/ps` → `[]`), and `"2m"` → `expires_at` = now + 2 min. | VERIFIED-LIVE | embed_test.py, ol.py `unload()` |
| `/api/ps` fields: `name, model, size, digest, details{format,family,families,parameter_size,quantization_level}, expires_at, size_vram, context_length`. **`context_length` is present** in the live response and in the docs.ollama.com schema, but not in the api.md example. On this machine `size_vram` = 0. | VERIFIED-LIVE + DOC | `/api/ps` output; docs.ollama.com/api/ps |
| On 0.32.6, GGUF models run through a bundled **llama-server**, one process per (model, num_ctx). **Changing `num_ctx` between requests restarts the server and reloads the model** (2.3-4 s for qwen3:1.7b). | VERIFIED-LIVE | journal: `msg="using llama-server for model"`, `starting llama-server ... -c 512` |
| **num_ctx overflow, single message**: a 1792-token prompt with num_ctx 512 → **HTTP 200, no API error**. The server log shows `WARN "truncating input prompt" limit=258 prompt=1792 keep=4 new=258`. It keeps the first 4 tokens and the **tail**, and **drops the start**, cutting to roughly num_ctx/2. `prompt_eval_count` = **258**. | VERIFIED-LIVE | `python3 -I work/ol.py qwen3_small` + journalctl |
| Overflow test: the marker "project codename is PELICAN-42" is at the start and the question is at the end. When the prompt fits (273 tok, ctx 512), the answer is `PELICAN-42`. When it overflows (1.2k words, ctx 512), the answer is **`Recycle`** (a word from the tail). Run 1 even quoted the spliced text: "The eligible for credit, we'll recycle…" ("The" kept from the head). | VERIFIED-LIVE | `work/qwen3_control2.out`, `work/qwen3_small.out` |
| **Overflow, multi-turn**: with system + long old turn + new question, Ollama **drops whole older messages** and keeps the system prompt and the last user message (59-73 tokens sent, no truncation WARN). Answer: `PELICAN-42`, which was in the system prompt. | VERIFIED-LIVE | `work/qwen3_control.out` + journal `task.n_tokens = 59` |
| Default context on docs.ollama.com: 4k when VRAM < 24 GiB. `OLLAMA_CONTEXT_LENGTH` overrides it. No doc describes the truncation behaviour. | VERIFIED-DOC | docs.ollama.com/context-length, /faq |
| qwen3:1.7b `think:false`: load 4.0 s, prompt 28 tok at 87.8 tok/s, gen 54 tok at **21.2 tok/s**, total 6.9 s. A 1792-token prompt ran at 69 tok/s. Resident size at ctx 4096 is 1.88 GB. | VERIFIED-LIVE | `work/qwen3_small.out` |
| **No rerank endpoint**: not in api.md (v0.32.6 or main) or docs.ollama.com. Live, `POST /api/rerank` → 404 and `POST /v1/rerank` → 404. Issue #3368 "Reranking models" is still open (updated 2026-08-08). PRs #14172 and #11328 are open and unmerged. Release notes up to v0.40.4 never mention rerank. | VERIFIED-DOC + LIVE | curl on 127.0.0.1:11434; `gh api search/issues q='repo:ollama/ollama rerank in:title'` |

**Still to verify**
- Whether nomic-embed-text in Ollama is v1.5, and therefore Matryoshka-trained so that `dimensions` keeps quality (milestone: embeddings/ingest).
- Whether to upgrade Ollama from 0.32.6 to 0.40.x for multimodal embeddings. This affects the trading app and needs the user's decision (milestone: multimodal ingest).
- `think:"max"` and level strings on a model that really supports levels, such as gpt-oss (milestone: generation settings; low priority).

**Recommendation for RAG Studio**
- Always send `think` explicitly. Qwen 3.5 defaults to thinking **on**, while Gemma 4 in Ollama defaults to **off** (see section 3).
- Pick one `num_ctx` per model (for example 8192) and never vary it per request, because every change reloads the model.
- Detect silent truncation by comparing `prompt_eval_count` with your own token estimate. If it is smaller, or close to num_ctx/2, show a "context truncated: the START of your prompt was dropped" warning. This is a good teaching point. Put the question last and keep the system prompt as a separate message, since Ollama keeps the system message on multi-turn overflow.
- Embeddings: send `truncate:false` and show the 400 as a "chunk too long" lesson. Use `dimensions` only with Matryoshka models, because Ollama ignores sizes above native and never warns.
- Reranking cannot happen inside Ollama 0.32.6. Do it in-app instead, for example with a cross-encoder through sentence-transformers or an LLM yes/no scorer through /api/chat.
- Use `/api/ps` (`size`, `context_length`, `expires_at`) for a "loaded models" panel. Unload with keep_alive 0 before 19:17 IST.

## 2. qwen3.5:4b (pulled 20:18 IST, 3.3 GB)

| Fact | Status | Source / command |
|---|---|---|
| `ollama show`: arch `qwen35`, 4.2B, **Q4_K_M**, context length 262144, requires 0.30.0, capabilities completion/vision/tools/thinking, CLIP projector 333.5M. Licence Apache 2.0. Default params: temperature 1, top_k 20, top_p 0.95, presence_penalty 1.5. | VERIFIED-LIVE | `ollama show qwen3.5:4b` |
| Registry: model layer 2.649 GB + projector 0.676 GB. `qwen3.5:2b` defaults to **Q8_0** (2.012 GB + 0.671 GB). | VERIFIED-DOC (registry) | `registry.ollama.ai/v2/library/qwen3.5/manifests/{4b,2b}` + config blobs |
| **`think:false` turns thinking off**: no `thinking` field, and the answer goes straight to `content`. | VERIFIED-LIVE | `python3 -I work/ol.py qwen35` |
| **`think:true` → reasoning in `message.thinking`** (2958 chars). With num_predict 700 the whole budget went to thinking: `content` = "" and `done_reason=length`, 142.7 s. | VERIFIED-LIVE | same |
| Omitting `think` → thinking **on** by default. `think:"low"` → HTTP 200 but output **identical** to the default (same 1040-char trace, hit the 400-token cap): **levels are not honoured** for Qwen 3.5 in 0.32.6. | VERIFIED-LIVE | same; consistent with `ThinkValue.Bool()` |
| RAG prompt: the given system prompt + 8 numbered passages taken from the iMac PER PDF = **1318 prompt tokens**. temperature 0, num_ctx 8192, think:false. | VERIFIED-LIVE | `work/passages.txt`, `work/system.txt` |
| **Cold** run: load **25.0 s**, prompt 1318 tok in 38.8 s = **34.0 tok/s**, gen 53 tok = **6.82 tok/s**, total **71.6 s**. | VERIFIED-LIVE | `work/qwen35.out` |
| Answer: "Apple reduced iMac emissions by 99 kg compared to their business-as-usual scenario [1]. Of the total reduction, about 13 percent was achieved through using more than 15 percent recycled content… [1]". Citations are correct. The passage's "more than 20 percent" was rendered as "99 kg", a figure that is also in [1]. | VERIFIED-LIVE | same |
| **Warm**, unanswerable question: prompt 1301 tok in 35.6 s (36.6 tok/s), gen 10 tok at 7.09 tok/s, total **37.4 s**. Output was exactly `I could not find this in the documents.` | VERIFIED-LIVE | same |
| Prefix cache: Qwen 3.5 is a hybrid recurrent model, so llama-server can only restore at **context checkpoints**. It reused the 294-token system-prompt checkpoint even though the shared prefix (LCP) was 1282 tokens. `prompt_eval_count` still reports the full prompt. | VERIFIED-LIVE | journal: `restored context checkpoint (... n_tokens = 294)` |
| Gen speed across runs was 6.4-7.1 tok/s, with one outlier at 4.1 tok/s (likely contention). Resident `size` at ctx 8192 was **3.36 GB**, `size_vram` 0. | VERIFIED-LIVE | `/api/ps`, `work/qwen35.out` |
| **Image input works**: page 2 of `apple_imac_per_2024_10.pdf` rendered at 100 dpi (850×1100 PNG) costs **918 image tokens** (954 prompt tokens in total). Prompt phase 57.5 s (image batch decode 31.8 s), gen 69 tok at 7.1 tok/s, **total 67.5 s**. | VERIFIED-LIVE | `python3 -I work/ol.py qwen35_image`; journal `n_tokens_batch = 918` |
| Image answer: "…comparing baseline emissions (445 kg) with reduced iMac emissions (346 kg)… over 99 kg of emissions were cut through… recycled materials, low-carbon electricity…, non-air shipping, renewable energy". All main numbers match the chart. Small flaws: "over 99 kg" and "low-carbon electricity for … product use". | VERIFIED-LIVE | `work/qwen35_image.out`; chart viewed in `work/page2.png` |
| The Ollama Qwen 3.5 renderer puts all images at the **front** of the message (`<|vision_start|><|image_pad|><|vision_end|>` before the text). | VERIFIED-DOC (source) | ollama v0.32.6 `model/renderers/qwen35.go` l.56-59 |

**Still to verify**
- Answer quality across a real question set, comparing qwen3.5:4b and qwen3:1.7b on refusal and citation accuracy (milestone: eval harness).
- Time-to-first-token with `stream:true`, which will be about prompt_s plus 0.15 s here (milestone: chat UI).
- Image cost at lower DPI or with a cropped chart (milestone: chart/vision ingest).
- Whether presence_penalty 1.5, Ollama's default, hurts temperature-0 RAG answers (milestone: generation settings).
- Behaviour under memory pressure when only 2 GB is free: swap risk (milestone: ops).

**Recommendation for RAG Studio**
- Make qwen3.5:4b the quality option and **always send `think:false`**. Thinking costs over 700 tokens, about 2+ minutes on this CPU, so expose it only as a "show reasoning" lesson with a warning.
- Prompt processing (~35 tok/s) dominates latency. Keep the context to about 4-6 chunks (~800-1000 tokens) and stream the output.
- Keep qwen3:1.7b as the fast option: ~21 tok/s gen and ~70-100 tok/s prompt, but weaker. In the overflow tests it misread a marker even when the prompt fitted.
- Treat image understanding as an **offline, ingest-time** step: about 1 minute per page at 100 dpi, never per query. Cache the captions.
- Check free RAM before loading, because qwen3.5:4b needs about 3.4 GB resident at 8k ctx.

## 3. Gemma 4 vs Qwen 3.5 (docs only; gemma4 not pulled)

| Fact | Status | Source |
|---|---|---|
| Ollama tags: `gemma4:e2b` 4.6-7.5 GB, `e4b` 6.6-9.5 GB (both 128K ctx), `12b` 7.7-8.0 GB (256K). Inputs: Text, Image. Recommended sampling: temp 1.0, top_p 0.95, top_k 64. Gemma 4 fixes appear in release notes from v0.20.6; e4b manifest `requires` 0.30.9. | VERIFIED-DOC | ollama.com/library/gemma4; registry config blob |
| **Default tag → quant**: `gemma4:latest` = `gemma4:e4b` (same config digest `7ca1ae56…`). file_type **Q4_K_M**, model_type 7.5B. Layers: model 5.493 GB + projector 0.992 GB + MTP `draft` 0.099 GB. Params `{draft_num_predict:3, temperature:1, top_k:64, top_p:0.95}`. | VERIFIED-DOC (registry) | `curl https://registry.ollama.ai/v2/library/gemma4/manifests/e4b` and `.../blobs/<config>` |
| `gemma4:e2b`: Q4_K_M, 4.6B, 3.502 + 0.987 + 0.098 GB. `gemma4:e4b-it-qat`: **Q4_0**, 5.155 + 0.992 GB. `gemma4:12b`: Q4_K_M, 11.9B, 7.381 + 0.175 + 0.465 GB. | VERIFIED-DOC (registry) | same, other tags |
| Gemma 4 thinking: put `<|think|>` at the start of the system prompt and remove it to disable. When disabled, E2B and E4B emit no empty thought block, while larger models emit `<|channel>thought\n<channel|>`. Do not carry earlier thoughts in the history. | VERIFIED-DOC | HF google/gemma-4-E4B-it card l.371-386; ollama.com/library/gemma4 |
| **Ollama mapping**: the 0.32.6 `gemma4.go` renderer writes `<|think|>\n` into the system turn **only if `think` is true or a level string**. `think:false` **or omitted** → no token → **thinking off by default** for Gemma 4. Level strings are not differentiated. | VERIFIED-DOC (source) | ollama v0.32.6 `model/renderers/gemma4.go` l.46-52 |
| The same mapping has not been run live on this machine (gemma4 not pulled). | UNVERIFIED | — |
| Image token budgets are 70/140/280/560/1120. Use higher budgets for OCR, document parsing and small text. Put **images before text** (and audio after text). | VERIFIED-DOC | HF Gemma 4 card l.397-401; ollama.com/library/gemma4 |
| How to choose a Gemma 4 image budget through the Ollama API: not documented. v0.34.4 auto-picks resolution on Apple Silicon (MLX) only. | UNVERIFIED | v0.34.4 release notes |
| Gemma 4 E4B (E2B) benchmarks: MMLU-Pro 69.4 (60.0); GPQA-D 58.6 (43.4); MMMU-Pro 52.6 (44.2); MATH-Vision 59.5 (52.4); **OmniDocBench 1.5 edit distance 0.181 (0.290)**, lower is better; MRCR v2 8-needle 128k 25.4 (19.1). **No DocVQA, ChartQA or OCRBench published.** | VERIFIED-DOC (vendor-reported) | HF gemma-4-E4B-it card table l.88-109 |
| Qwen 3.5-4B benchmarks: MMLU-Pro 79.1; GPQA-D 76.2; IFEval 89.8; AA-LCR 57.0; LongBench v2 50.0; MMMU 77.6; MMMU-Pro 66.3; **OmniDocBench1.5 86.2** (a score, higher is better, so **not comparable** to Gemma's edit distance); CharXiv(RQ) 70.8; MMLongBench-Doc 54.2; CC-OCR 76.7; AI2D 89.6; **OCRBench 85.0**. **No DocVQA or ChartQA published.** | VERIFIED-DOC (vendor-reported) | HF Qwen/Qwen3.5-4B README tables (parsed raw HTML) |
| Which mode (thinking or not) the Qwen 3.5 scores were measured in. The card says the models think by default. | UNVERIFIED | — |
| Qwen 3.5: thinks by default. Disable with `enable_thinking:false`; the `/think` and `/nothink` soft switches are **not** supported. Sampling for non-thinking general use: temp 0.7, top_p 0.8, top_k 20, presence_penalty 1.5. Thinking general: temp 1.0, top_p 0.95, top_k 20, presence_penalty 1.5. Context 262,144 native, extensible to ~1.01M. Licence Apache 2.0. | VERIFIED-DOC | HF Qwen3.5-4B card l.712, 835-838, 994 |
| Gemma 4 licence: Apache 2.0. | VERIFIED-DOC | HF card front matter; ai.google.dev/gemma/docs/gemma_4_license |
| Gemma 4 E4B CPU speed on Google's **LiteRT-LM** engine (not llama.cpp or Ollama): Windows Intel Lunar Lake CPU prefill 173 tok/s, decode 17 tok/s, TTFT 6.0 s, peak 9372 MB. Linux Arm CPU 82 / 18 tok/s. Quantization not stated. | VERIFIED-DOC (other engine) | developers.google.com/edge/litert-lm/models/gemma-4 |
| T4 numbers for either model in Ollama or llama.cpp: none found. willitrunai.com gives *modelled* figures (Qwen 3.5 4B on T4: ~56 tok/s). | CLAIM / UNVERIFIED | willitrunai.com |

**Still to verify**
- Gemma 4 E2B on this box (4.6 GB download): RAM, tok/s, whether `think:false` truly gives no thought block, and chart QA on the same page-2 PNG. Needs the user's approval to pull and a ≥5 GB free-RAM window (milestone: model comparison).
- A side-by-side on RAG Studio's own question set, since published benchmarks lack DocVQA/ChartQA for both models (milestone: eval harness).

**Recommendation for RAG Studio**
- Default to **qwen3.5:4b**. It is measured and working here, fits in about 3.4 GB, and its card reports stronger document and chart numbers (OCRBench 85.0, CharXiv 70.8, MMMU-Pro 66.3 vs Gemma E4B's 52.6).
- **Do not pull gemma4:e4b** on this machine: about 6.6 GB of weights plus projector plus KV, more than the free RAM. If a Gemma comparison is wanted, use `gemma4:e2b` in a dedicated window.
- In the UI, explain the opposite defaults: Qwen 3.5 thinks unless `think:false`, while Gemma 4 in Ollama stays silent unless `think:true`.
- For any vision model, send the image before the text. Ollama does this automatically for Qwen 3.5.

## Sources (all accessed 2026-10-14)
- https://github.com/ollama/ollama/blob/main/docs/api.md (and raw at tag: https://raw.githubusercontent.com/ollama/ollama/v0.32.6/docs/api.md)
- https://docs.ollama.com/api/embed, https://docs.ollama.com/api/chat, https://docs.ollama.com/api/ps
- https://docs.ollama.com/capabilities/thinking, https://docs.ollama.com/faq, https://docs.ollama.com/context-length
- https://github.com/ollama/ollama/releases (v0.20 to v0.40.4 notes, via `gh api`)
- https://raw.githubusercontent.com/ollama/ollama/v0.32.6/api/types.go, https://raw.githubusercontent.com/ollama/ollama/v0.32.6/model/renderers/gemma4.go, https://raw.githubusercontent.com/ollama/ollama/v0.32.6/model/renderers/qwen35.go
- https://github.com/ollama/ollama/issues/3368, https://github.com/ollama/ollama/pull/14172, https://github.com/ollama/ollama/pull/11328
- https://ollama.com/library/gemma4, https://ollama.com/library/qwen3.5
- https://registry.ollama.ai/v2/library/gemma4/manifests/{e4b,latest,e2b,e4b-it-qat,12b} (+ config/params blobs), https://registry.ollama.ai/v2/library/qwen3.5/manifests/{4b,2b}
- https://huggingface.co/Qwen/Qwen3.5-4B (raw README.md)
- https://huggingface.co/google/gemma-4-E4B-it (raw README.md), https://ai.google.dev/gemma/docs/gemma_4_license
- https://developers.google.com/edge/litert-lm/models/gemma-4
- https://willitrunai.com/ja/can-run/qwen-3.5-4b-on-t4-16gb (CLAIM, modelled numbers; seen via search snippet only)
