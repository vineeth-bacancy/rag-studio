# B: Embedding profiles and reranking for RAG Studio (research, 2026-10-14)

Status tags: VERIFIED-DOC = model card or official docs/source · VERIFIED-LIVE = I ran it here (command noted) · CLAIM = vendor, blog or community · UNVERIFIED · ESTIMATE = my calculation, reasoning shown.
No model weights were downloaded. "Range-read" means `curl -r 0-16777215 <blob>` (16 MB GGUF header only), parsed with `scratchpad/research/scripts/gguf_meta.py`.

## 0. Environment facts that drive the decisions

| Fact | Status | Source |
|---|---|---|
| Local Ollama is 0.32.6. The latest release is v0.40.4 (2026-10-12). | VERIFIED-LIVE | `curl localhost:11434/api/version` → `{"version":"0.32.6"}`; `gh api repos/ollama/ollama/releases` |
| The CPU is an i5-1135G7: 4 cores / 8 threads, AVX-512 (laptop class, not 8 physical cores). | VERIFIED-LIVE | `lscpu`, `/proc/cpuinfo` |
| During testing a concurrent `llama-server` used about 400% CPU and 4 GB RAM, leaving about 3.3 GB free. | VERIFIED-LIVE | `top`, `free -m` (20:24) |
| The rag-studio `.venv` (Python 3.12) has no torch, transformers or sentence-transformers yet. | VERIFIED-LIVE | `pyproject.toml` deps: fastapi, uvicorn, httpx |
| The system Python has transformers 4.46.2, which is older than the 4.51 that Qwen3 needs, and a broken numpy/scipy pairing (`ImportError: cannot import name 'Inf'`). It is not usable as a runtime. | VERIFIED-LIVE | `pip list`; failed import |
| Current PyPI versions: transformers 5.19.0 (2026-10-06), sentence-transformers 6.1.0 (2026-09-18), torch 2.14.1. | VERIFIED-LIVE | `pypi.org/pypi/<pkg>/json` |

## 1. Qwen3-Embedding (0.6B / 4B / 8B)

| Fact | Status | Source |
|---|---|---|
| Query format: `Instruct: {task}\nQuery:{query}`. There is no space after `Query:` in the reference code or the ST prompt. | VERIFIED-DOC | HF card `get_detailed_instruct`; `config_sentence_transformers.json` prompt `"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"` |
| The card's own TEI curl example writes `Query: ` with a space, so the card is inconsistent. Follow the code and ST config (no space). | VERIFIED-DOC | HF card, TEI section |
| Documents get no prefix: "No need to add instruction for retrieval documents"; ST `"document": ""`. | VERIFIED-DOC | HF card; ST config |
| Default retrieval task is "Given a web search query, retrieve relevant passages that answer the query". The card recommends a task-specific English instruction and says no instruct costs about 1–5%. | VERIFIED-DOC | HF card note and tip; GitHub README (no other task list) |
| Native dims are 1024 / 2560 / 4096. MRL accepts any output dim from 32 up to the native size. | VERIFIED-DOC | each HF card, "Embedding Dimension" |
| Pooling is the last token, then L2 normalisation, compared by cosine. | VERIFIED-DOC | `1_Pooling/config.json` `pooling_mode_lasttoken: true`, `include_prompt: true`; modules incl. `2_Normalize` |
| The HF tokenizer appends `<|endoftext|>` (151643) to every sequence, so the pooled token is EOS. | VERIFIED-LIVE | `tokenizer.json` post_processor TemplateProcessing; `encode("hi")` → `[6023, 151643]` |
| Context is 32K on the card; the reference code uses `max_length = 8192`. The GGUF `context_length` is 32768 (0.6b) and 40960 (4b, 8b). | VERIFIED-DOC / VERIFIED-LIVE | HF card; GGUF range-read |
| Needs transformers ≥ 4.51 (`KeyError: 'qwen3'` otherwise) and ST ≥ 2.7. | VERIFIED-DOC | HF card |
| MMTEB (multilingual) mean(task): 64.33 / 69.45 / 70.58. Retrieval: 64.64 / 69.60 / 70.88. | VERIFIED-DOC (self-reported) | HF card table |
| MTEB eng v2 mean(task): 70.70 / 74.60 / 75.22. Retrieval: 61.83 / 68.46 / 69.44. | VERIFIED-DOC (self-reported) | HF card table |
| Ollama tags: `:0.6b` is Q8_0 (639 MB, `ac6da0dfba84`); `:4b` is Q4_K_M (2.5 GB, `df5bd2e3c74c`); `:8b` = `:latest` is Q4_K_M (4.7 GB, `64b933495768`). q8_0 and fp16 variants also exist. | VERIFIED-DOC + VERIFIED-LIVE | tags page; registry config blobs `file_type: Q8_0` / `Q4_K_M`; GGUF `general.file_type` 7 / 15 |
| The Ollama GGUFs (0.6b, 4b, 8b) all have `qwen3.pooling_type=3` (LAST), `tokenizer.ggml.add_eos_token=True` and `eos_token_id=151643`. The manifest config has no `requires` field, so 0.32.6 should load them. | VERIFIED-LIVE | range-read of registry blobs; config blob JSON |
| The Ollama manifest has only a model layer (no template or params), so the client must add the `Instruct:` prefix itself. | VERIFIED-LIVE | `registry.ollama.ai/v2/library/qwen3-embedding/manifests/0.6b` |
| `/api/embed` accepts `dimensions` ("Number of dimensions to generate embeddings for"). | VERIFIED-DOC | docs.ollama.com/api/embed; `api/types.go` |
| `dimensions` truncates and re-normalises. Tested on nomic: d=256 gives norm 1.0 and a max diff of 1.4e-8 against the renormalised `full[:256]`. d=5000 silently returns the full 768, with no error. | VERIFIED-LIVE (nomic-embed-text, not qwen3) | two `curl /api/embed` calls compared in Python |
| fp16 overflow: Qwen3-Embedding-8B in fp16 gives NaN on some texts (e.g. ones starting "import…"). The Qwen maintainer advises fp32/bf16. Ollama `8b-fp16` returns a NaN error on "importance". | VERIFIED-DOC (issues) | QwenLM/Qwen3-Embedding#164; ollama#12921 |
| Early Qwen GGUFs needed a manual `<|endoftext|>`. The metadata is fixed now: HF `Qwen3-Embedding-0.6B-GGUF` also has `add_eos_token=True`. A community report says Ollama's library model matches llama.cpp with EOS added. | VERIFIED-LIVE (metadata) / CLAIM (parity) | llama.cpp#14234; HF discussion #17; range-read |
| Other open Ollama issues: all-zero vectors under sustained load (#17878); 0.6b crash on long inputs on macOS (#16582); about 5× slower than TEI on GPU (#12088). | VERIFIED-DOC (issues exist) | github.com/ollama/ollama issues |

**Still to verify**
- **M4:** Pull `qwen3-embedding:0.6b` and embed 20 baseline chunks and 5 queries through both Ollama and ST (fp32, CPU). I expect cosine ≥ 0.99 per vector because Q8_0 noise is small, but this is UNVERIFIED. Also confirm EOS is actually appended (compare with and without a manual `<|endoftext|>`).
- **M4:** Confirm `dimensions` works on qwen3-embedding (e.g. 256 and 512) and re-normalises.
- **M4:** Time the full build of 1,190 chunks on CPU. My ESTIMATE is 10–40 min, based on the PyTorch proxy in section 3 (about 130 tok/s contended); llama.cpp Q8_0 should be faster.
- **Colab milestone:** Run 4b/8b Q4_K_M on T4 under Ollama CUDA and check for NaN or zero vectors, especially on texts starting "import…".

**Recommendation for RAG Studio:** Use the Ollama library tags (`qwen3-embedding:0.6b|4b|8b`), not ad-hoc `hf.co/...` imports. Store the exact query template and task string, and never prefix documents. Pick one RAG-specific task string and keep it fixed per index, for example `Given a question about a PDF report, retrieve passages that answer the question`. Make the default-vs-custom task an eval experiment, not a silent change.

## 2. EmbeddingGemma 2 (`google/embeddinggemma-2`)

| Fact | Status | Source |
|---|---|---|
| The official card is `google/embeddinggemma-2` on HF: created 2026-09-14, last modified 2026-10-06, not gated. There is also ai.google.dev `model_card_2`. Launched 2026-10-06. | VERIFIED-LIVE + VERIFIED-DOC | HF API; ai.google.dev; Google blog |
| Licence is Apache 2.0 (HF card, ai.google.dev, blog). The card also says deployments must follow the Gemma Prohibited Use Policy. | VERIFIED-DOC | HF card header; ai.google.dev/gemma/apache_2 |
| Size: 740M total = 270M text (130M transformer + 140M embedder) + vision 170M + audio 300M. 24 layers, d_model 512, **mean pooling**, 512→768 projection. Context is 8,192 tokens; the config has `max_position_embeddings` 262144 and the Ollama page shows "256K". | VERIFIED-DOC | HF card "Model Overview"; `config.json` |
| Retrieval query prefix: `task: search result \| query: `. QA query prefix: `task: question answering \| query: `. Documents: `title: {title} \| text: {content}`, with `title: none` when there is no title. This is the same scheme as EmbeddingGemma 1. | VERIFIED-DOC | HF card "Task Instruction Prefixes"; `config_sentence_transformers.json` (`query`/`SearchQuery`, `document`/`Document`, `QuestionAnswering`) |
| `prompt_name="Document"` always means `title: none`, so titled documents must be formatted manually. | VERIFIED-DOC | HF card note |
| Image prefixes conflict. The card says "Prefixes apply to text only. Pass images, video, and audio without any prefix." The transformers docs mixed-batch example applies `prompt_name="document"` (system `title: none \| text: `) to images. | VERIFIED-DOC (conflict) | HF card §5; transformers `model_doc/embedding_gemma2` §4 |
| Image embedding in ST ≥ 6.1.0: `model.encode([{"image": path_or_url}])`, or `{"text":…, "image":…}` for one joint vector. In transformers ≥ 5.19: `AutoProcessor.apply_chat_template([[{"role":"user","content":[{"type":"image","url":…}]}]], tokenize=True, return_dict=True, return_tensors="pt")`, then mask-aware mean pooling and L2 normalisation in fp32. | VERIFIED-DOC | transformers docs "Multimodal embeddings" |
| Text and images share one 768-d space; docs: each modality "can be embedded on its own and compared directly against any other modality via cosine similarity". An image costs 280 soft tokens by default (configurable 70–1120). | VERIFIED-DOC | transformers docs; HF card context-limits table |
| Text-only loading: `config_kwargs={"vision_config": None, "audio_config": None}` gives 271M; `{"audio_config": None}` gives 439M (text+image). | VERIFIED-DOC | transformers docs §5 |
| Precision: bf16 or fp32 only. "In `float16` the model returns NaN or silently degraded embeddings." T4 has no native bf16, so use fp32 on T4 and on CPU. | VERIFIED-DOC | HF card §4 |
| Gotcha: `torch.cuda.is_bf16_supported()` defaults to `including_emulation=True`, so the card's dtype snippet may pick emulated bf16 on a T4. | VERIFIED-LIVE (source) / UNVERIFIED on T4 | `inspect.getsource` in local torch 2.5.1 |
| Runtime: transformers v5.19.0 added EmbeddingGemma2. The ST config requires `sentence-transformers>=6.1.0`; the checkpoint was saved with transformers 5.18.0.dev0. | VERIFIED-DOC | transformers v5.19.0 release notes; `config_sentence_transformers.json` `requirements` |
| Benchmarks at 768d: MTEB multilingual v2 61.36 (EG1 61.15); MTEB eng v2 68.46; MTEB code 78.68; MIEB-lite 64.64; MMEB v2 image 57.28 Hit@1; VisDoc 67.84 nDCG@5. At 256d: 60.41 / 67.78. At 128d: 57.89 / 65.68. No retrieval-only sub-score is published. | VERIFIED-DOC (self-reported) | HF card tables |
| No CPU latency is published for v2; the only figure, "<22 ms on EdgeTPU", is for v1. The blog reports Pixel RAM of about 191 MB (text) and 567 MB (full), quantised. | VERIFIED-DOC (absence) / CLAIM | ai.google.dev; Google blog |
| CPU cost ESTIMATE: about 0.23 GFLOP/token (24 layers × ~4.7M params × 2) against Qwen3-0.6B's 0.88. That is about 4× cheaper per token than qwen3-embedding:0.6b; the 140M per-layer embedder is lookups only. | ESTIMATE | derived from card dims and `config.json` |
| **All 17 Ollama tags are safetensors (nvfp4 / mxfp8 / bf16) for the MLX runner.** The config blob reads `{"model_format":"safetensors","file_type":"nvfp4","requires":"0.36.0",...}`. `:270m` is the same digest as `270m-nvfp4-text`. | VERIFIED-LIVE | `registry.ollama.ai/v2/library/embeddinggemma-2/manifests/270m` + config blob |
| On Linux the pull fails with "this model requires MLX support, but the MLX runtime is not available". A maintainer says: "not yet available on Linux and Windows". The issue is still open (0.40.x). | VERIFIED-DOC | ollama#18825, #18971 |
| Ollama v0.40.4 bumped llama.cpp to b11521 ("EmbeddingGemma 2 GGUFs can embed images and audio") and "Added support for multimodal embeddings through llama-server". | VERIFIED-DOC | release notes; PR #18917 |
| `/api/embed` `input` accepts `{"text","image"(base64),"audio"(base64)}` objects on `main`. docs.ollama.com/api/embed still shows text only; PR #18884 to update it is open. | VERIFIED-DOC (source) | `api/types.go` `EmbedRequest`; docs page |
| GGUF builds exist: `ggml-org/embeddinggemma-2-GGUF` Q8_0 (310 MB) + mmproj Q8_0 (555 MB); also BF16. | VERIFIED-LIVE | HF API `?blobs=true` |

**Still to verify**
- **M4:** Install the ST stack in the venv (CPU torch wheel, transformers ≥ 5.19, ST ≥ 6.1). Load text-only, embed 1,190 chunks with `title: none | text: ` (or the real PDF title), and measure time and RAM. The model download is 1.49 GB.
- **M4:** Does `truncate_dim` + `normalize_embeddings=True` in ST 6.1 return unit vectors? The transformers docs say `truncate_dim` slices already-normalised output, so renormalise explicitly anyway.
- **M4 (optional):** On Ollama ≥ 0.40.4, run `hf.co/ggml-org/embeddinggemma-2-GGUF:Q8_0` on Linux CPU and check parity with ST. UNVERIFIED whether it loads.
- **Image milestone:** Test the image-prefix conflict (none vs `title: none | text: `) by measuring text→image recall on 20 figure crops.

**Recommendation:** Do not use Ollama for EmbeddingGemma 2 on this Linux box for now; it is MLX-only and 0.32.6 is too old anyway. Build it as an in-process sentence-transformers profile: fp32, text-only config, explicit prefixes, explicit renormalisation after MRL. Add images later by loading with `audio_config=None`.

## 3. Qwen3-Reranker (0.6B / 4B / 8B)

| Fact | Status | Source |
|---|---|---|
| Score = P(yes) from the last-position logits: `softmax([logit("no"), logit("yes")])[1]`. | VERIFIED-DOC | HF card `compute_logits` |
| Template: `<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`. The default instruction is the web-search task string. | VERIFIED-DOC | HF card (transformers); `chat_template.jinja` |
| The vLLM example in the same card joins the fields with `\n\n`, while transformers and ST use `\n`. Treat `\n` as canonical. | VERIFIED-DOC (inconsistency) | HF card vLLM section |
| The ST `chat_template.jinja` renders byte-identical text and token IDs to the manual format (85 tokens for the sample). Token IDs: yes = 9693, no = 2152, matching `1_LogitScore/config.json`. Fixed template overhead is 73 tokens. | VERIFIED-LIVE | `scripts/rr_template_check.py` (jinja2 + tokenizers) |
| The reranker tokenizer does not append EOS; the embedding tokenizer does. Never share tokenizers between the two. | VERIFIED-LIVE | same script: `[6023]` vs `[6023, 151643]` |
| Official ST CrossEncoder support lives in the Qwen repos (commit "Integrate with Sentence Transformers v5.4", 2026-04-16). `LogitScore` = logit_yes − logit_no at the last position, with `logits_to_keep=1` and left padding enforced. sigmoid(a−b) = eᵃ/(eᵃ+eᵇ), so this equals the card's P(yes). | VERIFIED-DOC (source) + math | HF commits API; `sentence_transformers/cross_encoder/modules/logit_score.py` |
| ST v5.6.0 is titled "Fixes for Causal LM Rerankers". | VERIFIED-DOC | ST releases |
| `*-seq-cls` variants: `tomaarsen/Qwen3-Reranker-0.6B-seq-cls` and `-4B-seq-cls`, by the ST maintainer. They are converted to a 1-logit classifier (yes − no) and still need the same template, applied manually. Equivalent in principle; numeric parity UNVERIFIED. Third-party variants (e.g. `danielchalef/...-seq-cls-vllm-fixed`) are unreviewed. | VERIFIED-DOC / UNVERIFIED | tomaarsen card; vLLM docs |
| vLLM: `vllm serve Qwen/Qwen3-Reranker-0.6B --runner pooling --hf_overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' --chat-template examples/pooling/score/template/qwen3_reranker.jinja`, or serve the seq-cls model. | VERIFIED-DOC | docs.vllm.ai/en/stable/examples/pooling/score (via ctx7) |
| TEI has no merged Qwen3-Reranker support as of v1.9.4. PRs #835 and #886 are open; #795 was closed unmerged. | VERIFIED-LIVE | `gh api` TEI pulls |
| Ollama has no rerank endpoint: the routes on `main` have `/api/embed` but no rerank. Issue #3368 has been open since 2024. Maintainer: "we don't (yet) have an API to support reranking". | VERIFIED-DOC (source + issues) | `server/routes.go`; ollama#3368, #12601, #13183 |
| Community Ollama "Qwen3-Reranker" uploads exist (e.g. `dengcao/Qwen3-Reranker-8B`, 218K pulls). Their own readme says Ollama doesn't support rerankers, and in FastGPT reranking "can be added but is ineffective". | VERIFIED-DOC | ollama.com/dengcao/Qwen3-Reranker-0.6B |
| Why they mislead: `/api/embed` returns a pooled hidden state, not the yes/no logit. `/api/generate`/`chat` returns sampled text with no access to the logit difference. Nothing applies the rerank template. | Reasoning; not tested | — |
| llama.cpp can rerank natively: `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF` (639 MB) has `pooling_type=4` (RANK), `classifier.output_labels=[yes,no]` and an embedded `chat_template.rerank` matching the official template. That makes `llama-server --rerank` a candidate CPU path outside Ollama. | VERIFIED-LIVE (metadata) / UNVERIFIED (runtime) | range-read |
| Context is 32K; the reference uses `max_length=8192`. No batch size is recommended anywhere. | VERIFIED-DOC | HF card; QwenLM GitHub README |
| MTEB-R 65.80 / 69.76 / 69.02 · CMTEB-R 71.31 / 75.94 / 77.45 · MMTEB-R 66.36 / 72.74 / 72.94 · MLDR 67.28 / 69.97 / 70.19 · MTEB-Code 73.42 / 81.20 / 81.22 (top-100 from Qwen3-Embedding-0.6B, which alone scores MTEB-R 61.82). **8B does not beat 4B on English.** | VERIFIED-DOC (self-reported) | HF card evaluation table |
| Weight files are 1.19 / 8.04 / 16.38 GB (bf16). 4B: hidden 2560, 36 layers, tied embeddings. 8B: hidden 4096, untied. | VERIFIED-LIVE | HF API `?blobs=true`; `config.json` |
| No published CPU latency. | VERIFIED-DOC (absence) | card, GitHub, blog |
| **Proxy CPU benchmark** (random weights, exact Qwen3-0.6B block shapes, pure PyTorch fp32, 4 threads, 4 layers timed ×7; CPU contended by the 400% llama-server): 20 pairs × 390 tokens ≈ **70 s** (≈110 tok/s); 5 batches of 4 ≈ 73 s; 1 pair ≈ 3.7 s. | VERIFIED-LIVE (proxy) | `python3 scripts/proxy_bench.py 4 4` |
| Uncontended ESTIMATE: **20–35 s** for 20 pairs. Reasoning: 0.88 GFLOP/token × 7,800 tokens ≈ 6.9 TFLOP; at about 200–300 GFLOPS sustained fp32 on 4 Tiger Lake cores that is 23–35 s. Top-10 roughly halves it. | ESTIMATE | from the above |
| T4 fp16 risk: documented for Qwen3-8B embedding (NaN), Qwen3-Embedding-0.6B under TensorRT fp16 (zeros, RMSNorm suspected) and Ollama 8b-fp16. Nothing found for Reranker-4B specifically. | VERIFIED-DOC (others) / UNVERIFIED (4B) | QwenLM#164; NVIDIA forum; ollama#12921 |
| 4B on T4: fp16 weights are 8.04 GB and fit in 15 GB; fp32 (about 16 GB) does not. FlashAttention-2 needs Ampere, so use SDPA. | ESTIMATE | file sizes |
| 8B in 8-bit (bitsandbytes) ESTIMATE ≈ 9.5 GB: about 6.95B linear params as int8 + `embed_tokens` and `lm_head` at 1.24 GB each in fp16. LLM.int8 multiplies outlier columns in fp16, so overflow risk remains. bnb ≥ 0.45 claims 60–85% better int8 throughput on Turing; it is still expected to be slower than fp16 (UNVERIFIED). | ESTIMATE / CLAIM | bnb docs (ctx7) |
| Community report: Reranker-0.6B under vLLM went OOM on a T4 (no details given). | CLAIM | QwenLM#148 |

**Still to verify**
- **Reranker milestone (first task):** Download 0.6B (1.19 GB) and time ST CrossEncoder on 20 real (query, chunk) pairs with an idle CPU, at batch 4, 8 and 20. Assert that ST scores equal the manual-template transformers scores within 1e-4.
- **Reranker milestone:** If 20 pairs take more than about 15 s, A/B against `llama-server --rerank` with the ggml-org Q8_0 GGUF (latency and Spearman agreement with ST).
- **Colab milestone:** Run 4B fp16 on T4 over the eval set with a NaN/inf guard. Spot-check about 20 pairs against bf16 on Colab CPU. Record peak `torch.cuda.max_memory_allocated()`.

**Recommendation:** See §4.

## 4. Recommendation for RAG Studio

**M4 embedding profile order** (each one is a separate index; the manifest records everything below):
1. **`qwen3-embedding:0.6b` via local Ollama.** It uses the same runtime as the nomic baseline, so this is the smallest change. It runs on 0.32.6 (no `requires`) at Q8_0 and 1024d. It is the biggest expected quality step: MTEB eng v2 retrieval 61.83. First run the ST parity check on 20 chunks.
2. **`embeddinggemma-2` text-only, in-process via sentence-transformers.** It brings in the torch/ST stack that the reranker needs anyway. It costs about 4× less compute per token than Qwen 0.6B on CPU, and it opens the multimodal path later. **Not via Ollama**: the library tags are MLX-only on Linux, and local 0.32.6 is too old.
3. **`qwen3-embedding:4b` on Colab T4 via Ollama (Q4_K_M, 2.5 GB).** About +5 MMTEB-retrieval points over 0.6B. Also index an MRL-1024 variant so the 0.6B-vs-4B comparison holds dimension constant.
4. **`qwen3-embedding:8b` (optional).** Only +1.3 / +1.0 retrieval points over 4B. Build it only if Colab time allows.

**Prefixes and settings to store in the index manifest** (store the literal strings, the model digest/revision and the runtime version; refuse to query an index with a different template):

| Profile | doc template | query template | dims | other manifest fields |
|---|---|---|---|---|
| nomic-embed-text (baseline) | `search_document: {text}` | `search_query: {q}` | 768 | ollama digest `0a109f422b47`, Ollama 0.32.6 |
| qwen3-embedding:0.6b / 4b / 8b | `{text}` (no prefix) | `Instruct: {task}\nQuery:{q}` (no space after `Query:`) | 1024 / 2560 / 4096, or the MRL value passed as `dimensions` | `task` string, tag + digest (`ac6da0dfba84` / `df5bd2e3c74c` / `64b933495768`), quant Q8_0 / Q4_K_M, pooling=last+EOS, normalized=true |
| embeddinggemma-2 (ST) | `title: {title or "none"} \| text: {text}` | `task: search result \| query: {q}`; A/B `task: question answering \| query: {q}` | 768 (MRL 512/256/128 + renormalise) | HF revision sha `914f7f89…`, dtype=fp32, modalities=text, ST and transformers versions, pooling=mean |

**Reranker serving plan**
- **Local CPU:** Run Qwen3-Reranker-0.6B in-process with **sentence-transformers `CrossEncoder("Qwen/Qwen3-Reranker-0.6B")`** (pin ST ≥ 6.1, which is needed for EG2 anyway). Prefer it over hand-rolled transformers code: it ships the official template, the yes/no IDs, left padding and `logits_to_keep=1`, and I verified the template matches token for token. Settings: fp32, batch 4–8, `max_length` about 1024, `activation_fn=Sigmoid` for 0–1 display, and a fixed instruction stored with the run. Keep a 3-pair transformers reference test in `tests/`. Default to reranking the top 10 (expect about 10–35 s on this laptop), offer 20 as a "slow" option, and show the measured time in the UI.
- **Colab:** Run 4B through the same ST code behind a small FastAPI `/rerank` service on the T4: fp16, SDPA, batch 8–16, with a NaN/inf guard that falls back to local 0.6B and flags the run. Skip 8B: it is no better than 4B on MTEB-R/MMTEB-R and needs 8-bit on a T4. vLLM is optional later, and adds fp16-on-T4 and OOM risk.
- **Do not** use Ollama or the community "Qwen3-Reranker" Ollama models for reranking. Ollama has no rerank API.

## Sources (all accessed 2026-10-14)
- https://huggingface.co/Qwen/Qwen3-Embedding-0.6B (also -4B, -8B: README, config.json, config_sentence_transformers.json, 1_Pooling, tokenizer.json)
- https://huggingface.co/Qwen/Qwen3-Reranker-0.6B (also -4B, -8B: README, chat_template.jinja, 1_LogitScore, config_sentence_transformers.json, commits API)
- https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF and /discussions/17 · https://github.com/QwenLM/Qwen3-Embedding (+ issues 58, 148, 164)
- https://ollama.com/library/qwen3-embedding and /tags · https://registry.ollama.ai/v2/library/qwen3-embedding/manifests/{0.6b,4b,8b}
- https://ollama.com/library/embeddinggemma-2 and /tags · https://registry.ollama.ai/v2/library/embeddinggemma-2/manifests/{270m,440m,latest}
- https://docs.ollama.com/api/embed · https://github.com/ollama/ollama/blob/main/api/types.go · https://github.com/ollama/ollama/blob/main/server/routes.go
- Ollama issues and PRs: 3368, 10989, 12088, 12601, 12921, 13183, 16582, 17878, 18825, 18830, 18884, 18917, 18971, 18973; releases v0.40.0 and v0.40.4
- https://github.com/ggml-org/llama.cpp/issues/14234
- https://huggingface.co/google/embeddinggemma-2 (README, config.json, config_sentence_transformers.json, processor_config.json)
- https://huggingface.co/docs/transformers/main/en/model_doc/embedding_gemma2 · https://github.com/huggingface/transformers/releases/tag/v5.19.0
- https://ai.google.dev/gemma/docs/embeddinggemma · https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2
- https://huggingface.co/ggml-org/embeddinggemma-2-GGUF · https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF
- https://huggingface.co/tomaarsen/Qwen3-Reranker-0.6B-seq-cls · https://ollama.com/dengcao/Qwen3-Reranker-0.6B · https://ollama.com/search?q=qwen3-reranker
- https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/cross_encoder/modules/logit_score.py and releases
- https://docs.vllm.ai/en/stable/examples/pooling/score (via ctx7 `/websites/vllm_ai_en_stable`)
- https://github.com/huggingface/text-embeddings-inference PRs 795, 835, 886; releases
- bitsandbytes docs via ctx7 `/bitsandbytes-foundation/bitsandbytes` (quickstart, Linear8bitLt, benchmarking README)
- https://huggingface.co/datasets/hakari-bench/results/discussions/18 · https://forums.developer.nvidia.com/t/tensorrt-produce-all-zero-output-for-qwen3-embedding-0-6b/337047
- PyPI JSON for transformers, sentence-transformers, torch, ollama
