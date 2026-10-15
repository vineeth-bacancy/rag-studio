# Remote AI: a Colab GPU for the heavy jobs

This laptop has an i5-1135G7, 15.7 GB of RAM and no GPU. Qwen3-Embedding 0.6B
embeds **0.6 passages a second** here (35-40 minutes per index); on a Colab T4
it took 4 minutes, and the 8B 16-18. Google Colab lends a T4 GPU with 15 GB of
memory. This is how RAG Studio borrows one without moving anything else off
the laptop. The design is the trading app's (`~/Dev/trading/docs/17-remote-inference.md`),
adapted: spec §13.

```fish
./rag llm colab --copy     # the setup cell, on the clipboard: paste it into Colab and run it
./rag llm connect          # reads the connect code (or URL) from the clipboard, tests it
./rag index build docling-structure-qwen3-8b --on colab
./rag index check docling-structure-qwen3-0.6b    # the vector match check
./rag llm disconnect       # back to this computer only
```

The same acts are on **Settings › Model backend** (connect, test, the setup
cell) and on **Indexes** (build on Colab, check vectors).

## What crosses the network

The runtime is a computation backend, not a second app. One kind of thing
crosses: **a batch of passage texts (or one question) out, vectors back**. No
file is ever sent: not the PDFs, not an index, not the repository. The runtime
keeps nothing it is sent and is thrown away when Colab reclaims it.

```
THIS COMPUTER                                      COLAB RUNTIME (disposable)
parse, clean, chunk, the index (LanceDB)
search, prompts, answers, citations
the app and its API (127.0.0.1)                    Ollama (127.0.0.1:11434)
            │  https, bearer token                      ▲
            └──▶ trycloudflare.com tunnel ──▶ gateway (server.py, :8000)
```

## The pieces

| Piece | What it does |
|---|---|
| `tools/colab/server.py` | The gateway: standard library only. Serves GET `/api/version`, `/api/tags`, `/api/ps`, `/api/gpu` and POST `/api/chat`, `/api/embed`, `/api/embeddings`; **any other path is a 404 before the token is looked at**. Bearer token compared in constant time, 8 MB body cap, `stream` forced off for chat, no CORS, no Python version in headers, and a log of sizes and counts only (never a passage). `/healthz` is open and says nothing. |
| `tools/colab/start.sh` | Seven checked steps: GPU, `zstd` (Ollama's installer needs it), Ollama **pinned to the laptop's version**, the models (pulled, loaded, and checked to be on the GPU, each by its exact tag), the token, the gateway (its lock **proved on loopback** before any tunnel: 401 without the token, 200 with it), the cloudflared tunnel (401 = locked; 200 = open, so the tunnel is shut and the URL never printed; 000 = not routed yet). |
| `rag_studio/colab.py` | Builds the setup cell from those two files: gzip + base64, each checked against a sha256, written only when every file survived the paste. |
| `rag_studio/remote.py` | The laptop side: the connection, the remembered token, the connect code, connect checks, test, status. The CLI and the API both call it. |

## The token and the connect code

- The connection is kept in `~/.local/state/rag-studio/remote.json`, **mode 0600,
  written atomically**, outside the repository. No API response ever carries
  the token (`tests/test_embedding_profiles.py::test_no_llm_route_ever_returns_the_token`):
  a credential travels browser → API, never API → browser.
- **The browser's setup cell carries no token.** The runtime makes its own and
  shows it once inside a connect code (`rsc1.` + base64url JSON of URL, token,
  models, gateway version, time), in an output the cell blanks when you press
  Enter, because Colab saves cell output to Drive.
- **`./rag llm colab` embeds a token this laptop remembers** (`colab-token`,
  0600). After Colab reclaims the runtime (about 90 minutes idle), re-running
  the same cell and connecting with just the new URL is the whole reconnect.
  The cost: that token sits in the notebook Colab saves. `--new-token` rotates
  it; `./rag llm disconnect --forget-token` drops it.
- The connect code is **not encryption**: whoever holds it holds the token. It
  exists so a URL and a token cannot be half-copied or swapped. Two encoders
  exist (Python and `start.sh`'s bash); `tests/test_remote.py` decodes the bash
  one's output with the Python one.

Connect checks, in order, in one function: decode → https for any
non-loopback host → a token → save → **a real request** (embed one sentence
with every model the runtime serves) → the models the Colab indexes need. A
failed test still saves the connection, so the next step is fixing the
runtime, not re-pasting.

## Embedding on Colab

**Where each index embeds is set by its profile** (`EmbeddingSpec.build_on`,
`query_on`), and it is policy, not configuration: the same model and tag give
the same vectors on either machine (the vector match check proves it), so it is
not in the recorded config. The build's manifest records where it ran
(`embedding.built_on`, Ollama version, GPU).

| Model | Builds | Questions |
|---|---|---|
| nomic-embed-text, EmbeddingGemma 2 | this computer | this computer |
| Qwen3-Embedding 0.6B | here or on Colab | this computer |
| Qwen3-Embedding 8B | **Colab only** | **Colab only** (the user's choice, 2026-10-15: no 4.7 GB local pull) |

- **Strict, never silent.** An index whose questions are embedded on Colab
  refuses when no runtime is connected, or when the runtime's model is a
  different build (digest), with a message that says what to do. Nothing falls
  back to another model or machine.
- **Builds resume.** New vectors go through the vector cache, saved every 256
  passages, one file per (model digest, prefix, dimensions, options, machine).
  A build cut off when Colab reclaims the runtime starts again where it
  stopped. Colab vectors never feed a local build, or the other way round.
- **Options travel with the model.** The Qwen3 profiles send
  `options: {num_ctx: 1024}` with every embed call, here and on Colab, and
  record it. It saves memory (2.26 → 1.34 GB for the 0.6B); compared fairly it
  does not change a vector.
- **The same Ollama on both sides.** The setup cell pins the runtime's Ollama
  to the laptop's version (0.32.6), so a vector difference is the hardware's
  alone.
- Cloudflare ends any request after 100 s (HTTP 524). An embedding batch of 32
  on a T4 takes a few seconds; streaming answers through the tunnel is M6's
  problem.

## The vector match check

Spec §2.2: embed 50 passages both ways and require cosine ≥ 0.999.
`./rag index check <profile>` (or **Check vectors** on the Indexes page).

**First, what one machine does.** Measured on the T4 and here: through Ollama
0.32.6 **a text's vector depends on what the server embedded just before it**.
The server reuses the cached keys and values of a prefix shared with the
previous request (our passages share "Document: … Section: …"), computed in a
differently shaped pass, so the arithmetic order differs:

| Same texts, same model, on the T4 | Qwen3 8B (4-bit) | Qwen3 0.6B |
|---|---|---|
| one per request, same order, twice | 1.000000 | 1.000000 |
| one per request, shuffled order | 0.99868 | 0.99959 |
| a batch of 32, sent twice | 0.99857 | 0.99965 |
| 4 single requests in parallel | 0.99946 | 0.99956 |

So the check embeds **one text per request** and measures each machine's
**noise floor** (20 passages in two orders) before comparing machines:

- **across machines** (the spec's check): 50 passages spread through the
  build and 5 test questions, in the same order here and on the runtime.
  **Passes at ≥ 0.999.**
- **against the index**: the same passages against the vectors stored at
  build time (embedded in batches of 32). Reported, not judged.

With one machine (the 8B: not installed here by choice) there is nothing to
compare across: no verdict, the noise floor is shown.

| Measured 2026-10-15 | This computer vs Colab (passages / questions) | Noise floor here / Colab | vs the index here / Colab |
|---|---|---|---|
| PyMuPDF + structure, Qwen3 0.6B | **0.99954 / 0.99963: passed** | 1.00000 / 0.99967 | 0.99954 / 0.99966 |
| Docling + structure, Qwen3 0.6B | **0.99953 / 0.99963: passed** | 0.99972 / 0.99971 | 0.99954 / 0.99960 |
| PyMuPDF + structure, Qwen3 8B | Colab only | - / 0.99888 | - / 0.99834 |
| Docling + structure, Qwen3 8B | Colab only | - / 0.99899 | - / 0.99824 |

EmbeddingGemma 2, which runs in this app (sentence-transformers) rather than
through Ollama, is **exactly reproducible**: noise floor 1.000000 and identical
to its stored vectors on both indexes.

**This computer and the T4 agree about as well as either agrees with itself**,
so a Qwen3 0.6B index built on the T4 is searched from here. The 8B's own
noise sits below 0.999; whether that matters was measured too: a
retrieval-only rerun of both 8B indexes gave the **same top 5 for 61 of 62
questions**, and the one change swapped two answer pages 0.0004 apart. No
answer page moved.

## How fast, and what limits it

| Measured 2026-10-15 (T4, the free tunnel, laptop in India) | Value |
|---|---|
| Qwen3 8B: a build | 1,214 passages in 931 s; 1,473 in 1,069 s (about 1.3/s) |
| Qwen3 0.6B: a build | 1,214 in 227 s; 1,473 in 258 s (about 5.3/s; here on the CPU: 0.6/s, 35-40 min) |
| 8B, 32 passages: on the GPU / the round trip | 8.4 s / 29.7 s, while a build shared the runtime |
| What the extra time is | moving vectors back as JSON: 32 × 4,096 floats is 1.7 MB; about 2.5 s + 11 s per MB through the tunnel |
| A question during a build / when idle | 1-10 s / about 0.5-1 s |

The GPU is not the bottleneck; the tunnel is. Since then the gateway gzips
replies over 16 KB for a client that accepts it (httpx does); JSON floats
compress about 3×. **Not measured yet on a runtime**: the session above ran
the gateway before the change.

## Failures, and what each means

| Seen | Meaning | Do |
|---|---|---|
| "The Colab runtime is not answering" | the runtime is gone (reclaimed) or the tunnel is down | run the same cell again, connect with what it shows |
| "refused the token" | a runtime that made its own token issues a new one each time | paste its new connect code |
| "The tunnel answered … with an error page" | Cloudflare's own error: nothing behind the tunnel | as above |
| "does not serve qwen3-embedding:8b" | the cell was run with other models | run the default cell |
| "different build" | the runtime's model digest differs from the index's | rebuild the index on Colab |
| GPU "None" / a model "on the CPU" | Colab handed out a runtime without a GPU | Runtime › Change runtime type › T4 GPU, run the cell again |

## Not here yet, on purpose

- The rerank service on the runtime (`POST /rerank`, same token and cap): M5.
- Answer, vision and judge models on the GPU, and an optional loud local
  fallback for them: M6. M4's runtime serves embeddings.
- EmbeddingGemma 2 on Colab: it runs in this app (sentence-transformers); its
  image path arrives with figures in M7.
