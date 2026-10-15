# 5. Embed

**What it does:** turns each passage, and each question, into a vector: 768
numbers that place its meaning in a space where similar meanings sit close.

## Baseline (M1)

`rag_studio/embed.py` with the `baseline` index profile (`rag_studio/profiles.py`):

| Setting | Value |
|---|---|
| Model | `nomic-embed-text:latest` via Ollama `/api/embed`, F16, digest `0a109f422b47…` |
| Dimensions | 768, L2-normalised |
| Passage prefix | `search_document: ` |
| Question prefix | `search_query: ` |
| Batch | 32; `truncate: false`, so an over-long passage is an error, not a silent cut |

| Measured (2026-10-15) | Value |
|---|---|
| Embedding all 1,190 passages | 406 s (2.9 passages/s) with Ollama otherwise idle |
| Embedding one question (median over a run) | 0.47 s |

## Why profiles never mix

A vector means something only next to vectors made the same way. The index
profile records the model, its **digest**, the dimensions, both prefixes and the
parser + chunker that made the text. A query embedded any other way is refused
with a message that says why:

- the model is not installed → "Install it with: ollama pull …";
- the installed model is a different build (digest changed) → "Rebuild the index";
- the app's profile setup changed since the build → "Rebuild it with ./rag ingest".

`tests/test_index.py` sends each of these and expects the refusal.

## M4: three newer models, two backends

Ladder row 5 embeds the two row-3 indexes (PyMuPDF + structure, Docling +
structure) with three models; nothing else changes. `rag_studio/profiles.py`
holds the six profiles; `./rag index list` shows them.

| | Qwen3-Embedding 0.6B | EmbeddingGemma 2 | Qwen3-Embedding 8B |
|---|---|---|---|
| Runs through | Ollama (`qwen3-embedding:0.6b`, Q8_0, digest `ac6da0dfba84`) | sentence-transformers in this app (`google/embeddinggemma-2` @ `914f7f89`, float32, text tower only: 271M parameters) | Ollama on the Colab GPU (`qwen3-embedding:8b`, Q4_K_M) |
| Dimensions | 1,024 | 768 | 4,096 |
| Passage prefix | none | `title: none \| text: ` | none |
| Question prefix | `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:` (no space after `Query:`) | `task: search result \| query: ` | as 0.6B |
| Other | `options: {num_ctx: 1024}` (recorded) | `prompt=""` so no config prompt is added | as 0.6B |
| Builds | here or on Colab | here | Colab only |
| Questions | here | here | Colab only |

Why not Ollama for EmbeddingGemma 2: its Ollama tags need the MLX runner,
which Linux lacks (research B §2). Our prefix gives the same vector as
sentence-transformers' own `prompt_name="query"` (cosine 1.000000, checked by
`tests/test_embedding_profiles.py`).

| Measured on this CPU (2026-10-15) | Value |
|---|---|
| Qwen3-Embedding 0.6B, real passages | **0.6 passages/s** (32 in 52-58 s): about 34 min (PyMuPDF) / 41 min (Docling) per index |
| Qwen3-Embedding 0.6B in Ollama's memory | 2,261 MB at the default 4,096 context; **1,335 MB** at `num_ctx` 1,024 |
| `num_ctx` 1,024 vs the default | identical (cosine 1.00000000) on a fresh cache in the same order; an earlier 0.99963 was path noise (below) |
| EmbeddingGemma 2, a whole build | 1,214 passages in 896 s (**1.35 passages/s**); 2.2 GB peak RAM; 21.5 s to load |
| Same question vs the right and two wrong passages | Qwen3 0.64 vs 0.13 / 0.22; EmbeddingGemma 0.84 vs 0.57 / 0.55 (only the gap matters) |
| A Spanish question vs the English passage (EmbeddingGemma) | 0.81 vs 0.53 |

**A build that froze this laptop.** The first Qwen3 build loaded the model at
Ollama's default context (2.26 GB) with 2.7 GB free and deep swap; the desktop
froze and the session ended. Local builds now refuse below a measured need
(`profiles.BUILD_RAM_GB`), and loading EmbeddingGemma 2 in the app refuses
below 2.2 GB free, each with the numbers and what to do.

## Row 5 on the scorecard

`docs/ladder.md` has the full row. Sets A+D, Pass · Refused · Wrong, against
row 3's nomic-embed-text (3a 19 · 0 · 6, 3b 20 · 1 · 4):

| | Qwen3 0.6B | EmbeddingGemma 2 | Qwen3 8B |
|---|---|---|---|
| PyMuPDF + structure | 17 · 2 · 6 | 18 · 2 · 5 | 17 · 3 · 5 |
| Docling + structure | 15 · 1 · 9 | **20 · 3 · 2** | 19 · 2 · 4 |

Every new model fixes D9 (Spanish; nomic: rank over 50). Only EmbeddingGemma 2
finds A5 and reaches Set A recall 9/9 on both parsers. The Qwen3 models do not
improve on nomic here, and the 8B is no better than the much smaller
EmbeddingGemma 2. Bigger was not better on these documents.

## Path noise, and the vector match check

Through Ollama a text's vector depends slightly on what the server embedded
just before it (a shared prefix's cached keys and values are reused): the
same text moves to cosine 0.9996 (Qwen3 0.6B) or 0.9985 (the 4-bit 8B on a
T4). The vector match check therefore embeds one text per request and
measures each machine against itself first; this computer and the T4 agree to
0.9995 on Qwen3 0.6B, about as well as each agrees with itself (passed).
Under the 8B's larger noise, 61 of 62 top-5 lists were identical across two
runs. Details: `docs/02-remote-gpu.md`.

## How it works, step 5 (the embedding lab)

`/learn/embed` (`learn.embed_lab`): every passage of an index as a dot (PCA of
its vectors, coloured by document), a question landing among them with lines to
its true five nearest, the exact text the model read (prefix highlighted),
Matryoshka cuts, and a question embedded by another model searched in this
index. Measured on "What are the main risks to the global outlook?":

| | Result |
|---|---|
| The two PCA axes keep | 26.5% of the variance (EmbeddingGemma 2, PyMuPDF) |
| Matryoshka, EmbeddingGemma 2: same top 5 at 768 / 512 / 256 / 128 dimensions | 5 / 4 / 4 / 3 |
| A nomic-embed-text question searched in the EmbeddingGemma 2 index (both 768-d) | **0 of 5** right; every cosine 0.03-0.04 |

The last row is rule 3 made visible: same dimensions, valid arithmetic, and a
meaningless answer, because each model puts meaning in its own directions.
