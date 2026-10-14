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

## Next (M4)

Qwen3-Embedding 0.6B locally, then 8B built on Colab (with a local-vs-Colab
vector match of cosine ≥ 0.999), and EmbeddingGemma 2 through
sentence-transformers (research §0, §4).
