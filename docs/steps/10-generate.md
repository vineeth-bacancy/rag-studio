# 10. Generate and cite

**What it does:** the model writes the answer from the passages; code turns
each `[n]` into a document and a page.

## Baseline (M1)

`rag_studio/answer.py`, one answering path for the Ask page, `./rag eval` and
the Evaluation page.

| Setting | Value |
|---|---|
| Model | `qwen3:1.7b` (the Answer role) |
| Options | `think: false` (always sent), temperature 0, `num_ctx` 8192 |
| Streaming | Ollama `/api/chat` stream → server-sent events: `stage` per finished step, `ready` (sources and prompt budget), `delta` (text), `done` or `error` |

**Citations are built by code.** The model only writes `[n]`; the engine maps
n to the passage it was shown, and so to the document and 1-based page. A
number outside 1..k is reported as an invalid citation, never dropped or
"fixed".

**Refusal is strict:** the whole answer must be exactly "I could not find this
in the documents." A wrong fact followed by that sentence is an answer, not a
refusal (spec §5.7).

| Measured, Set A on qwen3:1.7b (2026-10-15) | Value |
|---|---|
| Time to answer, median | 23 s (21-38 s) |
| Prompt tokens | 1,258-1,815 |

## The stream through Next

The API sends `Cache-Control: no-cache, no-transform` and `X-Accel-Buffering:
no`, plus a `: ping` every 15 s of silence (`web/api/sse.py`). FastAPI 0.143's
built-in `EventSourceResponse` pings too, but it sets `Cache-Control: no-cache`
itself and a route can only append a second header. `no-transform` is what
stops Next's rewrite proxy gzipping the stream into one late burst. Measured
in the browser: the sources arrive within half a second, before the model
starts.

## Next (M6)

New answer models (qwen3.5:4b, gemma4:e4b), measured on the same scorecard.
