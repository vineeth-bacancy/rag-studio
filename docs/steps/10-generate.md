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

## How it works 10 (M6)

`learn.cite_lab(run, question)`, `/learn/cite`, on a recorded answer with its
passages read back from the run's own index build (as step 9):

- **From [n] to a page:** `answer.citation_marks(text, k)` finds every `[n]`
  group and its position; `answer.citations` maps each valid number to its
  passage, document and page. An invalid number is drawn in red and listed.
  Mapping the recorded text again gives the result the run saved (checked:
  `matches_recorded`), because it is code.
- **The refusal:** the strict rule, with whether the recorded answer is the
  exact refusal, contains it and goes on (an answer), or neither.
- **Change the answer:** an editable copy read by `answer_check` (POST
  `/learn/answer-check`): citations against the passages that run showed, the
  refusal rule, and `score_answer` by the key the run was scored with. Presets:
  cite a passage that was not shown, add a fact before the refusal, refuse.
- **How the model wrote it:** settings and the recorded timings (reading the
  prompt, first word, writing rate).
- **The same passages, other answer models:** every run on the same setup
  (index, embedding, k, retrieval, prompt) with another answer model: ladder
  row 10 question by question.

## New answer models (M6, ladder row 10)

| Model | Thinking off | Sets A+D | Set E | Answer, median | Peak memory |
|---|---|---|---|---|---|
| qwen3:1.7b (6d) | works | 22 · 1 · 2 | 7 · 0 · 1 | 17.2 s | - |
| qwen3.5:4b | works (0 thinking characters) | 22 · 1 · 2 | 7 · 0 · 1 | 43.7 s | ~6.7 GB |
| gemma4:e4b | works; system prompt verified by behaviour | **24 · 1 · 0** | **8 · 0 · 0** | 47.3 s | 8.0 GB |

`think: false` is always sent (`llm.chat_stream`). On this CPU qwen3.5:4b
reads a 1,339-token prompt at about 25 tokens/s and writes at 5.8;
gemma4:e4b reads at about 32 and writes at 7.5, after a 29 s load. Details and
what the Wrongs were: `docs/ladder.md`, row 10. The default answer model is
the user's call; it stays qwen3:1.7b until then.
