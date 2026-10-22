# 8. Query rewriting and multi-step search

**What it does:** asks the question in other words before searching, splits a
two-part question, turns a follow-up into a full question, or lets a model look
at what was found and decide to search again. All of it runs on **Remote AI
only** (qwen3:8b on the Colab runtime, the `rewrite` role; decided
2026-10-16): qwen3:8b takes minutes per call on this CPU, and a silent local
fallback would make a run's recorded setup untrue. When Remote AI is down these switches refuse with how to
connect.

## The techniques (`rag_studio/rewrite.py`)

Each is a versioned prompt with a JSON-schema answer (Ollama `format`), at
temperature 0 and one `num_ctx` (8,192: another value makes Ollama reload the
model). The question and the conversation are quoted data, never instructions.
Output that does not fit the schema is an error, never passed through.

| Switch (UI name) | Technical | Prompt | What is searched |
|---|---|---|---|
| Split into parts | decomposition | `decompose-v1` | the question, then each of up to 3 sub-questions; the lists are taken in turn (round robin), or the pool is reranked against the question when a reranker is on |
| Several phrasings | multi-query | `multi-v1` | the question and 3 rewordings, fused by RRF (K = 60) |
| Draft answer first | HyDE | `hyde-v1` | the question, and a drafted passage of at most about 100 words **embedded as a document** (the index's own model and document prefix, so rule 3 holds), fused by RRF |
| (follow-up) | standalone rewrite | `standalone-v1` | the follow-up rewritten with the conversation so far; "Ask as typed" skips it |

While Remote AI writes a rewrite, the original question is searched here, so
the tunnel's wait is not added to the search.

## Multi-step search (`rag_studio/agent.py`)

Round 1 is always the normal search. After it, the controller (the same
`rewrite` role, prompt `agent-v1`) sees the evidence (for 6 passages: id,
document, page, section, the first 300 characters, fenced as quoted data) and
the searches already tried, and answers with `{action: search | answer, query,
doc, reason}`.

- **The document filter** is a document id checked against the index's list
  and passed to LanceDB as a prefilter (`store.doc_filter`). Free text never
  reaches a `where` string.
- **It stops**, and records why: enough evidence, the round limit (3), the
  time budget (90 s), the token budget (8,000), a query it already tried, or
  output that is not a valid decision.
- **Answering:** every round's passages are pooled by RRF, reranked against
  the original question when a reranker is on, cut to the top 5, and answered
  by the local answer model as always.

## Conversations and memory (`rag_studio/chat.py`)

A conversation is one JSON file in `data/chat/` (a 32-hex id, so a request can
never name a path; atomic writes). **Memory acts only through the follow-up
rewrite**: the answer model sees the standalone question and the passages
under the same prompt v3, so an answer in a conversation is comparable with a
scorecard answer. Ask shows "Searched for: …" under a rewritten follow-up, with
**Ask as typed**. With Remote AI down a follow-up fails with "Follow-up
questions are rewritten by Remote AI, which is not connected. Connect it in
Settings, or ask the question in full." and offers Ask as typed.

## Remembered outputs (`rag_studio/remote_cache.py`)

Through Ollama a model's output depends on what it served just before (prefix
reuse; CLAUDE.md, M4). So every Remote AI output is cached by (model digest,
prompt hash, num_ctx, input) with the order it was generated in. A
retrieval-only run and the scorecard after it see the **same** rewrites and
decisions, and a scorecard replays them with Colab off. Each question records
hit or miss.

## Set E: follow-up questions

`data/golden/set_e.json`: 8 two-turn cases (a recorded first turn, then the
follow-up), each with the words the standalone question must contain
(`standalone_expect`), and for the topic switch the words it must not
(`standalone_forbid`), the answer pages (every page that states it, checked in
the page text) and the expected facts. Scored as a chain: rewrite right → Hit
→ Pass. It is remote-only, so `./rag eval` without `--set` leaves it out.

## Measured (2026-10-22, qwen3:8b Q4_K_M on a Colab T4)

Full tables: `docs/ladder.md`, rows 8 and 9 and Set E.

- **Rewrites:** 101 calls (93 for Sets A, B and D across the three techniques,
  8 Set E follow-ups) in 305 s, 1.2-4.7 s each through the tunnel; none failed
  its schema.
- **Row 8:** split into parts and draft answer first each add one Pass on
  PyMuPDF (D10, the two-document question: 20 · 2 · 3 → 21 · 1 · 3); several
  phrasings changes nothing. On Docling with the 4B reranker no technique
  changes an answer (all 22 · 1 · 2, as 6d).
- **Set E (memory):** rewritten 7 · 0 · 1 with 7/8 hits, against 5 · 1 · 2
  and 4/8 asked as typed. 6 of 8 rewrites held every expected word; "the old
  system" (E4) was not resolved, and "the year before" (E6) stayed relative.
- **Multi-step search (row 9's decisions), per base of 31 questions:**

  | | PyMuPDF (8a base) | Docling (6d base) |
  |---|---|---|
  | Stopped: enough evidence / bad output | 28 / 3 | 28 / 3 |
  | Searched a second time | 7 | 4 |
  | Controller calls per question | 1.2 | 1.1 |
  | Latency per question, p50 / p95 (live) | 4.1 s / 9.1 s | 4.0 s / 9.4 s |
  | Tokens per question (median) | 986 | 1,007 |
  | Searches limited to one document | 1 | 0 |

  Every bad output was the same: `action: "search"` with no usable query (the
  schema does not require `query` when the action is search). A loop never
  ran past its round limit, its time or its token budget. Asked again, one of
  four bad decisions came back valid: qwen3:8b through Ollama is not
  deterministic across sessions, which is why every decision is remembered,
  bad ones included.
- **Row 9's scorecards:** 9a (PyMuPDF 8a + multi-step) 21 · 1 · 3 and 9b
  (Docling 6d + multi-step) 22 · 1 · 2: **not one answer changed** against
  the rows they build on, nor did retrieval. With the 4B reranker on top, the
  first search already holds the answer pages here.
- **Obvious next step, not taken:** an `agent-v2` prompt whose schema requires
  `query` when `action` is `search`. It changes the prompt hash, so it would
  be a new measurement.
