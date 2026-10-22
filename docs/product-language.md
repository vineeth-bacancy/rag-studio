# Product language

**Status:** normative. Read this before writing any string a user reads: a label,
a heading, an error, an empty state, a tooltip, a CLI line, a status word, or
text the app asks a model to write. Adapted from the trading app's
specification (`~/Dev/trading/docs/product-language.md`), cut down to what a RAG
learning tool needs.

The goal: **make RAG easier to understand without making it less precise.**
RAG Studio is a learning tool and a research tool. Technical terms that carry
meaning (chunk, embedding, BM25, recall@5) stay; implementation details that
carry none (ports, file paths, Python names) move one layer down.

## 1. Principles

1. **One concept, one name, everywhere.** The sidebar, the page heading, the
   browser tab, ⌘K, the docs and the CLI use the same word. Page names live in
   one file, `web/ui/src/lib/routes.ts`, and a test fails if a page spells its
   own name.
2. **Plain language first, then the precise term, then the implementation.**
   Three layers:
   - *Primary:* what the user needs to understand at a glance.
   - *Explanation:* one or two sentences on what it means.
   - *Technical details:* the model tag, endpoint, file path, parameter. Behind
     a "▸ Technical details" disclosure, on System Health, or in `docs/`.
3. **Never hide a caveat to shorten a sentence.** "This figure was described by
   a model" is a caveat; it stays.
4. **Never invent.** A missing value shows as `-`, never as `0` and never as a
   guess. An empty result that is valid ("no passage states this") must not
   read like an error.

## 2. Canonical vocabulary

| Concept | Primary term | Technical term (allowed in explanations, details, docs) | Avoid |
|---|---|---|---|
| The app | **RAG Studio** | | RAG app, the studio |
| A piece of a document that is indexed and retrieved | **passage** | chunk (the step that makes passages is **Chunk**) | snippet, segment, node, document (for a piece) |
| A cited passage under an answer | **source** | | reference, context item |
| The PDF page a citation points to | **page N** (the 1-based PDF page) | printed page label ("printed 74") | page index, page 0 |
| A whole PDF | **document** | `doc_id` | file (in primary UI), report (except in prose) |
| Turning text into numbers | **embedding** (noun), **embed** (verb) | vector, dimensions | encoding |
| The searchable store of embedded passages | **index** | index profile, LanceDB table | vector DB, collection |
| One embedding setup bound to one index | **index profile** | manifest | config, variant |
| Search by meaning | **dense search** ("by meaning") | cosine similarity | semantic search (ambiguous), vector search |
| Search by words | **BM25** ("by words") | full-text index | keyword search |
| Both, fused | **hybrid search** | RRF (reciprocal rank fusion) | combined search |
| The full-text index BM25 searches | **word index** | full-text index (FTS), LanceDB's native inverted index | keyword index, text index |
| Moving extra passages of one section behind the others | **limit per section** | section cap | dedupe, diversity filter |
| Re-scoring the top results with a slower model | **reranking** | cross-encoder | re-sorting |
| A model's sentence placing a passage in its document | **context line** | contextual retrieval | summary, header |
| A thread of questions and answers kept on this computer | **conversation** (New conversation) | chat (`data/chat/`) | session, thread (in primary UI) |
| A question that leans on an earlier turn ("And for 2027?") | **follow-up question** | standalone rewrite | context query |
| What a follow-up was searched as, after Remote AI made it whole | **Searched for: …** (with **Ask as typed**) | standalone question | reformulated query |
| Asking the question in other words before searching | **query rewriting**: **Split into parts**, **Several phrasings**, **Draft answer first** | decomposition, multi-query, HyDE | query expansion (in primary UI) |
| Searching again after looking at what was found | **multi-step search**, in **rounds** ("Round 2") | agentic loop, controller | agent, autonomous search |
| A passage's section ("Ch. 1 > Box 1.1") | **section** | section path | breadcrumb, heading path |
| The document and section written in front of a passage before it is embedded, indexed and shown to the model | **section label** | contextual header (spec §5.2) | header, breadcrumb; not a context line (M5's model-written sentence) |
| The questions with verified answers | **test questions** | golden set | benchmark, dataset |
| One full scoring of a setup | **run** | evaluation run | experiment, job |
| The table of outcomes | **scorecard** | | results matrix |
| The step-by-step upgrade comparison | **the ladder** | | waterfall |
| A job the app gives a model | **role** (Answer, Embedding, Query rewriting, Context lines, Judge, Vision, Reranker) | | task, agent |
| The exact refusal | "I could not find this in the documents." | | "Not found", "No answer" |
| The rented GPU machine | **Colab runtime** (primary UI: **Remote AI**) | the gateway, the tunnel | server, instance, backend (in primary UI) |
| The one cell pasted into Colab | **setup cell** | bootstrap | script, notebook |
| The string that carries a runtime's URL and token | **connect code** | `rsc1.` prefix | handoff code, token string |
| Re-embedding stored passages on each computer and comparing | **vector match check** | cosine ≥ 0.999 | parity test |
| Making an index | **build** (Build here, Build on Colab) | ingest | index (as a verb), train |
| Cutting vectors shorter | **Matryoshka** (shorter vectors) | MRL, `dimensions` | truncation (alone) |

### Answer outcomes (scorecard)

| Word | Meaning | Tone |
|---|---|---|
| **Pass** | Every expected fact is present, or a trap question was refused | Healthy colour, check icon |
| **Refused** | An answerable question got the refusal. A safe failure | Neutral, never red |
| **Wrong** | A fact is missing or wrong, or a trap was answered. The dangerous failure | Red, with an icon |

Retrieval is reported beside the outcome as **Hit@N** (the best answer page was
at rank N) or **Miss**. The stored values stay `PASS` / `REFUSED` / `WRONG`.

### Model-generated content is always labelled

- "Figure (described by qwen3.5:4b)" for a vision description.
- "Figure (image match)" for an image-embedding hit.
- "Context line (written by qwen3:1.7b)" in the inspector.
- Query rewrites are shown as rewrites: "Searched for: …".

## 3. Status vocabulary

One set of words for every status in the browser:

| Word | Use |
|---|---|
| **Healthy** | Working as intended |
| **Running** | Active right now (a model loaded, a build in progress) |
| **Warning** | Works, but something will be slower or less accurate |
| **Needs attention** | Something must be fixed for a feature to work |
| **Unavailable** | Could not be reached or measured |
| **Info** | A valid state worth knowing ("Remote AI is not connected") |

Colour always comes with an icon and a word. Green means Healthy (and Pass) and
nothing else; it is never a chart series colour.

## 4. AI wording

| Situation | Primary | Technical details |
|---|---|---|
| Ollama down | **AI unavailable.** | Ollama is not running at 127.0.0.1:11434 |
| API down | **App service unavailable.** Start it with `./rag web`. | 127.0.0.1:8775 |
| Another app's model loaded | **AI busy.** Answers will be slower until it unloads. | qwen3:8b, unloads in 4 min |
| Which machine | **Local AI** / **Remote AI** | Backend: Ollama · Endpoint · Model |
| A role's model missing | **Not installed.** | `ollama pull <tag>` as a copyable fix |

Never show "ollama offline", "No usable model", "Python API" or a stack trace
in primary UI.

## 5. Patterns

- **Empty states have three parts:** what is missing, why it matters (only if
  not obvious), one action. "No search index yet. Build one to start asking
  questions. [Open Indexes]"
- **Errors:** "Short explanation. Action to recover." Never "Something went wrong".
- **Buttons:** short verbs. Ask, Build index, Run evaluation, Compare, Check
  again, Copy fix, Cancel.
- **Confirmations** answer: what will happen, is it reversible, what changes.
- **Commands in the browser** only when the user must run one, and always the
  public `./rag …` form, never a script path.
- **Three-sentence rule** for explanatory copy: what it is; why it matters;
  optionally what to do.
- **Numbers:** tabular figures; ranks as "rank 11"; scores to 3 decimals only in
  technical views (rank matters more than score: §6 of the build spec).

## 6. Terms that must never be simplified away

chunk, embedding, vector, dimensions, dense, BM25, hybrid, RRF, reranking,
cross-encoder, context window, token, prompt, temperature, quantization,
Matryoshka, recall@k, Hit@N, faithfulness, answer relevance, context precision,
context recall, HyDE, judge. These are what the user is here to learn. Explain
them; don't replace them.

## 7. The final test for any new string

1. Is it necessary? 2. Is it the canonical name? 3. Is it implementation detail
(move it down a layer)? 4. Is it understandable at once? 5. Does it say one
thing? 6. Does it keep its caveat? 7. Does another page say this differently
(fix both)? 8. Does the action say what to do?
