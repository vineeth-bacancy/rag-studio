<div align="center">

# RAG Studio

**Ask your PDF reports a question and get an answer that cites the exact page. Then open any step of the pipeline and see how it was produced.**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=nextdotjs&logoColor=white)](web/ui/package.json)
[![FastAPI](https://img.shields.io/badge/FastAPI-routes%20only-009688?logo=fastapi&logoColor=white)](web/api)
[![Ollama](https://img.shields.io/badge/Ollama-gemma4%3Ae4b-white?logo=ollama&logoColor=black)](docs/steps/10-generate.md)
[![LanceDB](https://img.shields.io/badge/LanceDB-0.40-CF3CFF)](docs/steps/06-store.md)
[![Tests](https://img.shields.io/badge/tests-574-brightgreen)](tests)
[![Loopback only](https://img.shields.io/badge/network-127.0.0.1%20only-blue)](#security-and-privacy)

</div>

![The Ask page: a question about South Asia's poverty projection, answered "approximately 18.9%" with numbered citations that open the cited PDF page](docs/assets/ask-cited-answer.webp)

---

Most RAG demos make claims and leave you to trust them. This one measures them.
It started from a RAG pipeline written by hand: PyMuPDF text, fixed
1,000-character chunks, `nomic-embed-text`, dense search and `qwen3:1.7b`. That
pipeline is rebuilt inside RAG Studio character for character and scored as
the **"before"** row. After that, each stage was upgraded one change at a time
and scored on the same test questions. **No upgrade was assumed to help.** Two
of the upgrades that every RAG tutorial recommends made the answers *worse*
here, and the scorecard caught both.

In the answer above, the 18.9% is not written anywhere in the report's text. It
is the height of a bar in a chart. A vision model described the chart while the
index was being built, the description became a searchable passage, and the
answer cites it. The model only picked the number `[1]`. Code turned `[1]` into
*Global Economic Prospects, page 97*, and clicking it opens that page with the
passage outlined.

Everything runs on this laptop at `127.0.0.1`. The heavier jobs can optionally
run on a Colab GPU that you start yourself; that is the only thing that ever
leaves the machine.

<table>
<tr>
<td align="center"><b>3</b><br>PDF reports<br><sub>294 pages</sub></td>
<td align="center"><b>1,710</b><br>passages<br><sub>1,473 text + 237 figures</sub></td>
<td align="center"><b>13</b><br>RAG steps<br><sub>each with an interactive page</sub></td>
<td align="center"><b>16</b><br>index profiles<br><sub>one per measured setup</sub></td>
<td align="center"><b>43</b><br>test questions<br><sub>5 sets, traps included</sub></td>
<td align="center"><b>11</b><br>ladder rows<br><sub>baseline → figures</sub></td>
</tr>
</table>

## Contents

- [What it does](#what-it-does)
  - [One question, end to end](#one-question-end-to-end)
  - [The model picks a number; code writes the citation](#the-model-picks-a-number-code-writes-the-citation)
- [What was measured](#what-was-measured) ← the main result
  - [The ladder](#the-ladder)
  - [What the measurements overturned](#what-the-measurements-overturned)
  - [How sure is any of this?](#how-sure-is-any-of-this)
- [How it works: thirteen steps you can open](#how-it-works-thirteen-steps-you-can-open)
- [Architecture](#architecture)
- [Security and privacy](#security-and-privacy)
- [Remote AI on a Colab GPU (optional)](#remote-ai-on-a-colab-gpu-optional)
- [Getting started](#getting-started)
- [Command reference](#command-reference)
- [Tests](#tests)
- [Project layout](#project-layout)
- [Documentation](#documentation)
- [Limitations](#limitations)

---

# What it does

RAG Studio is two things that share one engine.

**Ask** is the product. You type a question, the app retrieves the five
passages most likely to contain the answer, a local model writes at most three
sentences from them, and every fact carries a numbered citation that opens the
PDF page. If no passage states the answer, the model must reply with exactly one
sentence, *"I could not find this in the documents."*, and nothing else.

**How it works** is the course. Every RAG step has its own interactive page,
and each page runs on *your* documents: cut a page into passages, run three
searches side by side, read the exact prompt a recorded answer was given, and
watch the injection guard drop hidden text. Ask, How it works and the
scorecard all call the same engine functions, so a number cannot be right in
one place and wrong in another.

The library is three real reports that are awkward to read in useful ways:

| Document | Pages | Passages | What makes it hard |
|---|---:|---:|---|
| Apple *iMac Product Environmental Report* | 12 | 57 | infographic layout, a transparent overlay, a carbon chart whose sizes are never printed |
| U.S. GPO *2018 Annual Report* | 82 | 289 | hidden text (under fills, off the page), letter-spaced titles, two page-number series |
| World Bank *Global Economic Prospects, June 2026* | 200 | 1,364 | a font that drops the "h" from "Th" 325 times, 710 line-end hyphens, ~57 charts drawn as images with no text |

## One question, end to end

Test question C2, answered by the default setup. These are the recorded
values from the run, not estimates:

| Step | What happened | Time |
|---|---|---:|
| Question | *"In the World Bank's outlook figure for South Asia, what share of the region's population is projected to live in poverty in 2028?"* | |
| Embed | EmbeddingGemma 2 turns the question into 768 numbers | 114 ms |
| Search | exact cosine scan of 1,710 vectors; rank 1 is a **figure passage** (score 0.806) | 8 ms |
| Prompt | rules (v3) + 5 fenced, numbered passages + the question: 1,612 tokens | |
| Read prompt | `gemma4:e4b` reads it on the CPU (~34.5 tokens/s) | 46.7 s |
| Write answer | 78 tokens (~11.3 tokens/s) | 6.9 s |
| Cite | code finds `[1] [1] [3]`; all valid, both map to page 97 | |
| Score | the key accepts 18.6–19.2 (measured from the PDF's drawing: 18.93) | **Pass** |

About 86% of the 54 seconds goes on the model *reading the prompt*. Search takes
milliseconds. That single fact explains most of the design: every extra passage
and every extra line of system prompt costs real seconds on this CPU, and a GPU
would change the economics more than any search technique could.

## The model picks a number; code writes the citation

The model is never asked for a document name or a page. It writes `[n]`.
After generation, a regular expression finds every `[n]`, `[n][m]` and
`[n, m]`, and each valid `n` is mapped to the n-th passage the model was shown:

```python
valid   = sorted({n for n in nums if 1 <= n <= len(sources)})
invalid = sorted({n for n in nums if not 1 <= n <= len(sources)})
```

An out-of-range citation such as `[7]` from five passages is **shown in red
rather than dropped or "fixed"**. A model that invents a source should look like
one. The refusal check is equally strict: `text.strip() == REFUSAL`. A wrong
fact followed by "I could not find this…" counts as an answer, and it is scored
**Wrong**.

The system prompt is versioned to the byte and hashed, and every answer records
`messages_sha256`. How it works 9 can therefore rebuild a recorded prompt and
prove, by hash, that it is *byte for byte what the model read*. This matters
more than it sounds: on `qwen3:1.7b` at temperature 0, **one blank line** in the
system prompt moved two of eleven Set A outcomes.

---

# What was measured

**Scorecards read Pass · Refused · Wrong.** Pass means every expected fact is
present, or a trap question was refused. Refused is the safe failure. Wrong
is the dangerous one, and it is counted separately on purpose. **Recall@5**
asks whether a passage from an answer page reached the top five. Each run also
searches 50 deep so that MRR can show near-misses.

The test questions are split into five sets:

| Set | Questions | What it tests |
|---|---:|---|
| **A** | 11 | the main scorecard: nine questions written by hand, kept verbatim (a test checks), plus two traps |
| **B** | 6 | retrieval only: single facts stated on several pages |
| **C** | 4 | answers that exist *only* as a bar or segment in a chart; keys measured from the PDF's vector drawing |
| **D** | 14 | edge cases: near-miss numbers, table cells, arithmetic, typos, Spanish, two documents at once, a false premise, injected instructions |
| **E** | 8 | follow-up questions ("And for 2027?"), judged on the rewrite, the retrieval and the answer |

## The ladder

Each row changes **one thing** from the row it builds on. Answers came from
`qwen3:1.7b` up to row 9. A+D means Sets A and D together, 25 scored questions.

| Row | One change | Recall@5 A · B · D | A+D scorecard | Verdict |
|---|---|:---:|:---:|---|
| **1** | **Baseline**: PyMuPDF, fixed 1,000/150, nomic, dense, prompt v2 | 8/9 · 5/6 · 9/11 | 18 · 2 · 5 | the "before" |
| 2a | + full cleaning and the hidden-text guard | 9/9 · 6/6 · 10/11 | 15 · 1 · 9 | retrieval up, **answers down** |
| 3b | Docling + structure-aware passages with section labels | 8/9 · 6/6 · 10/11 | 20 · 1 · 4 | first setup to beat row 1 on every count |
| 4b | + hybrid search (RRF) | 8/9 · 6/6 · 10/11 | 19 · 2 · 4 | no help on labelled passages: off |
| 5d | EmbeddingGemma 2 instead of nomic | 9/9 · 5/6 · 10/11 | 20 · 3 · 2 | default from 16 Oct |
| 5f | Qwen3-Embedding **8B** instead (Colab) | 8/9 · 6/6 · 10/11 | 19 · 2 · 4 | no better than the 271M model |
| 6d | + Qwen3-Reranker 4B, top 20 → 5 (Colab) | 9/9 · 6/6 · **11/11** | 22 · 1 · 2 | best retrieval row |
| 7b | + context lines written by qwen3:8b | 9/9 · 6/6 · 10/11 | 21 · 2 · 2 | "rank better and answer no better" |
| 8–9 | + query rewriting / multi-step search | 9/9 · 6/6 · 11/11 | 22 · 1 · 2 | "not one answer changed" |
| 10b | 6d, answered by `gemma4:e4b` | 9/9 · 6/6 · 11/11 | **24 · 1 · 0** | first model with no Wrong answer |
| **11a** | **5d + 237 figure passages, `gemma4:e4b`** | 9/9 · – · 10/11 | 21 · 1 · 3 | **the default**; Set C 0 · 3 · 1 → **4 · 0 · 0** |

Every row and variant, with its run ids, is in
[docs/ladder.md](docs/ladder.md), and the Evaluation page draws the ladder from
the tagged runs:

![The ladder on the Evaluation page: each row's change, recall per set, Pass/Refused/Wrong scorecards and MRR](docs/assets/ladder.webp)

Read as a story:

- **Cleaner text alone made answers worse** (row 2). Retrieval improved on
  every set, but joining the lines of a flattened table produced false sentences:
  *"iMac (four ports) 512GB 391 kg CO2e"* read as one claim and gave the answer
  391 instead of 346.
- **Structure helped** (row 3). Whole paragraphs and tables, each with a
  *Document / Section* label, lifted the executive-summary question from rank 45
  to rank 1, because only the label says "Executive Summary".
- **The embedding model mattered and its size did not** (row 5). EmbeddingGemma 2,
  with 271M parameters, beat Qwen3-Embedding 8B.
- **A cross-encoder reranker was the biggest retrieval win** (row 6). It moved
  question B4 from rank 8 to rank 1, but it needs a GPU here: the 0.6B takes 39 s
  a question on this CPU.
- **LLM-driven retrieval added nothing on three documents** (rows 7–9). Context
  lines, three kinds of query rewriting and multi-step search were all built,
  cached, measured, and left switched off.
- **The answer model decided the Wrongs** (row 10), and **figures were a new
  capability**, not an improvement on the old questions (row 11).

The default is row 11a rather than the higher-scoring 10b because the 4B
reranker needs a Colab session. Without one, the app still has to answer.

## What the measurements overturned

Some hypotheses were overturned by measurement while this was being built.
Each one was corrected in the code and the docs:

| Expected | Measured |
|---|---|
| Docling won't fix the broken "Th" glyph | it does: it reads the font's `T_h` glyph name itself |
| Docling converts 0.6–1.3 pages/s (research) | **0.34 pages/s**, 866 s and 6.4 GB peak for 294 pages |
| Hidden text appears on one known page | the pixel guard found it on **9** pages |
| The documents contain 127 real hyphenated compounds | an independent oracle found **140**; all are kept |
| `num_ctx` changes Ollama's embedding vectors | it doesn't. Ollama reuses cached prefix keys, so **a text's vector depends on what was embedded before it** (cosine 0.9996). Compare one text per request |
| Paint-order rules catch hidden text | they flagged a transparent overlay and white table headers that are both visible; dropping them would have deleted real answers |

## How sure is any of this?

Not very, and the app says so. Each scorecard reports a Wilson 95% interval,
and a comparison of two runs gets an exact sign test on the questions that
changed:

| Observed | Wilson 95% | |
|---|---|---|
| 21 / 25 | 65–94% | the default (11a, Sets A+D) |
| 22 / 25 | 70–96% | 4B reranker (6d) |
| 24 / 25 | 80–99% | best row (10b): sign test vs 6d, 2 better / 0 worse, **p = 0.5** |

No two rows on the ladder are separated with statistical confidence. The
scorecard is treated as **evidence to read** (which question changed, and why)
rather than a number to maximise, and that is why the Evaluation page compares
runs question by question.

---

# How it works: thirteen steps you can open

The numbering follows the app's **How it works** page. Each step has an
interactive page in the app and a doc in [`docs/steps/`](docs/steps).

| # | Step | What the default does | Doc |
|---:|---|---|---|
| 1 | **Extract & parse** | Docling (tables kept as self-describing rows); PyMuPDF kept as a reference, glyph names used to repair "Th" | [01](docs/steps/01-parse.md) |
| 2 | **Clean** | nine ordered rules; every change kept as an inspectable mark; *the document is its own dictionary* for hyphens | [02](docs/steps/02-clean.md) |
| 3 | **Chunk** | structure-aware, ≤ 1,000 chars, never across a page or section, each passage starts with its section label | [03](docs/steps/03-chunk.md) |
| 4 | **Context lines** | built and measured, off by default | [04](docs/steps/04-contextual.md) |
| 5 | **Embed** | EmbeddingGemma 2, 768-d, in-process, pinned revision, exactly reproducible | [05](docs/steps/05-embed.md) |
| 6 | **Store** | LanceDB, exact cosine (8 ms), a word index for BM25, atomic pointer switch | [06](docs/steps/06-store.md) |
| 7 | **Search** | dense top 5; BM25, hybrid, limit per section and reranking are switches | [07](docs/steps/07-search.md) |
| 8 | **Rewrite & multi-step** | Remote AI only, every output cached for replay, off by default | [08](docs/steps/08-rewrite-agent.md) |
| 9 | **Prompt** | v3: six rules, passages fenced as quoted data, hashed | [09](docs/steps/09-prompt.md) |
| 10 | **Generate & cite** | `gemma4:e4b`, temperature 0, thinking off, citations mapped by code | [10](docs/steps/10-generate.md) |
| 11 | **Figures** | 288 charts found, described by `gemma4:12b`, 237 indexed as figure passages | [11](docs/steps/11-figures.md) |
| 12 | **Evaluate** | scorecard, ladder, recall@5, MRR, Wilson, sign test | [12](docs/steps/12-evaluate.md) |
| 13 | **Safety** | hidden text decided by pixels, injection phrases flagged, prompt fence | [13](docs/steps/13-safety.md) |

### The Search lab: three searches fail in three different ways

![The Search lab running question B4 three ways: BM25, hybrid and dense side by side, with the answer page marked in each list](docs/assets/search-lab.webp)

Question B4, *"What is the global growth forecast for 2026?"*, run against the
default index:

- **Dense** (by meaning) finds GDP table rows for Chad, the World and Guinea. It has the
  right topic and the wrong subject. The first answer page is at **rank 8**.
- **BM25** (by words) is distracted by publication lists that repeat "growth"
  and "global". The first answer page is at **rank 13**.
- **Hybrid** puts page 27, *"global growth is forecast to slow this year to 2.5
  percent"*, at **rank 2**, although it was 32nd by meaning and 34th by words.
  Reciprocal rank fusion scores it `1/92 + 1/94 ≈ 0.0215`. Dense's own #1 is
  missing from BM25's list and scores only `1/61 ≈ 0.0164`. A passage that is
  reasonably high in both lists beats one that is first in only one of them.

On the labelled passages the default index uses, hybrid search did not improve
the scorecard (row 4), so it stays a switch. The reranker solves B4 instead.

### Charts become passages

![How it works 11: the iMac carbon-footprint stacked bar chart beside the scattered labels its text layer had and the full description gemma4:12b wrote](docs/assets/figure-before-after.webp)

Some answers exist only as the height of a bar. Before step 11, the default
setup scored Set C **0 · 3 · 1**. Afterwards it scored **4 · 0 · 0**, while Sets A
and D did not change and the traps were still refused. Every Set C answer key
was *measured* from the PDF's vector drawing, with bar heights calibrated
against the axis tick labels. An eyeballed reading at 90 dpi had been 0.4
points off.

---

# Architecture

```
Browser
  │
  ▼
Next.js UI  127.0.0.1:3100   renders only · src/lib/api.ts is the one place that fetches
  │  rewrite /api/*
  ▼
RequestGuard                 pure ASGI: Host / Origin checks, never buffers a stream
  │
  ▼
FastAPI     127.0.0.1:8775   validates, calls the engine, returns · POST /api/ask streams (SSE)
  │
  ▼
rag_studio/                  the engine: every parse, chunk, rank, score and citation
  ├── Ollama 127.0.0.1:11434        gemma4:e4b writes answers
  ├── in-process models             EmbeddingGemma 2 · Qwen3-Reranker 0.6B
  ├── LanceDB  data/lancedb/        one table per build, pointer-file switch
  ├── data/                         manifests, caches, runs, conversations
  └ ─ ─ HTTPS tunnel (optional) ─ ─▶ your Colab T4: rerankers, rewrites, vision
```

**The one rule:** the engine computes, the API routes and the UI renders.
`./rag eval` in the terminal and the Evaluation page in the browser call the
same `eval.execute`. Ask and How it works call the same `retrieve.run` and
`answer.ask`.

A few decisions do most of the work:

- **Every artifact records its configuration.** Index manifests, runs and figure
  descriptions carry the model and its digest, the quantization, the prompt
  version and hash, and every parameter. Two numbers can only be compared if
  their recorded setups say they should be.
- **An index is bound to one embedding setup.** A query with a different model,
  digest or dimension is **refused loudly**, never silently answered with
  another model. A profile's recorded config is frozen: a new default-valued key
  would make every existing build refuse to open.
- **Builds are incremental by content hash and switch atomically.** Each build
  writes a new LanceDB table, verifies it, then swaps a pointer file written to a
  temp path and renamed. Adding 237 figure passages to the default index took
  282 s, because the other 1,473 vectors came from the cache.
- **The streaming contract is honest.** `stage` events are sent only *after* a
  step finishes, never as a prediction. `ready` (every source, the exact
  messages, the token budget) arrives about 0.5 s in, before any model text.
  The SSE writer is hand-written because FastAPI's own cannot send
  `no-transform`, and without it Next's proxy gzips the stream and every event
  arrives at the end.
- **Refuse rather than swap.** This laptop has 15.7 GB of RAM, has no GPU, and
  froze twice while this was being built. Builds, rerankers and answer models
  each check measured free-memory thresholds and refuse with the numbers
  instead of thrashing.

The full trace of one question through every layer, with line numbers, is in
[docs/01-architecture.md](docs/01-architecture.md).

---

# Security and privacy

A RAG system pastes document text straight into a model's prompt, so anyone who
can put a PDF in the library can try to steer the answers. **PDF text is data,
never instructions**, and three layers enforce that:

![How it works 13: a crafted test page through the guard, with five hidden-text tricks dropped and the visible injection sentence kept and flagged](docs/assets/guard-hidden-text.webp)

1. **Hidden text is decided by pixels, not rules.** Each page is rendered
   twice, with and without its text. A span whose pixels do not change is
   invisible to a reader and is dropped, with a reason attached: *too small,
   off the page, drawn invisible, covered, same colour, clipped*. Real hidden
   text was found on 9 pages of the three reports.
2. **Instruction-like text is flagged, never deleted.** Seven pattern families
   (override, own knowledge, role, new rules, reveal, chat markup, AI-addressed) are checked against passages *and* the
   typed question. A report may legitimately quote such a sentence, so it stays
   and the flag is shown in the trace. The three real reports raise zero flags.
3. **Prompt v3 fences the passages** between `<<<PASSAGES … >>>` markers, adds
   a rule that passages are data, and repeats the instruction after the
   question.

The fence fixed the question-borne injection for every model. It did **not**
stop `qwen3:1.7b` obeying an order hidden inside a passage, which `qwen3.5:4b`
and `gemma4:e4b` both ignored. The flag in the trace is the defence that works
for every model, and the docs say so plainly.

| Control | How |
|---|---|
| Loopback only | both servers bind `127.0.0.1` in every launch path; a test fails if one stops |
| DNS rebinding, cross-site writes | `Host` must be `127.0.0.1`, `localhost` or `::1`; a cross-site or foreign-`Origin` write gets HTTP 403 |
| Secrets | the Colab token travels browser → API only, is stored mode `0600` outside the repo, and is stripped from every response |
| File paths | documents are named by `doc_id`; runs and conversations by strict id patterns; no browser-sent path is ever opened |
| Logs | sizes and counts only, never document or answer content; no tracing service, no telemetry |

There is **no login**. That is deliberate: nothing can reach the app from
another machine. Anyone with a shell on this one can use it.

---

# Remote AI on a Colab GPU (optional)

Some jobs are too slow for this CPU: the 4B and 8B rerankers, query rewriting
with `qwen3:8b`, context lines, figure descriptions with `gemma4:12b`, and the
Qwen3 8B embedder. They run on a Colab T4 that you start yourself. **Answers are
always written on the laptop.**

```bash
./rag llm colab --copy   # puts a one-paste setup cell on the clipboard
#  → in Colab (Runtime → T4) paste and run; it prints an rsc1. connect code once
./rag llm connect        # reads the code, requires HTTPS, saves the token 0600, sends a real request
```

The cell starts Ollama behind a gateway that returns 404 for anything off its
allowlist *before* checking auth. It uses a constant-time bearer token, an
8 MB request cap and a Cloudflare tunnel. Before any downloads it checks that
enough GPU memory is free.

**Every Remote AI output is cached and replayed**, including bad ones:
rerank scores, rewrites, multi-step decisions, context lines and figure
descriptions. One bad controller decision came back *valid* when it was asked
again in a later session, which proved that remote LLM output cannot be assumed
to regenerate. The workflow is to fill the caches with a retrieval-only run
while Colab is up, disconnect, and run the scorecards locally. Nothing ever
falls back silently to a different model or machine, because that would make a
run's recorded setup untrue.

Details, including the GPU memory plan: [docs/02-remote-gpu.md](docs/02-remote-gpu.md).

---

# Getting started

| Needs | Version used here |
|---|---|
| Python + [uv](https://docs.astral.sh/uv/) | 3.12 (pinned `>=3.12,<3.13`); `torch` comes from the CPU index (~200 MB, not ~3 GB) |
| Node + pnpm | Node 22, pnpm 11; the first `./rag web` runs `pnpm install` |
| [Ollama](https://ollama.com/) | 0.32.6 at `127.0.0.1:11434` |
| RAM | ~8 GB free for an answer from `gemma4:e4b`; ~7 GB before a scorecard |

```bash
uv sync                                      # .venv from uv.lock
ollama pull gemma4:e4b                       # the answer model (6.6 GB)
ollama pull nomic-embed-text                 # the baseline's embedder
./rag index download google/embeddinggemma-2 # the in-process embedder (1.5 GB, pinned)

./rag ingest                                 # the BASELINE index, ~7 min
./rag figures import docling-structure-embeddinggemma2-fig
./rag index build    docling-structure-embeddinggemma2-fig   # the default, ~30 min the first time
./rag health                                 # "1,710 passages from 3 documents"
./rag web                                    # open http://localhost:3100
```

`./rag ingest` with no profile builds the **baseline** on purpose. On an empty
`data/`, building the default index starts a 15-minute Docling conversion and
needs the committed figure descriptions loaded first, which is what
`figures import` does. A build never calls a model. A `-fig` or `-ctx` profile
whose descriptions or context lines are missing refuses and prints the command
that fixes it.

Prefer the light answer model on a smaller machine: `./rag eval --model
qwen3:1.7b` (5.4 GB, ~20 s an answer), or choose it in Settings.

![System Health: Ollama, the model for each role, documents, indexes, memory and disk, all with one status word each](docs/assets/system-health.webp)

If anything looks wrong, `./rag health` (or **System Health** in the browser,
which runs the same checks) names it and prints the fix.

# Command reference

```
./rag web                      the app: API on 127.0.0.1:8775, UI on 127.0.0.1:3100 (--prod to build and serve)
./rag health                   Ollama, models per role, documents, indexes, memory, disk   [--json]
./rag ingest                   build the baseline index; a no-op when nothing changed      [--profile P] [--force]
./rag index ...                list | build <profile> [--on colab] | check <profile> | download <model>
./rag eval                     the scorecard                       [--set a b c d] [--mode retrieval]
                               [--model M] [--prompt v3] [--search hybrid] [--cap 2] [--rerank qwen3-4b]
                               [--rewrite decompose|multi|hyde] [--agent N] [--label "..."] [--ladder ROW] [--tag RUN_ID]
./rag figures ...              list | status | try | describe | check | export | import
./rag context ...              status | write | check | export | import
./rag rerank ...               download | bench
./rag llm ...                  colab | connect | status | test | disconnect | keepalive
./rag test                     pytest; arguments pass through (e.g. -m "not ollama")

scripts/eval_watchdog.sh       a scorecard that stops itself if free memory drops below 800 MB
scripts/run_progress.sh        a progress bar for the newest run, when the UI is stopped to save memory
```

Always go through `./rag`: it resolves the project's own virtual environment.
`./rag eval --mode retrieval` takes seconds and needs no answer model, so it is
the first thing to run after any change to parsing, cleaning, chunking or search.

---

# Tests

```console
$ ./rag test -q -m "not ollama and not colab and not st and not rerank"
568 passed, 6 deselected in 272.15s (0:04:32)
```

574 tests in 32 modules. The six deselected tests need a real model (Ollama,
EmbeddingGemma 2, the reranker) or a live Colab runtime, and they skip cleanly
when it is missing. The suite proves the claims this README makes:

| Claim | Proved by |
|---|---|
| The baseline is reproduced exactly | 1,190 passages identical to the hand-built ones, the mid-word chunk defect included |
| The scorer is right | fake answers, strict refusals, regex keys with look-arounds (`6.9` must not match `16.9`) |
| The guard works | a crafted PDF: each hidden trick found with its reason; the injection flagged and kept; real report prose not flagged |
| Nothing is reachable from outside | DNS rebinding via `Host` and via the UI proxy, cross-site POST, foreign and `null` `Origin`; launchers pinned to loopback |
| The Colab boundary holds | connect-code round trip, token never returned, allowlist 404 before auth, 8 MB cap, gzip bomb → 413 |
| Indexes are honest | recorded configs frozen, mismatched model refused, atomic switch, incremental reuse, the vector match check fails on drift |
| The UI says one thing per concept | page names from one list, no banned words in copy |

Before each milestone closed, the UI was checked in Chrome DevTools at
375 / 768 / 1024 / 1280 / 1440 px, in light and dark, by keyboard, and with a
clean console. The final matrix was 20 pages × 5 widths × 2 themes, **200
renders, all clean**, with axe-core included. Findings carry ids in
[docs/qa/](docs/qa).

# Project layout

```
rag                 bash launcher: always uses .venv/bin/python
rag_studio/         the engine, 36 modules: parse · clean · guard · structure · chunk · context
                    · vision · embed · store · retrieve · rewrite · agent · rerank · prompt
                    · answer · llm · eval · learn · health · remote · cli …
web/api/            FastAPI: main.py (routes) · guard.py (the security boundary) · sse.py
web/ui/             Next.js 16 · React 19 · Tailwind 4 · shadcn on Base UI
data/pdfs/          the three source PDFs                  (committed)
data/golden/        test questions, sets A–E               (committed)
data/figures/       figure descriptions                    (committed: ~50 min of GPU time)
data/context/       context lines                          (committed: ~1.4 h of GPU time)
data/*              everything else is rebuilt: lancedb/, indexes/, cache/, runs/, chat/
tools/colab/        what runs on the Colab VM: gateway, worker, start.sh
scripts/            run_web.sh · eval_watchdog.sh · run_progress.sh · find_pages.py · qa/
tests/              pytest; markers ollama, colab, st, rerank
docs/               see below
```

# Documentation

| Topic | File |
|---|---|
| 🟢 **New here?** What it is, how to run it, where to look | [docs/00-start-here.md](docs/00-start-here.md) |
| 📖 **The whole project in one page**, zero to hero: RAG from scratch, every step, every decision, every failure | [docs/rag-studio-bible.html](docs/rag-studio-bible.html) (open in a browser) |
| How the pieces fit, the one rule, the security model | [docs/01-architecture.md](docs/01-architecture.md) |
| Every ladder row, every variant, every run id | [docs/ladder.md](docs/ladder.md) |
| One doc per RAG step, mirroring How it works | [docs/steps/](docs/steps) |
| Remote AI: the gateway, the connect code, the GPU memory plan | [docs/02-remote-gpu.md](docs/02-remote-gpu.md) |
| Every choice, with sources and dates | [docs/research.md](docs/research.md) |
| UI QA reports, findings by id | [docs/qa/](docs/qa) |

[`docs/product-language.md`](docs/product-language.md) is normative rather than
explanatory. It fixes one name per concept (*passage*, not chunk, in the UI;
*Remote AI*, not "server"), the status words (Healthy · Running · Warning ·
Needs attention · Unavailable), and the error form *"Short explanation. Action
to recover."* Read it before changing any string a user reads.

---

## Limitations

**The evidence is thin, and the scorecard says so.** There are 25 scored A+D
questions and one run per setup. Every row's Wilson interval overlaps the
others, and keyword scoring proves that a fact is *present*, not that every
claim is supported. The faithfulness judge was researched and then cut.
`gemma4:e4b` is not word-for-word repeatable at temperature 0 (2 of 18
identical prompts were reworded).

**Three documents is a small library.** Techniques that changed nothing here
(multi-step search, query rewriting, context lines, ANN indexes) may matter at
a hundred times the size.

**The answers have known gaps.** Arithmetic across passages is not computed
(D3). A question about two documents needs both of them in the top five (D10's
iMac page ranked 13th). Figure descriptions are approximate readings of a scale,
and stacked bars are hard for vision models.

**It is slow on purpose rather than by accident.** An answer takes about 47 s
and 8 GB of RAM on this CPU, most of it spent reading the prompt. The best
retrieval needs a Colab session.

**The product is not finished.** There is no PDF upload in the browser yet:
copy a PDF into `data/pdfs/` and rebuild. Test questions cannot be edited in
the browser. QA covered Chrome only.

Every answer cites its page so that you can check it, and you should.
