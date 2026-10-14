# Start here

RAG Studio answers questions about PDF reports with page-level citations, and
shows how every step of retrieval-augmented generation works on your own
documents. It runs on this laptop (Ollama on the CPU), with an optional Colab
T4 GPU for the heavy jobs.

It is the upgrade of the RAG you built by hand in `~/Dev/RAG/advanced-rag`. That
pipeline (PyMuPDF, fixed-size chunks, `nomic-embed-text`, dense search,
`qwen3:1.7b`) is reproduced once as the measured **baseline**, then every stage
is upgraded and measured on the same scorecard. An upgrade becomes a default
only if the scorecard says it helps.

## Run it

```fish
./rag web                 # open http://localhost:3100
./rag ingest              # build the search index (about 7 min the first time; instant when nothing changed)
./rag eval --set a b      # the scorecard: answer and score the test questions (about 5 min on qwen3:1.7b)
./rag eval --mode retrieval   # recall@5 only, no model: about a second
./rag health              # the same checks as the System Health page, in the terminal
./rag test                # the test suite
```

The first `./rag web` installs the frontend packages. The Python environment is
`uv sync`. `./rag eval --help` lists the options (`--model`, `--prompt`,
`--label`, `--ladder ROW`, `--tag RUN_ID`).

## The pages

| Page | What it is for |
|---|---|
| **Ask** | Ask a question; get a streamed answer with `[n]` citations that open the cited page, outlined |
| **Documents** | The PDF library and the page inspector: each parser's text beside the page, defects highlighted (M2) |
| **How it works** | One interactive section per RAG step, on your own documents (M3-M7) |
| **Evaluation** | Test questions, runs with every question's evidence, and the upgrade ladder |
| **Indexes** | Index profiles: build locally or on Colab, versions, local-vs-Colab vector check (M4) |
| **Settings** | Which model does each job, Local AI or Remote AI, technique switches |
| **System Health** | Ollama, models per role, loaded models, documents, indexes, memory, disk |

## Milestones

| M | Deliverable | Status |
|---|---|---|
| M0 | Repo, `./rag`, the app shell with the trading design system, System Health, `docs/research.md` | **Done** (2026-10-14) |
| M1 | Baseline parity: PyMuPDF, NFKC, fixed chunks, nomic, exact search in LanceDB, prompt v2, the scorecard; Ask and Evaluation. Ladder row 1 | **Done** (2026-10-15) |
| M2 | Docling beside PyMuPDF, the cleaning order, sections, reference tagging, the injection guard; the page inspector. Ladder row 2 | Next |
| M3 | Word-boundary chunks, structure-aware chunks with section labels, hybrid search, a per-section cap; How it works 1-3, 6, 7. Ladder rows 3-4 | |
| M4 | Colab gateway and connect code, index profiles (Qwen3-Embedding 0.6B local, 4B/8B Colab, EmbeddingGemma 2), the Indexes page. Ladder row 5 | |
| M5 | Reranker, contextual retrieval, query rewriting, the agentic loop, chat memory; the Search lab. Ladder rows 6-9 | |
| M6 | New answer models (thinking off verified), the judge, RAGAS or DeepEval, tracing. Ladder row 10 | |
| M7 | Figures: vision descriptions and image embeddings. Ladder row 11 | |
| M8 | Upload any PDF end to end, polish, full QA, complete docs | |

## Where things are

- `CLAUDE.md`: the engineering rules and the gotchas, read by Claude Code.
- `docs/01-architecture.md`: the processes, the engine, the API, the security model.
- `docs/steps/`: one doc per RAG step, the baseline as built and its measured numbers.
- `docs/ladder.md`: the scorecard ladder, one row per upgrade.
- `docs/research.md`: every technology choice, with sources, dates and what is still unverified.
- `docs/product-language.md`: what every concept is called. Normative.
- `docs/qa/`: browser QA reports per milestone, with finding IDs.
- The build spec: `~/Dev/RAG/advanced-rag/RAG_STUDIO_PROMPT.md`.
