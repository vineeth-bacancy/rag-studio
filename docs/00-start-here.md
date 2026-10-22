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
./rag ingest --profile pymupdf-clean   # PyMuPDF + the full cleaning order and the guard (about 6 min)
./rag ingest --profile docling-clean   # Docling + cleaning (about 15 min of Docling the first time, then cached)
./rag ingest --profile pymupdf-structure   # structure-aware passages with section labels (about 7 min)
./rag ingest --profile docling-structure   # the same on Docling (about 8 min; needs docling-clean's conversion)
./rag index build docling-structure-embeddinggemma2   # the app's default index (row 5d; about 30 min the first time)
./rag eval --set a b      # the scorecard: answer and score the test questions (about 5 min on qwen3:1.7b)
./rag eval --mode retrieval   # recall@5 only, no model: a few seconds
./rag eval --search hybrid --cap 2   # search by meaning and by words (RRF), at most 2 passages per section
./rag index list          # every index profile: its model, where it builds, whether it is built
./rag index build docling-structure-embeddinggemma2   # EmbeddingGemma 2, in this app (about 17 min)
./rag index build docling-structure-qwen3-8b --on colab   # on a connected Colab GPU
./rag index check docling-structure-qwen3-0.6b         # the vector match check (this computer vs Colab)
./rag index download google/embeddinggemma-2           # the one model fetched from Hugging Face (1.5 GB)
./rag eval --rerank qwen3-0.6b            # rerank the top 20 here (39 s a question); qwen3-4b on Remote AI
./rag eval --rewrite multi --rerank qwen3-4b   # Remote AI only: decompose | multi | hyde; --agent 3 for multi-step search
./rag eval --set e         # follow-up questions (Remote AI); --as-typed searches them without the rewrite
./rag rerank download     # the local reranker, Qwen3-Reranker-0.6B (1.2 GB)
./rag context status docling-structure-embeddinggemma2-ctx   # context lines: status | write | check | export | import
./rag llm colab --copy    # the Colab setup cell, on the clipboard (docs/02-remote-gpu.md)
./rag llm colab --models qwen3:8b --worker context qwen3-0.6b-gpu qwen3-4b --copy   # rewriting, context lines, GPU rerankers
./rag llm connect         # connect to the runtime with the code it showed (reads the clipboard)
./rag llm status          # the runtime's GPU and models;  ./rag llm test  ·  ./rag llm disconnect
./rag health              # the same checks as the System Health page, in the terminal
./rag test                # the test suite
```

The first `./rag web` installs the frontend packages. The Python environment is
`uv sync`. `./rag eval --help` lists the options (`--profile`, `--model`,
`--prompt` (default v3), `--search`, `--cap`, `--rerank`, `--rewrite`, `--agent`, `--as-typed`,
`--fresh-rerank`, `--label`, `--ladder ROW`, `--tag RUN_ID`). Every Remote AI output (rerank scores,
rewrites, decisions, context lines) is remembered, so a scorecard replays a retrieval run's with Colab off.
Every index profile is listed by `./rag ingest --help`; `./rag ingest` builds the baseline unless told otherwise, and also
adds the word index (for BM25) to a build that lacks one, in place.

## The pages

| Page | What it is for |
|---|---|
| **Ask** | A conversation: ask, get a streamed answer with `[n]` citations that open the cited page, outlined; follow-up questions are made whole by Remote AI ("Searched for: …", Ask as typed). Switches for the reranker, query rewriting and multi-step search; conversations are saved on this computer |
| **Documents** | The PDF library and the page inspector: any page beside PyMuPDF's raw and cleaned text and Docling's, every rule's change highlighted, hidden text outlined |
| **How it works** | One interactive section per RAG step, on your own documents. Interactive now: steps 1-8 (extract and parse, clean, the chunk playground, contextual retrieval, embed, the vector store, the search lab with its reranked column, query rewriting and multi-step search) |
| **Evaluation** | Test questions, runs with every question's evidence, and the upgrade ladder |
| **Indexes** | Every index profile: build on this computer or on the Colab GPU with live progress, the vector match check, the earlier builds |
| **Settings** | Model backend (Local AI, or Remote AI on a Colab GPU: the setup cell for each job and Connect), which model does each job and where, and the defaults |
| **System Health** | Ollama, models per role, loaded models, documents, indexes, memory, disk |

## Milestones

| M | Deliverable | Status |
|---|---|---|
| M0 | Repo, `./rag`, the app shell with the trading design system, System Health, `docs/research.md` | **Done** (2026-10-14) |
| M1 | Baseline parity: PyMuPDF, NFKC, fixed chunks, nomic, exact search in LanceDB, prompt v2, the scorecard; Ask and Evaluation. Ladder row 1 | **Done** (2026-10-15) |
| M2 | Docling beside PyMuPDF, the cleaning order, sections, reference tagging, the injection guard; the page inspector. Ladder row 2 | **Done** (2026-10-15) |
| M3 | Word-boundary chunks, structure-aware chunks with section labels, hybrid search, a per-section cap; How it works 1-3, 6, 7. Ladder rows 3-4 | **Done** (2026-10-15) |
| M4 | Colab gateway and connect code, index profiles (Qwen3-Embedding 0.6B, EmbeddingGemma 2, Qwen3-Embedding 8B on Colab), the Indexes page, the vector match check, How it works 5. Ladder row 5 | **Done** (2026-10-15) |
| M5 | Reranker (0.6B here, 4B/8B on Colab), context lines, query rewriting, multi-step search, conversations with follow-up rewriting; the Search lab's reranked column; How it works 4 and 8. Ladder rows 6-9, Set E | **Done** (2026-10-22) |
| M6 | Answer models qwen3.5:4b and gemma4:e4b (thinking off verified), ladder row 10; How it works 9, 10, 12, 13 | **Done** (2026-10-22; gemma4:e4b 24 · 1 · 0. The judge, RAGAS and tracing were cut) |
| M7 | Figures as vision descriptions (Set C), ladder row 11; How it works 11 | Next (image embeddings cut) |
| M8 | Full QA, polish, complete docs | Planned (browser upload deferred to a later version) |

The reduced plan for M6-M8 (decided 2026-10-22) is `docs/plans/m6-m8.md`.

## Where things are

- `CLAUDE.md`: the engineering rules and the gotchas, read by Claude Code.
- `docs/01-architecture.md`: the processes, the engine, the API, the security model.
- `docs/02-remote-gpu.md`: Remote AI on a Colab GPU: the gateway, the connect code, the vector match check.
- `docs/steps/`: one doc per RAG step, the baseline as built and its measured numbers.
- `docs/ladder.md`: the scorecard ladder, one row per upgrade.
- `docs/research.md`: every technology choice, with sources, dates and what is still unverified.
- `docs/product-language.md`: what every concept is called. Normative.
- `docs/qa/`: browser QA reports per milestone, with finding IDs.
- The build spec: `~/Dev/RAG/advanced-rag/RAG_STUDIO_PROMPT.md`.
