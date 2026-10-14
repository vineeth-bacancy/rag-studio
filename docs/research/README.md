# Research appendices (2026-10-14)

The evidence behind `docs/research.md`. Each report was written by a research
pass for M0 and is kept unedited, so every table row still carries its status
label and source.

| Report | Covers |
|---|---|
| [A-ollama-llms.md](A-ollama-llms.md) | Ollama API on 0.32.6 (think, images, embed fields, num_ctx truncation, rerank), qwen3.5:4b measured live, Gemma 4 vs Qwen 3.5 |
| [B-embeddings-rerank.md](B-embeddings-rerank.md) | Qwen3-Embedding, EmbeddingGemma 2, Qwen3-Reranker, the M4 profile order and manifest fields |
| [C-docling-lancedb.md](C-docling-lancedb.md) | Docling API and the 15-defect hypotheses, LanceDB measured live (exact vs ANN, FTS, hybrid, versions) |
| [D-eval-tracing-techniques.md](D-eval-tracing-techniques.md) | RAGAS vs DeepEval, Phoenix vs Langfuse, and the evidence for each retrieval technique |

**Paths in the reports.** They were written in a temporary scratch folder, so
they cite scripts as `scratchpad/research/work/…`, `scratchpad/research/scripts/…`,
`scratchpad/scripts/…` or `scratchpad/lancetest/scripts/…`. All of those scripts
are now in [`scripts/`](scripts/); some still hardcode the old scratch path and
need it edited before a re-run. `run_isolated.sh` and `netguard.py` are the
loopback-only network namespace and socket guard used for the telemetry tests. Raw outputs, downloaded third-party pages and the LanceDB
test data were not kept; the reports quote the measured numbers and cite the
original URLs.

The scripts are probes, not part of the app. Run them in a throwaway
environment, for example
`uv run --no-project --with lancedb==0.40.0 --with pyarrow --with numpy python lance_probe.py /tmp/lancetest`.
