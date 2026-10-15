# 6. Vector store

**What it does:** keeps every passage, its vector and its metadata where the
closest ones can be found quickly.

## Baseline (M1)

`rag_studio/store.py`: **LanceDB 0.40.0**, embedded (a folder,
`data/lancedb/`).

- **One table per build:** `baseline__<UTC timestamp>`. Columns: `chunk_id`,
  `doc_id`, `page`, `section_path` (empty until M2-M3), `type`, `text`, `context`
  (M5), `content_hash`, `vector` (768 × float32), `parser`, `chunker`,
  `char_start`, `char_end`.
- **Atomic switch.** A rebuild writes a new table, checks the row count and that
  a stored vector finds its own passage first, then replaces the profile's
  pointer file (`data/indexes/baseline.json`, written to a temp file and
  renamed). Answers never read a half-built table. The previous build is kept
  for rollback. LanceDB OSS has no `rename_table`, which is why the pointer is
  ours.
- **Incremental by content hash.** A passage whose text is unchanged keeps its
  vector when the embedding digest matches. If nothing changed at all, no build
  is made ("The index is up to date").
- **The manifest** (the pointer file) records everything: profile, parser and
  version, chunker, embedding model + digest + quantization + prefixes, Ollama
  and LanceDB versions, per-document page and passage counts, timings.

## Exact search

At 1,190 passages, comparing the question with every vector takes about
**7-11 ms** (measured in the app's trace), so there is no approximate index to
lose recall to. The metric is always set to **cosine**: LanceDB's default is
squared L2.

## M3: the word index, versions, and why search stays exact

**Two more columns.** `label` (the section label, null when a chunker adds
none) and `search_text` (exactly what was embedded: label + passage). Builds
made before M3 have neither; a read fills `label` with null.

**The word index.** Every build gets LanceDB's native full-text index over
`search_text` (or `text` on older builds) for BM25 ("by words", see
`07-search.md`). It takes about 0.1 s to build and adds no row: on the
baseline build, LanceDB's version history shows **v1** (the table written) and
**v2** (the word index added), both with 1,190 rows. Adding it to a build made
before M3 keeps the build's name and its vectors, so earlier runs still point at
the table they used. The manifest records it under `word_index` (index name,
column, settings, LanceDB version, table version).

One side effect to know: running `./rag ingest` on the baseline after M3 made a
new baseline build (`baseline__20261015T080126…`), because the M1 manifest had
recorded no document titles and titles are part of the document list a build
compares. Every vector was reused (0 embedded), the passages are identical,
and the M1 build is kept beside it, so row 1's runs still point at a table that
exists.

**Exact or approximate?** An ANN index trades recall for speed. On this
corpus it buys nothing. `learn.ann_check()` (the "Measure again" button on
How it works, step 6) builds each index on a **scratch copy** of a build's
vectors and asks the 31 test questions, top 10, against exact search:

| Index | Recall@10 vs exact | Median query | Build |
|---|---|---|---|
| Exact (flat scan) | 1.000 | 3.95 ms | - |
| IVF_FLAT | 1.000 | 2.66 ms | 0.05 s |
| IVF_HNSW_SQ | 0.987 | 1.81 ms | 0.09 s |
| IVF_PQ | **0.794** | 2.30 ms | 0.20 s |

Measured 2026-10-15 16:13 IST with Ollama otherwise idle, on the baseline build
(1,190 × 768, nomic-embed-text), LanceDB 0.40.0, default index settings,
cosine. An earlier measurement while an index build ran gave IVF_PQ 0.800 and
slower times (exact 4.9 ms): PQ's training is randomised, so its recall moves a
little from build to build. IVF_PQ loses one passage in five from the top 10
to save under 2 ms. At about 1,200 passages the app never makes
that trade; ANN appears only on How it works, with its measured recall beside
exact.
