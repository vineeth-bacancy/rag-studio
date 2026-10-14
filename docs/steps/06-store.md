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

## Next

M3 adds the full-text (BM25) index for hybrid search; the Search lab shows an
ANN index only as a teaching toggle, with its measured recall beside exact.
