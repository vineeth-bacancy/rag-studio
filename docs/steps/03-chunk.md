# 3. Chunk

**What it does:** cuts each page's clean text into **passages** small enough to
search and to quote.

## Baseline (M1)

`rag_studio/chunk.py`, chunker id `fixed-1000-150`: the hand-built chunker,
line for line.

- 1,000 characters with 150 of overlap; never crosses a page.
- The end moves back to the last space, so a passage never ends mid-word.
- `chunk_id = "{doc_id}:p{page}:c{i}"`.
- **The known defect is kept on purpose:** the next start is `end - overlap`,
  not moved to a space, so passages often start mid-word ("n the upside",
  "agement (LSCM)"). The baseline must keep it to be an honest "before" row.

| Measured | Value |
|---|---|
| Passages | **1,190** (iMac 34, GPO 233, GEP 923) |
| Identical to `advanced-rag/outputs/chunks.jsonl` | yes: every id and every character (sha256 fingerprint pinned in `tests/test_baseline.py`) |

Each passage also stores its character span on the page (`char_start`,
`char_end`), which the page preview uses to outline it.

## Next (M3)

The word-boundary fix as a separate chunker, structure-aware passages with
section labels, and semantic chunking as an option. Each one is a scorecard row.
