# 7. Search

**What it does:** finds the passages most likely to answer a question.

## Baseline (M1)

`rag_studio/retrieve.py`: **dense search only** ("by meaning"). Embed the
question with `search_query: `, take the 5 nearest passages by cosine
similarity, exactly. Scores bunch between 0.6 and 0.9 on this corpus, so the
**rank** says more than the number.

## Measured: recall@5 (2026-10-15)

| Set | Hits | Misses |
|---|---|---|
| A (your 9) | **8/9** | A6 "According to the executive summary…": first answer page at **rank 45** |
| B (retrieval set) | **5/6** | B4 "global growth forecast for 2026": first answer page at **rank 11** |
| D (edge cases) | **9/11** | D6 Thailand's forecast (rank 6), D9 the Spanish question (rank 9) |

These are the hand-built project's numbers exactly (spec §7), including both
ranks. A run searches 50 deep, so a Miss always says where the answer page
really was.

Why the misses happen:
- **A6:** the risks paragraph never says "executive summary"; only the passage
  before it does. Passages carry no section label yet.
- **B4:** dense search matches the *topic* (growth), not the *subject*
  (global): the top 5 are U.S. and euro-area growth.
- **D6:** the regional chapter's prose about East Asia outranks the three
  tables that hold Thailand's number.
- **D9:** nomic-embed-text is trained mostly on English.

## Next

M3: hybrid search (BM25 + dense, fused with our own RRF) and a per-section cap.
M5: the reranker and query rewriting. Each must turn a Miss into a Hit on the
scorecard to become a default.
