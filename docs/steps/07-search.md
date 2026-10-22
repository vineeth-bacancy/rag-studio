# 7. Search

**What it does:** finds the passages most likely to answer a question.

## Baseline (M1)

`rag_studio/retrieve.py`: **dense search only** ("by meaning"). Embed the
question with `search_query: `, take the 5 nearest passages by cosine
similarity, exactly. Scores bunch between 0.6 and 0.9 on this corpus, so the
**rank** says more than the number.

## Measured: recall@5 on the baseline (2026-10-15)

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

## M3: BM25, hybrid and the limit per section

Three methods, each a switch (`./rag eval --search dense|bm25|hybrid`, the
Search picker on Ask and Evaluation):

- **Dense** ("by meaning"): as above.
- **BM25** ("by words"): LanceDB's native full-text index, the **word index**,
  over exactly the text that was embedded (section label + passage; the text
  alone for builds made before M3). Settings: simple tokenizer, English
  stemming, **stop words kept** (BM25's IDF already gives "the" almost no
  weight, and removing them is a choice the scorecard should make), lower
  case, accents folded, positions on.
- **Hybrid:** the dense list and the BM25 list, **50 deep each**, fused with
  **reciprocal rank fusion**: each passage scores Σ 1 / (60 + rank) over the
  lists it is in (1-based ranks). Only ranks count; cosine (0.6-0.9) and BM25
  (0-30) live on different scales, so adding the raw scores would let one list
  drown the other.

We fuse ourselves rather than use LanceDB's hybrid query, which returns only
the fused list. The trace, the Search lab and every run record where each
passage ranked in each list. `tests/test_search.py` checks our scores against
LanceDB's own `RRFReranker` on the same two lists. Ties keep the better dense
rank, then the better BM25 rank.

**Limit per section** (`--cap N`): after ranking, a passage moves behind the
others when its section already has N passages above it. Contents pages and
reference lists count as one section per document, capped at 1. Nothing is
dropped, so a 50-deep run still finds a moved passage. It targets spec §5.3
(GEP's Chapter 3 bibliography outranking the chapter's own summary). Passages
without a known section (the baseline index) are never moved.

**The first k of a deeper search are the same passages as a search with k**
for every method (tested), so a run can search 50 deep for the "first answer
page" rank and score the top 5.

The word index costs nothing to add to an existing build: no row or vector
changes, the build keeps its name, and earlier runs stay valid. `./rag ingest
--profile <p>` adds it to any build that lacks it.

Latency, median over the 31 test questions, 50 deep (the depth a run
searches), 1,098-1,473 passages, measured 2026-10-15 on this CPU:

| Step | Median |
|---|---|
| Embed the question (nomic-embed-text) | 66-89 ms |
| Dense search (exact cosine) | 7.8-9.7 ms |
| BM25 (word index) | 5.3-6.6 ms |
| Hybrid (both lists + RRF) | 14.9-18.3 ms |
| Limit per section | < 0.5 ms |
| Word index build (whole index) | 0.1-0.2 s |

Hybrid search costs about 8 ms more than dense per question: noise next to the
question's embedding, and nothing next to the answer model's 20-60 s.

## Measured: M3 retrieval

See [`docs/ladder.md`](../ladder.md), rows 3 and 4, for every index × method.

## M5: the reranker

A first-stage search compares one vector (or one bag of words) for the
question with one for each passage; neither has read the two together. A
**reranker** (a cross-encoder) does: it reads the question and a passage as one
input and answers "does this passage answer the question?". That is far slower
per passage, so it only re-reads the **dense top 20** and re-orders them; the
top 5 go to the answer model. Ranks 21-50 keep their order, so a search 5 deep
is still the first 5 of a search 50 deep.

- **Model:** Qwen3-Reranker (research §6), pinned by Hugging Face revision:
  0.6B on this computer (fp32, sentence-transformers `CrossEncoder`), and 0.6B
  fp16, 4B fp16 and 8B 8-bit on the Colab GPU worker. One instruction for all,
  recorded with its sha256: "Given a question about reports, retrieve the
  passage that states the answer". The score is the logit of "yes" minus "no"
  (only the order matters), with a NaN/inf guard.
- **What it reads:** the passage's `search_text` (section label + text), the
  same text that was embedded, against the question as asked.
- **Where it sits:** last in `retrieve.run()`: rewrite → search each query →
  fuse → limit per section → multi-step rounds → **rerank the top 20** → cut to
  k. Ask, evaluation and the search lab all call this one function.
- **Remembered scores:** `data/cache/rerank/` keeps every score by reranker,
  question and passages. A retrieval-only run on Colab fills it; the scorecard
  after it replays the same scores with Colab off. Each result records hit or
  miss; `--fresh-rerank` forces a new call (for timing).
- **Memory:** the 0.6B needs about 4 GB free beside EmbeddingGemma 2 in the
  same process (`RERANK_LOAD_GB`); it refuses with how to recover when that is
  not there. An evaluation releases both models before the answer model runs.

**Measured (row 6, `docs/ladder.md`):** on the default Docling index the
answer to B4 ("global growth") moves from rank 8 (a miss) to **rank 1** with
every size; on the baseline from 11 to 1-2. Docling + the 4B scores
**22 · 1 · 2** (row 5d: 20 · 3 · 2), the best row on the ladder. Cost per
question at depth 20: 39 s on this CPU (0.6B), 0.65 s / 3.3 s / 5.5 s of T4
time (0.6B / 4B / 8B) plus about 0.3 s of network.

In the Search lab, **Rerank** adds the fourth column on request (20-35 s here
the first time, a few seconds on Remote AI), and "Lines: Fusion | Reranking"
switches which neighbouring columns are joined.
